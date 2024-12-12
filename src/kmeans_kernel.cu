#include "kmeans.h"
#include <cuda_runtime.h>
// #include <cuda_occupancy.h>
#include <iostream>
#include "argparse.h"

// helper function that uses CUDA occupancy API to calculate optimal block size.
// template<typename T>
// int calculateOptimalBlockSize(T kernel, int sharedMemSize = 0, int maxThreadsPerBlock = 1024) {
//     int blockSize = 0;
//     int minGridSize = 0;
//     cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, kernel, sharedMemSize, maxThreadsPerBlock);
//     return blockSize;
// }

// CUDA kernel for assigning points to clusters
__global__ void assignPointsToClustersKernel(const double* data, const double* centroids, 
                                             int* assignments, int numPoints, int numClusters, 
                                             int numDimensions) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numPoints) {
        double minDistance = INFINITY;
        int closestCentroid = 0;
        
        for (int j = 0; j < numClusters; ++j) {
            double distance = 0.0;
            for (int d = 0; d < numDimensions; ++d) {
                double diff = data[idx * numDimensions + d] - centroids[j * numDimensions + d];
                distance += diff * diff;
            }
            if (distance < minDistance) {
                minDistance = distance;
                closestCentroid = j;
            }
        }
        assignments[idx] = closestCentroid;
    }

    // if (idx == 1){
    //     printf("Number of clusters: %d\n", numClusters);
    //     printf("Print Point %d: ", idx);
    //     for (int d = 0; d < numDimensions; d++){
    //         printf(" %lf", data[idx * numDimensions + d]);
    //     }
    //     for (int j = 0; j < numClusters; ++j){
    //         printf("\n centroid %d", j);
    //         for (int d = 0; d <numDimensions; ++d){
    //             printf(" %lf", centroids[j * numDimensions + d]);
    //         }
    //     }

    //     printf("\n Assignment: ");
    //     printf("%d\n", assignments[idx]);
    // }
}

// CUDA kernel for updating centroids
__global__ void updateCentroidsKernel(const double* data, double* centroids, 
                                      const int* assignments, int* clusterSizes,
                                      int numPoints, int numClusters, int numDimensions) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numPoints) {
        int cluster = assignments[idx];
        atomicAdd(&clusterSizes[cluster], 1);
        for (int d = 0; d < numDimensions; ++d) {
            atomicAdd(&centroids[cluster * numDimensions + d], data[idx * numDimensions + d]);
        }
    }
}

// CUDA kernel for normalizing centroids
__global__ void normalizeCentroidsKernel(double* centroids, const int* clusterSizes,
                                         int numClusters, int numDimensions) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numClusters) {
        if (clusterSizes[idx] > 0) {
            for (int d = 0; d < numDimensions; ++d) {
                centroids[idx * numDimensions + d] /= clusterSizes[idx];
            }
        }
    }
}

// CUDA kernel for calculating convergence check
__global__ void convergenceCheckKernel(const double* old_centroids, const double* centroids, 
				       int numClusters, int numDimensions, double threshold,
                                       int* hasConverged) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numClusters) {
        double squaredDistance = 0.0;
        for (int d = 0; d < numDimensions; ++d) {
            double diff = old_centroids[idx * numDimensions + d] - centroids[idx * numDimensions + d];
            squaredDistance += diff * diff;
        }
        if (squaredDistance > threshold * threshold) {
            *hasConverged = 0;
        }
    }
}

// Host function to compute K-means using CUDA
void computeKmeansCUDA(const std::vector<double>& data, 
                       std::vector<double>& centroids, 
                       std::vector<int>& assignments,
                       struct options_t *opts) {

    std::cout << "Executing computeKmeansCUDA" << std::endl;

    int numPoints = opts->num_points;
    int numDimensions = opts->dim;
    int numClusters = opts->num_cluster;
    cudaEvent_t start, stop;
    float milliseconds = 0;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Initialize centroids randomly
    std::cout << "Initializing centroids randomly" << std::endl;
    for (int i = 0; i < numClusters; i++){
        int index = kmeans_rand() % numPoints;
        for (int d = 0; d < numDimensions; ++d) {
            centroids[i * numDimensions + d] = data[index * numDimensions + d];
        }
    }

    // Allocate device memory
    double *d_data, *d_centroids, *d_old_centroids;
    int *d_assignments, *d_clusterSizes, *d_hasConverged;
    cudaMalloc(&d_data, numPoints * numDimensions * sizeof(double));
    cudaMalloc(&d_centroids, numClusters * numDimensions * sizeof(double));
    cudaMalloc(&d_old_centroids, numClusters * numDimensions * sizeof(double));
    cudaMalloc(&d_assignments, numPoints * sizeof(int));
    cudaMalloc(&d_clusterSizes, numClusters * sizeof(int));
    cudaMalloc(&d_hasConverged, sizeof(int));

    // Copy data to device
    cudaMemcpy(d_data, data.data(), numPoints * numDimensions * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_centroids, centroids.data(), numClusters * numDimensions * sizeof(double), cudaMemcpyHostToDevice);

    // Main K-means loop
    cudaEventRecord(start);
    int iteration = 0;
    for (int iter = 0; iter < opts->max_num_iter; ++iter) {

        //std::cout << "Iteration " << iter << std::endl;
        // Assign points to clusters
        int blockSize = 1024;
        int numBlocks = (numPoints + blockSize - 1) / blockSize;
        assignPointsToClustersKernel<<<numBlocks, blockSize>>>(d_data, d_centroids, d_assignments, 
                                                               numPoints, numClusters, numDimensions);
	    // Copy d_centroids to d_old_centroids
        cudaMemcpy(d_old_centroids, d_centroids, numClusters * numDimensions * sizeof(double), cudaMemcpyDeviceToDevice);
       	// Reset cluster sizes and centroids
        cudaMemset(d_clusterSizes, 0, numClusters * sizeof(int));
        cudaMemset(d_centroids, 0, numClusters * numDimensions * sizeof(double));

        // Update centroids
        updateCentroidsKernel<<<numBlocks, blockSize>>>(d_data, d_centroids, d_assignments, d_clusterSizes,
                                                        numPoints, numClusters, numDimensions);

        // Normalize centroids
        int centroidBlocks = (numClusters + blockSize - 1) / blockSize;
        normalizeCentroidsKernel<<<centroidBlocks, blockSize>>>(d_centroids, d_clusterSizes, 
                                                                numClusters, numDimensions);

        // Check for convergence
	
	    cudaMemset(d_hasConverged, 1, sizeof(int));
        convergenceCheckKernel<<<centroidBlocks, blockSize>>>(d_old_centroids, d_centroids, 
                                                              numClusters, numDimensions, 
                                                              opts->threshold, d_hasConverged);

        int hasConverged;
        cudaMemcpy(&hasConverged, d_hasConverged, sizeof(int), cudaMemcpyDeviceToHost);
        if (hasConverged) {
            iteration = iter + 1;
            std::cout << "Basic CUDA converged after " << iteration << " iterations." << std::endl;
            break;
        }
	
    }
    cudaEventRecord(stop);

    // Synchronize and calculate the elapsed time
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);

    // Print the execution time
    if (iteration == 0){
        std::cout << "Basic CUDA didn't converge after " << opts->max_num_iter << " iterations." << std::endl;
        iteration = opts->max_num_iter;
    }
    printf("K-means main loop execution time (CUDA): %.3f ms\n", milliseconds);
    float time_per_iter_in_ms = milliseconds / float(iteration);
    printf("Time per iteration: %.3f ms\n", time_per_iter_in_ms);

    // Copy results back to host
    cudaMemcpy(assignments.data(), d_assignments, numPoints * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(centroids.data(), d_centroids, numClusters * numDimensions * sizeof(double), cudaMemcpyDeviceToHost);
    

    // Free device memory
    cudaFree(d_data);
    cudaFree(d_centroids);
    cudaFree(d_old_centroids);
    cudaFree(d_assignments);
    cudaFree(d_clusterSizes);
    cudaFree(d_hasConverged);

}
