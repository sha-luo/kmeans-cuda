#include "kmeans.h"
#include <cuda_runtime.h>
#include <cuda_occupancy.h>
#include <iostream>
#include "argparse.h"
 
// CUDA kernel for assigning points to clusters
__global__ void assignPointsToClustersKernel_shmem(const double* data, const double* centroids, 
                                             int* assignments, int numPoints, int numClusters, 
                                             int numDimensions) {
    
    extern __shared__ double sharedCentroids[];

    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Cooperatively load centroids into shared memory
    for (int i = threadIdx.x; i < numClusters * numDimensions; i += blockDim.x) {
        sharedCentroids[i] = centroids[i];
    }
    __syncthreads();

    if (idx < numPoints) {
        double minDistance = INFINITY;
        int closestCentroid = 0;
        
        for (int j = 0; j < numClusters; ++j) {
            double distance = 0.0;
            for (int d = 0; d < numDimensions; ++d) {
                double diff = data[idx * numDimensions + d] - sharedCentroids[j * numDimensions + d];
                distance += diff * diff;
            }
            if (distance < minDistance) {
                minDistance = distance;
                closestCentroid = j;
            }
        }
        assignments[idx] = closestCentroid;
    }

}

// CUDA kernel for updating centroids
extern __global__ void updateCentroidsKernel(const double* data, double* centroids, 
                                      const int* assignments, int* clusterSizes,
                                      int numPoints, int numClusters, int numDimensions);

__global__ void updateCentroidsKernel_shmem(const double* data, double* centroids, 
                                      const int* assignments, int* clusterSizes,
                                      int numPoints, int numClusters, int numDimensions) {

    extern __shared__ double sharedMem[];
    double* sharedCentroids = sharedMem;
    int* sharedClusterSizes = (int*)&sharedMem[numClusters * numDimensions];

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Initialize shared memory
    for (int i = threadIdx.x; i < numClusters * numDimensions; i += blockDim.x) {
        sharedCentroids[i] = 0;
    }
    for (int c = threadIdx.x; c < numClusters; c += blockDim.x) {
        sharedClusterSizes[c] = 0;
    }
    __syncthreads();

    // Process points
    if (idx < numPoints) {
        int cluster = assignments[idx];
        atomicAdd(&sharedClusterSizes[cluster], 1);
        for (int d = 0; d < numDimensions; d++) {
            atomicAdd(&sharedCentroids[cluster * numDimensions + d], data[idx * numDimensions + d]);
        }
    }
    __syncthreads();

    // Accumulate results to global memory
    for (int c = threadIdx.x; c < numClusters; c += blockDim.x) {
        atomicAdd(&clusterSizes[c], sharedClusterSizes[c]);
        // for (int d = 0; d < numDimensions; d++) {
        //     atomicAdd(&centroids[c * numDimensions + d], sharedCentroids[c * numDimensions + d]);
        // }
    }

    for (int i = threadIdx.x; i < numClusters * numDimensions; i += blockDim.x) {
        atomicAdd(&centroids[i], sharedCentroids[i]);
    }
}

// CUDA kernel for normalizing centroids
extern __global__ void normalizeCentroidsKernel(double* centroids, const int* clusterSizes,
                                         int numClusters, int numDimensions);

__global__ void normalizeCentroidsKernel_shmem(double* centroids, const int* clusterSizes,
                                         int numClusters, int numDimensions) {
    extern __shared__ double sharedMem[];
    double* sharedCentroids = sharedMem;
    int* sharedClusterSizes = (int*)&sharedMem[numClusters * numDimensions];

    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Load centroids and cluster sizes into shared memory
    for (int i = threadIdx.x; i < numClusters * numDimensions; i += blockDim.x) {
        sharedCentroids[i] = centroids[i];
    }
    for (int c = threadIdx.x; c < numClusters; c += blockDim.x) {
        sharedClusterSizes[threadIdx.x] = clusterSizes[threadIdx.x];
    }
    __syncthreads();

    // Normalize centroids
    if (idx < numClusters) {
        if (sharedClusterSizes[idx] > 0) {
            for (int d = 0; d < numDimensions; ++d) {
                sharedCentroids[idx * numDimensions + d] /= sharedClusterSizes[idx];
            }
        }
    }
    __syncthreads();

    // Write back to global memory
    for (int i = threadIdx.x; i < numClusters * numDimensions; i += blockDim.x) {
        centroids[i] = sharedCentroids[i];
    }
}

// CUDA kernel for calculating convergence check
extern __global__ void convergenceCheckKernel(const double* old_centroids, const double* centroids, 
				       int numClusters, int numDimensions, double threshold,
                                       int* hasConverged);
                                        
__global__ void convergenceCheckKernel_shmem(const double* old_centroids, const double* centroids, 
				       int numClusters, int numDimensions, double threshold,
                                       int* hasConverged) {
    extern __shared__ double sharedMem[];
    // double* sharedOldCentroids = sharedMem;
    // double* sharedCentroids = &sharedMem[numClusters * numDimensions];

    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // // Cooperatively load centroids into shared memory
    // for (int i = threadIdx.x; i < numClusters * numDimensions; i += blockDim.x) {
    //     sharedOldCentroids[i] = old_centroids[i];
    //     sharedCentroids[i] = centroids[i];
    // }
    // __syncthreads();

    if (idx < numClusters) {
        double squaredDistance = 0.0;
        for (int d = 0; d < numDimensions; ++d) {
            // double diff = sharedOldCentroids[idx * numDimensions + d] - sharedCentroids[idx * numDimensions + d];
            double diff = old_centroids[idx * numDimensions + d] - centroids[idx * numDimensions + d];
            squaredDistance += diff * diff;
        }
        if (squaredDistance > threshold * threshold) {
            *hasConverged = 0;
        }
    }
}


// Host function to compute K-means using CUDA
void computeKmeansCUDA_shmem(const std::vector<double>& data, 
                       std::vector<double>& centroids, 
                       std::vector<int>& assignments,
                       struct options_t *opts) {

    std::cout << "Executing computeKmeansCUDA_shmem" << std::endl;

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
        int sharedMemSize = numClusters * numDimensions * sizeof(double);
        // int blockSize = calculateOptimalBlockSize(assignPointsToClustersKernel_shmem, sharedMemSize);
        // std::cout << "optimal block size for assign points to cluster kernel is: " << blockSize << std::endl;
        int blockSize = 128;
        int numBlocks = (numPoints + blockSize - 1) / blockSize;
        assignPointsToClustersKernel_shmem<<<numBlocks, blockSize, sharedMemSize>>>(d_data, d_centroids, d_assignments, 
                                                               numPoints, numClusters, numDimensions);
	    // Copy d_centroids to d_old_centroids
        cudaMemcpy(d_old_centroids, d_centroids, numClusters * numDimensions * sizeof(double), cudaMemcpyDeviceToDevice);
       	// Reset cluster sizes and centroids
        cudaMemset(d_clusterSizes, 0, numClusters * sizeof(int));
        cudaMemset(d_centroids, 0, numClusters * numDimensions * sizeof(double));

        // Update centroids
        sharedMemSize = numClusters * numDimensions * sizeof(double) + numClusters * sizeof(int);
        updateCentroidsKernel_shmem<<<numBlocks, blockSize, sharedMemSize>>>(d_data, d_centroids, d_assignments, d_clusterSizes,
                                                        numPoints, numClusters, numDimensions);
        // updateCentroidsKernel<<<numBlocks, blockSize>>>(d_data, d_centroids, d_assignments, d_clusterSizes,
        //                                                 numPoints, numClusters, numDimensions);

        // Normalize centroids
        sharedMemSize = numClusters * numDimensions * sizeof(double) + numClusters * sizeof(int);
        blockSize = 128;
        int centroidBlocks = (numClusters + blockSize - 1) / blockSize;
        // normalizeCentroidsKernel<<<centroidBlocks, blockSize>>>(d_centroids, d_clusterSizes, 
        //                                                         numClusters, numDimensions);
        normalizeCentroidsKernel_shmem<<<centroidBlocks, blockSize, sharedMemSize>>>(d_centroids, d_clusterSizes, 
                                                                numClusters, numDimensions);

        // Check for convergence
	
	    cudaMemset(d_hasConverged, 1, sizeof(int));
        sharedMemSize = 2 * numClusters * numDimensions * sizeof(double) + numClusters * sizeof(int);
        // convergenceCheckKernel<<<centroidBlocks, blockSize>>>(d_old_centroids, d_centroids, 
        //                                                       numClusters, numDimensions, 
        //                                                       opts->threshold, d_hasConverged);
        convergenceCheckKernel_shmem<<<centroidBlocks, blockSize, sharedMemSize>>>(d_old_centroids, d_centroids, 
                                                              numClusters, numDimensions, 
                                                              opts->threshold, d_hasConverged);

        int hasConverged;
        cudaMemcpy(&hasConverged, d_hasConverged, sizeof(int), cudaMemcpyDeviceToHost);
        if (hasConverged) {
            iteration = iter + 1;
            std::cout << "CUDA shared memory converged after " << iteration << " iterations." << std::endl;
            break;
        }
	
    }
    cudaEventRecord(stop);

    // Copy results back to host
    cudaMemcpy(assignments.data(), d_assignments, numPoints * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(centroids.data(), d_centroids, numClusters * numDimensions * sizeof(double), cudaMemcpyDeviceToHost);

    // Synchronize and calculate the elapsed time
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);

    // Print the execution time
    if (iteration == 0){
        std::cout << "CUDA shared memory didn't converge after " << opts->max_num_iter << " iterations." << std::endl;
        iteration = opts->max_num_iter;
    }
    printf("K-means main loop execution time (CUDA shared memory): %.3f ms\n", milliseconds);
    float time_per_iter_in_ms = milliseconds / float(iteration);
    printf("Time per iteration: %.3f ms\n", time_per_iter_in_ms);
    

    // Free device memory
    cudaFree(d_data);
    cudaFree(d_centroids);
    cudaFree(d_old_centroids);
    cudaFree(d_assignments);
    cudaFree(d_clusterSizes);
    cudaFree(d_hasConverged);

}