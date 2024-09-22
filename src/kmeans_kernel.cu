#include "kmeans.h"
#include <cuda_runtime.h>
#include <iostream>
#include "argparse.h"

 
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

// Host function to compute K-means using CUDA
void computeKmeansCUDA(const std::vector<std::vector<double>>& data, 
                       std::vector<std::vector<double>>& centroids, 
                       std::vector<int>& assignments,
                       struct options_t *opts) {
    int numPoints = data.size();
    int numDimensions = data[0].size();
    int numClusters = opts->num_cluster;

    // Allocate device memory
    double *d_data, *d_centroids;
    int *d_assignments, *d_clusterSizes;
    cudaMalloc(&d_data, numPoints * numDimensions * sizeof(double));
    cudaMalloc(&d_centroids, numClusters * numDimensions * sizeof(double));
    cudaMalloc(&d_assignments, numPoints * sizeof(int));
    cudaMalloc(&d_clusterSizes, numClusters * sizeof(int));

    // Copy data to device
    cudaMemcpy(d_data, data.data(), numPoints * numDimensions * sizeof(double), cudaMemcpyHostToDevice);

    // Initialize centroids randomly
    std::cout << "Initializing centroids randomly" << std::endl;
    for (int i = 0; i < numClusters; i++) {
        int index = kmeans_rand() % numPoints;
        centroids[i] = data[index];
    }
    cudaMemcpy(d_centroids, centroids.data(), numClusters * numDimensions * sizeof(double), cudaMemcpyHostToDevice);

    // Main K-means loop
    for (int iter = 0; iter < opts->max_num_iter; ++iter) {
        std::cout << "Iteration: " << iter << std::endl;

        // Assign points to clusters
        int blockSize = 256;
        int numBlocks = (numPoints + blockSize - 1) / blockSize;
        assignPointsToClustersKernel<<<numBlocks, blockSize>>>(d_data, d_centroids, d_assignments, 
                                                               numPoints, numClusters, numDimensions);

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

        // Check for convergence (simplified for brevity)
        // In a full implementation, you would copy centroids back to host and check for convergence
    }

    // Copy results back to host
    cudaMemcpy(assignments.data(), d_assignments, numPoints * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(centroids.data(), d_centroids, numClusters * numDimensions * sizeof(double), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_data);
    cudaFree(d_centroids);
    cudaFree(d_assignments);
    cudaFree(d_clusterSizes);
}


