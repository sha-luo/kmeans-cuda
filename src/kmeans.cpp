#include "kmeans.h"
#include <iostream>
#include <chrono>
#include "argparse.h"

void assignPointsToClusters(const std::vector<double>& data, 
                            const std::vector<double>& centroids, 
                            std::vector<int>& assignments, 
                            int numPoints, int numDimensions, int numClusters) {
    assignments.resize(numPoints);
    for (int i = 0; i < numPoints; ++i) {
        float minDistance = std::numeric_limits<float>::max();
        int closestCentroid = 0;
        for (int j = 0; j < numClusters; ++j) {
            float distance = 0.0f;
            for (int d = 0; d < numDimensions; ++d) {
                float diff = data[i * numDimensions + d] - centroids[j * numDimensions + d];
                distance += diff * diff;
            }
            if (distance < minDistance) {
                minDistance = distance;
                closestCentroid = j;
            }
        }
        assignments[i] = closestCentroid;
    }
}

void updateCentroids(const std::vector<double>& data, 
                     std::vector<double>& centroids, 
                     const std::vector<int>& assignments, 
                     int numPoints, int numDimensions, int numClusters) {

    std::vector<int> clusterSizes(numClusters, 0);

    // Initialize centroids to zero
    std::fill(centroids.begin(), centroids.end(), 0.0);

    // Sum up all points for each cluster
    for (int i = 0; i < numPoints; ++i) {
        int cluster = assignments[i];
        clusterSizes[cluster]++;
        for (int d = 0; d < numDimensions; ++d) {
            centroids[cluster * numDimensions + d] += data[i * numDimensions + d];
        }
    }

    // Calculate average to get new centroids
    for (int c = 0; c < numClusters; ++c) {
        if (clusterSizes[c] > 0) {
            for (int d = 0; d < numDimensions; ++d) {
                centroids[c * numDimensions + d] /= clusterSizes[c];
            }
        }
    }
}

void computeKmeansCPU(const std::vector<double>& data, 
                      std::vector<double>& centroids, 
                      std::vector<int>& assignments,
                      struct options_t *opts) {

    std::cout << "Executing computeKmeansCPU..." << std::endl;

    // Initialize centroids randomly
    std::cout << "Initializing centroids randomly" << std::endl;
    for (int i = 0; i < opts->num_cluster; i++){
        int index = kmeans_rand() % opts->num_points;
        for (int d = 0; d < opts->dim; ++d) {
            centroids[i * opts->dim + d] = data[index * opts->dim + d];
        }
    }
    // printCentroids(centroids, opts->num_cluster, opts->dim);

    // Main K-means loop
    auto start = std::chrono::high_resolution_clock::now();
    int iteration = 0;
    for (int iter = 0; iter < opts->max_num_iter; ++iter) {
        // Assign points to clusters
        assignPointsToClusters(data, centroids, assignments, opts->num_points, opts->dim, opts->num_cluster);

        // Update centroids
        std::vector<double> oldCentroids = centroids;
        updateCentroids(data, centroids, assignments, opts->num_points, opts->dim, opts->num_cluster);

        // Check for convergence
        float maxChange = 0.0f;
        for (int i = 0; i < opts->num_cluster; ++i) {
            float change = 0.0f;
            for (int d = 0; d < opts->dim; ++d) {
                float diff = oldCentroids[i * opts->dim + d] - centroids[i * opts->dim + d];
                change += diff * diff;
            }
            maxChange = std::max(maxChange, change);
        }

        if (std::sqrt(maxChange) < opts->threshold) {
            iteration = iter + 1;
            std::cout << "Converged after " << iteration << " iterations." << std::endl;
            break;
        }
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto diff = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    if (iteration == 0){
        std::cout << "K-means CPU didn't converge after " << opts->max_num_iter << " iterations." << std::endl;
        iteration = 150;
    }
    float time_per_iter_in_ms = (float)diff.count() / (float)iteration;
    printf("K-means main loop execution time (CPU): %.3f ms\n", (float)diff.count());
    std::cout << "Time per iteration: " << time_per_iter_in_ms << " ms" << std::endl;
}

void printCentroids(const std::vector<double>& centroids, int numClusters, int numDimensions) {
    std::cout << "centroids: " << std::endl;
    for (int i = 0; i < numClusters; i++) {
        std::cout << i << " ";
        for (int j = 0; j < numDimensions; j++) {
            std::cout << centroids[i * numDimensions + j] << " ";
        }
        std::cout << std::endl;
    }
}

void printClusters(const std::vector<int>& assignments) {
    std::cout << "clusters: " << std::endl;
    for (size_t i = 0; i < assignments.size(); i++) {
        std::cout << assignments[i] << " ";
    }
    std::cout << std::endl; 
}

