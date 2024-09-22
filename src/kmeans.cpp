#include "kmeans.h"
#include <iostream>
#include <chrono>
#include "argparse.h"

void assignPointsToClusters(const std::vector<std::vector<double>>& data, 
                            const std::vector<std::vector<double>>& centroids, 
                            std::vector<int>& assignments) {
    assignments.resize(data.size());
    for (size_t i = 0; i < data.size(); ++i) {
        double minDistance = std::numeric_limits<double>::max();
        int closestCentroid = 0;
        for (size_t j = 0; j < centroids.size(); ++j) {
            double distance = euclideanDistance(data[i], centroids[j]);
            if (distance < minDistance) {
                minDistance = distance;
                closestCentroid = j;
            }
        }
        assignments[i] = closestCentroid;
    }
}

void updateCentroids(const std::vector<std::vector<double>>& data, 
                     std::vector<std::vector<double>>& centroids, 
                     const std::vector<int>& assignments) {

    int numClusters = centroids.size();
    int numDimensions = centroids[0].size();
    std::vector<int> clusterSizes(numClusters, 0);

    // Initialize centroids to zero
    for (auto& centroid : centroids) {
        centroid = std::vector<double>(numDimensions, 0.0);
    }

    // Sum up all points for each cluster
    for (size_t i = 0; i < data.size(); ++i) {
        int cluster = assignments[i];
        clusterSizes[cluster]++;
        for (int d = 0; d < numDimensions; ++d) {
            centroids[cluster][d] += data[i][d];
        }
    }

    // Calculate average to get new centroids
    for (int c = 0; c < numClusters; ++c) {
        if (clusterSizes[c] > 0) {
            for (int d = 0; d < numDimensions; ++d) {
                centroids[c][d] /= clusterSizes[c];
            }
        }
    }
}

void computeKmeansCPU(const std::vector<std::vector<double>>& data, 
                    std::vector<std::vector<double>>& centroids, 
                    std::vector<int>& assignments,
                    struct options_t *opts) {

    // Initialize centroids randomly
    std::cout << "Initializing centroids randomly" << std::endl;
    for (int i = 0; i < opts->num_cluster; i++){
        int index = kmeans_rand() % opts->num_points;
        centroids[i] = data[index];
    }

    
    // Main K-means loop
    auto start = std::chrono::high_resolution_clock::now();
    int iteration = 0;
    for (int iter = 0; iter < opts->max_num_iter; ++iter) {
        // Assign points to clusters
        std::cout << "Iteration: " << iter << std::endl;
        assignPointsToClusters(data, centroids, assignments);

        // Update centroids
        std::vector<std::vector<double>> oldCentroids = centroids;
        updateCentroids(data, centroids, assignments);

        // Check for convergence
        double maxChange = 0.0;
        for (size_t i = 0; i < centroids.size(); ++i) {
            double change = euclideanDistance(oldCentroids[i], centroids[i]);
            maxChange = std::max(maxChange, change);
        }

        if (maxChange < opts->threshold) {
            iteration = iter + 1;
            break;
        }
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto diff = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    double time_per_iter_in_ms = (double)diff.count() / (double)iteration;
    std::cout << "Converged after " << iteration << " iterations." << std::endl;
    std::cout << "Time per iteration: " << time_per_iter_in_ms << " ms." << std::endl;
}

void printCentroids(const std::vector<std::vector<double>>& centroids) {
    std::cout << "centroids: " << std::endl;
    for (size_t i = 0; i < centroids.size(); i++) {
        std::cout <<  i << " ";
        for (size_t j = 0; j < centroids[i].size(); j++) {
            std::cout << centroids[i][j] << " ";
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

