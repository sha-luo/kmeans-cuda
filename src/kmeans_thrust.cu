#include "kmeans.h"
#include <iostream>
#include <chrono>
#include "argparse.h"
#include <thrust/sort.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/transform.h>
#include <thrust/transform_reduce.h>
#include <thrust/reduce.h>
#include <thrust/functional.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/random.h>


struct assign_cluster {
    const double* data;
    const double* centroids;
    int num_dimensions;
    int num_clusters;

    assign_cluster(const double* _data, const double* _centroids, int _num_dimensions, int _num_clusters)
        : data(_data), centroids(_centroids), num_dimensions(_num_dimensions), num_clusters(_num_clusters) {}

    __device__
    int operator()(int idx) const {
        const double* point = data + idx * num_dimensions;
        
        double min_distance = INFINITY;
        int closest_centroid = 0;

        for (int j = 0; j < num_clusters; ++j) {
            double distance = 0.0;
            for (int d = 0; d < num_dimensions; ++d) {
                double diff = point[d] - centroids[j * num_dimensions + d];
                distance += diff * diff;
            }
            if (distance < min_distance) {
                min_distance = distance;
                closest_centroid = j;
            }
        }
        return closest_centroid;
    }
};


void computeKmeansThrust(const std::vector<double>& data,
                       std::vector<double>& centroids,
                       std::vector<int>& assignments,
                       struct options_t* opts) {
    
    std::cout << "Executing computeKmeansThrust..." << std::endl;

    int num_points = opts->num_points;
    int num_dimensions = opts->dim;
    int num_clusters = opts->num_cluster;
    cudaEvent_t start, stop;
    float milliseconds = 0;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Initialize centroids randomly
    std::cout << "Initializing centroids randomly" << std::endl;
    for (int i = 0; i < num_clusters; i++){
        int index = kmeans_rand() % num_points;
        for (int d = 0; d < num_dimensions; ++d) {
            centroids[i * num_dimensions + d] = data[index * num_dimensions + d];
        }
    }

    // Transfer data to device
    thrust::device_vector<double> d_data = data;
    thrust::device_vector<double> d_centroids = centroids;
    thrust::device_vector<int> d_assignments(num_points);

    cudaEventRecord(start);
    int iteration = 0;
    for (int iter = 0; iter < opts->max_num_iter; ++iter) {
        // Create an instance of the assign_cluster functor
        assign_cluster cluster_functor(thrust::raw_pointer_cast(d_data.data()),
                                   thrust::raw_pointer_cast(d_centroids.data()),
                                   num_dimensions, num_clusters);
        // Assign points to clusters
        thrust::transform(
            thrust::counting_iterator<int>(0),
            thrust::counting_iterator<int>(num_points),
            d_assignments.begin(),
            cluster_functor
        );

        // std::cout << "DEBUG: printing assignment" << std::endl;
        // std::vector<int> temp4 = std::vector<int>(d_assignments.begin(), d_assignments.end());
        // for (int i = 0; i < temp4.size(); i++) {
        //     std::cout << temp4[i] << " ";
        // }
        // std::cout << std::endl;

        // Update centroids
        // Prepare device vectors to hold results
        thrust::device_vector<int> d_keys_out(num_clusters * num_dimensions);
        thrust::device_vector<double> d_new_centroids(num_clusters * num_dimensions);  // This will hold the sums
        thrust::device_vector<int> d_cluster_counts(num_clusters * num_dimensions);     // This will hold the counts
        // Prepare keys for reduce_by_key
        thrust::device_vector<int> d_keys(num_points * num_dimensions);
        const int* d_assignments_ptr = thrust::raw_pointer_cast(d_assignments.data());
        thrust::transform(
            thrust::make_counting_iterator(0),
            thrust::make_counting_iterator(num_points * num_dimensions),
            d_keys.begin(),
            [=] __device__ (int idx) {
                return d_assignments_ptr[idx / num_dimensions] * num_dimensions + (idx % num_dimensions);
            }
        );

        // Perform reduce_by_key to sum the values for each combined key
        thrust::sort_by_key(d_keys.begin(), d_keys.end(), d_data.begin());
        thrust::reduce_by_key(d_keys.begin(), d_keys.end(), 
                                d_data.begin(),
                                d_keys_out.begin(), 
                                d_new_centroids.begin(),
                                thrust::equal_to<int>(),
                                thrust::plus<double>());

        // Calculate counts for each combined key (number of occurrences of each key)
        thrust::device_vector<int> d_ones(num_points * num_dimensions, 1);
        thrust::reduce_by_key(d_keys.begin(), d_keys.end(), 
                            d_ones.begin(),
                            d_keys_out.begin(), d_cluster_counts.begin());

        // Compute averages using Thrust
        thrust::transform(d_new_centroids.begin(), d_new_centroids.end(),
                        d_cluster_counts.begin(),
                        d_new_centroids.begin(),
                        thrust::divides<double>());

        // // Check for convergence
        thrust::device_vector<double> d_distance(num_clusters);
        auto calc_sum_of_squares = [d_centroids_ptr = thrust::raw_pointer_cast(d_centroids.data()),
                                    d_new_centroids_ptr = thrust::raw_pointer_cast(d_new_centroids.data()),
                                    num_dimensions] __device__ (int idx) {
            double sum_sq = 0.0;
            for (int i = 0; i < num_dimensions; ++i)
            {
                double value = d_centroids_ptr[idx * num_dimensions + i] - d_new_centroids_ptr[idx * num_dimensions + i];
                sum_sq += value * value;
            }
            return sum_sq;
        };

        // Apply the lambda function to each index and store the result in d_distance
        thrust::transform(
            thrust::make_counting_iterator(0),
            thrust::make_counting_iterator(num_clusters),
            d_distance.begin(),
            calc_sum_of_squares
        );
        // Find the maximum value in d_output
        double max_change = thrust::reduce(d_distance.begin(), d_distance.end(), 0.0, thrust::maximum<double>());
        if (std::sqrt(max_change) < opts->threshold) {
            iteration = iter + 1;
            std::cout << "Thrust converged after " << iteration << " iterations." << std::endl;
            break;
        }

        // update centroids and reload data (which has been overwritten in sort_by_key)
        d_centroids = d_new_centroids;
        d_data = data;

    }
    cudaEventRecord(stop);

    // Transfer results back to host
    centroids = std::vector<double>(d_centroids.begin(), d_centroids.end());
    assignments = std::vector<int>(d_assignments.begin(), d_assignments.end());

    // Synchronize and calculate the elapsed time
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);

    if (iteration == 0){
        std::cout << "Thrust didn't converge after " << opts->max_num_iter << " iterations." << std::endl;
        iteration = opts->max_num_iter;
    }
    printf("K-means main loop execution time (CUDA thrust): %.3f ms\n", milliseconds);
    float time_per_iter_in_ms = milliseconds / float(iteration);
    printf("Time per iteration: %.3f ms\n", time_per_iter_in_ms);
}

