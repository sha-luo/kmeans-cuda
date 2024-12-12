#include <stdio.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <sstream>
#include <iomanip>
#include "io.h"
#include "kmeans.h"

int main(int argc, char **argv)
{
    // Parse args
    struct options_t opts;
    get_opts(argc, argv, &opts);

    // Print all arguments
    std::cout << "Input file: " << opts.input_filename << std::endl;
    std::cout << "Number of clusters: " << opts.num_cluster << std::endl;
    std::cout << "Number of dimensions: " << opts.dim << std::endl;
    std::cout << "Number of iterations: " << opts.max_num_iter << std::endl;
    std::cout << "Threshold: " << opts.threshold << std::endl;
    std::cout << "Seed: " << opts.seed << std::endl;
    std::cout << "Centroid output: " << (opts.centroid_output ? "true" : "false") << std::endl;
    std::cout << "Approach: " << opts.approach << std::endl;

    // read data from file
    std::vector<double> data;
    std::vector<double> centroids(opts.num_cluster * opts.dim);
    std::vector<int> assignments;
    readDataFromFile(&opts, data);

    // std::cout << "Data size: " << data.size() << std::endl;
    // for (int i = 0; i < opts.num_points; i++) {
    //     for (int j = 0; j < opts.dim; j++) {
    //         std::cout << data[i * opts.dim + j] << " ";
    //     }
    //     std::cout << std::endl;
    // }

    
    kmeans_srand(opts.seed);
    if (opts.approach == 1){
        computeKmeansCPU(data, centroids, assignments, &opts);
    }
    else if (opts.approach == 2){
        computeKmeansCUDA(data, centroids, assignments, &opts);
    }
    else if (opts.approach == 3){
        computeKmeansCUDA_shmem(data, centroids, assignments, &opts);
    }
    else if (opts.approach == 4){
        computeKmeansThrust(data, centroids, assignments, &opts);
    }
    

    // Output results: if -c is specified, print centroids; otherwise, print clusters
    if (opts.centroid_output) {
        printCentroids(centroids, opts.num_cluster, opts.dim);
    }
    else{
        printClusters(assignments);
    }

    // ... rest of your code ...
}

/*
TO DO: 
1. test different block size (API calls?)
2. shmem implementation: try saving into register?
*/