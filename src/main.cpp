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

    // read data from file
    std::vector<std::vector<double>> data;
    readDataFromFile(&opts, data);

    // start kmeans
    std::vector<std::vector<double>> centroids(opts.num_cluster, std::vector<double>(opts.dim));
    std::vector<int> assignments;
    kmeans_srand(opts.seed);
    computeKmeansCPU(data, centroids, assignments, &opts);
    // Output results: if -c is specified, print centroids; otherwise, print clusters
    if (opts.centroid_output) {
        printCentroids(centroids);
    }
    else{
        printClusters(assignments);
    }

    computeKmeansCUDA(data, centroids, assignments, &opts);


    return 0;
}