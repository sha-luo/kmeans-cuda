#include "helper.h"

void assignPointsToClusters(const std::vector<double>& data, 
                            const std::vector<double>& centroids, 
                            std::vector<int>& assignments, 
                            int numPoints, int numDimensions, int numClusters);

void updateCentroids(const std::vector<double>& data, 
                     std::vector<double>& centroids, 
                     const std::vector<int>& assignments, 
                     int numPoints, int numDimensions, int numClusters);

void computeKmeansCPU(const std::vector<double>& data, 
                      std::vector<double>& centroids, 
                      std::vector<int>& assignments,
                      struct options_t *opts);

void printClusters(const std::vector<int>& assignments);

void printCentroids(const std::vector<double>& centroids, int numClusters, int numDimensions);

void computeKmeansCUDA(const std::vector<double>& data, 
                       std::vector<double>& centroids, 
                       std::vector<int>& assignments,
                       struct options_t *opts);

void computeKmeansCUDA_shmem(const std::vector<double>& data, 
                             std::vector<double>& centroids, 
                             std::vector<int>& assignments,
                             struct options_t *opts);

void computeKmeansThrust(const std::vector<double>& data,
                       std::vector<double>& centroids,
                       std::vector<int>& assignments,
                       struct options_t* opts);