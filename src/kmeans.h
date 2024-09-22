#include "helper.h"

void assignPointsToClusters(const std::vector<std::vector<double>>& data, 
                            const std::vector<std::vector<double>>& centroids, 
                            std::vector<int>& assignments);

void updateCentroids(const std::vector<std::vector<double>>& data, 
                     std::vector<std::vector<double>>& centroids, 
                     const std::vector<int>& assignments);

void computeKmeansCPU(const std::vector<std::vector<double>>& data, 
                    std::vector<std::vector<double>>& centroids, 
                    std::vector<int>& assignments,
                    struct options_t *opts);

void printClusters(const std::vector<int>& assignments);

void printCentroids(const std::vector<std::vector<double>>& centroids);

void computeKmeansCUDA(const std::vector<std::vector<double>>& data, 
                       std::vector<std::vector<double>>& centroids, 
                       std::vector<int>& assignments,
                       struct options_t *opts);

