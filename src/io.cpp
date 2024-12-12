#include <sstream>
#include <vector>
#include "io.h"

void readDataFromFile(struct options_t* args,
               std::vector<double>& data) {

  	std::ifstream inputFile(args->input_filename);
    if (!inputFile.is_open()) {
        std::cerr << "Unable to open file: " << args->input_filename << std::endl;
        return;
    }

    inputFile >> args->num_points;
    std::cout << "Read " << args->num_points << " number of records" << std::endl;

    std::string line;
    std::getline(inputFile, line); // Consume the newline after the first line

    while (std::getline(inputFile, line)) {
        std::istringstream iss(line);
        int index;
        iss >> index;  // Read the index

        double value;
        while (iss >> value) {
            data.push_back(value);
        }
    }

    inputFile.close();
}

