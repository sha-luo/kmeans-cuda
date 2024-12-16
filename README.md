# kmeans-cuda

### Introduction
K-means is a machine learning algorithm most commonly used for unsupervised learning. This exercise focuses on leveraging CUDA and its libraries to accelerate the K-means algorithm. Four distinct methods of implementation are examined:
* **Sequential**: The sequential implementation using CPU serves as the baseline for performance comparison.
* **CUDA global memory**: This basic CUDA implementation parallelizes the primaries steps of K-means algorithm, finding nearest centroid and assigning each data point to the nearest centroid, in each iteration using global memory.
* **CUDA shared memory**: Shared memory is a special type of memory located in each SM. It provides significantly lower latency compared to accessing global memory. It is explicitly controlled by the programmer and is shared among all threads in a block that are running on the same SM. 
* **Thrust**: Thrust is a high-level parallel programming library in CUDA, which allows developers to write GPU code in a more abstract way without dealing with low-level CUDA details such as kernel launch and memory management.

### Performance


### To run the program 
```bash
Usage:
        --num_cluster or -k <an integer specifying the number of clusters>
        --dim or -d <an integer specifying the dimension of the data>
        --input_filename or -i <a string specifying the input filename>
        --max_num_iter or -m <an integer specifying the maximum number of iterations>
        --threshold or -t <a double specifying the threshold for convergence>
        --seed or -s <an integer specifying the seed for rand()>
        --approach or -a <an integer specifying which approach to be invoked to solve kmeans 1 for CPU; 2 for CUDA gmem; 3 for CUDA shmem; 4 for thrust
        [Optional]--centroid_output or -c <a flag to control whether output the centroids or labels>
```
```bash
./bin/kmeans -i input/random-n2048-d16-c16.txt -k 16 -d 16 -m 150 -t 1e-6 -s 8675309 -a 4 -c
```
