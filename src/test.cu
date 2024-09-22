#include <stdio.h>

// CUDA kernel using atomicAdd for double precision
__global__ void sumArray(double *arr, double *result, int N) {
    // Calculate global thread index
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Ensure the index is within array bounds
    if (idx < N) {
        // Use atomicAdd to safely add element to the global result
        atomicAdd(result, arr[idx]);
    }
}

int main() {
    int N = 1000;  // Array size
    double *h_arr = (double *)malloc(N * sizeof(double));
    double *h_result = (double *)malloc(sizeof(double));

    // Initialize array and result
    for (int i = 0; i < N; i++) {
        h_arr[i] = 1.0;  // Fill with 1.0 for simplicity
    }
    *h_result = 0.0;

    // Allocate device memory
    double *d_arr, *d_result;
    cudaMalloc((void **)&d_arr, N * sizeof(double));
    cudaMalloc((void **)&d_result, sizeof(double));

    // Copy host array to device memory
    cudaMemcpy(d_arr, h_arr, N * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_result, h_result, sizeof(double), cudaMemcpyHostToDevice);

    // Define block size and grid size
    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;

    // Launch the kernel
    sumArray<<<gridSize, blockSize>>>(d_arr, d_result, N);

    // Copy the result back to host
    cudaMemcpy(h_result, d_result, sizeof(double), cudaMemcpyDeviceToHost);

    // Print the result
    printf("Sum: %f\n", *h_result);

    // Free device and host memory
    cudaFree(d_arr);
    cudaFree(d_result);
    free(h_arr);
    free(h_result);

    return 0;
}

