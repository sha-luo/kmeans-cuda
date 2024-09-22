#include <stdio.h>

#define N 128  // Size of the array
#define RADIUS 3  // Window size for stencil operation
#define BLOCK_SIZE 32 

// CUDA kernel for stencil operation with a window size of 7
__global__ void stencil_1d(int *input, int *output) {
    
    __shared__ int temp[BLOCK_SIZE + 2 * RADIUS];
    int gindex = threadIdx.x + blockIdx.x * blockDim.x;
    int lindex = threadIdx.x + RADIUS;

    // read input elements into shared memory
    temp[lindex] = input[gindex];
    if (threadIdx.x < RADIUS) {
            temp[lindex - RADIUS] = input[gindex - RADIUS];
            temp[lindex + BLOCK_SIZE] = input[gindex + BLOCK_SIZE];
    }

    // Ensure all threads have loaded their values into shared memory
    __syncthreads(); 

    // Perform stencil operation using values from shared memory
    int result = 0;
    for (int i = -RADIUS; i <= RADIUS; i++) {
        result += temp[lindex + i];  // Example stencil: sum over neighbors
    }
    output[gindex] = result / (2 * RADIUS + 1);  // Example: average
    
}

// Helper function to fill an array with random integers
void random_ints(int *x, int n) {
    for (int i = 0; i < n; i++) {
        x[i] = rand() % 100;  // Fill with random values between 0 and 99
    }
}

int main() {

    int *input, *output;
    int *d_input, *d_output;
    int size = N * sizeof(int);

    input = (int *)malloc(size); 
    random_ints(input, N);
    output = (int *)malloc(size);

    // Allocate space on the device
    cudaMalloc((void **)&d_input, size);
    cudaMalloc((void **)&d_output, size);

    // Copy the input array to the device
    cudaMemcpy(d_input, input, size, cudaMemcpyHostToDevice);

    // Launch stencil kernel with N threads
    stencil_1d<<<(N + BLOCK_SIZE -1) / BLOCK_SIZE, BLOCK_SIZE>>>(d_input, d_output);

    // Copy the output array back to the host
    cudaMemcpy(output, d_output, size, cudaMemcpyDeviceToHost);

    // Print the result
    printf("Input array:  ");
    for (int i = 0; i < N; i++) {
        printf("%d ", input[i]);
    }
    printf("\nOutput array: ");
    for (int i = 0; i < N; i++) {
        printf("%d ", output[i]);
    }
    printf("\n");

    // Clean up
    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}
