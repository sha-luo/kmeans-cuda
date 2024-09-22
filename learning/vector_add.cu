#include <stdio.h>
#include <cuda.h>

#define N 100  // Define the size of the vector

// CUDA Kernel for vector addition
__global__ void add(int *a, int *b, int *c) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < N) {
        c[index] = a[index] + b[index];
    }
}

// Helper function to fill an array with random integers
void random_ints(int *x, int n) {
    for (int i = 0; i < n; i++) {
        x[i] = rand() % 100;  // Fill with random values between 0 and 99
    }
}

int main(void) {
    int *a, *b, *c;           // Host copies of a, b, c
    int *d_a, *d_b, *d_c;     // Device copies of a, b, c
    int size = N * sizeof(int);

    // Allocate space for device copies of a, b, c
    cudaMalloc((void **)&d_a, size);
    cudaMalloc((void **)&d_b, size);
    cudaMalloc((void **)&d_c, size);

    // Allocate space for host copies of a, b, c and setup initial values
    a = (int *)malloc(size); random_ints(a, N);
    b = (int *)malloc(size); random_ints(b, N);
    c = (int *)malloc(size);

    // Copy inputs to device (GPU memory)
    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);

    // Launch add() kernel on GPU with N blocks and 1 thread per block
    add<<<(N+31)/32, 32>>>(d_a, d_b, d_c);

    // Copy result back to host from device
    cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);

    // Print the results of vector addition
    printf("Vector addition result:\n");
    for (int i = 0; i < N; i++) {
        printf("i = %d: %d +%d = %d\n", i, a[i], b[i], c[i]);
    }

    // Cleanup
    free(a); free(b); free(c);
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);

    return 0;
}

