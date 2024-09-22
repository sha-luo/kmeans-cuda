#include <stdio.h>

// CUDA kernel function
__global__ void helloFromGPU() {
    printf("Hello World from the GPU!\n");
}

__global__ void add(int *a, int *b, int *c){
    *c = *a + *b;
}

int main() {
    // Launch the kernel on the GPU
    helloFromGPU<<<1, 1>>>();
    
    // Wait for the GPU to finish before accessing printf output
    cudaDeviceSynchronize();

    printf("Hello World from the CPU!\n");
    return 0;
}
