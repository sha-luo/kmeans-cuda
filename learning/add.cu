#include <stdio.h>

__global__ void add(int *a, int *b, int *c){
    *c = *a + *b;
}

int main(void){
    int a, b, c;
    int *d_a, *d_b, *d_c;
    int size = sizeof(int);

    a = 3;
    b = 8;

    // allocate space for device copies of a, b, c
    cudaMalloc((void **)&d_a, size);
    cudaMalloc((void **)&d_b, size);
    cudaMalloc((void **)&d_c, size);

    // copy inputs to device
    cudaMemcpy(d_a, &a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, &b, size, cudaMemcpyHostToDevice);

    // Launch add() kernal on GPU
    add<<<1,1>>>(d_a, d_b, d_c);

    //copy result back to host
    cudaMemcpy(&c, d_c, size, cudaMemcpyDeviceToHost);

    // print result
    printf("The result of %d + %d is %d\n", a, b, c);

    //clean up
    cudaFree(d_a); 
    cudaFree(d_b); 
    cudaFree(d_c);
    return 0;
}