#include <stdio.h>

__global__ void printHelloWorld(){
    printf("global thread id: %d Hello World\n", threadIdx.x + blockIdx.x * blockDim.x);
}

int main(){
    printHelloWorld<<<4,2>>>();
    cudaDeviceSynchronize();
    return 0;
}