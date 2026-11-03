#include <stdio.h>

__global__ void hello_from_gpu(){
    int bid = blockIdx.x;
    int tid = threadIdx.x;
    int id = tid + bid * blockDim.x;
    printf("hello from gpu %d\n", id);

}

int main(void){
    hello_from_gpu<<<4,2>>>();
    cudaDeviceSynchronize();
    return 0;
}