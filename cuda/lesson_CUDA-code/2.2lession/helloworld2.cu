#include <stdio.h>

__global__ void hello_from_gpu(){
    printf("hello from gpu\n");

}


int main(void){
    hello_from_gpu<<<3,2>>>();
    cudaDeviceSynchronize();
    return 0;
}