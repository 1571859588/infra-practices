#include <stdio.h>

int main(void){
    int iDeviceCount = 0;
    cudaError_t error_code = cudaGetDeviceCount(&iDeviceCount);
    if( error_code != cudaSuccess || iDeviceCount == 0){
        printf("no available Cuda Device\n");
        exit(-1);
    }
    int iDevice = 0;
    error_code = cudaSetDevice(iDevice);
    if(error_code != cudaSuccess){
        printf("fail to set device\n");
        exit(-1);
    }
    else{
        printf("set sucess\n");
    }
    return 0;
}