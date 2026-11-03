#pragma once
#include <stdio.h>
#include <stdlib.h>

#define CUDA_CHECK(expr_to_check) do{ \
    cudaError_t result = expr_to_check; \
    if(result != cudaSuccess){ \
        fprintf(stderr, "CUDA RunTime Error Name %s: %s:%i:%d = %s\n", cudaGetErrorName(result), __FILE__, __LINE__, result, cudaGetErrorString(result)); \
    } \
}while(0)

void setGPU(){
    int iGPUCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&iGPUCount));
    printf("GPU Count: %d\n", iGPUCount);
    CUDA_CHECK(cudaSetDevice(0));
}