#include <stdio.h>
#include "../tools/common2.cuh"

#define NUM_REPEATS 10

void vec_serial_add(float *A, float *B, float *C, int size){
    for(int i=0;i<size;i++){
        C[i] = A[i] + B[i];
    }
}

void initialData(float *fpHost, int iSize){
    for(int i=0;i<iSize;i++){
        fpHost[i] = float(rand()&0xFF)/10.0f;
    }
}

bool compare_cpu_gpu_results(float *fpHostC, float *fpCPUResult, int iSize, float min_error=0.0001){
    for(int i=0;i<iSize;i++){
        if(fabs(fpHostC[i] - fpCPUResult[i]) > min_error){
            printf("vec num %d: %f, %f\n", i, fpHostC[i], fpCPUResult[i]);
            return false;
        }
    }
    return true;
}

__global__ void vec_add_from_gpu(float *fpDevA, float *fpDevB, float *fpDevC, int iSize){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < iSize){
        fpDevC[i] = fpDevA[i] + fpDevB[i];
    }
}

int main(void){
    setGPU();
    int iSize = 4096;
    size_t stBytesCount = iSize * sizeof(float);
    float *fpHostA = (float*)malloc(stBytesCount);
    float *fpHostB = (float*)malloc(stBytesCount);
    float *fpHostC = (float*)malloc(stBytesCount);
    float *fpCPUResult = (float*)malloc(stBytesCount);
    if(fpHostA != NULL && fpHostB != NULL && fpHostC != NULL){
        memset(fpHostA, 0, stBytesCount);
        memset(fpHostB, 0, stBytesCount);
        memset(fpHostC, 0, stBytesCount);
        printf("success allocate memory to host\n");
    }else{
        printf("failed allocate memory to host\n");
        return -1;
    }

    srand(666);
    initialData(fpHostA, iSize);
    initialData(fpHostB, iSize);

    float *fpDevA = NULL;
    float *fpDevB = NULL;
    float *fpDevC = NULL;
    CUDA_CHECK(cudaMalloc(&fpDevA, stBytesCount));
    CUDA_CHECK(cudaMalloc(&fpDevB, stBytesCount));
    CUDA_CHECK(cudaMalloc(&fpDevC, stBytesCount));
    if(fpDevA != NULL && fpDevB != NULL && fpDevC != NULL){
        CUDA_CHECK(cudaMemset(fpDevA, 0, stBytesCount));
        CUDA_CHECK(cudaMemset(fpDevB, 0, stBytesCount));
        CUDA_CHECK(cudaMemset(fpDevC, 0, stBytesCount));
        printf("success allocate memory to device\n");
    }else{
        free(fpHostA);
        free(fpHostB);
        free(fpHostC);
        printf("failed allocate memory to device\n");
        return -1;
    }
    CUDA_CHECK(cudaMemcpy(fpDevA, fpHostA, stBytesCount, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(fpDevB, fpHostB, stBytesCount, cudaMemcpyHostToDevice));

    vec_serial_add(fpHostA, fpHostB, fpCPUResult, iSize);

    dim3 block(32);
    dim3 grid(iSize + block.x-1/block.x);

    float total_time = 0;
    for(int i=0;i<NUM_REPEATS;i++){
        cudaEvent_t start, stop;
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));
        CUDA_CHECK(cudaEventRecord(start));
        cudaEventQuery(start);

        vec_add_from_gpu<<<grid, block>>>(fpDevA, fpDevB, fpDevC, iSize);

        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float elapsed_time = 0;
        CUDA_CHECK(cudaEventElapsedTime(&elapsed_time, start, stop));
        if(i>0){
            total_time += elapsed_time;
        }
        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));
    }
    CUDA_CHECK(cudaMemcpy(fpHostC, fpDevC, stBytesCount, cudaMemcpyDeviceToHost));

    if(compare_cpu_gpu_results(fpHostC, fpCPUResult, iSize)){
        printf("the results between cpu and gpu are same\n");
    }else{
        printf("the results between cpu and gpu are not same\n");
        return -1;
    }
    
    for(int i = 0;i<10;i++){
        printf("vec num %d: %f\n", i, fpHostC[i]);
    }
    free(fpHostA);
    free(fpHostB);
    free(fpHostC);
    free(fpCPUResult);
    CUDA_CHECK(cudaFree(fpDevA));
    CUDA_CHECK(cudaFree(fpDevB));
    CUDA_CHECK(cudaFree(fpDevC));
    CUDA_CHECK(cudaDeviceReset());
    return 0;

}