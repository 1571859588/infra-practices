#include <stdio.h>
#include "../tools/common2.cuh"
void initialData(float *a, int size){
    for(int i = 0; i < size; i++){
        a[i] = (float)(rand() & 0xFF) / 10.0f;
    }
}

void add_from_cpu(float *a, float *b, float *c, int size){
    for(int i = 0; i < size; i++){
        c[i] = a[i] + b[i];
    }
}

__global__ void add_from_gpu(float *a, float *b, float *c, const int size){
    const int bid = blockIdx.x;
    const int tid = threadIdx.x;
    const int i = bid * blockDim.x + tid;
    if(i<size){
        c[i] = a[i] + b[i];
    }
}

bool compare_cpu_gpu_results(float *a, float *b, int size, float min_error=0.0001f){
    for(int i =0;i<size;i++){
        if(fabs(a[i]-b[i])>min_error){
            return false;
        }
    }
    return true;
}

int main(void){

    setGPU();

    int iSize = 513;
    size_t iElementSize = iSize * sizeof(float);
    float *fpHostA = (float *)malloc(iElementSize);
    float *fpHostB = (float *)malloc(iElementSize);
    float *fpHostC = (float *)malloc(iElementSize);
    float *CPUResults = (float *)malloc(iElementSize);
    
    if(fpHostA!=NULL && fpHostB!=NULL&& fpHostC!=NULL){
        memset(fpHostA, 0, iElementSize);
        memset(fpHostB, 0, iElementSize);
        memset(fpHostC, 0, iElementSize);
    }else{
        printf("allocate memory failed\n");
        exit(-1);
    }
    
    srand(666);
    initialData(fpHostA, iSize);
    initialData(fpHostB, iSize);
    
    float *fpDevA = nullptr;
    float *fpDevB = nullptr;
    float *fpDevC = nullptr;

    cudaMalloc(&fpDevA, iElementSize);
    cudaMalloc(&fpDevB, iElementSize);
    cudaMalloc(&fpDevC, iElementSize);
    
    if(fpDevA !=NULL && fpDevB !=NULL && fpDevC !=NULL){
        cudaMemset(fpDevA, 0, iElementSize);
        cudaMemset(fpDevB, 0, iElementSize);
        cudaMemset(fpDevC, 0, iElementSize);
    }else{
        printf("allocate memory failed\n");
        free(fpHostA);
        free(fpHostB);
        free(fpHostC);
        exit(-1);
    }
    
    cudaMemcpy(fpDevA, fpHostA, iElementSize, cudaMemcpyHostToDevice);
    cudaMemcpy(fpDevB, fpHostB, iElementSize, cudaMemcpyHostToDevice);
    
    dim3 blocks(16);
    dim3 blocksPerGrid(iSize + blocks.x - 1 / blocks.x);
    add_from_gpu<<<blocksPerGrid, blocks>>>(fpDevA, fpDevB, fpDevC, iSize);
    cudaDeviceSynchronize();
    cudaMemcpy(fpHostC, fpDevC, iElementSize, cudaMemcpyDeviceToHost);

    add_from_cpu(fpHostA, fpHostB, CPUResults, iSize);

    if(!compare_cpu_gpu_results(fpHostC, CPUResults, iSize)){
        printf("compare_cpu_gpu_results failed\n");
        exit(-1);
    }else{
        printf("compare_cpu_gpu_results success\n");
    }

    for(int i =0;i<10;i++){
        printf("matrixSum1D_CPU:fpHostC[%d] = %f\n", i, fpHostC[i]);
    }

    cudaFreeHost(fpHostA);
    cudaFreeHost(fpHostB);
    cudaFreeHost(fpHostC);
    cudaFree(fpDevA);
    cudaFree(fpDevB);
    cudaFree(fpDevC);
    free(CPUResults);
    cudaDeviceReset();
    return 0;
}