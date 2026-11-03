#include <stdio.h>
// #include <stdlib.h>
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


int main(void){
    int iSize = 512;
    size_t iElementSize = iSize * sizeof(float);
    float *fpHostA = (float *)malloc(iElementSize);
    float *fpHostB = (float *)malloc(iElementSize);
    float *fpHostC = (float *)malloc(iElementSize);
    
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

    add_from_cpu(fpHostA, fpHostB, fpHostC, iSize);

    for(int i =0;i<10;i++){
        printf("matrixSum1D_CPU:fpHostC[%d] = %f\n", i, fpHostC[i]);
    }
    free(fpHostA);
    free(fpHostB);
    free(fpHostC);
    return 0;
}