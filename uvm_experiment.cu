#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <sys/time.h>
#include <time.h>

#include "experiment_utils.h"

__global__ void uvm_access(int *array, int length, __clock_t *timer, int *dummy);
__device__ int uvm_access1(int *array, int length, __clock_t *timer, int *dummy);

/* This experiment is designed to find the the bandwidth for UVM, direct CudaMemCpy transfer
   and compare the results with rdma transfers 
   
   */

int main(int argc, char **argv){

    // if (argc != 6)
    //     usage(argv[0]);
    

    int num_msg = (unsigned long) atoi(argv[4]);
    int mesg_size = (unsigned long) atoi(argv[5]);

    int *array, *dev_array, size = 64*1024+1;
    clock_t timer[2], *dtimer;
    cudaError_t cudaState;
    // array = (int *) malloc(sizeof(int) * size);

    // cudaState = cudaHostRegister(array, sizeof(array), cudaHostRegisterMapped);
    // // printf("Function: %s line number: %d cudaState: %d\n",__func__, __LINE__, cudaState);
    // // comment the below lines when gpu buffer used for qp->buf.buf
    // printf("Function: %s line number: %d\n",__func__, __LINE__);
    // if(cudaState != cudaSuccess && cudaState != cudaErrorHostMemoryAlreadyRegistered) 
    //     exit(0);
    // // printf("Function: %s line number: %d\n",__func__, __LINE__);
    // if(cudaHostGetDevicePointer(&dev_array, array, 0) != cudaSuccess)
    //     exit(0);

    cudaState = cudaMallocManaged(&array, size*sizeof(int));

    if(cudaState != cudaSuccess) exit(-1);
    if(cudaMalloc((void **) &dtimer, 2*sizeof(clock_t)) != cudaSuccess)
    {
        printf("dtimer allocation error\n");
        exit(-1);   
    }

    for(int i = 0; i < size; i++)
        array[i] = 1;

    int *dummy, *hdummy, size_dummy = 1;
    hdummy = (int *) malloc(size_dummy*sizeof(int));
    cudaError_t ret = cudaMalloc((void **)&dummy, size_dummy*sizeof(int));
    
    struct timeval start, end;

    cudaDeviceSynchronize();
    gettimeofday(&start, NULL);
    uvm_access<<<1, 1>>>(array, size, dtimer, dummy);
    gettimeofday(&end, NULL);
    cudaDeviceSynchronize();


    cudaMemcpy(hdummy, dummy, sizeof(int) * size_dummy, cudaMemcpyDeviceToHost);
    cudaMemcpy(timer, dtimer, sizeof(clock_t) * (2), cudaMemcpyDeviceToHost);
    cudaFree(dummy);
    cudaFree(dtimer);

    double time_taken;

    time_taken = (end.tv_sec - start.tv_sec) * 1e6;
    time_taken = (time_taken + (end.tv_usec - 
                            start.tv_usec)) * 1e-6;

    cudaDeviceProp devProp;
    cudaGetDeviceProperties(&devProp, 0);

    printf("Cuda device clock rate = %d\n", devProp.clockRate);
    printf("array[0]: %d\n", array[0]);
    printf("array[1]: %d\n", array[1]);

    float freq = (float)1/(devProp.clockRate*1000);
    float g_usec = (float)((float)1/(devProp.clockRate*1000))*((timer[1]-timer[0])) * 1000000;
	printf("POLLING - INTERNAL MEASUREMENT: %f seconds to execute \n", time_taken);

    return 1;
}

__global__ void uvm_access(int *array, int length, __clock_t *timer, int *dummy){
    // clock_t start, end;
    // int dummy_var, tempDummy;
    // start = clock();
    // tempDummy = 2*start;
    // // tempDummy = array[0]*((int)(start>1203));
    // // tempDummy = array[tempDummy]*tempDummy;
    // dummy_var = tempDummy + start;
    // end = clock() + dummy_var - start*3;
    // timer[0] = start;
    // timer[1] = end;
    // *dummy = dummy_var;
    timer[0] = clock64();
    // int ret = uvm_access1(array, length, timer, dummy);
    // array[0] = length;
    // array[length-1] = array[0]*2 + 6;
    *dummy = array[0] + array[16*1024-1];
    *dummy = *dummy*2;
    *dummy = *dummy*array[16*1024] + array[32*1024-1];
    *dummy = *dummy*2;
    *dummy = *dummy*array[32*1024] + array[48*1024-1];
    *dummy = *dummy*2;
    *dummy = *dummy*array[48*1024] + array[64*1024-1];
    *dummy = *dummy*2;
    *dummy = *dummy*array[64*1024+1];
    *dummy = *dummy*2;
    timer[1] = clock64();
    printf("cycles : %lld\n", timer[1] - timer[0]);
}

__device__ int uvm_access1(int *array, int length, __clock_t *timer, int *dummy){
    // clock_t start, end;
    // int dummy_var, tempDummy;
    // start = clock();
    // tempDummy = 2*start;
    // // tempDummy = array[0]*((int)(start>1203));
    // // tempDummy = array[tempDummy]*tempDummy;
    // dummy_var = tempDummy + start;
    // end = clock() + dummy_var - start*3;
    // timer[0] = start;
    // timer[1] = end;
    // *dummy = dummy_var;
    // array[0] = length;
    // array[length-1] = array[0]*2 + 6;
    // *dummy = array[0]; // + array[length-1];

    // *(array + 1) = 4;
    // *(array + 2) = 4;

}