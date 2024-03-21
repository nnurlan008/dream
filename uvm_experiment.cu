#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <sys/time.h>
#include <time.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <errno.h>
#include <string.h>

#include "experiment_utils.h"

// #define PROT_READ	0x1		/* Page can be read.  */
// #define PROT_WRITE	0x2		/* Page can be written.  */
// #define PROT_EXEC	0x4		/* Page can be executed.  */

__global__ void uvm_access(int *array, int length, __clock_t *timer, int *dummy);
__device__ int uvm_access1(int *array, int length, __clock_t *timer, int *dummy);

/* This experiment is designed to find the the bandwidth for UVM, direct CudaMemCpy transfer
   and compare the results with rdma transfers 
   
   */

int main(int argc, char **argv){

    // if (argc != 6)
    //     usage(argv[0]);
    
    // int fd;
    // char *path = "/dev/infiniband/uverbs0";

    // if (asprintf(&path, RDMA_CDEV_DIR "/%s", devname_hint) < 0)
	// 	return -1;

	// fd = open(path, O_RDWR | O_CLOEXEC);
    // printf("fd: %d\n", fd);
    // void *ptr;
    

    // if(posix_memalign(&ptr, 64*1024, 64*4096)) {
	// 	printf("Failed to allocated memory for DB register\n");
    //     return 1;
    // }

    // ptr = mmap(NULL, 32*4096, PROT_READ | PROT_WRITE, MAP_ANON, fd, 0);
    // if (ptr == MAP_FAILED) {
    //     printf("mmap failed: %s\n", strerror(errno));
    //     exit(EXIT_FAILURE);
    // }
    // printf("mmap: 0x%llx\n", ptr);

  
    cudaError_t cudaState;
    // array = (int *) malloc(sizeof(int) * size);

    // cudaState = cudaHostRegister(ptr, 32*4096, cudaHostRegisterIoMemory);
    // printf("Function: %s line number: %d cudaState: %d, errno: %s\n", __func__, __LINE__, cudaState, cudaGetErrorString(cudaState));
    // // comment the below lines when gpu buffer used for qp->buf.buf
    // // printf("Function: %s line number: %d\n",__func__, __LINE__);
    // if(cudaState != cudaSuccess && cudaState != cudaErrorHostMemoryAlreadyRegistered) 
    //     exit(0);
    // printf("Function: %s line number: %d cudaState: %d\n",__func__, __LINE__, cudaState);

      int *array, *dev_array, size = 64*8*1024+1, size_timer = 20;
    clock_t timer[size_timer], *dtimer;
    // printf("Function: %s line number: %d\n",__func__, __LINE__);
    // if(cudaHostGetDevicePointer(&dev_array, ptr, 0) != cudaSuccess)
    //     exit(0);
    // printf("Function: %s line number: %d cudaState: %d\n",__func__, __LINE__, cudaState);
    cudaState = cudaMallocManaged(&array, size*sizeof(int));

    if(cudaState != cudaSuccess) exit(-1);
    if(cudaMalloc((void **) &dtimer, size_timer*sizeof(clock_t)) != cudaSuccess)
    {
        printf("dtimer allocation error\n");
        exit(-1);   
    }
    printf("Function: %s line number: %d cudaState: %d\n",__func__, __LINE__, cudaState);
    for(int i = 0; i < size; i++)
        array[i] = 1;

    int *dummy, *hdummy, size_dummy = 1;
    hdummy = (int *) malloc(size_dummy*sizeof(int));
    cudaError_t ret = cudaMalloc((void **)&dummy, size_dummy*sizeof(int));
    printf("Function: %s line number: %d cudaState: %d\n",__func__, __LINE__, cudaState);
    struct timeval start, end;

    cudaDeviceSynchronize();
    gettimeofday(&start, NULL);
    uvm_access<<<1, 1>>>(array, size, dtimer, dummy);
    gettimeofday(&end, NULL);
    cudaDeviceSynchronize();


    cudaMemcpy(hdummy, dummy, sizeof(int) * size_dummy, cudaMemcpyDeviceToHost);
    cudaMemcpy(timer, dtimer, sizeof(clock_t) * size_timer, cudaMemcpyDeviceToHost);
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
    
    int ret = uvm_access1(array, length, timer, dummy);
    *dummy = ret;
    
    printf("cycles : %lld\n", timer[1] - timer[0]);
    printf("cycles : %lld\n", timer[2] - timer[1]);
    printf("cycles : %lld\n", timer[3] - timer[2]);
    printf("cycles : %lld\n", timer[4] - timer[3]);
    printf("cycles : %lld\n", timer[5] - timer[4]);
}

__device__ int uvm_access1(int *array, int length, __clock_t *timer, int *dummy){
    timer[0] = clock64();
    array[0] = length;
    timer[1] = clock64();
    // array[length-1] = array[0]*2 + 6;
    // timer[2] = clock64();
    *dummy = array[0] + array[16*1024-1] + timer[1];
    timer[2] = clock64();
    // *dummy = *dummy*2;
    *dummy = *dummy*array[16*1024] + array[32*1024-1];
    timer[3] = clock64();
    // *dummy = *dummy*2;
    *dummy = *dummy*array[32*1024] + array[48*1024-1];
    timer[4] = clock64();
    // *dummy = *dummy*2;
    *dummy = *dummy*array[48*1024] + array[64*1024-1];
    timer[5] = clock64();
    // *dummy = *dummy*2;
   
   return (*dummy)+3;

}