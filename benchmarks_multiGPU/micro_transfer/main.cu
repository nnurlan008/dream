#include <cstdio>
#include <cuda.h>
#include <cuda_runtime.h>
#include <string>
#include <cstring>
#include <unistd.h>
#include <stdio.h>
#include <time.h>
#include <sys/time.h>
#include <stdlib.h>
#include <stdarg.h>

#include "../../include/runtime.h"

double rtclock()
{
    struct timezone Tzp;
    struct timeval Tp;
    int stat;
    stat = gettimeofday (&Tp, &Tzp);
    if (stat != 0) printf("Error return from gettimeofday: %d",stat);
    return(Tp.tv_sec + Tp.tv_usec*1.0e-6);
}

// Main program
int main(int argc, char **argv)
{   
    init_gpu(0);
    int *h_buffer, *d_buffer;
    size_t size = 1024*1024*1024llu;

    // if(cudaSuccess != cudaMallocHost((void **) &h_buffer, size*sizeof(int))){
    //     printf("Error on cudaMallocHost\n");
    //     return -1;
    // }

    h_buffer = (int *) malloc(size*sizeof(int));
    if(h_buffer == NULL){
        printf("Error on malloc\n");
        return -1;
    }

    if(cudaSuccess != cudaMalloc((void **) &d_buffer, size*sizeof(int))){
        printf("Error on cudaMalloc\n");
        return -1;
    }
    
    for (size_t i = 0; i < size; i++)
    {
        h_buffer[i] = i;
    }

    int ret = cudaDeviceSynchronize();
    if(cudaSuccess != ret){
        printf("Error on cudaDeviceSynchronize: %d\n", ret);
        return -1;
    }

    double t_start, t_end;

    t_start = rtclock();
    ret = cudaMemcpy(d_buffer, h_buffer, size*sizeof(int), cudaMemcpyHostToDevice);
    if(cudaSuccess != ret){
        printf("Error on cudaMemcpy: %d\n", ret);
        return -1;
    }
    t_end = rtclock();

    fprintf(stdout, "Copy time: %0.6lfs\n", t_end - t_start);
    

	return 0;
}
