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

#include "polybenchUtilFuncts.h"

using namespace std;
// using namespace std;


// extern "C"{
//   #include "rdma_utils.h"
// }

// #include "../../src/rdma_utils.cuh"
#include <time.h>
#include "../../include/runtime.h"


// Size of array
// #define N 1*1024*1024llu


// Kernel
__global__ void add_vectors_uvm(int *a, int *b, int *c, int size)
{
	int id = blockDim.x * blockIdx.x + threadIdx.x;
	// if(id < size) {
		c[id] = a[id] + b[id];
		// printf("c[%d]: %d\n", id, c[id]);
	// }
}

#define htonl(x)  ((((uint32_t)(x) & 0xff000000) >> 24) |\
                   (((uint32_t)(x) & 0x00ff0000) >>  8) |\
                   (((uint32_t)(x) & 0x0000ff00) <<  8) |\
                   (((uint32_t)(x) & 0x000000ff) << 24))


void delay(int number_of_seconds)
{
    // Converting time into milli_seconds
    int milli_seconds = 1000000 * number_of_seconds;
 
    // Storing start time
    clock_t start_time = clock();
 
    // looping till required time is not achieved
    while (clock() < start_time + milli_seconds)
        ;
}

enum { NS_PER_SECOND = 1000000000 };

void sub_timespec(struct timespec t1, struct timespec t2, struct timespec *td)
{
    td->tv_nsec = t2.tv_nsec - t1.tv_nsec;
    td->tv_sec  = t2.tv_sec - t1.tv_sec;
    if (td->tv_sec > 0 && td->tv_nsec < 0)
    {
        td->tv_nsec += NS_PER_SECOND;
        td->tv_sec--;
    }
    else if (td->tv_sec < 0 && td->tv_nsec > 0)
    {
        td->tv_nsec -= NS_PER_SECOND;
        td->tv_sec++;
    }
}

void usage(const char *argv0)
{
  fprintf(stderr, "usage: %s <mode> <server-address> <server-port>\n  mode = \"read\", \"write\"\n", argv0);
  exit(1);
}

//define the error threshold for the results "not matching"
#define PERCENT_DIFF_ERROR_THRESHOLD 0.05

#define GPU_DEVICE 0

/* Problem size */
#define N1 (4096*8llu)

/* Thread block dimensions */
#define DIM_THREAD_BLOCK_X 256
#define DIM_THREAD_BLOCK_Y 1

/* Declared constant values for ALPHA and BETA (same as values in PolyBench 2.0) */
#define ALPHA 43532.0f
#define BETA 12313.0f

#define PREF 0

/* Can switch DATA_TYPE between float and double */
typedef float DATA_TYPE;



void gesummv(DATA_TYPE *A, DATA_TYPE *B, DATA_TYPE *x, DATA_TYPE *y, DATA_TYPE *tmp)
{
	size_t i, j;
	
	for (i = 0; i < N1; i++)
	{
		tmp[i] = 0;
		y[i] = 0;
		for (j = 0; j < N1; j++)
		{
            
			tmp[i] = A[i*N1 + j] * x[j] + tmp[i];
			y[i] = B[i*N1 + j] * x[j] + y[i];
            // if(j == 1){
            //     printf("index: %llu A[i*N1 + j]: %f, B[i*N1 + j]: %f\n", i*N1 + j, A[i*N1 + j], B[i*N1 + j]);
            //     printf("i: %llu tmp_: %f, y_: %f\n", i, tmp[i], y[i]);
            // }
           
		}
		// printf("i: %llu tmp_: %f, y_: %f ALPHA * tmp[i] + BETA * y[i]: %f\n", i, tmp[i], y[i], ALPHA * tmp[i] + BETA * y[i]);
		y[i] = ALPHA * tmp[i] + BETA * y[i];
        // printf("i: %llu tmp_: %f, y_: %f\n", i, tmp[i], y[i]);
        // printf("y[%llu]: %f\n", i, y[i]);
	}
}


void init(DATA_TYPE *A, DATA_TYPE *B, DATA_TYPE *x, rdma_buf<DATA_TYPE> *A_gpu, rdma_buf<DATA_TYPE> *B_gpu,\
          rdma_buf<DATA_TYPE> *x_gpu)
{
  	int i, j;

    printf("A[i*N1]: %d\n", A[i*N1]);
 	for (i = 0; i < N1; i++)
    {
		x[i] = ((DATA_TYPE) i) / N1;
		x_gpu->local_buffer[i] = x[i]; // ((DATA_TYPE) i) / N1;
      	printf("A[%d]: %d\n", i*N1, A[i*N1]);
		for (j = 0; j < N1; j++)
		{
			A[i*N1 + j] = ((DATA_TYPE) i*j) / N1;
			B[i*N1 + j] = ((DATA_TYPE) i*j) / N1;
			A_gpu->local_buffer[i*N1 + j] = A[i*N1 + j]; // ((DATA_TYPE) i*j) / N1;
			B_gpu->local_buffer[i*N1 + j] = B[i*N1 + j] ; // ((DATA_TYPE) i*j) / N1;
		}
    }
}


void compareResults(DATA_TYPE* y, rdma_buf<DATA_TYPE> *y_outputFromGpu)
{
	int i, fail;
	fail = 0;
	
	for (i=0; i<(N1); i++) 
	{
		if (percentDiff(y[i], y_outputFromGpu->local_buffer[i]) > PERCENT_DIFF_ERROR_THRESHOLD) 
		{
            // printf("i: %llu y[i]: %f, y_outputFromGpu->local_buffer[i]: %f\n", i, y[i], y_outputFromGpu->local_buffer[i]);
			fail++;
		}
	}
	
	// Print results
	printf("Non-Matching CPU-GPU Outputs Beyond Error Threshold of %4.2f Percent: %d\n", PERCENT_DIFF_ERROR_THRESHOLD, fail);
}


void GPU_argv_init()
{
	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, GPU_DEVICE);
	printf("setting device %d with name %s\n",GPU_DEVICE,deviceProp.name);
	cudaSetDevice( GPU_DEVICE );
}


__global__ void gesummv_kernel(rdma_buf<DATA_TYPE> *a, rdma_buf<DATA_TYPE> *b, rdma_buf<DATA_TYPE> *x,\
                               rdma_buf<DATA_TYPE> *y, rdma_buf<DATA_TYPE> *tmp)
{
	size_t i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < N1)
	{
		size_t j;
        DATA_TYPE tmp_var = 0, y_var = 0;
		for(j = 0; j < N1; j++)
		{	
            size_t index = i * N1 + j;
            DATA_TYPE x_ = (*x)[j];
            DATA_TYPE a_ = (*a)[index];
            DATA_TYPE b_ = (*b)[index];

            tmp_var += a_*x_;
            y_var += b_*x_;
            
		}
		// y[i] = ALPHA * tmp[i] + BETA * y[i];
        // tmp_ = (*tmp)[i];
        // y_ = (*y)[i];
        // printf("i: %llu tmp_: %f, y_: %f ALPHA*tmp_ + BETA*y_: %f\n", i, (*tmp)[i], (*y)[i], ALPHA*tmp_var + BETA*y_var);
        (*y).rvalue(i, ALPHA*tmp_var + BETA*y_var);
        // printf("i: %llu tmp_var: %f, y_: %f\n", i, tmp_var, (*y)[i]);
        // printf("(*y)[%llu]: %f\n", i, (*y)[i]);
	}
}

// void gesummvCuda(DATA_TYPE* A_gpu, DATA_TYPE* B_gpu, DATA_TYPE* x_gpu, DATA_TYPE* y_gpu, DATA_TYPE* tmp_gpu)
// {
// 	cudaStream_t stream1;
// 	cudaStream_t stream2;
// 	cudaStream_t stream3;
// 	cudaStream_t stream4;
// 	cudaStream_t stream5;
// 	cudaStreamCreate(&stream1);
// 	cudaStreamCreate(&stream2);
// 	cudaStreamCreate(&stream3);
// 	cudaStreamCreate(&stream4);
// 	cudaStreamCreate(&stream5);


// 	if(1){
// 		printf("Pref: 1\n");
// 		double t_start, t_end;		
// 		dim3 block(DIM_THREAD_BLOCK_X, DIM_THREAD_BLOCK_Y);
// 		dim3 grid((unsigned int)ceil( ((float)N) / ((float)block.x) ), 1);
// 		// cudaMemPrefetchAsync(A_gpu,N*N*sizeof(DATA_TYPE), GPU_DEVICE, stream1 );
// 		// cudaMemPrefetchAsync(B_gpu,N*N*sizeof(DATA_TYPE), GPU_DEVICE, stream2 );
// 		// cudaMemPrefetchAsync(x_gpu,N*sizeof(DATA_TYPE), GPU_DEVICE, stream3 );
// 		// cudaMemPrefetchAsync(y_gpu,N*sizeof(DATA_TYPE), GPU_DEVICE, stream4 );
// 		// cudaMemPrefetchAsync(tmp_gpu,N*sizeof(DATA_TYPE), GPU_DEVICE, stream5 );
// 		// cudaStreamSynchronize(stream1);
// 		// cudaStreamSynchronize(stream2);
// 		// cudaStreamSynchronize(stream3);
// 		// cudaStreamSynchronize(stream4);
// 		// cudaStreamSynchronize(stream5);
// 		cudaDeviceSynchronize();
// 		t_start = rtclock();
// 		for (int i = 0; i < 1; i++){
// 		gesummv_kernel<<< grid, block, 0>>>(A_gpu,B_gpu,x_gpu, y_gpu, tmp_gpu);
// 		cudaDeviceSynchronize();
// 		}
// 		t_end = rtclock();
// 		fprintf(stdout, "GPU Runtime: %0.6lfs\n", t_end - t_start);
// 	}
// 	else{
// 		printf("Pref: 0\n");
// 		cudaDeviceSynchronize();
// 		double t_start, t_end;		
// 		dim3 block(DIM_THREAD_BLOCK_X, DIM_THREAD_BLOCK_Y);
// 		dim3 grid((unsigned int)ceil( ((float)N) / ((float)block.x) ), 1);
// 		t_start = rtclock();
// 		for (int i = 0; i < 1; i++){
// 		gesummv_kernel<<< grid, block>>>(A_gpu,B_gpu,x_gpu, y_gpu, tmp_gpu);
// 		cudaDeviceSynchronize();
// 		}
// 		t_end = rtclock();
// 		fprintf(stdout, "GPU Runtime: %0.6lfs\n", t_end - t_start);
// 	}
// 	// #endif
// }

int alloc_global_cont(struct post_content *post_cont, struct poll_content *poll_cont, struct post_content2 *post_cont2){
    struct post_content *d_post;
    struct poll_content *d_poll;
    struct post_content2 *d_post2;

    cudaError_t ret0 = cudaMalloc((void **)&d_post, sizeof(struct post_content));
    if(ret0 != cudaSuccess){
        printf("Error on allocation post content!\n");
        return -1;
    }
    ret0 = cudaMalloc((void **)&d_poll, sizeof(struct poll_content));
    if(ret0 != cudaSuccess){
        printf("Error on allocation poll content!\n");
        return -1;
    }
    printf("sizeof(struct post_content): %d, sizeof(struct poll_content): %d\n", sizeof(struct post_content), sizeof(struct poll_content));
    ret0 = cudaMemcpy(d_post, post_cont, sizeof(struct post_content), cudaMemcpyHostToDevice);
    if(ret0 != cudaSuccess){
        printf("Error on post copy!\n");
        return -1;
    }
    ret0 = cudaMemcpy(d_poll, poll_cont, sizeof(struct poll_content), cudaMemcpyHostToDevice);
    if(ret0 != cudaSuccess){
        printf("Error on poll copy!\n");
        return -1;
    }

    ret0 = cudaMalloc((void **)&d_post2, sizeof(struct post_content2));
    if(ret0 != cudaSuccess){
        printf("Error on allocation post content!\n");
        return -1;
    }
    ret0 = cudaMemcpy(d_post2, post_cont2, sizeof(struct post_content2), cudaMemcpyHostToDevice);
    if(ret0 != cudaSuccess){
        printf("Error on poll copy!\n");
        return -1;
    }

    alloc_content<<<1,1>>>(d_post, d_poll);
    alloc_global_content<<<1,1>>>(d_post, d_poll, d_post2);
    ret0 = cudaDeviceSynchronize();
    if(ret0 != cudaSuccess){
        printf("Error on alloc_content!\n");
        return -1;
    }
    return 0;
}

// __global__ void test_uvm(DATA_TYPE *a, DATA_TYPE *b, size_t size){
    
//     size_t id = blockDim.x * blockIdx.x + threadIdx.x;
//     size_t stride = blockDim.x * gridDim.x;
    

//     for(size_t i = id; i < size; i += stride){
//         int k = (*a)[i] + (*b)[i];
        
//     }

    
// }

__global__ void test(rdma_buf<DATA_TYPE> *a, rdma_buf<DATA_TYPE> *b, size_t size){
    
    size_t id = blockDim.x * blockIdx.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;
    

    for(size_t i = id; i < size; i += stride){
        int k = (*a)[i] + (*b)[i];
        
    }

    
}

// Main program
int main(int argc, char **argv)
{   
    if (argc != 7)
        usage(argv[0]);
    
    init_gpu(0);
    printf("Function: %s line number: %d 1024MB: %d bytes REQUEST_SIZE: %d\n",__func__, __LINE__, MB(1024), REQUEST_SIZE);
    int num_msg = (unsigned long) atoi(argv[4]);
    int mesg_size = (unsigned long) atoi(argv[5]);
    int num_bufs = (unsigned long) atoi(argv[6]);

    struct context *s_ctx = (struct context *)malloc(sizeof(struct context));
    struct post_content post_cont, *d_post, host_post;
    struct poll_content poll_cont, *d_poll, host_poll;
    struct post_content2 post_cont2, *d_post2;
    struct host_keys keys;

    int num_iteration = num_msg;
    s_ctx->n_bufs = num_bufs;
    s_ctx->gpu_buf_size = 12*1024*1024*1024llu; // N*sizeof(int)*3llu;

    // remote connection:
    int ret = connect(argv[2], s_ctx);

    // local connect
    // char *mlx_name = "mlx5_0";
    // int ret = local_connect(mlx_name, s_ctx);

    cudaError_t ret1 = cudaDeviceSynchronize();
    printf("ret: %d\n", ret1);
    if(cudaSuccess != ret1){    
        return -1;
    }

    ret = prepare_post_poll_content(s_ctx, &post_cont, &poll_cont, &post_cont2, \
                                    &host_post, &host_poll, &keys);
    if(ret == -1) {
        printf("Post and poll contect creation failed\n");    
        return -1;
    }

    printf("alloc synDev ret: %d\n", cudaDeviceSynchronize());
    alloc_global_cont(&post_cont, &poll_cont, &post_cont2);
    // if(cudaSuccess != ){    
    printf("alloc synDev ret1: %d\n", cudaDeviceSynchronize());
        // return -1;
    // }

    ret1 = cudaDeviceSynchronize();
    printf("ret: %d\n", ret1);
    if(cudaSuccess != ret1){    
        return -1;
    }

    printf("function: %s line: %d\n", __FILE__, __LINE__);
    alloc_global_host_content(host_post, host_poll, keys);
    printf("function: %s line: %d\n", __FILE__, __LINE__);

    ret1 = cudaDeviceSynchronize();
    printf("ret: %d\n", ret1);
    if(cudaSuccess != ret1){    
        return -1;
    }

    // // rdma_buf<int> buf((uint64_t) s_ctx->gpu_buffer, (uint64_t) s_ctx->server_mr.addr, 100);
    // rdma_buf<int> *buf1, *buf2, *buf3;
    // cudaMallocManaged(&buf1, sizeof(rdma_buf<int>));
    // cudaMallocManaged(&buf2, sizeof(rdma_buf<int>));
    // cudaMallocManaged(&buf3, sizeof(rdma_buf<int>));
   
    // // printf("s_ctx->gpu_buffer: %p, buf1->size: %d, Address_Offset: %d\n", s_ctx->gpu_buffer, buf1->size, Address_Offset);
    
    // // buf1->start(N*sizeof(int));
    // // printf("s_ctx->gpu_buffer: %p, buf1->size: %d, Address_Offset: %d\n", s_ctx->gpu_buffer, buf1->size, Address_Offset);
    // // buf2->start(N*sizeof(int));
    
    // printf("s_ctx->gpu_buffer: %p, buf1->size: %d, Address_Offset: %d\n", s_ctx->gpu_buffer, buf1->size, Address_Offset);
    
    
    
    double t_start, t_end;

	DATA_TYPE* A;
	DATA_TYPE* B;  
	DATA_TYPE* x;  
	DATA_TYPE* y;
	DATA_TYPE* tmp;

	// DATA_TYPE *A_gpu;
	// DATA_TYPE *B_gpu;
	// DATA_TYPE *x_gpu;
	// DATA_TYPE *y_gpu;
	// DATA_TYPE *tmp_gpu;

    rdma_buf<DATA_TYPE> *A_gpu;
    rdma_buf<DATA_TYPE> *B_gpu;
    rdma_buf<DATA_TYPE> *x_gpu;
    rdma_buf<DATA_TYPE> *y_gpu;
    rdma_buf<DATA_TYPE> *tmp_gpu;
	
	A = (DATA_TYPE*)malloc(N1*N1*sizeof(DATA_TYPE));
	B = (DATA_TYPE*)malloc(N1*N1*sizeof(DATA_TYPE));
	x = (DATA_TYPE*)malloc(N1*sizeof(DATA_TYPE)); 
	y = (DATA_TYPE*)malloc(N1*sizeof(DATA_TYPE));
	tmp = (DATA_TYPE*)malloc(N1*sizeof(DATA_TYPE));

	// cudaDeviceProp devProp;
    // cudaGetDeviceProperties(&devProp, 0);
	// // Calculate memory utilization
    // size_t totalMemory = devProp.totalGlobalMem;
    // size_t freeMemory;
    // size_t usedMemory;
	// float workload_size = ((float) sizeof(DATA_TYPE)*N*N*2 + sizeof(DATA_TYPE)*N*3);
    // cudaMemGetInfo(&freeMemory, &totalMemory);
	// usedMemory = totalMemory - freeMemory;
	// printf("Total GPU Memory: %.2f MiB\n", (float) totalMemory / (1024 * 1024));
	// printf("Free GPU Memory: %.2f MiB\n", (float) freeMemory / (1024 * 1024));
	// printf("Used GPU Memory: %.2f MiB\n", (float) usedMemory / (1024 * 1024));

	// printf("Workload size: %.2f\n", workload_size/1024/1024);
	// float oversubs_ratio = 0;
	// void *tmp_ptr;
	// cudaMalloc(&tmp_ptr, (size_t) (freeMemory - workload_size));
	// cudaMemGetInfo(&freeMemory, &totalMemory);
	// printf("Free GPU Memory: %.2f MiB\n", (float) freeMemory / (1024 * 1024));
	// if(oversubs_ratio > 0){
		
	// 	void *over_ptr;
	// 	long long unsigned int os_size = freeMemory - workload_size /(1 + oversubs_ratio);
	// 	printf("workload: %.2f\n",  workload_size);
	// 	printf("workload: %llu\n",  os_size);
	// 	cudaMalloc(&over_ptr, os_size); 
	// 	printf("os_size: %u\n", os_size/1024/1024);
	// }

	// printf("Free GPU Memory after allocation: %.2f\n", (float) freeMemory - workload_size );

	// cudaMallocManaged((void **)&A_gpu, sizeof(DATA_TYPE) * N * N);
	// cudaMallocManaged((void **)&B_gpu, sizeof(DATA_TYPE) * N * N);
	// cudaMallocManaged((void **)&x_gpu, sizeof(DATA_TYPE) * N);
	// cudaMallocManaged((void **)&y_gpu, sizeof(DATA_TYPE) * N);
	// cudaMallocManaged((void **)&tmp_gpu, sizeof(DATA_TYPE) * N);

    cudaMallocManaged((void **)&A_gpu, sizeof(rdma_buf<DATA_TYPE>));
	cudaMallocManaged((void **)&B_gpu, sizeof(rdma_buf<DATA_TYPE>));
	cudaMallocManaged((void **)&x_gpu, sizeof(rdma_buf<DATA_TYPE>) );
	cudaMallocManaged((void **)&y_gpu, sizeof(rdma_buf<DATA_TYPE>));
	cudaMallocManaged((void **)&tmp_gpu, sizeof(rdma_buf<DATA_TYPE>));

    A_gpu->start(sizeof(DATA_TYPE) * N1 * N1);
    B_gpu->start(sizeof(DATA_TYPE) * N1 * N1);
    x_gpu->start(sizeof(DATA_TYPE) * N1);
    y_gpu->start(sizeof(DATA_TYPE) * N1);
    tmp_gpu->start(sizeof(DATA_TYPE) * N1);

    // for (size_t i = 0; i < N1*N1; i++)
    // {
    //     A_gpu->local_buffer[i] = 12.4;
    // }
    // printf("debug - line: %d function: %s A_gpu->local_buffer: %p\n", __LINE__, __func__, A_gpu->local_buffer);
    // A_gpu->memcpyHostToServer();
    // A_gpu->local_buffer[A_gpu->size/sizeof(DATA_TYPE)-1] = 9.0;
    // A_gpu->local_buffer[0] = 10.0;
    // A_gpu->memcpyServerToHost();
    // memcpyServerToHost_global();
    // // printf("debug - line: %d function: %s transfer_size: %llu n_regions: %d\n", __LINE__, __func__, transfer_size, n_regions);
    // for (size_t i = 0; i < 256; i++)
    // {
        // printf("A_gpu[%d]: %f ", 0, A_gpu->local_buffer[0]);
        // printf("A_gpu[%d]: %f ", A_gpu->size/sizeof(DATA_TYPE)-1, A_gpu->local_buffer[A_gpu->size/sizeof(DATA_TYPE)-1]);
    // }

    

	// Check if allocations are successful
    if (A == NULL || B == NULL || x == NULL || A_gpu->local_buffer == NULL || B_gpu->local_buffer == NULL || x_gpu->local_buffer == NULL) {
        printf("Memory allocation failed!\n");
        return -1;
    }

	printf("init... \n");
	init(A, B, x, A_gpu, B_gpu, x_gpu);
    A_gpu->memcpyHostToServer();
    B_gpu->memcpyHostToServer();
    x_gpu->memcpyHostToServer();
	printf("init finished\n");

	GPU_argv_init();
	// gesummvCuda(A_gpu, B_gpu, x_gpu, y_gpu, tmp_gpu);
	
    printf("Pref: 0\n");
    ret1 = cudaDeviceSynchronize();
    printf("cudaDeviceSynchronize: %d\n", ret1); 
    if(cudaSuccess != ret1){  
        printf("cudaDeviceSynchronize error: %d\n", ret1);  
        exit(-1);
    }
    printf("cudaGetLastError(): %d\n", cudaGetLastError());
    // double t_start, t_end;		
    dim3 block(DIM_THREAD_BLOCK_X, DIM_THREAD_BLOCK_Y);
    dim3 grid((unsigned int)ceil( ((float)N1) / ((float)block.x) ), 1);

    cudaEvent_t event1, event2;
    cudaEventCreate(&event1);
    cudaEventCreate(&event2);

    t_start = rtclock();
    cudaEventRecord(event1, (cudaStream_t)0); //where 0 is the default stream
    // gesummv_kernel<<< grid, block>>>(A_gpu,B_gpu,x_gpu, y_gpu, tmp_gpu);
    test<<<2048, 512>>>(A_gpu, B_gpu, A_gpu->size/sizeof(DATA_TYPE));
    cudaEventRecord(event2, (cudaStream_t) 0);
    printf("cudaGetLastError(): %d\n", cudaGetLastError());
    ret1 = cudaDeviceSynchronize();   
    
    // //synchronize
    cudaEventSynchronize(event1); //optional
    cudaEventSynchronize(event2); //wait for the event to be executed!

    // //calculate time
    float dt_ms;
    cudaEventElapsedTime(&dt_ms, event1, event2);
    printf("Cuda time: %f\n", dt_ms);
  
    t_end = rtclock();
    fprintf(stdout, "GPU Runtime: %0.6lfs\n", t_end - t_start);

	t_start = rtclock();
	gesummv(A, B, x, y, tmp);
	t_end = rtclock();
	fprintf(stdout, "CPU Runtime: %0.6lfs\n", t_end - t_start);
	y_gpu->memcpyDtoH();
    y_gpu->memcpyServerToHost();
	compareResults(y, y_gpu);

	free(A);
	free(B);  
	free(x);  
	free(y);
	free(tmp);
	cudaFree(A_gpu);
	cudaFree(B_gpu);  
	cudaFree(x_gpu);  
	cudaFree(y_gpu);
	cudaFree(tmp_gpu);
    

	return 0;
}
