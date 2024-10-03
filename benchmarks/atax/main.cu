#include <cstdio>
#include <cuda.h>
#include <cuda_runtime.h>
#include <string>
#include <cstring>
#include <math.h>
#include <chrono>

using namespace std;
// using namespace std;


// extern "C"{
//   #include "rdma_utils.h"
// }

// #include "../../src/rdma_utils.cuh"
#include <time.h>
// #include "../../include/runtime_prefetching.h"
#include "../../include/runtime_eviction.h"

//define the error threshold for the results "not matching"
#define PERCENT_DIFF_ERROR_THRESHOLD 0.05

//define a small float value
#define SMALL_FLOAT_VAL 0.00000001f

/* Problem size. */
#define NX 4096*16llu
#define NY 4096*16llu

#define BLOCK_NUM 1024ULL
#define MYINFINITY 2147483647llu

#define BLOCK_SIZE 1024
#define WARP_SHIFT 5
#define WARP_SIZE 32

typedef float DATA_TYPE;

/* Thread block dimensions */
#define DIM_THREAD_BLOCK_X 1024
#define DIM_THREAD_BLOCK_Y 1

#ifndef M_PI
#define M_PI 3.14159
#endif

__device__ rdma_buf<unsigned int> D_adjacencyList;

__global__ void test(rdma_buf<unsigned int> *a/*, rdma_buf<int> *b, rdma_buf<int> *c*/);


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

#define WARP_SIZE 32

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


#define check_cuda_error(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }

}

__global__ void transfer(size_t size, rdma_buf<DATA_TYPE> *d_adjacencyList)
{
    size_t id = blockDim.x * blockIdx.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;
    
        for (size_t i = id; i < size ; i += stride)
        {
            DATA_TYPE y = (*d_adjacencyList)[i];
        }
}

__global__ void check(size_t size, rdma_buf<DATA_TYPE> *d_adjacencyList, DATA_TYPE *a)
{
    size_t id = blockDim.x * blockIdx.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;
    
        for (size_t i = id; i < size ; i += stride)
        {
            DATA_TYPE y = (*d_adjacencyList)[i];
            if(a[i] != y){
                printf("y: %f %f ", y, a[i]);
            }
        }
}

__global__ void assign_array(rdma_buf<unsigned int> *adjacencyList){
    D_adjacencyList = *adjacencyList;
    printf("D_adjacencyList.d_TLB[0].state: %d\n", D_adjacencyList.d_TLB[0].state);
    printf("D_adjacencyList.d_TLB[0].device_address: %p\n", D_adjacencyList.d_TLB[0].device_address);
}

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

__device__ size_t sum_page_faults = 0;

__global__ void
print_retires(void){
    // size_t max = cq_wait[0];
    // for (size_t i = 0; i < 128; i++)
    // {
    //     if(max < cq_wait[i]) max = cq_wait[i];
    // }
    sum_page_faults += g_qp_index;
    printf("g_qp_index: %llu sum page fault: %llu\n", g_qp_index, sum_page_faults);
    g_qp_index = 0;
    // for (size_t i = 0; i < 128; i++)
    // {
    //     max = 0;
    // }
}

void oversubs(float os, size_t size){
    cudaDeviceProp devProp;
    cudaGetDeviceProperties(&devProp, 0);
    // Calculate memory utilization
    size_t totalMemory = devProp.totalGlobalMem;
    size_t freeMemory;
    size_t usedMemory;
    float workload_size = ((float) size);
    cudaMemGetInfo(&freeMemory, &totalMemory);
    usedMemory = totalMemory - freeMemory;
    printf("Total GPU Memory: %.2f MiB\n", (float) totalMemory / (1024 * 1024));
    printf("Free GPU Memory: %.2f MiB\n", (float) freeMemory / (1024 * 1024));
    printf("Used GPU Memory: %.2f MiB\n", (float) usedMemory / (1024 * 1024));

    printf("Workload size: %.2f\n", workload_size/1024/1024);
    float oversubs_ratio = (float) os;
    void *tmp_ptr;
    cudaMalloc(&tmp_ptr, (size_t) (freeMemory - workload_size));
    cudaMemGetInfo(&freeMemory, &totalMemory);
    printf("Free GPU Memory: %.2f MiB\n", (float) freeMemory / (1024 * 1024));
    if(oversubs_ratio > 0){
        
        void *over_ptr;
        long long unsigned int os_size = freeMemory - workload_size /(1 + oversubs_ratio);
        printf("workload: %.2f\n",  workload_size);
        printf("workload: %llu\n",  os_size);
        cudaMalloc(&over_ptr, os_size); 
        printf("os_size: %u\n", os_size/1024/1024);
    }
    cudaMemGetInfo(&freeMemory, &totalMemory);
    printf("Free GPU Memory: %.2f MiB\n", (float) freeMemory / (1024 * 1024));
}

__global__
void print_utilization() {
    printf("GPU_address_offset: %llu \n", GPU_address_offset);
}

void atax_cpu(DATA_TYPE *A, DATA_TYPE *x, DATA_TYPE *&y, DATA_TYPE *&tmp)
{
	size_t i,j;
	
	for (i= 0; i < NY; i++)
	{
    	y[i] = 0;
	}
  
	for (i = 0; i < NX; i++)
 	{
      	tmp[i] = 0;

      	for (j = 0; j < NY; j++)
		{
			tmp[i] = tmp[i] + A[i*NY + j] * x[j];
		}
		
      	for (j = 0; j < NY; j++)
		{
			y[j] = y[j] + A[i*NY + j] * tmp[i];
		}
    }
}

float absVal(float a)
{
	if(a < 0)
	{
		return (a * -1);
	}
   	else
	{ 
		return a;
	}
}


float percentDiff(double val1, double val2)
{
	if ((absVal(val1) < 0.01) && (absVal(val2) < 0.01))
	{
		return 0.0f;
	}

	else
	{
    		return 100.0f * (absVal(absVal(val1 - val2) / absVal(val1 + SMALL_FLOAT_VAL)));
	}
} 


void init_array(DATA_TYPE *x, DATA_TYPE *A)
{
	size_t i, j;

	for (i = 0; i < NX; i++)
	{
		x[i] = i * M_PI;
		for (j = 0; j < NY; j++)
		{
			A[i*NY + j] = ((DATA_TYPE) i*(j)) / NX;
		}
	}
}

void compareResults(DATA_TYPE *z, DATA_TYPE *z_outputFromGpu)
{
	size_t i, fail;
	fail = 0;

	for (i = 0; i < NY; i++)
	{
        // printf("1: %f %f\n", z[i], z_outputFromGpu[i]);
		if (percentDiff(z[i], z_outputFromGpu[i]) > PERCENT_DIFF_ERROR_THRESHOLD)
		{
            // printf("1: %f %f\n", z[i], z_outputFromGpu[i]);
			fail++;
		}		
	}
	
	// print results
	printf("Non-Matching CPU-GPU Outputs Beyond Error Threshold of %4.2f Percent: %d\n", PERCENT_DIFF_ERROR_THRESHOLD, fail);
}

/******************************* CUDA Imlementation BEGIN ***************************************/
__global__ void atax_kernel1(DATA_TYPE *A, DATA_TYPE *x, DATA_TYPE *tmp)
{
	size_t i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < NX)
	{
		size_t j;
		for(j=0; j < NY; j++)
		{
			tmp[i] += A[i * NY + j] * x[j];
		}
	}
}

__global__ void atax_kernel2(DATA_TYPE *A, DATA_TYPE *y, DATA_TYPE *tmp)
{
	size_t j = blockIdx.x * blockDim.x + threadIdx.x;
	
	if (j < NY)
	{
		size_t i;
		for(i=0; i < NX; i++)
		{
			y[j] += A[i * NY + j] * tmp[i];
		}
	}
}


void ataxGpu(DATA_TYPE* A, DATA_TYPE* x, DATA_TYPE* y)
{
    printf("Starting CUDA UVM/Direct\n");
	DATA_TYPE *A_gpu, *x_gpu, *y_gpu, *tmp_gpu;
    DATA_TYPE *res_gpu, *tmp_cpu, *y_cpu;
    cudaError_t ret;

    y_cpu = (DATA_TYPE *) malloc(sizeof(DATA_TYPE)*NY);
    memcpy(y_cpu, y, sizeof(DATA_TYPE) * NY);

    bool uvm = true;
    if(uvm){
        printf("UVM in action\n");
        check_cuda_error(cudaMallocManaged((void **)&A_gpu, sizeof(DATA_TYPE)*NX*NY));
        memcpy(A_gpu, A, sizeof(DATA_TYPE)*NX*NY);
        check_cuda_error(cudaMemAdvise(A_gpu, sizeof(DATA_TYPE)*NX*NY, cudaMemAdviseSetReadMostly, 0));
    }
    else{
        check_cuda_error(cudaMalloc((void **)&A_gpu, sizeof(DATA_TYPE) * NX * NY));
        check_cuda_error(cudaMemcpy(A_gpu, A, sizeof(DATA_TYPE) * NX * NY, cudaMemcpyHostToDevice));
    }

    check_cuda_error(cudaMalloc((void **)&x_gpu, sizeof(DATA_TYPE) * NY));
    check_cuda_error(cudaMalloc((void **)&y_gpu, sizeof(DATA_TYPE) * NY));
    check_cuda_error(cudaMalloc((void **)&tmp_gpu, sizeof(DATA_TYPE) * NX));

    
    check_cuda_error(cudaMemcpy(x_gpu, x, sizeof(DATA_TYPE) * NY, cudaMemcpyHostToDevice));
    check_cuda_error(cudaMemcpy(y_gpu, y, sizeof(DATA_TYPE) * NY, cudaMemcpyHostToDevice));

    res_gpu = (DATA_TYPE *) malloc(sizeof(DATA_TYPE) * NY);
    tmp_cpu = (DATA_TYPE *) malloc(sizeof(DATA_TYPE) * NX);
    
    for (size_t i = 0; i < NX; i++)
    {
        tmp_cpu[i] = 0;
    }

    check_cuda_error(cudaMemcpy(tmp_gpu, tmp_cpu, sizeof(DATA_TYPE) * NX, cudaMemcpyHostToDevice));
	
	dim3 block(DIM_THREAD_BLOCK_X/2, DIM_THREAD_BLOCK_Y);
	dim3 grid1((size_t)(ceil( ((float)NX) / ((float)block.x) )), 1);
	dim3 grid2((size_t)(ceil( ((float)NY) / ((float)block.x) )), 1);

    // oversubs(0.07, sizeof(DATA_TYPE)*NX*NY);

    printf("Starting kernels\n");
	auto start = std::chrono::steady_clock::now();
	atax_kernel1<<< grid1, block >>>(A_gpu, x_gpu, tmp_gpu);
	ret= cudaDeviceSynchronize();
    printf("ret: %d cudaGetLastError: %d\n", ret, cudaGetLastError());
	atax_kernel2<<< grid2, block >>>(A_gpu, y_gpu, tmp_gpu);
	ret= cudaDeviceSynchronize();
    printf("ret: %d cudaGetLastError: %d\n", ret, cudaGetLastError());
	auto end = std::chrono::steady_clock::now();
    long duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    printf("Elapsed time for normal cuda in milliseconds: %li ms.d\n\n", duration);

    check_cuda_error(cudaMemcpy(res_gpu, y_gpu, sizeof(DATA_TYPE) * NY, cudaMemcpyDeviceToHost));

    check_cuda_error(cudaFree(A_gpu));  
    check_cuda_error(cudaFree(x_gpu));  
    check_cuda_error(cudaFree(y_gpu));  
    check_cuda_error(cudaFree(tmp_gpu));

    //run the algorithm on the CPU
    printf("Running on CPU\n");
	atax_cpu(A, x, y_cpu, tmp_cpu);
    printf("Comparing CPU with CUDA-direct-uvm\n");
    compareResults(res_gpu, y_cpu);

    free(y_cpu);   
    free(tmp_cpu); 
    free(res_gpu); 
}
/******************************* CUDA Imlementation END ***************************************/


/******************************* RDMA Imlementation BEGIN ***************************************/
__global__ void atax_kernel1_rdma(rdma_buf<DATA_TYPE> *A, DATA_TYPE *x, DATA_TYPE *tmp)
{
	size_t i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < NX)
	{
		size_t j;
		for(j=0; j < NY; j++)
		{
			tmp[i] += (*A)[i * NY + j] * x[j];
		}
	}
}

__global__ void atax_kernel2_rdma(rdma_buf<DATA_TYPE> *A, DATA_TYPE *y, DATA_TYPE *tmp)
{
	size_t j = blockIdx.x * blockDim.x + threadIdx.x;
	
	if (j < NY)
	{
		size_t i;
		for(i=0; i < NX; i++)
		{
			y[j] += (*A)[i * NY + j] * tmp[i];
		}
	}
}


void ataxGpu_rdma(DATA_TYPE* A, DATA_TYPE* x, DATA_TYPE* y)
{
	DATA_TYPE *A_gpu, *x_gpu, *y_gpu, *tmp_gpu;
    DATA_TYPE *res_gpu, *tmp_cpu, *y_cpu;
    cudaError_t ret;

    y_cpu = (DATA_TYPE *) malloc(sizeof(DATA_TYPE) * NY);
    memcpy(y_cpu, y, sizeof(DATA_TYPE) * NY);

    // check_cuda_error(cudaMalloc((void **)&A_gpu, sizeof(DATA_TYPE) * NX * NY));
    check_cuda_error(cudaMalloc((void **)&x_gpu, sizeof(DATA_TYPE) * NY));
    check_cuda_error(cudaMalloc((void **)&y_gpu, sizeof(DATA_TYPE) * NY));
    check_cuda_error(cudaMalloc((void **)&tmp_gpu, sizeof(DATA_TYPE) * NX));

    // check_cuda_error(cudaMemcpy(A_gpu, A, sizeof(DATA_TYPE) * NX * NY, cudaMemcpyHostToDevice));
    check_cuda_error(cudaMemcpy(x_gpu, x, sizeof(DATA_TYPE) * NY, cudaMemcpyHostToDevice));
    check_cuda_error(cudaMemcpy(y_gpu, y, sizeof(DATA_TYPE) * NY, cudaMemcpyHostToDevice));

    res_gpu = (DATA_TYPE *) malloc(sizeof(DATA_TYPE) * NY);
    tmp_cpu = (DATA_TYPE *) malloc(sizeof(DATA_TYPE) * NX);
	
	dim3 block(DIM_THREAD_BLOCK_X/2, DIM_THREAD_BLOCK_Y);
	dim3 grid1((size_t)(ceil( ((float)NX) / ((float)block.x) )), 1);
	dim3 grid2((size_t)(ceil( ((float)NY) / ((float)block.x) )), 1);

    rdma_buf<DATA_TYPE> *rdma_a;
    check_cuda_error(cudaMallocManaged((void **) &rdma_a, sizeof(rdma_buf<DATA_TYPE>)));
    rdma_a->start(NX*NY*sizeof(DATA_TYPE));
    for(size_t i = 0; i < NX * NY; i++){
        rdma_a->local_buffer[i] = A[i];
    }

	auto start = std::chrono::steady_clock::now();
	atax_kernel1_rdma<<< grid1, block >>>(rdma_a, x_gpu, tmp_gpu);
	ret= cudaDeviceSynchronize();
    printf("ret: %d cudaGetLastError: %d\n", ret, cudaGetLastError());
	atax_kernel2_rdma<<< grid2, block >>>(rdma_a, y_gpu, tmp_gpu);
	ret= cudaDeviceSynchronize();
    printf("ret: %d cudaGetLastError: %d\n", ret, cudaGetLastError());

    // atax_kernel1<<< grid1, block >>>(A_gpu, x_gpu, tmp_gpu);
	// ret= cudaDeviceSynchronize();
    // printf("ret: %d cudaGetLastError: %d\n", ret, cudaGetLastError());
	// atax_kernel2<<< grid2, block >>>(A_gpu, y_gpu, tmp_gpu);
	// ret= cudaDeviceSynchronize();
    // printf("ret: %d cudaGetLastError: %d\n", ret, cudaGetLastError());
	auto end = std::chrono::steady_clock::now();
    long duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    printf("Elapsed time for rdma cuda in milliseconds: %li ms.d\n\n", duration);

    check_cuda_error(cudaMemcpy(res_gpu, y_gpu, sizeof(DATA_TYPE) * NY, cudaMemcpyDeviceToHost));

    // check_cuda_error(cudaFree(A_gpu));
    check_cuda_error(cudaFree(x_gpu));
    check_cuda_error(cudaFree(y_gpu));
    check_cuda_error(cudaFree(tmp_gpu));

    //run the algorithm on the CPU
    printf("Running on CPU\n");
	atax_cpu(A, x, y_cpu, tmp_cpu);
    printf("Comparing CPU with RDMA\n");
    compareResults(res_gpu, y_cpu);

    free(res_gpu);
    free(tmp_cpu);
    free(y_cpu);
}

/******************************* RDMA Imlementation END ***************************************/


// Main program
int main(int argc, char **argv)
{   
    init_gpu(0);

    DATA_TYPE* A;
	DATA_TYPE* x;
    DATA_TYPE* A_rdma;
	DATA_TYPE* x_rdma;
	DATA_TYPE* y_rdma;
    DATA_TYPE* y_direct;
	DATA_TYPE* tmp;

	// DATA_TYPE* tmp;
	A = (DATA_TYPE*)malloc(NX*NY*sizeof(DATA_TYPE));
	x = (DATA_TYPE*)malloc(NY*sizeof(DATA_TYPE));
    A_rdma = (DATA_TYPE*)malloc(NX*NY*sizeof(DATA_TYPE));
	x_rdma = (DATA_TYPE*)malloc(NY*sizeof(DATA_TYPE));
	y_rdma = (DATA_TYPE*)malloc(NY*sizeof(DATA_TYPE));
    y_direct = (DATA_TYPE*)malloc(NY*sizeof(DATA_TYPE));
	tmp = (DATA_TYPE*)malloc(NX*sizeof(DATA_TYPE));

    printf("Initializing arrays\n");
    init_array(x, A);
    init_array(x_rdma, A_rdma);
    
    for (size_t i = 0; i < NY; i++)
    {
        y_rdma[i] = 0;
        y_direct[i] = 0;
    }

    


    bool rdma_flag = true;
    cudaError_t ret1;
    struct context *s_ctx = (struct context *)malloc(sizeof(struct context));
    if(rdma_flag){
        
        int num_msg = (unsigned long) atoi(argv[4]);
        int mesg_size = (unsigned long) atoi(argv[5]);
        int num_bufs = (unsigned long) atoi(argv[6]);

        
        struct post_content post_cont, *d_post, host_post;
        struct poll_content poll_cont, *d_poll, host_poll;
        struct post_content2 post_cont2, *d_post2;
        struct host_keys keys;

        int num_iteration = num_msg;
        s_ctx->n_bufs = num_bufs;

        s_ctx->gpu_buf_size = 16*1024*1024*1024llu; // N*sizeof(int)*3llu;

        // // remote connection:
        // int ret = connect(argv[2], s_ctx);

        // local connect
        char *mlx_name = "mlx5_2";
        int ret = local_connect(mlx_name, s_ctx);

        ret = prepare_post_poll_content(s_ctx, &post_cont, &poll_cont, &post_cont2, \
                                        &host_post, &host_poll, &keys);
        if(ret == -1) {
            printf("Post and poll contect creation failed\n");    
            exit(-1);
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

        size_t restricted_gpu_mem = rest_memory;// 16*1024*1024*1024llu;
        // restricted_gpu_mem = restricted_gpu_mem / 3;
        const size_t page_size = REQUEST_SIZE;
        // const size_t numPages = ceil((double)restricted_gpu_mem/page_size);

        printf("function: %s line: %d\n", __FILE__, __LINE__);
        alloc_global_host_content(host_post, host_poll, keys);
        printf("function: %s line: %d\n", __FILE__, __LINE__);

        ret1 = cudaDeviceSynchronize();
        printf("ret: %d\n", ret1);
        if(cudaSuccess != ret1){    
            return -1;
        }

        ret1 = cudaDeviceSynchronize();
        printf("ret: %d\n", ret1);
        if(cudaSuccess != ret1){    
            return -1;
        }
        
        printf("restricted_gpu_mem: %zu\n", restricted_gpu_mem);
        start_page_queue<<<1, 1>>>(/*s_ctx->gpu_buf_size*/restricted_gpu_mem, page_size);
        ret1 = cudaDeviceSynchronize();
        printf("ret: %d\n", ret1);
        if(cudaSuccess != ret1){    
            return -1;
        }
    }

    
    

    if(rdma_flag){
        printf("Allocation finished. Calling mvtRDMA\n");
        ataxGpu_rdma(A_rdma, x_rdma, y_rdma);
        cudaFree(s_ctx->gpu_buffer);
    }

    ataxGpu(A, x, y_direct);

    printf("oversubs ratio: %d\n", oversubs_ratio_macro-1);
    
	return 0;
}

__global__ void kernel_coalesce_new_repr_rdma(bool *curr_visit, size_t n, bool *next_visit, uint64_t new_size, unsigned int *new_vertex_list,
                                unsigned int *new_offset, rdma_buf<unsigned int> *edgeList, unsigned long long *comp, bool *changed) {
    size_t warp_size = 32;
    // Page size in elements (64KB / 4 bytes per unsigned int)
    const size_t pageSize = 4*1024 / sizeof(unsigned int);
    // Elements per warp
    const size_t elementsPerWarp = pageSize / warp_size; // warpSize;
    // Global thread ID
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Warp ID within the block
    size_t warpId = tid / warp_size; // warpSize;

    // Thread lane within the warp
    size_t lane = threadIdx.x % warp_size; // warpSize;

    // Determine which page this warp will process
    size_t pageStart = warpId * pageSize;

    // Ensure we don't process out-of-bounds pages
    if (pageStart < n * pageSize) {
        // Process elements within the page
        for (size_t i = 0; i < elementsPerWarp; ++i) {
            size_t elementIdx = pageStart + lane + i * warp_size; // warpSize;
            if (elementIdx < new_size && curr_visit[new_vertex_list[elementIdx]] == true) {
                unsigned int start_vertex = new_vertex_list[elementIdx];
                
                // const uint64_t start = new_offset[elementIdx];
                // const uint64_t shift_start = start & MEM_ALIGN;
                // const uint64_t end = new_offset[elementIdx+1];

                // Process adjacent nodes
                for(size_t j = new_offset[elementIdx]/*&MEM_ALIGN*/; j < new_offset[elementIdx+1]; j += 1) {
                    // if(j >= new_offset[elementIdx]){
                        uint next = (*edgeList)[j];
                        if (comp[next] != comp[start_vertex]) {
                            if (comp[start_vertex] < comp[next]) {
                                // next_target = next;
                                // comp_target = comp[elementIdx];
                                atomicMin(&comp[next], comp[start_vertex]);
                                next_visit[next] = true;
                            }
                            else {
                                // next_target = elementIdx;
                                // comp_target = comp[next];
                                atomicMin(&comp[start_vertex], comp[next]);
                                next_visit[start_vertex] = true;   
                            }
                            *changed = true;
                        }
                    // }
                }
            }
        }
    }
}

__global__ void kernel_coalesce_new_repr_uvm(bool *curr_visit, size_t n, bool *next_visit, uint64_t new_size, unsigned int *new_vertex_list,
                                unsigned int *new_offset, unsigned int *edgeList, unsigned long long *comp, bool *changed) {
    size_t warp_size = 32;
    // Page size in elements (64KB / 4 bytes per unsigned int)
    const size_t pageSize = 4*1024 / sizeof(unsigned int);
    // Elements per warp
    const size_t elementsPerWarp = pageSize / warp_size; // warpSize;
    // Global thread ID
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Warp ID within the block
    size_t warpId = tid / warp_size; // warpSize;

    // Thread lane within the warp
    size_t lane = threadIdx.x % warp_size; // warpSize;

    // Determine which page this warp will process
    size_t pageStart = warpId * pageSize;

    // Ensure we don't process out-of-bounds pages
    if (pageStart < n * pageSize) {
        // Process elements within the page
        for (size_t i = 0; i < elementsPerWarp; ++i) {
            size_t elementIdx = pageStart + lane + i * warp_size; // warpSize;
            if (elementIdx < new_size && curr_visit[new_vertex_list[elementIdx]] == true) {
                unsigned int start_vertex = new_vertex_list[elementIdx];
                
                // const uint64_t start = new_offset[elementIdx];
                // const uint64_t shift_start = start & MEM_ALIGN;
                // const uint64_t end = new_offset[elementIdx+1];

                // Process adjacent nodes
                for(size_t j = new_offset[elementIdx]/*&MEM_ALIGN*/; j < new_offset[elementIdx+1]; j += 1) {
                    // if(j >= new_offset[elementIdx]){
                        uint next = edgeList[j];
                        if (comp[next] != comp[start_vertex]) {
                            if (comp[start_vertex] < comp[next]) {
                                // next_target = next;
                                // comp_target = comp[elementIdx];
                                atomicMin(&comp[next], comp[start_vertex]);
                                next_visit[next] = true;
                            }
                            else {
                                // next_target = elementIdx;
                                // comp_target = comp[next];
                                atomicMin(&comp[start_vertex], comp[next]);
                                next_visit[start_vertex] = true;   
                            }
                            *changed = true;
                        }
                    // }
                }
            }
        }
    }
}