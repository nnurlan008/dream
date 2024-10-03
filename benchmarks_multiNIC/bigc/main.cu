#include <cstdio>
#include <cuda.h>
#include <cuda_runtime.h>
#include <string>
#include <cstring>

#include <chrono>

using namespace std;
// using namespace std;


// extern "C"{
//   #include "rdma_utils.h"
// }

// #include "../../src/rdma_utils.cuh"
#include <time.h>
// #include "../../include/runtime_prefetching.h"
// #include "../../include/runtime_eviction.h"
// #include "../../include/runtime_prefetching_2nic.h"
#include "../../include/runtime_eviction_2nic.h"

//define the error threshold for the results "not matching"
#define PERCENT_DIFF_ERROR_THRESHOLD 0.05

//define a small float value
#define SMALL_FLOAT_VAL 0.00000001f

/* Problem size. */
#define NX (16*4096llu)
#define NY (16*4096llu)

#define BLOCK_NUM 1024ULL
#define MYINFINITY 2147483647llu

#define BLOCK_SIZE 1024
#define WARP_SHIFT 5
#define WARP_SIZE 32


#define GPU 0

typedef float DATA_TYPE;

/* Thread block dimensions */
#define DIM_THREAD_BLOCK_X 1024
#define DIM_THREAD_BLOCK_Y 1

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

void init_array(DATA_TYPE *A, DATA_TYPE *p, DATA_TYPE *r, DATA_TYPE *A_gpu, DATA_TYPE *p_gpu, DATA_TYPE *r_gpu)
{
	int i, j;

  	for (i = 0; i < NX; i++)
	{
			r[i] = i * M_PI;
			r_gpu[i] = i * M_PI;

    		for (j = 0; j < NY; j++)
		{
				  A[i*NY + j] = ((DATA_TYPE) i*j) / NX;
				  A_gpu[i*NY + j] = ((DATA_TYPE) i*j) / NX;
		}
 	}
	
	for (i = 0; i < NY; i++)
	{
			p[i] = i * M_PI;
			p_gpu[i] = i * M_PI;
	}
}


void compareResults(DATA_TYPE* s, DATA_TYPE* s_outputFromGpu, DATA_TYPE* q, DATA_TYPE* q_outputFromGpu)
{
	size_t i,fail;
	fail = 0;

	// Compare s with s_cuda
	for (i=0; i<NX; i++)
	{
		if (percentDiff(q[i], q_outputFromGpu[i]) > PERCENT_DIFF_ERROR_THRESHOLD)
		{
			fail++;
		}
	}

	for (i=0; i<NY; i++)
	{
		if (percentDiff(s[i], s_outputFromGpu[i]) > PERCENT_DIFF_ERROR_THRESHOLD)
		{
			fail++;
		}		
	}
	
	// print results
	printf("Non-Matching CPU-GPU Outputs Beyond Error Threshold of %4.2f Percent: %d\n", PERCENT_DIFF_ERROR_THRESHOLD, fail);
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

int alloc_global_cont(struct post_content *post_cont, struct poll_content *poll_cont, struct server_content_2nic *post_cont2, 
                      struct gpu_memory_info gpu_mem){
    struct post_content *d_post;
    struct poll_content *d_poll;
    struct server_content_2nic *d_post2;

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

    ret0 = cudaMalloc((void **)&d_post2, sizeof(struct server_content_2nic));
    if(ret0 != cudaSuccess){
        printf("Error on allocation post content!\n");
        return -1;
    }
    ret0 = cudaMemcpy(d_post2, post_cont2, sizeof(struct server_content_2nic), cudaMemcpyHostToDevice);
    if(ret0 != cudaSuccess){
        printf("Error on poll copy!\n");
        return -1;
    }

    // cudaSetDevice(0);
    alloc_content<<<1,1>>>(d_post, d_poll);
    alloc_global_content<<<1,1>>>(d_post, d_poll, d_post2, gpu_mem);
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
    printf("overfsubscrion level: %f\n", os);
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

void bicg_cpu(DATA_TYPE* A, DATA_TYPE* r, DATA_TYPE* s, DATA_TYPE* p, DATA_TYPE* q)
{
	size_t i,j;
	
  	for (i = 0; i < NY; i++)
	{
		s[i] = 0.0;
	}

    for (i = 0; i < NX; i++)
    {
		q[i] = 0.0;
		for (j = 0; j < NY; j++)
	  	{
	    		s[j] = s[j] + r[i] * A[i*NY + j];
	    		q[i] = q[i] + A[i*NY + j] * p[j];
	  	}
	}
}


/******************************* CUDA Imlementation BEGIN ***************************************/
//Distributed (split) from initial loop and permuted into reverse order to allow parallelism...
__global__ void bicg_kernel1(DATA_TYPE *A, DATA_TYPE *r, DATA_TYPE *s)
{
	size_t j = blockIdx.x * blockDim.x + threadIdx.x;
	
	if (j < NY)
	{
		s[j] = 0.0f;

		int i;
		for(i = 0; i < NX; i++)
		{
			s[j] += A[i * NY + j] * r[i];
		}
	}	
}


//Distributed (split) from initial loop to allow parallelism
__global__ void bicg_kernel2(DATA_TYPE *A, DATA_TYPE *p, DATA_TYPE *q)
{
	size_t i = blockIdx.x * blockDim.x + threadIdx.x;
	
	if (i < NX)
	{
		q[i] = 0.0f;

		int j;
		for(j=0; j < NY; j++)
		{
			q[i] += A[i * NY + j] * p[j];
		}
	}
}


void bicgCuda()
{
	double t_start, t_end;

    DATA_TYPE* A;
	DATA_TYPE* r;
	DATA_TYPE* s;
	DATA_TYPE* p;
	DATA_TYPE* q;

	DATA_TYPE *A_gpu;
	DATA_TYPE *q_gpu, *q_cpu;
	DATA_TYPE *p_gpu, *p_cpu;
	DATA_TYPE *r_gpu, *r_cpu;
	DATA_TYPE *s_gpu, *s_cpu;
 	
	A = (DATA_TYPE*)malloc(NX*NY*sizeof(DATA_TYPE));
	r = (DATA_TYPE*)malloc(NX*sizeof(DATA_TYPE));
	s = (DATA_TYPE*)malloc(NY*sizeof(DATA_TYPE));
	p = (DATA_TYPE*)malloc(NY*sizeof(DATA_TYPE));
	q = (DATA_TYPE*)malloc(NX*sizeof(DATA_TYPE));

    r_cpu = (DATA_TYPE*)malloc(NX*sizeof(DATA_TYPE));
	s_cpu = (DATA_TYPE*)malloc(NY*sizeof(DATA_TYPE));
	p_cpu = (DATA_TYPE*)malloc(NY*sizeof(DATA_TYPE));
	q_cpu = (DATA_TYPE*)malloc(NX*sizeof(DATA_TYPE));

	check_cuda_error(cudaMallocManaged(&A_gpu, sizeof(DATA_TYPE) * NX * NY));
	check_cuda_error(cudaMalloc(&r_gpu, sizeof(DATA_TYPE) * NX));
	check_cuda_error(cudaMalloc(&s_gpu, sizeof(DATA_TYPE) * NY));
	check_cuda_error(cudaMalloc(&p_gpu, sizeof(DATA_TYPE) * NY));
	check_cuda_error(cudaMalloc(&q_gpu, sizeof(DATA_TYPE) * NX));

	init_array(A, p, r, A_gpu, p_cpu, r_cpu);

    check_cuda_error(cudaMemcpy(p_gpu, p_cpu, sizeof(DATA_TYPE) * NY, cudaMemcpyHostToDevice)); 
    check_cuda_error(cudaMemcpy(r_gpu, r_cpu, sizeof(DATA_TYPE) * NX, cudaMemcpyHostToDevice));  

    check_cuda_error(cudaMemAdvise(A_gpu, sizeof(DATA_TYPE) * NX * NY, cudaMemAdviseSetReadMostly, 0));

	dim3 block(DIM_THREAD_BLOCK_X, DIM_THREAD_BLOCK_Y);
	dim3 grid1((size_t)(ceil( ((float)NY) / ((float)block.x) )), 1);
	dim3 grid2((size_t)(ceil( ((float)NX) / ((float)block.x) )), 1);

    // oversubs(0.37, sizeof(DATA_TYPE) * NX * NY);

    auto start = std::chrono::steady_clock::now();
	bicg_kernel1<<< grid1, block >>>(A_gpu, r_gpu, s_gpu);
	cudaDeviceSynchronize();
	bicg_kernel2<<< grid2, block >>>(A_gpu, p_gpu, q_gpu);
	cudaDeviceSynchronize();

    auto end = std::chrono::steady_clock::now();
    long duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    printf("GPU Runtime cuda in milliseconds: %li ms.d\n\n", duration);

    check_cuda_error(cudaMemcpy(q_cpu, q_gpu, sizeof(DATA_TYPE) * NX, cudaMemcpyDeviceToHost)); 
    check_cuda_error(cudaMemcpy(s_cpu, s_gpu, sizeof(DATA_TYPE) * NY, cudaMemcpyDeviceToHost));  

    start = std::chrono::steady_clock::now();
	bicg_cpu(A, r, s, p, q);
	end = std::chrono::steady_clock::now();
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    printf("CPU Runtime cuda in milliseconds: %li ms.d\n\n", duration);

	compareResults(s, s_cpu, q, q_cpu);

	free(A);
	free(r);
	free(s);
	free(p);
	free(q);

    free(r_cpu);
	free(s_cpu);
	free(p_cpu);
	free(q_cpu);
	
	cudaFree(A_gpu);
	cudaFree(r_gpu);
	cudaFree(s_gpu);
	cudaFree(p_gpu);
	cudaFree(q_gpu);
	
}

__global__ __launch_bounds__(1024,2)
void bicg_kernel1_rdma(rdma_buf<DATA_TYPE> *A, DATA_TYPE *r, DATA_TYPE *s)
{
	int j = blockIdx.x * blockDim.x + threadIdx.x;
	
	if (j < NY)
	{
		s[j] = 0.0f;

		int i;
		for(i = 0; i < NX; i++)
		{
			s[j] += (*A)[i * NY + j] * r[i];
		}
	}	
}


//Distributed (split) from initial loop to allow parallelism
__global__ __launch_bounds__(1024,2)
void bicg_kernel2_rdma(rdma_buf<DATA_TYPE> *A, DATA_TYPE *p, DATA_TYPE *q)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	
	if (i < NX)
	{
		q[i] = 0.0f;

		int j;
		for(j=0; j < NY; j++)
		{
			q[i] += (*A)[i * NY + j] * p[j];
		}
	}
}

void bicgRDMA()
{
	double t_start, t_end;
    cudaError_t ret;
    rdma_buf<DATA_TYPE> *A_rdma;
    DATA_TYPE *A;
	DATA_TYPE *r;
	DATA_TYPE *s;
	DATA_TYPE *p;
	DATA_TYPE *q;

	
	DATA_TYPE *q_gpu, *q_cpu;
	DATA_TYPE *p_gpu, *p_cpu;
	DATA_TYPE *r_gpu, *r_cpu;
	DATA_TYPE *s_gpu, *s_cpu;
 	
	A = (DATA_TYPE*)malloc(NX*NY*sizeof(DATA_TYPE));
    check_cuda_error(cudaMallocManaged(&A_rdma, sizeof(rdma_buf<DATA_TYPE>)));
	r = (DATA_TYPE*)malloc(NX*sizeof(DATA_TYPE));
	s = (DATA_TYPE*)malloc(NY*sizeof(DATA_TYPE));
	p = (DATA_TYPE*)malloc(NY*sizeof(DATA_TYPE));
	q = (DATA_TYPE*)malloc(NX*sizeof(DATA_TYPE));

    r_cpu = (DATA_TYPE*)malloc(NX*sizeof(DATA_TYPE));
	s_cpu = (DATA_TYPE*)malloc(NY*sizeof(DATA_TYPE));
	p_cpu = (DATA_TYPE*)malloc(NY*sizeof(DATA_TYPE));
	q_cpu = (DATA_TYPE*)malloc(NX*sizeof(DATA_TYPE));


	r_cpu = (DATA_TYPE*)malloc(NX*sizeof(DATA_TYPE));
	s_cpu = (DATA_TYPE*)malloc(NY*sizeof(DATA_TYPE));
	p_cpu = (DATA_TYPE*)malloc(NY*sizeof(DATA_TYPE));
	q_cpu = (DATA_TYPE*)malloc(NX*sizeof(DATA_TYPE));

	// check_cuda_error(cudaMallocManaged(&A_gpu, sizeof(DATA_TYPE) * NX * NY));
	check_cuda_error(cudaMalloc(&r_gpu, sizeof(DATA_TYPE) * NX));
	check_cuda_error(cudaMalloc(&s_gpu, sizeof(DATA_TYPE) * NY));
	check_cuda_error(cudaMalloc(&p_gpu, sizeof(DATA_TYPE) * NY));
	check_cuda_error(cudaMalloc(&q_gpu, sizeof(DATA_TYPE) * NX));

    A_rdma->start(NX*NY*sizeof(DATA_TYPE), GPU, NULL);
	init_array(A, p, r, A, p_cpu, r_cpu);

    for(size_t i = 0; i < NX*NY; i++) A_rdma->local_buffer[i] = A[i];

    check_cuda_error(cudaMemcpy(p_gpu, p_cpu, sizeof(DATA_TYPE) * NY, cudaMemcpyHostToDevice)); 
    check_cuda_error(cudaMemcpy(r_gpu, r_cpu, sizeof(DATA_TYPE) * NX, cudaMemcpyHostToDevice));  


	dim3 block(DIM_THREAD_BLOCK_X, DIM_THREAD_BLOCK_Y);
	dim3 grid1((size_t)(ceil( ((float)NY) / ((float)block.x) )), 1);
	dim3 grid2((size_t)(ceil( ((float)NX) / ((float)block.x) )), 1);
    cudaDeviceSynchronize();
    auto start = std::chrono::steady_clock::now();
	bicg_kernel2_rdma<<< grid1, block >>>(A_rdma, r_gpu, s_gpu);
	ret = cudaDeviceSynchronize();
    printf("ret: %d cudaGetLastError: %d\n", ret, cudaGetLastError());
	bicg_kernel1_rdma<<< grid2, block >>>(A_rdma, p_gpu, q_gpu);
	printf("ret: %d cudaGetLastError: %d\n", ret, cudaGetLastError());

    auto end = std::chrono::steady_clock::now();
    long duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    printf("GPU Runtime rdma in milliseconds: %li ms.d\n\n", duration);

    check_cuda_error(cudaMemcpy(q_cpu, q_gpu, sizeof(DATA_TYPE) * NX, cudaMemcpyDeviceToHost)); 
    check_cuda_error(cudaMemcpy(s_cpu, s_gpu, sizeof(DATA_TYPE) * NY, cudaMemcpyDeviceToHost));  

    start = std::chrono::steady_clock::now();
	bicg_cpu(A, r, s, p, q);
	end = std::chrono::steady_clock::now();
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    printf("CPU Runtime rdma in milliseconds: %li ms.d\n\n", duration);

	compareResults(s, s_cpu, q, q_cpu);

	free(A);
	free(r);
	free(s);
	free(p);
	free(q);

    free(r_cpu);
	free(s_cpu);
	free(p_cpu);
	free(q_cpu);
	
	cudaFree(A_rdma);
	cudaFree(r_gpu);
	cudaFree(s_gpu);
	cudaFree(p_gpu);
	cudaFree(q_gpu);
	
}


// void mvtCuda(DATA_TYPE* a, DATA_TYPE* &x1, DATA_TYPE* &x2, DATA_TYPE* y_1, DATA_TYPE* y_2)
// {
    
// 	DATA_TYPE* a_gpu;
// 	DATA_TYPE* x1_gpu;
// 	DATA_TYPE* x2_gpu;
// 	DATA_TYPE* y_1_gpu;
// 	DATA_TYPE* y_2_gpu;
// 	DATA_TYPE* x1_cpu;
// 	DATA_TYPE* x2_cpu;

//     x1_cpu = (DATA_TYPE*)malloc(N*sizeof(DATA_TYPE));
// 	x2_cpu = (DATA_TYPE*)malloc(N*sizeof(DATA_TYPE));

//     memcpy(x1_cpu, x1, N*sizeof(DATA_TYPE));
//     memcpy(x2_cpu, x2, N*sizeof(DATA_TYPE));
//     printf("Allocating on GPU\n");
	
// 	check_cuda_error(cudaMalloc(&x1_gpu, sizeof(DATA_TYPE) * N));   
// 	check_cuda_error(cudaMalloc(&x2_gpu, sizeof(DATA_TYPE) * N));   
// 	check_cuda_error(cudaMalloc(&y_1_gpu, sizeof(DATA_TYPE) * N));  
// 	check_cuda_error(cudaMalloc(&y_2_gpu, sizeof(DATA_TYPE) * N));  
    
//     printf("Initializing finished on GPU. Transferring to GPU...\n");
// 	check_cuda_error(cudaMemcpy(y_1_gpu, y_1, sizeof(DATA_TYPE) * N, cudaMemcpyHostToDevice));  
// 	check_cuda_error(cudaMemcpy(y_2_gpu, y_2, sizeof(DATA_TYPE) * N, cudaMemcpyHostToDevice));  
//     check_cuda_error(cudaMemcpy(x1_gpu, x1, sizeof(DATA_TYPE) * N, cudaMemcpyHostToDevice));  
// 	check_cuda_error(cudaMemcpy(x2_gpu, x2, sizeof(DATA_TYPE) * N, cudaMemcpyHostToDevice));  

//     bool uvm = true;
//     if(uvm) {
//         printf("UVM in action\n");
//         check_cuda_error(cudaMallocManaged(&a_gpu, sizeof(DATA_TYPE) * N * N));
        
//         memcpy(a_gpu, a, sizeof(DATA_TYPE) * N * N);

//         check_cuda_error(cudaMemAdvise(a_gpu, sizeof(DATA_TYPE) * N * N, cudaMemAdviseSetReadMostly, 0));
//         // check_cuda_error(cudaMemAdvise(a_gpu, sizeof(DATA_TYPE) * N * N, cudaMemAdviseSetAccessedBy, 0));

//         // oversubs(0.33, sizeof(DATA_TYPE) * N * N);
//     }
//     else{
//         check_cuda_error(cudaMalloc(&a_gpu, sizeof(DATA_TYPE) * N * N));
//         check_cuda_error(cudaMemcpy(a_gpu, a, sizeof(DATA_TYPE) * N * N, cudaMemcpyHostToDevice));
//     }
//     DATA_TYPE* a_direct;
//     check_cuda_error(cudaMalloc(&a_direct, sizeof(DATA_TYPE) * N * N));
//     check_cuda_error(cudaMemcpy(a_direct, a, sizeof(DATA_TYPE) * N * N, cudaMemcpyHostToDevice));

// 	double t_start, t_end;
// 	dim3 block(DIM_THREAD_BLOCK_X, DIM_THREAD_BLOCK_Y);
// 	dim3 grid((size_t)ceil((float)N/ ((float)DIM_THREAD_BLOCK_X)), 1);

//     printf("Starting Kernels\n");
// 	auto start = std::chrono::steady_clock::now();
// 	mvt_kernel1<<<grid,block>>>(a_gpu,x1_gpu,y_1_gpu);
// 	mvt_kernel2<<<grid,block>>>(a_gpu,x2_gpu,y_2_gpu, a_direct);
// 	cudaDeviceSynchronize();
// 	auto end = std::chrono::steady_clock::now();
//     long duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
//     printf("Elapsed time for normal cuda in milliseconds: %li ms.d\n\n", duration);

//     check_cuda_error(cudaMemcpy(x1, x1_gpu, sizeof(DATA_TYPE) * N, cudaMemcpyDeviceToHost));   
// 	check_cuda_error(cudaMemcpy(x2, x2_gpu, sizeof(DATA_TYPE) * N, cudaMemcpyDeviceToHost)); 

//     check_cuda_error(cudaFree(a_gpu));
//     check_cuda_error(cudaFree(x1_gpu));
//     check_cuda_error(cudaFree(x2_gpu));
//     check_cuda_error(cudaFree(y_1_gpu));
//     check_cuda_error(cudaFree(y_2_gpu));

//     check_cuda_error(cudaFree(a_direct));

//     // //run the algorithm on the CPU
//     // printf("Running on CPU\n");
// 	// runMvt(a, x1_cpu, x2_cpu, y_1, y_2);  
//     // printf("Comparing Results for CPU and Direct transfer\n");
//     // compareResults(x1_cpu, x1, x2_cpu, x2);


// }
// /******************************* CUDA Imlementation END ***************************************/



// /******************************* RDMA Imlementation BEGIN ***************************************/
// __global__ void mvt_kernel1_rdma(rdma_buf<DATA_TYPE> *a, DATA_TYPE *x1, DATA_TYPE *y_1)
// {
// 	size_t i = blockIdx.x * blockDim.x + threadIdx.x;

// 	if (i < N)
// 	{
// 		size_t j;
// 		for(j=0; j < N; j++)
// 		{
// 			x1[i] += (*a)[i * N + j] * y_1[j];
// 		}
// 	}
// }


// __global__ void mvt_kernel2_rdma(rdma_buf<DATA_TYPE> *a, DATA_TYPE *x2, DATA_TYPE *y_2, DATA_TYPE *a_gpu)
// {
// 	size_t i = blockIdx.x * blockDim.x + threadIdx.x;

// 	if (i < N)
// 	{
// 		size_t j;
// 		for(j=0; j < N; j++)
// 		{
//             size_t index = j * N + i;
//             DATA_TYPE tmp = (*a)[index]; 
// 			// x2[i] += tmp * y_2[j];
//             // if(tmp != a_gpu[index]){
//             //     printf("tmp: %f %f ", tmp, a_gpu[index]);
//             // }	
//             x2[i] += tmp * y_2[j];
// 		}
// 	}
// }

// void mvtCuda_rdma(DATA_TYPE* a, DATA_TYPE* &x1, DATA_TYPE* &x2, DATA_TYPE* y_1, DATA_TYPE* y_2)
// {
//     // DATA_TYPE* a_gpu;
// 	DATA_TYPE* x1_gpu;
// 	DATA_TYPE* x2_gpu;
// 	DATA_TYPE* y_1_gpu;
// 	DATA_TYPE* y_2_gpu;
// 	DATA_TYPE* x1_cpu;
// 	DATA_TYPE* x2_cpu;

//     x1_cpu = (DATA_TYPE*)malloc(N*sizeof(DATA_TYPE));
// 	x2_cpu = (DATA_TYPE*)malloc(N*sizeof(DATA_TYPE));
//     printf("Allocating on GPU\n");
    
//     memcpy(x1_cpu, x1, N*sizeof(DATA_TYPE));
//     memcpy(x2_cpu, x2, N*sizeof(DATA_TYPE));
    
// 	// check_cuda_error(cudaMalloc(&a_gpu, sizeof(DATA_TYPE) * N * N));
// 	check_cuda_error(cudaMalloc(&x1_gpu, sizeof(DATA_TYPE) * N));   
// 	check_cuda_error(cudaMalloc(&x2_gpu, sizeof(DATA_TYPE) * N));   
// 	check_cuda_error(cudaMalloc(&y_1_gpu, sizeof(DATA_TYPE) * N));  
// 	check_cuda_error(cudaMalloc(&y_2_gpu, sizeof(DATA_TYPE) * N));

//     printf("Initializing finished on GPU. Transferring to GPU...\n"); 
// 	// check_cuda_error(cudaMemcpy(a_gpu, a, sizeof(DATA_TYPE) * N * N, cudaMemcpyHostToDevice));
// 	check_cuda_error(cudaMemcpy(y_1_gpu, y_1, sizeof(DATA_TYPE) * N, cudaMemcpyHostToDevice));  
// 	check_cuda_error(cudaMemcpy(y_2_gpu, y_2, sizeof(DATA_TYPE) * N, cudaMemcpyHostToDevice));  
//     check_cuda_error(cudaMemcpy(x1_gpu, x1, sizeof(DATA_TYPE) * N, cudaMemcpyHostToDevice));  
// 	check_cuda_error(cudaMemcpy(x2_gpu, x2, sizeof(DATA_TYPE) * N, cudaMemcpyHostToDevice));    

// 	dim3 block(DIM_THREAD_BLOCK_X/2, DIM_THREAD_BLOCK_Y);
// 	dim3 grid((size_t)ceil((float)N/ ((float)DIM_THREAD_BLOCK_X/2)), 1);
//     rdma_buf<DATA_TYPE> *rdma_a;

//     check_cuda_error(cudaMallocManaged((void **) &rdma_a, sizeof(rdma_buf<unsigned int>)));
    
//     rdma_a->start(N*N*sizeof(DATA_TYPE), GPU, NULL);

//     for(size_t i = 0; i < N*N; i++){
//         rdma_a->local_buffer[i] = a[i];
//     }

//     // transfer<<<2048, 512>>>(rdma_a->size/sizeof(DATA_TYPE), rdma_a);
//     cudaError_t ret = cudaDeviceSynchronize();
//     // check<<<2048, 512>>>(rdma_a->size/sizeof(DATA_TYPE), rdma_a, a_gpu);
//     printf("ret: %d cudaGetLastError: %d for transfer\n", ret, cudaGetLastError());

//     printf("Starting Kernels\n");
// 	auto start = std::chrono::steady_clock::now();
// 	mvt_kernel1_rdma<<<grid,block>>>(rdma_a, x1_gpu, y_1_gpu);
//     printf("ret: %d cudaGetLastError: %d for kernel1\n", ret, cudaGetLastError());
//     mvt_kernel2_rdma<<<grid,block>>>(rdma_a, x2_gpu, y_2_gpu, y_2_gpu);
// 	ret = cudaDeviceSynchronize();
//     printf("ret: %d cudaGetLastError: %d\n", ret, cudaGetLastError());
// 	auto end = std::chrono::steady_clock::now();
//     long duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    
//     printf("Elapsed time for GPU RDMA in milliseconds: %li ms.d\n\n", duration);

//     ret = cudaDeviceSynchronize();
//     print_retires<<<1,1>>>();

//     check_cuda_error(cudaMemcpy(x1, x1_gpu, sizeof(DATA_TYPE) * N, cudaMemcpyDeviceToHost));   
// 	check_cuda_error(cudaMemcpy(x2, x2_gpu, sizeof(DATA_TYPE) * N, cudaMemcpyDeviceToHost));

//     // check_cuda_error(cudaFree(a_gpu));
//     check_cuda_error(cudaFree(x1_gpu));
//     check_cuda_error(cudaFree(x2_gpu));
//     check_cuda_error(cudaFree(y_1_gpu));
//     check_cuda_error(cudaFree(y_2_gpu));

//     // start = std::chrono::steady_clock::now();
//     // //run the algorithm on the CPU
//     // printf("Running on CPU\n");
// 	// runMvt(a, x1_cpu, x2_cpu, y_1, y_2); 
//     // end = std::chrono::steady_clock::now();
//     // duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    
//     // printf("Elapsed time for CPU in milliseconds: %li ms.d\n\n", duration); 
//     // printf("Comparing Results for RDMA and CPU\n");
//     // compareResults(x1_cpu, x1, x2_cpu, x2);

//     free(x1_cpu);
//     free(x2_cpu);    
// }
// /******************************* RDMA Imlementation END ***************************************/


// Main program
int main(int argc, char **argv)
{   
    init_gpu(0);

    bool rdma_flag = true;
    cudaError_t ret1;
    struct context_2nic *s_ctx = (struct context_2nic *)malloc(sizeof(struct context_2nic));
    if(rdma_flag){
        s_ctx->gpu_cq = NULL;
        s_ctx->wqbuf = NULL;
        s_ctx->cqbuf = NULL;
        s_ctx->gpu_qp = NULL;


        int num_msg = (unsigned long) atoi(argv[4]);
        int mesg_size = (unsigned long) atoi(argv[5]);
        int num_bufs = (unsigned long) atoi(argv[6]);

        
        struct post_content post_cont, *d_post, host_post;
        struct poll_content poll_cont, *d_poll, host_poll;
        // struct post_content2 /*post_cont2,*/ *d_post2;
        struct server_content_2nic post_cont2, *d_post2;
        struct host_keys keys;
        struct gpu_memory_info gpu_mem;

        int num_iteration = num_msg;
        s_ctx->n_bufs = num_bufs;

        s_ctx->gpu_buf_size = 16*1024*1024*1024llu; // N*sizeof(int)*3llu;
        s_ctx->gpu_buffer = NULL;

        // // remote connection:
        // int ret = connect(argv[2], s_ctx);

        // local connect
        char *mlx_name = "mlx5_0";
        // int ret = local_connect(mlx_name, s_ctx);
        int ret = local_connect_2nic(mlx_name, s_ctx, 0, GPU);

        mlx_name = "mlx5_2";
        // int ret = local_connect(mlx_name, s_ctx);
        ret = local_connect_2nic(mlx_name, s_ctx, 1, GPU);

        ret = prepare_post_poll_content_2nic(s_ctx, &post_cont, &poll_cont, &post_cont2, \
                                        &host_post, &host_poll, &keys, &gpu_mem);
        if(ret == -1) {
            printf("Post and poll contect creation failed\n");    
            exit(-1);
        }

        printf("alloc synDev ret: %d\n", cudaDeviceSynchronize());
        cudaSetDevice(GPU);
        alloc_global_cont(&post_cont, &poll_cont, &post_cont2, gpu_mem);

        // if(cudaSuccess != ){    
        printf("alloc synDev ret1: %d\n", cudaDeviceSynchronize());
            // return -1;
        // }

        ret1 = cudaDeviceSynchronize();
        printf("ret: %d\n", ret1);
        if(cudaSuccess != ret1){    
            return -1;
        }

        size_t restricted_gpu_mem = 16*1024*1024*1024llu;
        // restricted_gpu_mem = restricted_gpu_mem / 3;
        const size_t page_size = REQUEST_SIZE;
        // const size_t numPages = ceil((double)restricted_gpu_mem/page_size);

        printf("function: %s line: %d\n", __FILE__, __LINE__);
        alloc_global_host_content(host_post, host_poll, keys, gpu_mem);
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
        cudaSetDevice(GPU);
        start_page_queue<<<1, 1>>>(/*s_ctx->gpu_buf_size*/restricted_gpu_mem, page_size);
        ret1 = cudaDeviceSynchronize();
        printf("ret: %d\n", ret1);
        if(cudaSuccess != ret1){    
            return -1;
        }
    }

    if(rdma_flag){
        printf("Allocation finished. Calling mvtRDMA\n");
        bicgRDMA();
        cudaFree(s_ctx->gpu_buffer);
    }

    printf("Allocation finished. Calling mvtCUDA\n");
    bicgCuda();

    

    printf("oversubs ratio: %d\n", oversubs_ratio_macro-1);
    
	return 0;
}

