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
#include "../../include/runtime_prefetching_2nic.h"

//define the error threshold for the results "not matching"
#define PERCENT_DIFF_ERROR_THRESHOLD 0.05

//define a small float value
#define SMALL_FLOAT_VAL 0.00000001f

// Size of array
/* Problem size */
#define NI (32*256llu)
#define NJ (32*256llu)
#define NK (32*256llu)

#define BLOCK_NUM 1024ULL
#define MYINFINITY 2147483647llu

#define BLOCK_SIZE 1024
#define WARP_SHIFT 5
#define WARP_SIZE 32


#define GPU 0

#define TILE_WIDTH 16 // Adjust tile width based on shared memory size


/* Declared constant values for ALPHA and BETA (same as values in PolyBench 2.0) */
#define ALPHA 32412.0f
#define BETA 2123.0f

typedef float DATA_TYPE;

/* Thread block dimensions */
#define DIM_THREAD_BLOCK_X 32
#define DIM_THREAD_BLOCK_Y 8

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

void compareResults(DATA_TYPE* C, DATA_TYPE* C_outputFromGpu)
{
	int i, j, fail;
	fail = 0;
	
	// Compare C1 and C2
	for (i=0; i < NI; i++) 
	{
		for (j=0; j < NJ; j++) 
		{
			if (percentDiff(C[i*NJ + j], C_outputFromGpu[i*NJ + j]) > PERCENT_DIFF_ERROR_THRESHOLD) 
			{
				fail++;
			}
		}
	}
	
	// Print results
	printf("Non-Matching CPU-GPU Outputs Beyond Error Threshold of %4.2f Percent: %d\n", PERCENT_DIFF_ERROR_THRESHOLD, fail);
}

void init(DATA_TYPE *A, DATA_TYPE *B, DATA_TYPE *C, DATA_TYPE *A_gpu, DATA_TYPE *B_gpu, DATA_TYPE *C_gpu)
{
	int i, j;

  	for (i = 0; i < NI; i++)
	{
    	for (j = 0; j < NK; j++)
		{
			A[i*NK + j] = ((DATA_TYPE) i*j) / NI;
			A_gpu[i*NK + j] = ((DATA_TYPE) i*j) / NI;
		}
	}

  	for (i = 0; i < NK; i++)
	{
    	for (j = 0; j < NJ; j++)
		{
			  B[i*NJ + j] = ((DATA_TYPE) i*j + 1) / NJ;
			  B_gpu[i*NJ + j] = ((DATA_TYPE) i*j + 1) / NJ;
		}
	}

  	for (i = 0; i < NI; i++)
	{
    	for (j = 0; j < NJ; j++)
		{
			  C[i*NJ + j] = ((DATA_TYPE) i*j + 2) / NJ;
			//   C_gpu[i*NJ + j] = ((DATA_TYPE) i*j + 2) / NJ;
		}
	}
}


/******************************* CUDA Imlementation BEGIN ***************************************/
__global__ void gemm_kernel(DATA_TYPE *a, DATA_TYPE *b, DATA_TYPE *c)
{
	int j = blockIdx.x * blockDim.x + threadIdx.x;
	int i = blockIdx.y * blockDim.y + threadIdx.y;

	if ((i < NI) && (j < NJ))
	{	
		c[i * NJ + j] *= BETA;
		int k;
		for(k=0; k < NK; k++)
		{
			c[i * NJ + j] += ALPHA * a[i * NK + k] * b[k * NJ +j];
		}
	}
}

__global__ void gemm_kernel_tiled(DATA_TYPE *A, DATA_TYPE *B, DATA_TYPE *C) {
    // Block row and column indices
    int blockRow = blockIdx.y;
    int blockCol = blockIdx.x;
    
    // Thread row and column within the block
    int row = threadIdx.y;
    int col = threadIdx.x;

    // Submatrices in shared memory
    __shared__ DATA_TYPE shared_A[TILE_WIDTH][TILE_WIDTH];
    __shared__ DATA_TYPE shared_B[TILE_WIDTH][TILE_WIDTH];

    // Initialize the accumulation register
    DATA_TYPE cValue = 0.0;

    // Loop over the tiles of A and B that are required to compute Csub
    for (int t = 0; t < (NK + TILE_WIDTH - 1) / TILE_WIDTH; t++) {
        // Load tile of A into shared memory
        if ((blockRow * TILE_WIDTH + row < NI) && (t * TILE_WIDTH + col < NK)) {
            shared_A[row][col] = A[(blockRow * TILE_WIDTH + row) * NK + t * TILE_WIDTH + col];
        } else {
            shared_A[row][col] = 0.0;
        }

        // Load tile of B into shared memory
        if ((t * TILE_WIDTH + row < NK) && (blockCol * TILE_WIDTH + col < NJ)) {
            shared_B[row][col] = B[(t * TILE_WIDTH + row) * NJ + blockCol * TILE_WIDTH + col];
        } else {
            shared_B[row][col] = 0.0;
        }

        // Synchronize threads to ensure all data is loaded before proceeding
        __syncthreads();

        // Compute dot product for this tile
        for (int k = 0; k < TILE_WIDTH; k++) {
            cValue += shared_A[row][k] * shared_B[k][col];
        }

        // Synchronize threads to ensure that the computation is done before loading new tiles
        __syncthreads();
    }

    // Write the result to the output matrix C
    int globalRow = blockRow * TILE_WIDTH + row;
    int globalCol = blockCol * TILE_WIDTH + col;

    if (globalRow < NI && globalCol < NJ) {
        C[globalRow * NJ + globalCol] = ALPHA * cValue + BETA * C[globalRow * NJ + globalCol];
    }
}


void gemmCuda(int u_case)
{
	double t_start, t_end;


    DATA_TYPE* A;
	DATA_TYPE* B;  
	DATA_TYPE* C; 
	DATA_TYPE *A_gpu;
	DATA_TYPE *B_gpu;
	DATA_TYPE *C_gpu; 

    printf("Allocating A, B, C on CPU\n");
	A = (DATA_TYPE*)malloc(NI*NK*sizeof(DATA_TYPE)); 
	B = (DATA_TYPE*)malloc(NK*NJ*sizeof(DATA_TYPE));   
	C = (DATA_TYPE*)malloc(NI*NJ*sizeof(DATA_TYPE)); 

    printf("Allocating A, B, C on GPU\n");
	cudaMallocManaged(&A_gpu, sizeof(DATA_TYPE) * NI * NK);
	cudaMallocManaged(&B_gpu, sizeof(DATA_TYPE) * NK * NJ);
	cudaMalloc(&C_gpu, sizeof(DATA_TYPE) * NI * NJ);

    check_cuda_error(cudaMemAdvise(A_gpu, sizeof(DATA_TYPE) * NI * NK, cudaMemAdviseSetReadMostly, 0));
    check_cuda_error(cudaMemAdvise(B_gpu, sizeof(DATA_TYPE) * NK * NJ, cudaMemAdviseSetReadMostly, 0));
    

    printf("Initializing A, B, C\n");
	init(A, B, C, A_gpu, B_gpu, C_gpu);

    check_cuda_error(cudaMemcpy(C_gpu, C, sizeof(DATA_TYPE) * NI * NJ, cudaMemcpyHostToDevice));

    printf("Launching the kernel\n");
    auto start = std::chrono::steady_clock::now();

    switch(u_case){
        case 1:{
            printf("gemm_kernel\n");
            dim3 block(DIM_THREAD_BLOCK_X, DIM_THREAD_BLOCK_Y);
	        dim3 grid((size_t)(ceil( ((float)NI)/ ((float)block.x) )),(size_t)(ceil( ((float)NJ)/ ((float)block.y) )));

            gemm_kernel<<< grid, block >>>(A_gpu, B_gpu, C_gpu);
	        cudaDeviceSynchronize();
            break;
        }
        case 2:{
            printf("gemm_kernel_tiled\n");
            dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);
            dim3 dimGrid((NJ + TILE_WIDTH - 1) / TILE_WIDTH, (NI + TILE_WIDTH - 1) / TILE_WIDTH);

           
            gemm_kernel_tiled<<< dimGrid, dimBlock >>>(A_gpu, B_gpu, C_gpu);
	        cudaDeviceSynchronize();
            break;
        }
        default:{
            break;
        }
    }

	auto end = std::chrono::steady_clock::now();
    long duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    printf("Elapsed time for normal cuda in milliseconds: %li ms.d\n\n", duration);

	// compareResults(C, C_gpu);

	free(A);
	free(B);  
	free(C);  
	cudaFree(A_gpu);
	cudaFree(B_gpu);	  
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
/******************************* CUDA Imlementation END ***************************************/



/******************************* RDMA Imlementation BEGIN ***************************************/
__global__ // __launch_bounds__(1024,2)
void gemm_kernel_rdma(rdma_buf<DATA_TYPE> *a, rdma_buf<DATA_TYPE> *b, DATA_TYPE *c)
{
	int j = blockIdx.x * blockDim.x + threadIdx.x;
	int i = blockIdx.y * blockDim.y + threadIdx.y;

	if ((i < NI) && (j < NJ))
	{	
		c[i * NJ + j] *= BETA;
		int k;
		for(k=0; k < NK; k++)
		{
			c[i * NJ + j] += ALPHA * (*a)[i * NK + k]; // * (*b)[k * NJ +j];
		}
	}
}

__global__ void gemm_kernel_rdma_tiled(rdma_buf<DATA_TYPE> *A, rdma_buf<DATA_TYPE> *B, DATA_TYPE *C) {
    // Block row and column indices
    int blockRow = blockIdx.y;
    int blockCol = blockIdx.x;
    
    // Thread row and column within the block
    int row = threadIdx.y;
    int col = threadIdx.x;

    // Submatrices in shared memory
    __shared__ DATA_TYPE shared_A[TILE_WIDTH][TILE_WIDTH];
    __shared__ DATA_TYPE shared_B[TILE_WIDTH][TILE_WIDTH];

    // Initialize the accumulation register
    DATA_TYPE cValue = 0.0;

    // Loop over the tiles of A and B that are required to compute Csub
    for (int t = 0; t < (NK + TILE_WIDTH - 1) / TILE_WIDTH; t++) {
        // Load tile of A into shared memory
        if ((blockRow * TILE_WIDTH + row < NI) && (t * TILE_WIDTH + col < NK)) {
            shared_A[row][col] = (*A)[(blockRow * TILE_WIDTH + row) * NK + t * TILE_WIDTH + col];
        } else {
            shared_A[row][col] = 0.0;
        }

        // Load tile of B into shared memory
        if ((t * TILE_WIDTH + row < NK) && (blockCol * TILE_WIDTH + col < NJ)) {
            shared_B[row][col] = (*B)[(t * TILE_WIDTH + row) * NJ + blockCol * TILE_WIDTH + col];
        } else {
            shared_B[row][col] = 0.0;
        }

        // Synchronize threads to ensure all data is loaded before proceeding
        __syncthreads();

        // Compute dot product for this tile
        for (int k = 0; k < TILE_WIDTH; k++) {
            cValue += shared_A[row][k] * shared_B[k][col];
        }

        // Synchronize threads to ensure that the computation is done before loading new tiles
        __syncthreads();
    }

    // Write the result to the output matrix C
    int globalRow = blockRow * TILE_WIDTH + row;
    int globalCol = blockCol * TILE_WIDTH + col;

    if (globalRow < NI && globalCol < NJ) {
        C[globalRow * NJ + globalCol] = ALPHA * cValue + BETA * C[globalRow * NJ + globalCol];
    }
}

void gemmRDMA()
{
	double t_start, t_end;
    cudaError_t ret; // = cudaSuccess_t;

    rdma_buf<DATA_TYPE> *A_rdma;
	rdma_buf<DATA_TYPE> *B_rdma;  
	DATA_TYPE* C; 
	// DATA_TYPE *A_gpu;
	// DATA_TYPE *B_gpu;
	DATA_TYPE *C_gpu; 

    printf("Allocating A, B, C on CPU\n");
    cudaMallocManaged(&A_rdma, sizeof(rdma_buf<DATA_TYPE>));
	cudaMallocManaged(&B_rdma, sizeof(rdma_buf<DATA_TYPE>));

    A_rdma->start(NI*NK*sizeof(DATA_TYPE), GPU, NULL);
    B_rdma->start(NK*NJ*sizeof(DATA_TYPE), GPU, NULL);

	// A = (DATA_TYPE*)malloc(NI*NK*sizeof(DATA_TYPE)); 
	// B = (DATA_TYPE*)malloc(NK*NJ*sizeof(DATA_TYPE));   
	C = (DATA_TYPE*)malloc(NI*NJ*sizeof(DATA_TYPE)); 

    printf("Allocating A, B, C on GPU\n");
	
	cudaMalloc(&C_gpu, sizeof(DATA_TYPE) * NI * NJ);
    

    printf("Initializing A, B, C\n");
	init(A_rdma->local_buffer, B_rdma->local_buffer, C, A_rdma->local_buffer, B_rdma->local_buffer, C_gpu);

    // for(size_t i = 0; i < NI*NK; i++){
    //     A_rdma->local_buffer[i] = A[i]; 
    // }

    // for(size_t i = 0; i < NK*NJ; i++){
    //     B_rdma->local_buffer[i] = B[i]; 
    // }

    check_cuda_error(cudaMemcpy(C_gpu, C, sizeof(DATA_TYPE) * NI * NJ, cudaMemcpyHostToDevice));

    dim3 block(DIM_THREAD_BLOCK_X, DIM_THREAD_BLOCK_Y);
	dim3 grid((size_t)(ceil( ((float)NI)/ ((float)block.x) )),(size_t)(ceil( ((float)NJ)/ ((float)block.y) )));
	
    printf("Launching the kernel\n");
    auto start = std::chrono::steady_clock::now();

    int u_case = 2;

    switch(u_case){
        case 1:{
            printf("gemm_rdma\n");
            dim3 block(DIM_THREAD_BLOCK_X, DIM_THREAD_BLOCK_Y);
	        dim3 grid((size_t)(ceil( ((float)NI)/ ((float)block.x) )),(size_t)(ceil( ((float)NJ)/ ((float)block.y) )));

            gemm_kernel_rdma<<< grid, block >>>(A_rdma, A_rdma, C_gpu);
            ret = cudaDeviceSynchronize();
            printf("ret: %d cudaGetLastError: %d\n", ret, cudaGetLastError());
            break;
        }
        case 2:{
            printf("gemm_rdma_tiled\n");
            dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);
            dim3 dimGrid((NJ + TILE_WIDTH - 1) / TILE_WIDTH, (NI + TILE_WIDTH - 1) / TILE_WIDTH);

           
            gemm_kernel_rdma_tiled<<< dimGrid, dimBlock >>>(A_rdma, B_rdma, C_gpu);
	        cudaDeviceSynchronize();
            break;
        }
        default:{
            break;
        }
    }

	auto end = std::chrono::steady_clock::now();
    long duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    printf("Elapsed time for normal cuda in milliseconds: %li ms.d\n\n", duration);

	// compareResults(C, C_gpu);

	// free(A);
	// free(B);  
	free(C);  
	cudaFree(A_rdma);
	// cudaFree(B_rdma);
    cudaFree(C_gpu);	  
}

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
/******************************* RDMA Imlementation END ***************************************/


// Main program
int main(int argc, char **argv)
{   
    init_gpu(0);

    cudaSetDevice(GPU);

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

    unsigned long long *direct_new_repr;
    if(rdma_flag){
        printf("Allocation finished. Calling gesummRDMA\n");
        gemmRDMA();
        printf("Finished gesummRDMA\n");
        cudaFree(s_ctx->gpu_buffer);
    }

    printf("Allocation finished. Calling gemmCuda\n");
    gemmCuda(1);
    gemmCuda(2);

    printf("oversubs ratio: %d\n", oversubs_ratio_macro-1);
    
	return 0;
}