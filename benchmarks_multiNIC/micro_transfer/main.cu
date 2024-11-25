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
// #include "../../include/runtime_eviction.h"
// #include "../../include/runtime_micro.h"
#include "../../include/runtime_prefetching_2nic.h"


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

#define GPU 0

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

__global__ __launch_bounds__(1024,2) 
void
calculate_opt(size_t n, size_t size, rdma_buf<unsigned int> *rdma_array, unsigned int *array) {


    // Page size in elements (64KB / 4 bytes per unsigned int)
    const size_t pageSize = REQUEST_SIZE/4 / sizeof(unsigned int);
    // Elements per warp
    const size_t elementsPerWarp = pageSize / warpSize;

    // Global thread ID
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    // if(tid == 0) printf("warpSize: %d\n", warpSize);
    // Warp ID within the block
    size_t warpId = tid / warpSize;

    // Thread lane within the warp
    size_t lane = threadIdx.x % warpSize;

    // Determine which page this warp will process
    size_t pageStart = warpId * pageSize;

    // Ensure we don't process out-of-bounds pages
    if (pageStart < n * pageSize) {
        
        // Process elements within the page
        // for (size_t i = 0; i < elementsPerWarp; ++i) {
        //     size_t elementIdx = pageStart + lane + i * warpSize;
            uint end = (warpId + 1)*pageSize > size ? size : (warpId + 1)*pageSize;
            for(size_t j = warpId*pageSize + lane; j < end; j += warpSize) {
                uint end_edge = (*rdma_array)[j]; // shared_data[j - pageStart];
                array[j] = end_edge;
            }
        // }
    }
}


__global__ void transfer_opt(size_t n, rdma_buf<unsigned int> *rdma_array) {


    // Page size in elements (64KB / 4 bytes per unsigned int)
    const size_t pageSize = 4*1024 / sizeof(unsigned int);
    // Elements per warp
    const size_t elementsPerWarp = pageSize / warpSize;

    // Global thread ID
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    // if(tid == 0) printf("warpSize: %d\n", warpSize);
    // Warp ID within the block
    size_t warpId = tid / warpSize;

    // Thread lane within the warp
    size_t lane = threadIdx.x % warpSize;

    // Determine which page this warp will process
    size_t pageStart = warpId * pageSize;

    // Ensure we don't process out-of-bounds pages
    if (pageStart < n * pageSize) {
        
        // Process elements within the page
        // for (size_t i = 0; i < elementsPerWarp; ++i) {
        //     size_t elementIdx = pageStart + lane + i * warpSize;
                
            for(size_t j = warpId*pageSize + lane; j < (warpId + 1)*pageSize; j += warpSize) {
                uint end_edge = (*rdma_array)[j]; // shared_data[j - pageStart];
            }
        // }
    }
}

void compute_benchmark(){

    cudaError_t ret;
    unsigned int *cuda_array, *h_cuda_array;
    uint64_t numblocks_update, numthreads, numblocks_kernel;
    double avg_milliseconds;
    float milliseconds;
    size_t num_elements, size = 12*1024*1024*1024llu; 
    num_elements = size/sizeof(uint);
    cudaEvent_t start, end;

    rdma_buf<unsigned int> *rdma_array;
    check_cuda_error(cudaMallocManaged((void **) &rdma_array, sizeof(rdma_buf<unsigned int>)));
    rdma_array->start(num_elements *sizeof(unsigned int), GPU, NULL);
    for(size_t i = 0; i < num_elements; i++){
        rdma_array->local_buffer[i] = 14;
    }

    check_cuda_error(cudaEventCreate(&start));
    check_cuda_error(cudaEventCreate(&end));

    // h_cuda_array = new uint[num_elements];
    auto start_chrono = std::chrono::steady_clock::now();
    check_cuda_error(cudaMallocHost((void **)&h_cuda_array, size));
    for(size_t i = 0; i < num_elements; i++){
        h_cuda_array[i] = 9;
    }
    check_cuda_error(cudaMalloc((void **) &cuda_array, size));
    cudaDeviceSynchronize();
    
    check_cuda_error(cudaEventRecord(start, (cudaStream_t) 1));

    check_cuda_error(cudaMemcpy(cuda_array, h_cuda_array, size, cudaMemcpyHostToDevice));
    
    check_cuda_error(cudaEventRecord(end, (cudaStream_t) 1));
    cudaDeviceSynchronize();
    auto end_chrono = std::chrono::steady_clock::now();
    
    
    check_cuda_error(cudaEventSynchronize(start));
    check_cuda_error(cudaEventSynchronize(end));
    check_cuda_error(cudaEventElapsedTime(&milliseconds, start, end));

    long duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_chrono - start_chrono).count();
    printf("ret: %d cudaGetLastError: %d\n", ret, cudaGetLastError());
    printf("Elapsed time for cudaMemcpy in milliseconds : %li ms bw: %.2f\n\n", duration, (float) size/(duration*0.001*1024*1024*1024) );
    printf("CUDA elapsed time for cudaMemcpy in milliseconds : %.2f ms bw: %.2f\n\n", milliseconds, (float) size/(milliseconds*0.001*1024*1024*1024) );

    // numblocks_update = ((numVertex + numthreads) / numthreads);
    dim3 blockDim_kernel(numthreads, (numblocks_kernel+numthreads)/numthreads);
    // dim3 blockDim_kernel(BLOCK_SIZE, (numblocks_kernel+BLOCK_SIZE)/BLOCK_SIZE);
    numthreads = BLOCK_SIZE;
    // numblocks_update = ((numVertex + numthreads) / numthreads);
    dim3 blockDim_update(BLOCK_SIZE, (numblocks_update+BLOCK_SIZE)/BLOCK_SIZE);

    avg_milliseconds = 0.0f;

    ret = cudaDeviceSynchronize();
    printf("ret: %d cudaGetLastError: %d\n", ret, cudaGetLastError());
    start_chrono = std::chrono::steady_clock::now();
    printf("starting kernel\n");
    size_t n_pages = size/(REQUEST_SIZE/4); // (REQUEST_SIZE);

    check_cuda_error(cudaEventRecord(start, (cudaStream_t) 1));
    numthreads = 1024;
<<<<<<< HEAD
    calculate_opt<<< (n_pages*32)/numthreads+1, numthreads >>>(num_elements, num_elements, rdma_array, cuda_array);
=======
    calculate_opt<<<(n_pages*32)/numthreads+1, numthreads>>>(num_elements, num_elements, rdma_array, cuda_array);
>>>>>>> origin/cloudlab
    check_cuda_error(cudaEventRecord(end, (cudaStream_t) 1));

    ret = cudaDeviceSynchronize();               
    end_chrono = std::chrono::steady_clock::now();
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_chrono - start_chrono).count();
    print_retires<<<1,1>>>();
    printf("ret: %d cudaGetLastError: %d\n", ret, cudaGetLastError());
    printf("Elapsed time in milliseconds : %li ms\n\n", duration);
    // printf("ret: %d cudaGetLastError: %d\n", ret, cudaGetLastError());
    
    check_cuda_error(cudaMemcpy(h_cuda_array, cuda_array, size, cudaMemcpyDeviceToHost));
    size_t num_errs = 0;
    for (size_t i = 0; i < num_elements; i++)
    {
        if(h_cuda_array[i] != 14){
            num_errs++;
            // printf("i: %d h_cuda_array[i]: %d\n", i, h_cuda_array[i]);
        }
    }
    

    
    check_cuda_error(cudaEventSynchronize(start));
    check_cuda_error(cudaEventSynchronize(end));
    check_cuda_error(cudaEventElapsedTime(&milliseconds, start, end));
    printf("CUDA elapsed time in milliseconds : %0.3f ms num_errs: %llu bw: %.2f REQUEST_SIZE: %d\n\n", milliseconds, num_errs, (float) size/(milliseconds*0.001*1024*1024*1024), REQUEST_SIZE/1024);

}

void transfer_benchmark(){
    cudaError_t ret;
    
    uint64_t numblocks_update, numthreads, numblocks_kernel;
    double avg_milliseconds;
    float milliseconds;
    size_t num_elements, size = 1*1024*1024*1024llu; 
    num_elements = size/sizeof(uint);
    
    cudaEvent_t start, end;

    rdma_buf<unsigned int> *rdma_array;
    check_cuda_error(cudaMallocManaged((void **) &rdma_array, sizeof(rdma_buf<unsigned int>)));

    rdma_array->start(num_elements *sizeof(unsigned int), GPU, NULL);

    for(size_t i = 0; i < num_elements; i++){
        rdma_array->local_buffer[i] = 14;
    }

   
    check_cuda_error(cudaEventCreate(&start));
    check_cuda_error(cudaEventCreate(&end));
   

    // numblocks_update = ((numVertex + numthreads) / numthreads);

    dim3 blockDim_kernel(numthreads, (numblocks_kernel+numthreads)/numthreads);
    // dim3 blockDim_kernel(BLOCK_SIZE, (numblocks_kernel+BLOCK_SIZE)/BLOCK_SIZE);
    numthreads = BLOCK_SIZE;
    // numblocks_update = ((numVertex + numthreads) / numthreads);
    dim3 blockDim_update(BLOCK_SIZE, (numblocks_update+BLOCK_SIZE)/BLOCK_SIZE);

    avg_milliseconds = 0.0f;

    printf("Initialization done\n");
    fflush(stdout);

    
    ret = cudaDeviceSynchronize();
    printf("ret: %d cudaGetLastError: %d\n", ret, cudaGetLastError());
    

    auto start_chrono = std::chrono::steady_clock::now();
    
    printf("starting kernel\n");
    size_t n_pages = size/(4*1024);

    check_cuda_error(cudaEventRecord(start, (cudaStream_t) 1));
    transfer_opt<<<(n_pages*32)/512+1, 512>>>(num_elements, rdma_array);
    check_cuda_error(cudaEventRecord(end, (cudaStream_t) 1));

    ret = cudaDeviceSynchronize();               
    auto end_chrono = std::chrono::steady_clock::now();
    long duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_chrono - start_chrono).count();
    print_retires<<<1,1>>>();
    printf("ret: %d cudaGetLastError: %d\n", ret, cudaGetLastError());
    printf("Elapsed time in milliseconds : %li ms\n\n", duration);
    // printf("ret: %d cudaGetLastError: %d\n", ret, cudaGetLastError());
    

    
    check_cuda_error(cudaEventSynchronize(start));
    check_cuda_error(cudaEventSynchronize(end));
    check_cuda_error(cudaEventElapsedTime(&milliseconds, start, end));
    printf("CUDA elapsed time in milliseconds : %li ms\n\n", milliseconds);

}

// Main program
int main(int argc, char **argv)
{   
    cudaSetDevice(GPU);

    // init_gpu(0);
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    printf("deviceCount: %d\n", deviceCount);

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

        s_ctx->gpu_buf_size = 12*1024*1024*1024llu; // N*sizeof(int)*3llu;
        s_ctx->gpu_buffer = NULL;

        // // remote connection:
        // int ret = connect(argv[2], s_ctx);

        // local connect
        char *mlx_name = "mlx5_0";
        // int ret = local_connect(mlx_name, s_ctx);
        int ret = local_connect_2nic(mlx_name, s_ctx, 0, GPU);

        mlx_name = "mlx5_3";
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
        compute_benchmark();
        // transfer_benchmark();
        cudaFree(s_ctx->gpu_buffer);
    }

    
    // printf("oversubs ratio: %d\n", oversubs_ratio_macro-1);
    
	return 0;
}