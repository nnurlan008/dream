#include <cstdio>
#include <cuda.h>
#include <cuda_runtime.h>
#include <string>
#include <cstring>
#include <unistd.h>
#include "graph.h"
#include "bfsCPU.h"
#include "bfs.cuh"

using namespace std;
// using namespace std;


// extern "C"{
//   #include "rdma_utils.h"
// }

// #include "../../src/rdma_utils.cuh"
#include <time.h>
#include "../../include/runtime_eviction_2nic.h"
// #include "../../include/runtime_prefetching.h"
// #include "../../include/runtime_prefetching_2nic.h"


// Size of array
#define N 1*1024*1024llu

#define BLOCK_NUM 1024ULL
#define MYINFINITY 2147483647llu

#define BLOCK_SIZE 1024
#define WARP_SHIFT 5
#define WARP_SIZE (1 << WARP_SHIFT)

#define MEM_ALIGN_64 (~(0xfULL))
#define MEM_ALIGN MEM_ALIGN_64

#define CHUNK_SHIFT 3
#define CHUNK_SIZE (1 << CHUNK_SHIFT)

#define GPU 0

// __device__ size_t transfer_time;

__device__ rdma_buf<unsigned int> D_adjacencyList;

__global__ void test(rdma_buf<unsigned int> *a/*, rdma_buf<int> *b, rdma_buf<int> *c*/);

__global__
void simpleBfs_normal(size_t n, size_t vertexCount, unsigned int level, unsigned int *d_edgeList, unsigned int *d_vertex_list,
                      unsigned int *d_distance, unsigned int *changed);

__global__
void simpleBfs_normal_rdma(size_t n, size_t vertexCount, unsigned int level, rdma_buf<unsigned int> *d_edgeList, unsigned int *d_vertex_list,
                      unsigned int *d_distance, unsigned int *changed);

__global__
void simpleBfs_uvm(size_t n, unsigned int level, unsigned int *d_adjacencyList, unsigned int *d_edgesOffset,
                    unsigned int *d_edgesSize, unsigned int *d_distance, unsigned int *changed);

__global__ // __launch_bounds__(1024,2)
void simpleBfs_rdma(size_t n, unsigned int level, rdma_buf<unsigned int> *d_adjacencyList, uint64_t *d_edgesOffset,
                    unsigned int *d_edgesSize, unsigned int *d_distance, unsigned int *changed);

__global__
void simpleBfs_rdma_2(size_t n, unsigned int level, rdma_buf<unsigned int> *d_adjacencyList, unsigned int *d_edgesOffset,
                      unsigned int *d_edgesSize, unsigned int *d_distance, unsigned int *changed);    

__global__
void simpleBfs_rdma_3(size_t n, unsigned int level, rdma_buf<unsigned int> *d_adjacencyList, unsigned int *vertexList, unsigned int *changed);               

__global__ __launch_bounds__(128,16)
void kernel_coalesce_ptr_pc(unsigned int *ptr, unsigned int *label, const uint32_t level, const uint64_t vertex_count, unsigned int *edgeOffset,
                            unsigned int * edgeSize, unsigned *edgeList, unsigned int *changed);

__global__
void simpleBfs_normal_rdma_optimized(int numEdgesPerThread, size_t numEdges, size_t vertexCount, unsigned int level, rdma_buf<unsigned int> *d_edgeList, unsigned int *d_vertex_list,
                      unsigned int *d_distance, unsigned int *changed);

__global__ __launch_bounds__(128,32)
void kernel_coalesce_ptr_pc_rdma(rdma_buf<unsigned int> *ptr, unsigned int *label, const uint32_t level, const uint64_t vertex_count, unsigned int *edgeOffset,
                            unsigned int * edgeSize, unsigned int *changed);

__global__
void simpleBfs_normal_rdma_optimized2(size_t n, size_t vertexCount, unsigned int level, rdma_buf<unsigned int> *d_edgeList, unsigned int *d_vertex_list,
                                     unsigned int *d_distance, unsigned int *changed);

__global__
void simpleBfs_rdma_optimized_warp(size_t n, size_t numVertices, unsigned int level, rdma_buf<unsigned int> *d_adjacencyList, unsigned int *d_edgesOffset,
                    unsigned int *d_edgesSize, unsigned int *d_distance, unsigned int *changed);

__global__
void simpleBfs_rdma_dynamic_page(size_t n, unsigned int level, rdma_buf<unsigned int> *d_adjacencyList, unsigned int *d_edgesOffset,
                                 unsigned int *d_edgesSize, unsigned int *d_distance, unsigned int *changed);

__global__
void simpleBfs_rdma_optimized_dynamic(size_t n, unsigned int level, rdma_buf<unsigned int> *d_adjacencyList, unsigned int *d_edgesOffset,
                    unsigned int *d_edgesSize, unsigned int *d_distance, unsigned int *changed);

__global__
void simpleBfs_rdma_optimized_warp2(size_t n, unsigned int level, rdma_buf<unsigned int> *d_adjacencyList, unsigned int *d_edgesOffset,
                                   unsigned int *d_edgesSize, unsigned int *d_distance, unsigned int *changed);

__global__
void simpleBfs_csr(size_t n, unsigned int level, unsigned int *d_adjacencyList,
                    uint64_t *vertex_list, unsigned int *d_distance, unsigned int *changed);

__global__
void simpleBfs_csr_warp_opt(size_t n, size_t numVertices, unsigned int level, unsigned int *d_adjacencyList, unsigned int *d_edgesOffset,
                    unsigned int *d_edgesSize, unsigned int *d_distance, unsigned int *changed);

__global__
void simpleBfs_modVertexList_rdma(size_t n, unsigned int level, rdma_buf<unsigned int> *d_adjacencyList,
                    unsigned int *vertex_list, unsigned int *d_distance, unsigned int *changed);

__global__
void simpleBfs_rdma_optimized_thread_different_page(size_t n, size_t numVertices, unsigned int level, rdma_buf<unsigned int> *d_adjacencyList, unsigned int *d_edgesOffset,
                    unsigned int *d_edgesSize, unsigned int *d_distance, unsigned int *changed);

__global__ void kernel_coalesce_chunk_rdma(unsigned int *label, unsigned int level, unsigned int vertex_count,
                                           unsigned int *vertexList, rdma_buf<unsigned int> *edgeList, uint *changed);

__global__ void 
kernel_baseline(uint32_t *label, const uint32_t level, const uint64_t vertex_count, const uint64_t *vertexList, 
                const uint64_t *edgeList, bool *changed);

__global__ void kernel_coalesce(uint32_t *label, const uint32_t level, const uint64_t vertex_count, 
                                const uint64_t *vertexList, const uint64_t *edgeList, bool *changed);

__global__ // __launch_bounds__(1024, 512)  
void kernel_coalesce_new_repr_rdma(uint level, size_t n, size_t numVertex, const uint64_t new_size, unsigned int *d_distance, 
                    uint64_t *new_offset, unsigned int *new_vertex_list, rdma_buf<unsigned int> *edgeList, unsigned int *changed);

__global__ void 
kernel_coalesce_new_repr(uint level, size_t n, size_t numVertex, const uint64_t new_size, unsigned int *d_distance, 
                        uint64_t *new_offset, unsigned int *new_vertex_list, unsigned int *edgeList, unsigned int *changed);

__global__ void 
bfs_kernel_coalesce_chunk(unsigned int *label, unsigned int level, const uint64_t vertex_count, uint64_t *vertexList, unsigned int *edgeList, unsigned int *changed);

__global__ void 
check_edgeList(rdma_buf<unsigned int> *a, unsigned int *b, size_t size);

// __global__ void kernel_coalesce_chunk(unsigned int *label, const uint level, const uint64_t vertex_count, const uint64_t *vertexList,
//                                       const uint64_t *edgeList, uint *changed);

__global__ void kernel_coalesce_chunk(uint32_t *label, const uint32_t level, const uint64_t vertex_count, const uint64_t *vertexList,
                                      const uint64_t *edgeList, bool *changed) {
    const uint64_t tid = blockDim.x * BLOCK_SIZE * blockIdx.y + blockDim.x * blockIdx.x + threadIdx.x;
    const uint64_t warpIdx = tid >> WARP_SHIFT;
    const uint64_t laneIdx = tid & ((1 << WARP_SHIFT) - 1);
    const uint64_t chunkIdx = warpIdx * CHUNK_SIZE;
    uint64_t chunk_size = CHUNK_SIZE;

    if((chunkIdx + CHUNK_SIZE) > vertex_count) {
        if ( vertex_count > chunkIdx )
            chunk_size = vertex_count - chunkIdx;
        else
            return;
    }

    for(uint32_t i = chunkIdx; i < chunk_size + chunkIdx; i++) {
        if(label[i] == level) {
            const uint64_t start = vertexList[i];
            const uint64_t shift_start = start & MEM_ALIGN;
            const uint64_t end = vertexList[i+1];

            for(uint64_t j = shift_start + laneIdx; j < end; j += WARP_SIZE) {
                if (j >= start) {
                    const uint64_t next = edgeList[j];

                    // if(label[next] == MYINFINITY) {
                    if(label[next] > level + 1) {
                        label[next] = level + 1;
                        *changed = true;
                    }
                }
            }
        }
    }
}


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



void runCpu(int startVertex, Graph &G, std::vector<int> &distance,
            std::vector<int> &parent, std::vector<bool> &visited) {
    printf("Starting sequential bfs.\n");
    auto start = std::chrono::steady_clock::now();
    bfsCPU(startVertex, G, distance, parent, visited);
    auto end = std::chrono::steady_clock::now();
    long duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    printf("Elapsed time in milliseconds : %li ms.\n\n", duration);
}


#define checkError(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }

}

rdma_buf<unsigned int> *rdma_adjacencyList;
// rdma_buf<unsigned int> *u_edgesOffset;
// rdma_buf<unsigned int> *u_edgesSize;
// rdma_buf<unsigned int> *u_distance;
unsigned int *u_adjacencyList;
unsigned int *uvm_adjacencyList;
uint64_t *u_edgesOffset;
unsigned int *u_edgesSize;
unsigned int *u_distance;
unsigned int *u_startVertices;
rdma_buf<unsigned int> *u_parent;
rdma_buf<unsigned int> *u_currentQueue;
rdma_buf<unsigned int> *u_nextQueue;
rdma_buf<unsigned int> *u_degrees;

// uint *u_adjacencyList;
// uint *u_edgesOffset;
// uint *u_edgesSize;
// uint *u_distance;
// uint *u_parent;
// uint *u_currentQueue;
// uint *u_nextQueue;
// uint *u_degrees;

uint *incrDegrees;


void initCuda(Graph &G) {

    checkError(cudaMallocManaged(&rdma_adjacencyList, sizeof(rdma_buf<unsigned int>)));
    // checkError(cudaMallocManaged(&uvm_adjacencyList, G.numEdges*sizeof(unsigned int)));
    
    auto start = std::chrono::steady_clock::now();
    // checkError(cudaMallocHost(&u_adjacencyList, G.numEdges*sizeof(unsigned int)));
    u_adjacencyList = (unsigned int *) malloc(G.numEdges*sizeof(uint)); // new uint[G.numEdges];
    // u_adjacencyList = new uint[G.numEdges];
    auto end = std::chrono::steady_clock::now();
    long duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    printf("Elapsed time in milliseconds for cudaMallocHost: %li ms.\n", duration);

    
    
    // checkError(cudaMallocManaged(&u_edgesOffset, sizeof(rdma_buf<unsigned int>)));
    // checkError(cudaMallocManaged(&u_edgesSize, sizeof(rdma_buf<unsigned int>)));
    // checkError(cudaMallocManaged(&u_distance, sizeof(rdma_buf<unsigned int>)));
    
    // checkError(cudaMallocHost(&u_startVertices, G.numEdges*sizeof(unsigned int)));
    checkError(cudaMallocHost(&u_distance, G.numVertices*sizeof(unsigned int)));
    checkError(cudaMallocHost(&u_edgesOffset, (G.numVertices+1)*sizeof(uint64_t)));
    checkError(cudaMallocHost(&u_edgesSize, G.numVertices*sizeof(unsigned int)));

    // rdma_adjacencyList->start(G.numEdges *sizeof(unsigned int));

    // u_edgesOffset->start(G.numVertices *sizeof(unsigned int));
    // u_edgesSize->start(G.numVertices *sizeof(unsigned int));
    // u_distance->start(G.numVertices *sizeof(unsigned int));
    // u_parent->start(G.numVertices *sizeof(unsigned int));
    // u_currentQueue->start(G.numVertices *sizeof(unsigned int));
    // u_nextQueue->start(G.numVertices *sizeof(unsigned int));
    // u_degrees->start(G.numVertices *sizeof(unsigned int));

    // checkError(cudaMallocManaged(&u_adjacencyList, G.numEdges * sizeof(int) ));
    // checkError(cudaMallocManaged(&u_edgesOffset, G.numVertices * sizeof(int) ));
    // checkError(cudaMallocManaged(&u_edgesSize, G.numVertices * sizeof(int)) );
    // checkError(cudaMallocManaged(&u_distance, G.numVertices * sizeof(int) ));
    // checkError(cudaMallocManaged(&u_parent, G.numVertices * sizeof(int) ));
    // checkError(cudaMallocManaged(&u_currentQueue, G.numVertices * sizeof(int) ));
    // checkError(cudaMallocManaged(&u_nextQueue, G.numVertices * sizeof(int) ));
    // checkError(cudaMallocManaged(&u_degrees, G.numVertices * sizeof(int) ));



    // checkError(cudaMallocHost((void **) &incrDegrees, sizeof(int) * G.numVertices));


    // memcpy(u_adjacencyList, G.adjacencyList.data(), G.numEdges * sizeof(int));
    // memcpy(u_edgesOffset, G.edgesOffset.data(), G.numVertices * sizeof(int));
    // memcpy(u_edgesSize, G.edgesSize.data(), G.numVertices * sizeof(int));
    // checkError(cudaMemcpy(d_adjacencyList, G.adjacencyList.data(), G.numEdges * sizeof(int), cudaMemcpyHostToDevice));
    // checkError(cudaMemcpy(d_edgesOffset, G.edgesOffset.data(), G.numVertices * sizeof(int), cudaMemcpyHostToDevice));
    // checkError(cudaMemcpy(d_edgesSize, G.edgesSize.data(), G.numVertices * sizeof(int), cudaMemcpyHostToDevice ));

}

void finalizeCuda() {

    checkError(cudaFree(u_adjacencyList));
    checkError(cudaFree(u_edgesOffset));
    checkError(cudaFree(u_edgesSize));
    checkError(cudaFree(u_distance));
    checkError(cudaFree(u_parent));
    checkError(cudaFree(u_currentQueue));
    checkError(cudaFree(u_nextQueue));
    checkError(cudaFree(u_degrees));
    checkError(cudaFreeHost(incrDegrees));
}

// void checkOutput(std::vector<int> &distance, std::vector<int> &expectedDistance, Graph &G) {
//     for (int i = 0; i < G.numVertices; i++) {
//         if (expectedDistance[i] != *(u_distance+i) ) {
//             printf("%d %d %d\n", i, distance[i], expectedDistance[i]);
//             printf("Wrong output!\n");
//             exit(1);
//         }
//     }

//     printf("Output OK!\n\n");
// }

void checkOutput_rdma(uint *u_distance, uint *expectedDistance, Graph &G) {
    size_t num_errors = 0;
    for (int i = 0; i < G.numVertices; i++) {
        if(i < 10){
                printf("%d ", i);
                printf("%llu ", u_distance[i]);
                printf("%d\n", expectedDistance[i]);
            }
        if (expectedDistance[i] != u_distance[i] ) {
            // 
            
            // printf("Wrong output!\n");
            num_errors++;
            // exit(1);
        }
    }

    printf("num_errors: %llu Output OK!\n\n", num_errors);
}


void initializeCudaBfs(int startVertex, std::vector<int> &distance, std::vector<int> &parent, Graph &G) {
    //initialize values
    std::fill(distance.begin(), distance.end(), 100000/*std::numeric_limits<int>::max()*/);
    std::fill(parent.begin(), parent.end(), std::numeric_limits<int>::max());
    distance[startVertex] = 0;
    parent[startVertex] = 0;

    // checkError(cudaMemcpy(d_distance, distance.data(), G.numVertices * sizeof(int), cudaMemcpyHostToDevice));
    // checkError(cudaMemcpy(d_parent, parent.data(), G.numVertices * sizeof(int), cudaMemcpyHostToDevice));
    // memcpy(u_distance->local_buffer, distance.data(), G.numVertices * sizeof(int));
    // memcpy(u_parent->local_buffer, parent.data(), G.numVertices * sizeof(int));
    printf("printing samples from parent and distance vectors initializations...\n");
    for (size_t i = 0; i < G.numVertices; i++)
    {
        u_distance[i] = 2147483647; // distance.data()[i];
        // u_parent->local_buffer[i] = parent.data()[i];
    }
    u_distance[startVertex] = 0;

    for (size_t i = 0; i < 5; i++)
    {
        printf("u_distance->local_buffer[%llu]: %llu; distance.data()[%llu]: %llu\n", i, u_distance[i], i, distance.data()[i]);
        // printf("u_parent->local_buffer[%llu]: %llu; parent.data()[%llu]: %llu\n", i, u_parent->local_buffer[i], i, parent.data()[i]);
        
    }
    

    int firstElementQueue = startVertex;
    // cudaMemcpy(d_currentQueue, &firstElementQueue, sizeof(int), cudaMemcpyHostToDevice);
    // *u_currentQueue->local_buffer = firstElementQueue;
}

void finalizeCudaBfs(std::vector<int> &distance, std::vector<int> &parent, Graph &G) {
    //copy memory from device
    // checkError(cudaMemcpy(distance.data(), d_distance, G.numVertices * sizeof(int), cudaMemcpyDeviceToHost));
    // checkError(cudaMemcpy(parent.data(), d_parent, G.numVertices * sizeof(int), cudaMemcpyDeviceToHost));
}

__global__ void transfer(size_t size, rdma_buf<unsigned int> *d_adjacencyList, unsigned int *changed)
{
    size_t id = blockDim.x * blockIdx.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;
    
        for (size_t i = id; i < size ; i += stride)
        {
            unsigned int y = (*d_adjacencyList)[i];
            // y++;
            // *changed += y;
        }
}

__global__ void assign_array(rdma_buf<unsigned int> *adjacencyList){
    D_adjacencyList = *adjacencyList;
    printf("D_adjacencyList.d_TLB[0].state: %d\n", D_adjacencyList.d_TLB[0].state);
    // printf("D_adjacencyList.d_TLB[0].device_address: %p\n", D_adjacencyList.d_TLB[0].page_number);
}

__global__ void test2(size_t size, int level, unsigned int *d_distance, rdma_buf<unsigned int> *d_edgesOffset,
                      rdma_buf<unsigned int> *d_edgesSize, rdma_buf<unsigned int> *d_adjacencyList, unsigned int *changed)
{
    size_t id = blockDim.x * blockIdx.x + threadIdx.x;
    int valueChange = 0;
    size_t stride = blockDim.x * gridDim.x;
    // size_t size = d_distance->size/sizeof(unsigned int);
    
    if(id < size){
        unsigned int k = d_distance[id];
        uint edgesOffset = (*d_edgesOffset)[id];
        uint edgesSize = (*d_edgesSize)[id];
        for (size_t i = edgesOffset; i < edgesOffset + edgesSize /*d_adjacencyList->size/sizeof(unsigned int)*/; i += 1)
        {
            unsigned int y = (*d_adjacencyList)[i];
            if(k == level){
            //     if(i < edgesOffset + edgesSize && i >= edgesOffset){
            //         unsigned int dist = (*d_distance)[y];
                    if (level + 1 < d_distance[y]) {
                    
                        unsigned int new_dist = level + 1;
                        d_distance[i] = new_dist /*(int) level + 1*/;
                        valueChange = 1;
                    }
            //     }
            }
        }
    }
        if (valueChange) {
            *changed = valueChange;
        }
    // }
    // a->rvalue(id, id);
    // c->rvalue(id, (*a)[id] + (*b)[id]); 
    // if(id == 0) printf("(*b)[%d]: %d\n", id, (*b)[id]);
}

void runCudaSimpleBfs(int startVertex, Graph &G, std::vector<int> &distance,
                      std::vector<int> &parent) 
{
    initializeCudaBfs(startVertex, distance, parent, G);

    for (size_t i = 0; i < G.numVertices; i++)
    {
        if(u_distance[i] == 0)
            printf("%zu-%zu ", u_edgesOffset[i], u_edgesOffset[i] + u_edgesSize[i]);
            // printf("%zu-%zu ", u_edgesOffset->local_buffer[i], u_edgesOffset->local_buffer[i] + u_edgesSize->local_buffer[i]);
        // printf("%llu ", u_edgesOffset->local_buffer[i]);
    }

    for (size_t i = 0; i < G.numVertices; i++)
    {
        if(u_distance[i] == 0){
            printf("u_distance->host_buffer[%llu]: %u\n", i, u_distance[i]);
        }
    }

    unsigned int *d_distance, *d_edgesSize, *d_edgesOffset, *d_adjacencyList, *d_startVertices;
    // rdma_buf<unsigned int> *d_adjacencyList;
    // checkError(cudaMalloc((void **) &d_adjacencyList, sizeof(rdma_buf<unsigned int>)));
    checkError(cudaMalloc((void **) &d_startVertices, G.numEdges*sizeof(unsigned int)));
    checkError(cudaMemcpy(d_startVertices, u_startVertices, G.numEdges*sizeof(unsigned int), cudaMemcpyHostToDevice));

    checkError(cudaMalloc((void **) &d_adjacencyList, G.numEdges*sizeof(unsigned int)));
    checkError(cudaMemcpy(d_adjacencyList, u_adjacencyList, G.numEdges*sizeof(unsigned int), cudaMemcpyHostToDevice));
    // checkError(cudaMemcpy(d_adjacencyList, u_adjacencyList, sizeof(rdma_buf<unsigned int>), cudaMemcpyHostToDevice));

    checkError(cudaMalloc((void **) &d_distance, G.numVertices*sizeof(unsigned int)));
    checkError(cudaMemset(d_distance, 100000, G.numVertices*sizeof(unsigned int)));
    checkError(cudaMemset(&d_distance[startVertex], 0, 1*sizeof(unsigned int)));

    checkError(cudaMalloc((void **) &d_edgesSize, G.numVertices*sizeof(unsigned int)));
    checkError(cudaMalloc((void **) &d_edgesOffset, G.numVertices*sizeof(unsigned int)));
    
    checkError(cudaMemcpy(d_edgesOffset, u_edgesOffset, G.numVertices*sizeof(unsigned int), cudaMemcpyHostToDevice));
    checkError(cudaMemcpy(d_edgesSize, u_edgesSize, G.numVertices*sizeof(unsigned int), cudaMemcpyHostToDevice));

    uint *changed;
    checkError(cudaMallocHost((void **) &changed, sizeof(unsigned int)));


    //launch kernel
    printf("Starting simple parallel bfs.\n");

    cudaError_t ret1 = cudaDeviceSynchronize();
    if(cudaSuccess != ret1){  
        printf("cudaDeviceSynchronize error: %d\n", ret1);  
        exit(-1);
    }
    auto start = std::chrono::steady_clock::now();
    
    cudaEvent_t event1, event2;
    cudaEventCreate(&event1);
    cudaEventCreate(&event2);

    ret1 = cudaDeviceSynchronize();

    cudaEventRecord(event1, (cudaStream_t)1);
    
    unsigned int level;
    // checkError(cudaMallocManaged((void **) &level, sizeof(unsigned int)));
    level = 0;
    // transfer<<<2048, 512>>>(rdma_adjacencyList->size/sizeof(unsigned int), rdma_adjacencyList, changed);
    
    // printf("cudaDeviceSynchronize for transfer: %d\n", ret1); 
    
    cudaEventRecord(event2, (cudaStream_t) 1);
    ret1 = cudaDeviceSynchronize();
    assign_array<<< 1 , 1 >>>(rdma_adjacencyList);
    ret1 = cudaDeviceSynchronize();
    printf("cudaDeviceSynchronize for transfer: %d\n", ret1);  
    if(cudaSuccess != ret1){  
        printf("cudaDeviceSynchronize error for transfer: %d\n", ret1);  
        exit(-1);
    }
    cudaEventSynchronize(event1); //optional
    cudaEventSynchronize(event2); //wait for the event to be executed!
    float dt_ms;
    cudaEventElapsedTime(&dt_ms, event1, event2);
    printf("Elapsed time for transfer with cudaEvent: %f\n", dt_ms);
    ret1 = cudaDeviceSynchronize();
    int numEdgesPerThread = 8;
    int numThreadsPerBlock = 512;
    // int numBlock = (numNodes) / (numThreadsPerBlock) + 1;
    int numBlock = (G.numEdges) / (numThreadsPerBlock * numEdgesPerThread) + 1;
    cudaEventRecord(event1, (cudaStream_t)1);
    *changed = 1;
    while (*changed) {
        *changed = 0;
        // void *args[] = {&G.numVertices, &level, &d_adjacencyList, &d_edgesOffset, &d_edgesSize, &d_distance, &d_parent,
        //                 &changed};
        // checkError(cuLaunchKernel(cuSimpleBfs, G.numVertices / 1024 + 1, 1, 1,
        //                           1024, 1, 1, 0, 0, args, 0));
        // ret1 = cudaDeviceSynchronize();
        // printf("cudaDeviceSynchronize: %d\n", ret1);  
        // if(cudaSuccess != ret1){  
        //     printf("cudaDeviceSynchronize error: %d\n", ret1);  
        //     exit(-1);
        // }
        // printf("G.numVertices: %llu\n", G.numVertices); 
        // simpleBfs_uvm<<< G.numVertices / 512 + 1, 512 >>>(G.numVertices, level, uvm_adjacencyList, d_edgesOffset, d_edgesSize, d_distance, changed);  
        // simpleBfs_rdma<<< G.numVertices / 256 + 1, 256 >>>(G.numVertices, level, rdma_adjacencyList, d_edgesOffset, d_edgesSize, d_distance, changed);    
        // kernel_coalesce_hash_ptr_pc<<< G.numVertices / 512 + 1, 512 >>>(u_adjacencyList, d_distance, level, G.numVertices); 
        // simpleBfs_normal_rdma<<< /*G.numVertices / 512 + 1, 512*/ 2048, 512 >>>(G.numEdges, G.numVertices, level, rdma_adjacencyList, d_startVertices, d_distance, changed);
        // simpleBfs_normal_rdma_optimized<<<numBlock, numThreadsPerBlock>>>(numEdgesPerThread, G.numEdges, G.numVertices, level, rdma_adjacencyList, d_startVertices, d_distance, changed);
        // simpleBfs_normal<<< G.numVertices / 512 + 1, 512 >>>(G.numEdges, G.numVertices, level, uvm_adjacencyList/*d_adjacencyList u_adjacencyList*/, d_startVertices, d_distance, changed);
        
        // for cudamemcpy 
        // simpleBfs_normal<<< /*G.numVertices / 512 + 1, 512*/ 2048, 8 >>>(G.numEdges, G.numVertices, level, uvm_adjacencyList, d_startVertices, d_distance, changed);      
        ret1 = cudaDeviceSynchronize();
        // test2<<< /*G.numVertices/256+1, 256*/ G.numVertices/256+1, 256>>>(G.numVertices, level, u_distance, u_edgesOffset, u_edgesSize, u_adjacencyList, changed);
        // test<<< 2, 1024>>>(u_adjacencyList);
        

        level++;
    }
    cudaEventRecord(event2, (cudaStream_t) 1);
    printf("cudaGetLastError(): %d level: %d\n", cudaGetLastError(), level);
        ret1 = cudaDeviceSynchronize();
        printf("cudaDeviceSynchronize: %d *changed: %d\n", ret1, *changed);  
        if(cudaSuccess != ret1){  
            printf("cudaDeviceSynchronize error: %d\n", ret1);  
            exit(-1);
        }


    
    cudaEventSynchronize(event1); //optional
    cudaEventSynchronize(event2); //wait for the event to be executed!

    // calculate time
    dt_ms;
    cudaEventElapsedTime(&dt_ms, event1, event2);
    printf("Elapsed time with cudaEvent: %f\n", dt_ms);

    auto end = std::chrono::steady_clock::now();
    long duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    printf("Elapsed time in milliseconds : %li ms.\n", duration);

    checkError(cudaMemcpy(u_distance, d_distance, G.numVertices*sizeof(unsigned int), cudaMemcpyHostToDevice));

    finalizeCudaBfs(distance, parent, G);
}

__global__ 
void test_batch(rdma_buf<unsigned int> *a, unsigned int *b, size_t size){
    size_t tid = blockDim.x * blockIdx.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;
    // size_t index = tid*16*1024;

    for (size_t i = tid; i < size; i += stride)
    {
        unsigned int ab = (*a)[i];
        if(ab != b[i]){
            printf("unmatched: %llu expected: %d got: %d\n", i, b[i], ab);
        }
    }
    

    // if(index < size){
    //     unsigned int ab = (*a)[index];
    //     if(ab != b[index]){
    //         printf("unmatched: %llu expected: %d got: %d\n", index/16384, b[index], ab);
    //     }
    // }
}

unsigned int *runCudaSimpleBfs_emogi(int startVertex, Graph &G) 
{
    // initializeCudaBfs(startVertex, distance, parent, G);
    
    printf("fuction: %s, line: %d\n", __func__, __LINE__);

    unsigned int *d_distance, *d_edgesSize, *d_startVertices, *d_vertexList;
    uint64_t *d_adjacencyList, *d_edgesOffset;
    // rdma_buf<unsigned int> *d_adjacencyList;
    // checkError(cudaMalloc((void **) &d_adjacencyList, sizeof(rdma_buf<unsigned int>)));

    // checkError(cudaMalloc((void **) &d_startVertices, G.numEdges*sizeof(unsigned int)));
    // checkError(cudaMemcpy(d_startVertices, u_startVertices, G.numEdges*sizeof(unsigned int), cudaMemcpyHostToDevice));

    uint64_t *u_adjacencyList_64;
    u_adjacencyList_64 = (uint64_t *) malloc(sizeof(uint64_t)*G.numEdges);  
    // checkError(cudaMallocHost((void **) &u_adjacencyList_64, sizeof(uint64_t)*G.numEdges));
    for (size_t i = 0; i < G.numEdges; i++)
    {
       u_adjacencyList_64[i] = u_adjacencyList[i];
    }

    int mem = 2;
    size_t edge_size = G.numEdges*sizeof(uint64_t);
    if(mem == 0){
        checkError(cudaMalloc((void **) &d_adjacencyList, G.numEdges*sizeof(uint64_t)));
        cudaDeviceSynchronize();
        auto start = std::chrono::steady_clock::now();
        checkError(cudaMemcpy(d_adjacencyList, u_adjacencyList_64, G.numEdges*sizeof(uint64_t), cudaMemcpyHostToDevice));
        auto end = std::chrono::steady_clock::now();
        long duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        printf("Elapsed time for tranfser of edge list in milliseconds : %li ms.\n\n", duration);
    }
    else if (mem == 1) {   
        // case UVM_READONLY:
            checkError(cudaMallocManaged((void**)&d_adjacencyList, edge_size));
            // file.read((char*)edgeList_d, edge_size);
            memcpy(d_adjacencyList, u_adjacencyList_64, edge_size);


            auto start2 = std::chrono::steady_clock::now();
            checkError(cudaMemAdvise(d_adjacencyList, edge_size, cudaMemAdviseSetReadMostly, 0));

            auto end2 = std::chrono::steady_clock::now();
            long duration2 = std::chrono::duration_cast<std::chrono::milliseconds>(end2 - start2).count();
            printf("Elapsed time for memAdvise SetReadMostly in milliseconds : %li ms.\n\n", duration2);
    }
     
    else if(mem == 2){ 
          // break;
        // case UVM_DIRECT:
            checkError(cudaMallocManaged((void**)&d_adjacencyList, edge_size));
            // file.read((char*)edgeList_d, edge_size);
            std::cout << "line: " << __LINE__  <<  " edge_size: " << edge_size << std::endl;
            memcpy(d_adjacencyList, u_adjacencyList_64, edge_size);
            // for (size_t i = 0; i < edge_count; i++)
            // {
            //     // std::cout << "i: " << i << " edgeList_d[i]: "<< edgeList_d[i] << std::endl;
            //     edgeList_d[i] = temp_edgeList[i];
                
            // }
            std::cout << "line: " << __LINE__ << std::endl;

            auto start1 = std::chrono::steady_clock::now();
            // checkError(cudaMemAdvise(d_adjacencyList, edge_size, cudaMemAdviseSetAccessedBy, 0));
            auto end1 = std::chrono::steady_clock::now();
            long duration1 = std::chrono::duration_cast<std::chrono::milliseconds>(end1 - start1).count();
            printf("Elapsed time for memAdvise SetAccessedBy in milliseconds : %li ms.\n\n", duration1);

            cudaDeviceProp devProp;
            cudaGetDeviceProperties(&devProp, 0);
            // Calculate memory utilization
            size_t totalMemory = devProp.totalGlobalMem;
            size_t freeMemory;
            size_t usedMemory;
            float workload_size = (float) G.numEdges*sizeof(uint);
            cudaMemGetInfo(&freeMemory, &totalMemory);
            usedMemory = totalMemory - freeMemory;
            printf("Total GPU Memory: %.2f MiB\n", (float) totalMemory / (1024 * 1024));
            printf("Free GPU Memory: %.2f MiB\n", (float) freeMemory / (1024 * 1024));
            printf("Used GPU Memory: %.2f MiB\n", (float) usedMemory / (1024 * 1024));

            printf("Workload size: %.2f\n", workload_size/1024/1024);
            float oversubs_ratio = 0;
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
            // break;
    }
    
    // checkError(cudaMemcpy(d_adjacencyList, u_adjacencyList, sizeof(rdma_buf<unsigned int>), cudaMemcpyHostToDevice));

    checkError(cudaMalloc((void **) &d_distance, G.numVertices*sizeof(unsigned int)));
    uint *h_distance = new uint[G.numVertices];
    for (size_t i = 0; i < G.numVertices; i++)
    {
        h_distance[i] = 100000;
    }
    h_distance[startVertex] = 0;
    checkError(cudaMemcpy(d_distance, h_distance, G.numVertices*sizeof(unsigned int), cudaMemcpyHostToDevice));
    // checkError(cudaMemset(d_distance, 100000 /*std::numeric_limits<int>::max()*/, G.numVertices*sizeof(unsigned int)));
    // checkError(cudaMemset(&d_distance[startVertex], 0, sizeof(unsigned int)));

    checkError(cudaMalloc((void **) &d_edgesSize, G.numVertices*sizeof(unsigned int)));
    checkError(cudaMemcpy(d_edgesSize, u_edgesSize, G.numVertices*sizeof(unsigned int), cudaMemcpyHostToDevice));

    checkError(cudaMalloc((void **) &d_edgesOffset, (G.numVertices+1)*sizeof(uint64_t)));

    uint64_t *u_edgesOffset_64 = (uint64_t *) malloc(sizeof(uint64_t)*(G.numVertices+1)); 
    for (size_t i = 0; i < G.numVertices+1; i++)
    {
       u_edgesOffset_64[i] = u_edgesOffset[i];
    }
    
    checkError(cudaMemcpy(d_edgesOffset, u_edgesOffset_64, (G.numVertices+1)*sizeof(uint64_t), cudaMemcpyHostToDevice));

    printf("fuction: %s, line: %d\n", __func__, __LINE__);

    auto start = std::chrono::steady_clock::now();
    // cudaEventRecord(event1, (cudaStream_t)1);
    
    //launch kernel
    printf("Starting simple parallel bfs.\n");
    cudaError_t ret1 = cudaDeviceSynchronize();
    if(cudaSuccess != ret1){  
        printf("cudaDeviceSynchronize error: %d\n", ret1);  
        exit(-1);
    }

    ret1 = cudaDeviceSynchronize();
    // ret1 = cudaDeviceSynchronize();
    // cudaEventRecord(event2, (cudaStream_t) 1);
    auto end = std::chrono::steady_clock::now();

    // cudaEventSynchronize(event1); //optional
    // cudaEventSynchronize(event2); //wait for the event to be executed!

    float dt_ms;
    // cudaEventElapsedTime(&dt_ms, event1, event2);
    // printf("Elapsed time for transfer: %f\n", dt_ms);
    long duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    printf("Elapsed time in milliseconds for transfer : %li ms.\n", duration);

    uint64_t numblocks, numthreads;
    numthreads = BLOCK_SIZE;
    int type = 0;
    uint64_t vertex_count = G.numVertices;
    switch (type) {
        case 0:
            numblocks = ((vertex_count + numthreads) / numthreads);
            break;
        case 1:
            numblocks = ((vertex_count * WARP_SIZE + numthreads) / numthreads);
            break;
        case 2:
            numblocks = ((vertex_count * (WARP_SIZE / CHUNK_SIZE) + numthreads) / numthreads);
            break;
        default:
            fprintf(stderr, "Invalid type\n");
            exit(1);
            break;
    }
    dim3 blockDim(BLOCK_SIZE, (numblocks+BLOCK_SIZE)/BLOCK_SIZE);

    unsigned int level;
    level = 0;
    printf("fuction: %s, line: %d\n", __func__, __LINE__);
    
    

    bool *changed_d, changed_h;
    // checkError(cudaMallocHost((void **) &changed_h, sizeof(bool)));
    checkError(cudaMalloc((void **) &changed_d, sizeof(bool)));

    ret1 = cudaDeviceSynchronize();
    cudaEvent_t event1, event2;
    cudaEventCreate(&event1);
    cudaEventCreate(&event2);
    start = std::chrono::steady_clock::now();
    cudaEventRecord(event1,  (cudaStream_t) 1);
    changed_h = true;

    do {
            changed_h = false;
            checkError(cudaMemcpy(changed_d, &changed_h, sizeof(bool), cudaMemcpyHostToDevice));

            switch (type) {
                case 0:
                    kernel_baseline<<<blockDim, numthreads>>>
                    (d_distance, level, vertex_count, d_edgesOffset, d_adjacencyList, changed_d);
                    break;
                case 1:
                    kernel_coalesce<<<blockDim, numthreads>>>
                    (d_distance, level, vertex_count, d_edgesOffset, d_adjacencyList, changed_d);
                    break;
                case 2:
                    kernel_coalesce_chunk<<<blockDim, numthreads>>>
                    (d_distance, level, vertex_count, d_edgesOffset, d_adjacencyList, changed_d);
                    break;
                default:
                    fprintf(stderr, "Invalid type\n");
                    exit(1);
                    break;
            }
            // printf("cudaDeviceSynchronize(): %d\n", cudaDeviceSynchronize());
            cudaDeviceSynchronize();
            printf("level: %d\n", level);
            level++;

            checkError(cudaMemcpy(&changed_h, changed_d, sizeof(bool), cudaMemcpyDeviceToHost));
        } while(changed_h);

    cudaEventRecord(event2,  (cudaStream_t) 1);
    cudaEventSynchronize(event1); //optional
    cudaEventSynchronize(event2); //wait for the event to be executed!
    cudaEventElapsedTime(&dt_ms, event1, event2);
    printf("cudaGetLastError(): %d level: %d\n", cudaGetLastError(), level);
        ret1 = cudaDeviceSynchronize();
        printf("cudaDeviceSynchronize: %d *changed: %d\n", ret1, changed_h);  
        if(cudaSuccess != ret1){  
            printf("cudaDeviceSynchronize error: %d\n", ret1);  
            exit(-1);
        }

    // calculate time
    // float dt_ms;
    
    printf("Elapsed time with cudaEvent: %f\n", dt_ms);

    end = std::chrono::steady_clock::now();
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    printf("Elapsed time in milliseconds : %li ms.\n", duration);

    checkError(cudaMemcpy(h_distance, d_distance, G.numVertices*sizeof(uint), cudaMemcpyDeviceToHost));
    return h_distance;
}

__device__ size_t sum_page_faults = 0;
__device__ size_t correct_results = 0;

__global__ void
print_retires(void){
    // size_t max = cq_wait[0];
    // for (size_t i = 0; i < 128; i++)
    // {
    //     if(max < cq_wait[i]) max = cq_wait[i];
    // }
    
    // printf("g_qp_index: %llu cq_wait: %llu\n", g_qp_index, max);
    // g_qp_index = 0;
    // for (size_t i = 0; i < 128; i++)
    // {
    //     max = 0;
    // }
    sum_page_faults += g_qp_index;
    printf("g_qp_index: %llu sum page fault: %llu\n", g_qp_index, sum_page_faults);
    // g_qp_index = 0;
}

__global__ void
correct_retires(void){
    printf("correct results: %llu\n", correct_results);

}

void runCudaSimpleBfs_optimized(int startVertex, Graph &G, std::vector<int> &distance,
                      std::vector<int> &parent, bool rdma, bool uvm) 
{
    // initializeCudaBfs(startVertex, distance, parent, G);

    printf("fuction: %s, line: %d\n", __func__, __LINE__);

    unsigned int *d_distance, *d_edgesSize, *d_edgesOffset, *d_adjacencyList, *d_startVertices, *d_vertexList;
    // rdma_buf<unsigned int> *d_adjacencyList;
    // checkError(cudaMalloc((void **) &d_adjacencyList, sizeof(rdma_buf<unsigned int>)));

    // checkError(cudaMalloc((void **) &d_startVertices, G.numEdges*sizeof(unsigned int)));
    // checkError(cudaMemcpy(d_startVertices, u_startVertices, G.numEdges*sizeof(unsigned int), cudaMemcpyHostToDevice));

    if(!rdma)
    checkError(cudaMalloc((void **) &d_adjacencyList, G.numEdges*sizeof(unsigned int)));
    
    // checkError(cudaMemcpy(d_adjacencyList, u_adjacencyList, sizeof(rdma_buf<unsigned int>), cudaMemcpyHostToDevice));

    checkError(cudaMalloc((void **) &d_distance, G.numVertices*sizeof(unsigned int)));
    uint *h_distance = new uint[G.numVertices];
    for (size_t i = 0; i < G.numVertices; i++)
    {
        h_distance[i] = 100000;
    }
    h_distance[startVertex] = 0;
    checkError(cudaMemcpy(d_distance, h_distance, G.numVertices*sizeof(unsigned int), cudaMemcpyHostToDevice));
    free(h_distance);
    // checkError(cudaMemcpy(d_distance, u_distance, G.numVertices*sizeof(unsigned int), cudaMemcpyHostToDevice));
    // checkError(cudaMemset(d_distance, 100000, G.numVertices*sizeof(unsigned int)));
    // checkError(cudaMemset(&d_distance[startVertex], 0, 1*sizeof(unsigned int)));

    checkError(cudaMalloc((void **) &d_edgesSize, G.numVertices*sizeof(unsigned int)));
    // checkError(cudaMemcpy(d_edgesSize, u_edgesSize, G.numVertices*sizeof(unsigned int), cudaMemcpyHostToDevice));

    checkError(cudaMalloc((void **) &d_edgesOffset, (G.numVertices+1)*sizeof(unsigned int)));
    checkError(cudaMemcpy(d_edgesOffset, u_edgesOffset, (G.numVertices+1)*sizeof(unsigned int), cudaMemcpyHostToDevice));

    // checkError(cudaMalloc((void **) &d_vertexList, (G.numVertices + 1)*sizeof(unsigned int)));
    // checkError(cudaMemcpy(d_vertexList, u_edgesOffset, G.numVertices*sizeof(unsigned int), cudaMemcpyHostToDevice));
    
    unsigned int changed_h, *d_changed;
    checkError(cudaMalloc((void **) &d_changed, sizeof(unsigned int)));

    printf("fuction: %s, line: %d\n", __func__, __LINE__);

    //launch kernel
    printf("Starting simple parallel bfs.\n");
    cudaError_t ret1 = cudaDeviceSynchronize();
    if(cudaSuccess != ret1){  
        printf("cudaDeviceSynchronize error: %d\n", ret1);  
        exit(-1);
    }
    
    
    cudaEvent_t event1, event2;
    cudaEventCreate(&event1);
    cudaEventCreate(&event2);

    uint64_t threadsPerBlock = 256;
    uint64_t numblocks = ((G.numVertices + threadsPerBlock - 1) / threadsPerBlock);
    size_t sharedMemorySize = threadsPerBlock * sizeof(unsigned int);

    ret1 = cudaDeviceSynchronize();
    auto start = std::chrono::steady_clock::now();
    cudaEventRecord(event1, (cudaStream_t)1);
    if(!rdma)
        checkError(cudaMemcpy(d_adjacencyList, u_adjacencyList, G.numEdges*sizeof(unsigned int), cudaMemcpyHostToDevice));
    // ret1 = cudaDeviceSynchronize();
    cudaEventRecord(event2, (cudaStream_t) 1);
    auto end = std::chrono::steady_clock::now();

    cudaEventSynchronize(event1); //optional
    cudaEventSynchronize(event2); //wait for the event to be executed!

    float dt_ms;
    cudaEventElapsedTime(&dt_ms, event1, event2);
    printf("Elapsed time for transfer: %f\n", dt_ms);
    long duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    printf("Elapsed time in milliseconds for transfer : %li ms.\n", duration);

    unsigned int level;
    level = 0;
    printf("fuction: %s, line: %d\n", __func__, __LINE__);

    cudaDeviceProp devProp;
    cudaGetDeviceProperties(&devProp, 0);
    printf("Cuda device clock rate = %d\n", devProp.clockRate);

    ret1 = cudaDeviceSynchronize();
    cudaEventRecord(event1, (cudaStream_t)1);

    if(uvm){
        cudaDeviceProp devProp;
        cudaGetDeviceProperties(&devProp, 0);
        // Calculate memory utilization
        size_t totalMemory = devProp.totalGlobalMem;
        size_t freeMemory;
        size_t usedMemory;
        float workload_size = ((float) G.numEdges*sizeof(uint));
        cudaMemGetInfo(&freeMemory, &totalMemory);
        usedMemory = totalMemory - freeMemory;
        printf("Total GPU Memory: %.2f MiB\n", (float) totalMemory / (1024 * 1024));
        printf("Free GPU Memory: %.2f MiB\n", (float) freeMemory / (1024 * 1024));
        printf("Used GPU Memory: %.2f MiB\n", (float) usedMemory / (1024 * 1024));

        printf("Workload size: %.2f\n", workload_size/1024/1024);
        float oversubs_ratio = 0;
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

    uint64_t numthreads = BLOCK_SIZE;
    numblocks = ((G.numVertices * (WARP_SIZE / CHUNK_SIZE) + numthreads) / numthreads);
    dim3 blockDim(BLOCK_SIZE, (numblocks+BLOCK_SIZE)/BLOCK_SIZE);
    start = std::chrono::steady_clock::now();
    changed_h = 1;
    while (changed_h) {
        changed_h = 0;
        checkError(cudaMemcpy(d_changed, &changed_h, sizeof(unsigned int), cudaMemcpyHostToDevice));
        auto start1 = std::chrono::steady_clock::now();
        if(!rdma){
            // kernel_coalesce_ptr_pc<<< blockDim1, numthreads>>>(d_adjacencyList, d_distance, level, (const uint64_t) G.numVertices, d_edgesOffset,
            //                     d_edgesSize, d_adjacencyList, changed);
            
            // simpleBfs_normal<<< G.numVertices / 1024 + 1, 1024 >>>(G.numEdges, G.numVertices, level, uvm_adjacencyList /* d_adjacencyList u_adjacencyList */, d_startVertices, d_distance, changed);
            // simpleBfs_uvm<<<G.numVertices / 1024 + 1, 1024>>>(G.numVertices, level, d_adjacencyList, d_edgesOffset, d_edgesSize, d_distance, changed); 
            
            // simpleBfs_csr<<<G.numVertices / 1024 + 1, 1024>>>(G.numVertices, level, d_adjacencyList, d_edgesOffset, d_distance, d_changed); 
            // kernel_coalesce_chunk<<<blockDim, numthreads>>>(d_distance, level, G.numVertices, d_edgesOffset,
            //                       d_adjacencyList, d_changed);
            
            // kernel_coalesce(unsigned int *d_distance, unsigned int level, size_t n, 
            //                     unsigned int *vertexList, unsigned int *d_adjacencyList, unsigned int *changed) 
        
        }
        else{
            // kernel_coalesce_ptr_pc_rdma<<< blockDim1, numthreads>>>(rdma_adjacencyList, d_distance, level, (const uint64_t) G.numVertices, d_edgesOffset,
            //                     d_edgesSize, changed);
            // simpleBfs_normal_rdma<<< /*G.numVertices / 512 + 1, 512*/ 2048, 512 >>>(G.numEdges, G.numVertices, level, rdma_adjacencyList, d_startVertices, d_distance, changed);
            // simpleBfs_rdma<<< /*G.numVertices / 256 + 1, 512*/ 2048, 512 >>>(G.numVertices, level, rdma_adjacencyList, d_edgesOffset, d_edgesSize, d_distance, changed); 
            // simpleBfs_normal_rdma_optimized2<<<1024, 512>>>(G.numEdges, G.numVertices, level, rdma_adjacencyList, d_startVertices, d_distance, changed);
            

            // simpleBfs_rdma_optimized_dynamic<<< numblocks, threadsPerBlock, sharedMemorySize >>>(G.numVertices, level, rdma_adjacencyList, d_edgesOffset, d_edgesSize, d_distance, changed);

            // check_edgeList<<< G.numEdges/512+1, 512>>>
            // (rdma_adjacencyList, uvm_adjacencyList, G.numEdges);

            // the next one for friendster:
            size_t n_pages = G.numVertices*sizeof(unsigned int)/(4*1024)+1;
            // if(level >= 2)
                simpleBfs_rdma_optimized_warp<<</*G.numVertices / 256 + 1, 256*/(n_pages*32)/512 + 1, 512>>>(
                        n_pages, G.numVertices, level, rdma_adjacencyList, d_edgesOffset, d_edgesSize, d_distance, d_changed);
            // else
                // simpleBfs_modVertexList_rdma<<< (G.numVertices*32)/512 + 1, 512>>>(
                //         G.numVertices, level, rdma_adjacencyList, d_edgesOffset, d_distance, d_changed); 

                // simpleBfs_rdma<<< G.numVertices/512 + 1, 512>>>
                // (G.numVertices, level, rdma_adjacencyList, d_edgesOffset, d_edgesSize, d_distance, d_changed);


            // simpleBfs_rdma_optimized_thread_different_page<<</*G.numVertices / 256 + 1, 256*/1024, 512>>>
            // (n_pages,  G.numVertices, level, rdma_adjacencyList, d_edgesOffset, d_edgesSize, d_distance, changed);
            
            // test_batch <<< 2048*2, 512 >>> (rdma_adjacencyList, d_adjacencyList, rdma_adjacencyList->size/sizeof(unsigned int));

            // kernel_coalesce_chunk_rdma<<<blockDim, numthreads>>>
            // (d_distance, level, G.numVertices, d_edgesOffset, rdma_adjacencyList, d_changed);

            // simpleBfs_rdma_optimized_warp2<<</*G.numVertices / 256 + 1, 256*/1024, 512>>>(G.numVertices, level, rdma_adjacencyList, d_edgesOffset, d_edgesSize, d_distance, changed);

            // simpleBfs_rdma_dynamic_page<<<G.numVertices / 512 + 1, 512>>>(G.numVertices, level, rdma_adjacencyList, d_edgesOffset, d_edgesSize, d_distance, changed);
        
            // print_retires<<<1,1>>>();
            printf("cudaGetLastError(): %d level: %d\n", cudaGetLastError(), level);
            
        }
        ret1 = cudaDeviceSynchronize();
        auto end1 = std::chrono::steady_clock::now();
        long duration1 = std::chrono::duration_cast<std::chrono::milliseconds>(end1 - start1).count();
        printf("Elapsed time in milliseconds : %li ms.\n\n", duration);
        
        checkError(cudaMemcpy(&changed_h, d_changed, sizeof(unsigned int), cudaMemcpyDeviceToHost));
        level++;
    }
    cudaEventRecord(event2, (cudaStream_t) 1);
    printf("cudaGetLastError(): %d level: %d\n", cudaGetLastError(), level);
        ret1 = cudaDeviceSynchronize();
        printf("cudaDeviceSynchronize: %d *changed: %d\n", ret1, changed_h);  
        if(cudaSuccess != ret1){  
            printf("cudaDeviceSynchronize error: %d\n", ret1);  
            exit(-1);
        }


    
    cudaEventSynchronize(event1); //optional
    cudaEventSynchronize(event2); //wait for the event to be executed!
    

    // calculate time
    // float dt_ms;
    cudaEventElapsedTime(&dt_ms, event1, event2);
    printf("Elapsed time with cudaEvent: %f\n", dt_ms);

    end = std::chrono::steady_clock::now();
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    printf("Elapsed time in milliseconds : %li ms.\n", duration);

    checkError(cudaMemcpy(u_distance, d_distance, G.numVertices*sizeof(unsigned int), cudaMemcpyDeviceToHost));

    finalizeCudaBfs(distance, parent, G);
}


__global__ void test(rdma_buf<unsigned int> *a/*, rdma_buf<int> *b, rdma_buf<int> *c*/){
    size_t id = blockDim.x * blockIdx.x + threadIdx.x;
    int k = (*a)[id]; // + (*b)[id];
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



__global__
void print_utilization() {
    printf("GPU_address_offset: %llu \n", GPU_address_offset);
}

__global__
void print_transferTime(void) {
    printf("transfer time: %llu \n", transfer_time);
}

float time_total = 0;
float time_readmostly_total = 0;

uint *runRDMA(int startVertex, Graph &G, bool rdma, unsigned int *new_vertex_list, uint64_t *new_offset, unsigned int new_size,
             unsigned int *u_adjacencyList, uint64_t *u_edgesOffset, int u_case) 
{
    printf("fuction: %s, line: %d\n", __func__, __LINE__);
    // u_case: 0: direct transfer; 1: rdma edgelist rep.; 2: new rep.

    unsigned int *d_distance, *d_edgesSize, *d_adjacencyList, *d_startVertices, *d_vertexList;
    uint64_t *d_edgesOffset, *d_new_offset;
    unsigned int *d_new_vertexList;
    unsigned int *return_distance;
    // checkError(cudaMemcpy(d_adjacencyList, u_adjacencyList, sizeof(rdma_buf<unsigned int>), cudaMemcpyHostToDevice));

    return_distance = new uint[G.numVertices];
    checkError(cudaMalloc((void **) &d_distance, G.numVertices*sizeof(unsigned int)));
    uint *h_distance = new uint[G.numVertices];
    
    for (size_t i = 0; i < G.numVertices; i++)
    {
        h_distance[i] = 100000;
    }

    h_distance[startVertex] = 0;
    checkError(cudaMemcpy(d_distance, h_distance, G.numVertices*sizeof(unsigned int), cudaMemcpyHostToDevice));
    free(h_distance);
    checkError(cudaMalloc((void **) &d_edgesSize, G.numVertices*sizeof(unsigned int)));
    checkError(cudaMalloc((void **) &d_edgesOffset, (G.numVertices+1)*sizeof(uint64_t)));
    checkError(cudaMemcpy(d_edgesOffset, u_edgesOffset, (G.numVertices+1)*sizeof(uint64_t), cudaMemcpyHostToDevice));

    if(u_case == 2 || u_case == 3 || u_case == 4){
        checkError(cudaMalloc((void **) &d_new_vertexList, new_size*sizeof(unsigned int)));
        checkError(cudaMemcpy(d_new_vertexList, new_vertex_list, new_size*sizeof(unsigned int), cudaMemcpyHostToDevice));

        checkError(cudaMalloc((void **) &d_new_offset, (new_size+1)*sizeof(uint64_t)));
        checkError(cudaMemcpy(d_new_offset, new_offset, (new_size+1)*sizeof(uint64_t), cudaMemcpyHostToDevice));
    }
    
    unsigned int changed_h, *d_changed;
    checkError(cudaMalloc((void **) &d_changed, sizeof(unsigned int)));

    printf("fuction: %s, line: %d\n", __func__, __LINE__);

    //launch kernel
    printf("Starting simple parallel bfs.\n");
    cudaError_t ret1 = cudaDeviceSynchronize();
    if(cudaSuccess != ret1){  
        printf("cudaDeviceSynchronize error: %d\n", ret1);  
        exit(-1);
    }
    
    
    cudaEvent_t event1, event2;
    cudaEventCreate(&event1);
    cudaEventCreate(&event2);

    uint64_t threadsPerBlock = 256;
    uint64_t numblocks = ((G.numVertices + threadsPerBlock - 1) / threadsPerBlock);
    size_t sharedMemorySize = threadsPerBlock * sizeof(unsigned int);

    ret1 = cudaDeviceSynchronize();
    if(u_case == 0 || u_case == 3){
        checkError(cudaMalloc((void **) &d_adjacencyList, G.numEdges*sizeof(unsigned int)));

        ret1 = cudaDeviceSynchronize();
        auto start = std::chrono::steady_clock::now();
        cudaEventRecord(event1, (cudaStream_t)1);
        
        checkError(cudaMemcpy(d_adjacencyList, u_adjacencyList, G.numEdges*sizeof(unsigned int), cudaMemcpyHostToDevice));
        // ret1 = cudaDeviceSynchronize();
        cudaEventRecord(event2, (cudaStream_t) 1);
        auto end = std::chrono::steady_clock::now();

        cudaEventSynchronize(event1); //optional
        cudaEventSynchronize(event2); //wait for the event to be executed!
        float dt_ms;
        cudaEventElapsedTime(&dt_ms, event1, event2);
        printf("Elapsed time for transfer: %f\n", dt_ms);
        long duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        printf("Elapsed time in milliseconds for transfer : %li ms. data: %f\n", duration, (double) G.numEdges*sizeof(unsigned int)/(1024*1024*1024llu));
        // exit(0);
    }

    auto start1 = std::chrono::steady_clock::now();
    // checkError(cudaMemAdvise(uvm_adjacencyList, G.numEdges*sizeof(unsigned int), cudaMemAdviseSetAccessedBy, 0));
    auto end1 = std::chrono::steady_clock::now();
    long duration1 = std::chrono::duration_cast<std::chrono::milliseconds>(end1 - start1).count();
    printf("Elapsed time for memAdvise SetAccessedBy in milliseconds : %li ms.\n\n", duration1);

    unsigned int level;
    level = 0;
    printf("fuction: %s, line: %d\n", __func__, __LINE__);

    
    if(u_case == 6 || u_case == 7){
        if(!uvm_adjacencyList)
            checkError(cudaMallocManaged(&uvm_adjacencyList, G.numEdges*sizeof(unsigned int)));

        for(size_t i = 0; i < G.numEdges; i++){
            uvm_adjacencyList[i] = u_adjacencyList[i];
        }
        ret1 = cudaDeviceSynchronize();
        auto start1 = std::chrono::steady_clock::now();
        cudaEventRecord(event1, (cudaStream_t)1);
        checkError(cudaMemAdvise(uvm_adjacencyList, G.numEdges*sizeof(unsigned int), cudaMemAdviseSetReadMostly, 0));
        ret1 = cudaDeviceSynchronize();
        cudaEventRecord(event2, (cudaStream_t) 1);
        cudaEventSynchronize(event1); //optional
        cudaEventSynchronize(event2); //wait for the event to be executed!
        // calculate time
        float dt_ms;
        cudaEventElapsedTime(&dt_ms, event1, event2);
        printf("Elapsed time for cudaMemAdviseSetReadMostly with cudaEvent: %f startVertex: %d\n", dt_ms, startVertex);
        auto end1 = std::chrono::steady_clock::now();
        long duration1 = std::chrono::duration_cast<std::chrono::milliseconds>(end1 - start1).count();
        printf("Elapsed time for cudaMemAdviseSetReadMostly in milliseconds : %li ms.\n\n", duration1);
        time_readmostly_total += dt_ms;
        cudaDeviceProp devProp;
        cudaGetDeviceProperties(&devProp, 0);
        // Calculate memory utilization
        size_t totalMemory = devProp.totalGlobalMem;
        size_t freeMemory;
        size_t usedMemory;
        float workload_size = (float) G.numEdges*sizeof(uint);
        cudaMemGetInfo(&freeMemory, &totalMemory);
        usedMemory = totalMemory - freeMemory;
        printf("Total GPU Memory: %.2f MiB\n", (float) totalMemory / (1024 * 1024));
        printf("Free GPU Memory: %.2f MiB\n", (float) freeMemory / (1024 * 1024));
        printf("Used GPU Memory: %.2f MiB\n", (float) usedMemory / (1024 * 1024));

        printf("Workload size: %.2f\n", workload_size/1024/1024);
        float oversubs_ratio = 0;
        void *tmp_ptr;
        // cudaMalloc(&tmp_ptr, (size_t) (freeMemory - workload_size));
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

    cudaDeviceProp devProp;
    cudaGetDeviceProperties(&devProp, 0);
    printf("Cuda device clock rate = %d\n", devProp.clockRate);

    

    uint64_t numthreads = BLOCK_SIZE;
    numblocks = ((G.numVertices * (WARP_SIZE / CHUNK_SIZE) + numthreads) / numthreads);
    dim3 blockDim(BLOCK_SIZE, (numblocks+BLOCK_SIZE)/BLOCK_SIZE);
    delay(10);
    // transfer<<<2048, 256>>>(rdma_adjacencyList->size/sizeof(unsigned int), rdma_adjacencyList, NULL);
    ret1 = cudaDeviceSynchronize();

    auto start = std::chrono::steady_clock::now();
    changed_h = 1;
    ret1 = cudaDeviceSynchronize();
    

    printf("2. Cuda device clock rate = %d\n", devProp.clockRate);
        
     switch (u_case)
        {
        case 0:
            printf("direct transfer\n");
            break;
        case 1:
            printf("rdma csr edgelist representation\n");
            break;
        case 2: 
            printf("rdma new representation\n");
            break;
        case 3:
            printf("direct new representation\n");
            break;
        case 4: 
            printf("uvm new representation\n");
            break;
        case 5:
            printf("uvm transfer edge list\n");
            break;
        case 6:
            printf("uvm transfer\n");
            break;
        case 7:
            printf("veryfing rdma transfer\n");
        default:
            break;
        }

    cudaEventRecord(event1, (cudaStream_t)1);

    while (changed_h) {
        changed_h = 0;
        checkError(cudaMemcpy(d_changed, &changed_h, sizeof(unsigned int), cudaMemcpyHostToDevice));
        auto start = std::chrono::steady_clock::now();
        switch (u_case)
        {

        case 0:{
            // printf("direct transfer\n");
            simpleBfs_csr<<<G.numVertices / 1024 + 1, 1024>>>
            (G.numVertices, level, d_adjacencyList, d_edgesOffset, d_distance, d_changed); 
            ret1 = cudaDeviceSynchronize();
            break;
        }
        case 1:{
            // printf("rdma csr edgelist representation\n");
            size_t n_pages = G.numVertices*sizeof(unsigned int)/(8*1024);
            // __launch_bounds__(1024,2)
            numthreads = BLOCK_SIZE/2;
            numblocks = ((G.numVertices * (WARP_SIZE / CHUNK_SIZE) + numthreads) / numthreads);
            dim3 blockDim(numthreads, (numblocks+numthreads)/numthreads);
            simpleBfs_rdma<<< blockDim, numthreads /*(G.numVertices / numthreads)*(1 << WARP_SHIFT) + 1, numthreads*/ >>>(G.numVertices, level, rdma_adjacencyList, d_edgesOffset,
                    d_edgesSize, d_distance, d_changed);

            // simpleBfs_rdma_optimized_warp<<</*G.numVertices / 256 + 1, 256*/(n_pages*32)/512 + 1, 512>>>(
            //             n_pages, G.numVertices, level, rdma_adjacencyList, d_edgesOffset, d_edgesSize, d_distance, d_changed); 
            ret1 = cudaDeviceSynchronize();
            break;
        }
        case 2: {// new representation{
            // printf("rdma new representation\n");
            numthreads = 512;
            numblocks = ((new_size * (WARP_SIZE / CHUNK_SIZE) + numthreads) / numthreads);
            dim3 blockDim(numthreads, (numblocks+numthreads)/numthreads);
            size_t n_pages = (new_size*sizeof(uint64_t))/(8*1024);
            kernel_coalesce_new_repr_rdma<<< /*blockDim, numthreads*/ (n_pages*32)/numthreads + 1, numthreads >>>
            (level, n_pages, G.numVertices, new_size, d_distance, d_new_offset, d_new_vertexList, rdma_adjacencyList,
            d_changed);
            ret1 = cudaDeviceSynchronize(); 
            break;
        }
        case 3:{
            // printf("direct new representation\n");
            size_t n_pages = new_size*sizeof(unsigned int)/(8*1024)+1;
            kernel_coalesce_new_repr<<<(n_pages*32)/512 + 1, 512>>>
            (level, n_pages, G.numVertices, new_size, d_distance, d_new_offset, d_new_vertexList, d_adjacencyList,
            d_changed);
            ret1 = cudaDeviceSynchronize();
            break;
        }
        case 4: {
            // printf("uvm new representation\n");
            size_t n_pages = new_size*sizeof(unsigned int)/(4*1024)+1;
            kernel_coalesce_new_repr<<<(n_pages*32)/512 + 1, 512>>>
            (level, n_pages, G.numVertices, new_size, d_distance, d_new_offset, d_new_vertexList, uvm_adjacencyList,
            d_changed);
            ret1 = cudaDeviceSynchronize();
            break;
        }
        case 5:{
            // printf("uvm transfer edge list\n");
            // simpleBfs_csr<<<G.numVertices / 1024 + 1, 1024>>>
            // (G.numVertices, level, uvm_adjacencyList, d_edgesOffset, d_distance, d_changed); 
            numthreads = BLOCK_SIZE;
            numblocks = ((G.numVertices * (WARP_SIZE / CHUNK_SIZE) + numthreads) / numthreads);
            dim3 blockDim(BLOCK_SIZE, (numblocks+BLOCK_SIZE)/BLOCK_SIZE);
            bfs_kernel_coalesce_chunk<<<blockDim, numthreads>>>
            (d_distance, level, G.numVertices, d_edgesOffset, uvm_adjacencyList, d_changed);
            ret1 = cudaDeviceSynchronize();
            break;
        }
        case 6:{
            // printf("uvm transfer\n");
            size_t n_pages = G.numVertices*sizeof(unsigned int)/(64*1024);
            numthreads = 1024;
            numblocks = ((new_size * (WARP_SIZE / CHUNK_SIZE) + numthreads) / numthreads);
            dim3 blockDim(numthreads, (numblocks+numthreads)/numthreads);
            simpleBfs_csr<<<blockDim, numthreads /*(G.numVertices / numthreads)*(1 << 2)/4 + 1, numthreads*/>>>
            (G.numVertices, level, /*d_adjacencyList*/uvm_adjacencyList, d_edgesOffset, d_distance, d_changed); 

            // simpleBfs_csr_warp_opt<<< (n_pages*32)/1024 + 1, 1024>>>
            //         (n_pages, G.numVertices, level, uvm_adjacencyList, d_edgesOffset, d_edgesSize, d_distance, d_changed); 
            ret1 = cudaDeviceSynchronize();
            break;
        }
        case 7:{
            numthreads = 512;
            check_edgeList<<< (G.numEdges/numthreads + 1), numthreads>>>(rdma_adjacencyList, uvm_adjacencyList, G.numEdges);
            ret1 = cudaDeviceSynchronize();
            correct_retires<<<1, 1>>>();
        }
        default:
            break;
        }
        if(u_case == 1 || u_case == 2){
            print_retires<<<1,1>>>();
            ret1 = cudaDeviceSynchronize();
        }
        
        printf("ret1: %d cudaGetLastError(): %d level: %d\n", ret1, cudaGetLastError(), level);
        auto end = std::chrono::steady_clock::now();
        long duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        printf("Elapsed time in milliseconds : %li ms.\n\n", duration);
        
        checkError(cudaMemcpy(&changed_h, d_changed, sizeof(unsigned int), cudaMemcpyDeviceToHost));
        level++;
    }
    cudaEventRecord(event2, (cudaStream_t) 1);
    printf("cudaGetLastError(): %d level: %d\n", cudaGetLastError(), level);
        ret1 = cudaDeviceSynchronize();
        printf("cudaDeviceSynchronize: %d *changed: %d\n", ret1, changed_h);  
        if(cudaSuccess != ret1){  
            printf("cudaDeviceSynchronize error: %d\n", ret1);  
            exit(-1);
        }


    
    cudaEventSynchronize(event1); //optional
    cudaEventSynchronize(event2); //wait for the event to be executed!
    

    // calculate time
    float dt_ms;
    cudaEventElapsedTime(&dt_ms, event1, event2);
    printf("Elapsed time with cudaEvent: %f startVertex: %d\n", dt_ms, startVertex);
    time_total += dt_ms;

    auto end = std::chrono::steady_clock::now();
    long duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    printf("Elapsed time in milliseconds : %li ms.\n", duration);

    checkError(cudaMemcpy(return_distance, d_distance, G.numVertices*sizeof(unsigned int), cudaMemcpyDeviceToHost));

    if(u_case == 2 || u_case == 3 || u_case == 4){
        checkError(cudaFree(d_new_offset));
        checkError(cudaFree(d_new_vertexList));
    }

    if(u_case == 0 || u_case == 3){ // direct transfer
        checkError(cudaFree(d_adjacencyList));
    }
    if(u_case == 4 || u_case == 5){
        cudaFree(uvm_adjacencyList);
    }

    checkError(cudaFree(d_distance));
    checkError(cudaFree(d_changed));
    checkError(cudaFree(d_edgesOffset));
    checkError(cudaFree(d_edgesSize));
    checkError(cudaFree(uvm_adjacencyList));

    uvm_adjacencyList = NULL;

    return return_distance;
    // finalizeCudaBfs(distance, parent, G);
}

// Main program
int main(int argc, char **argv)
{   
    if (argc != 9)
        usage(argv[0]);

    cudaSetDevice(0);

    printf("Function: %s line number: %d 1024MB: %d bytes REQUEST_SIZE: %d\n",__func__, __LINE__, MB(1024), REQUEST_SIZE);
    
    // Get the process ID
    pid_t pid = getpid();
    
    // Print the process ID
    printf("The process ID is: %d\n", pid);

    // read graph from standard input
    Graph G;
    Graph_m G_m;
    unsigned int *tmp_edgesOffset, *tmp_edgesSize, *tmp_adjacencyList;
    
    unsigned int startVertex = atoi(argv[7]);
    // printf("function: %s line: %d u_edgesOffset->local_buffer: %p\n", __FILE__, __LINE__, u_edgesOffset->local_buffer);

    // readGraph(G, argc, argv);
    readfile(G, G_m, argc, argv, tmp_edgesOffset, tmp_edgesSize, tmp_adjacencyList);
    printf("reading done\n");
    initCuda(G);
    printf("initCuda done\n");

    for(size_t i = 0; i < G.numEdges; i++){
        // uvm_adjacencyList[i] = G.adjacencyList_r[i];
        u_adjacencyList[i] = G.adjacencyList_r[i];
    }
    for(size_t i = 0; i < G.numVertices+1; i++){
        u_edgesOffset[i] = G.edgesOffset_r[i];
    }
    printf("copying Gparh structure done, G.numVertices: %llu G.numEdges: %llu\n", G.numVertices, G.numEdges);
    // ------------------- new representation for rdma only: -----------------//
    auto start = std::chrono::steady_clock::now();                
    size_t new_size = 0, treshold = 128;
    for (size_t i = 0; i < G.numVertices; i++)
    {
        uint64_t degree = u_edgesOffset[i+1] - u_edgesOffset[i];
        
        if(degree <= treshold){
            new_size++;
        }
        else{
            size_t count = degree/treshold + 1;
            new_size += count;
        }
    }
    printf("new representation being created new_size: %llu\n", new_size);
    unsigned int *new_vertex_list;
    uint64_t *new_offset;
    size_t index_zero = 0;
    new_vertex_list = new uint[new_size];
    new_offset = new uint64_t[new_size+1];
    new_offset[0] = 0;
    for (size_t i = 0; i < G.numVertices; i++)
    {
        uint64_t degree = u_edgesOffset[i+1] - u_edgesOffset[i];
        
        if(degree <= treshold){
            new_vertex_list[index_zero] = i;
            new_offset[index_zero+1] = u_edgesOffset[i+1];
            index_zero++;
        }
        else{
            size_t count = degree/treshold + 1;
            size_t total = degree;
            for (size_t k = 0; k < count; k++)
            {
                new_vertex_list[index_zero] = i;
                if(total > treshold) new_offset[index_zero+1] = new_offset[index_zero] + treshold;
                else new_offset[index_zero+1] = u_edgesOffset[i+1];
                index_zero++;
                total = total - treshold;
            }
        }
    }
    auto end = std::chrono::steady_clock::now();
    long duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    printf("Elapsed time for preprocessing in milliseconds : %li ms.\n\n", duration);
    /***********************************************************/

    uint *direct_distance;
    direct_distance  = runRDMA(startVertex, G, false, new_vertex_list, new_offset, new_size,
             u_adjacencyList, u_edgesOffset, 6);

    int number_of_vertices = 0;
    int active_vertices = 0;
    for (size_t i = 0; i < number_of_vertices; i++)
    {
        startVertex = i;
        printf("vertex %d has degree of %d\n", startVertex, u_edgesOffset[i+1] - u_edgesOffset[i]);
        if(u_edgesOffset[i+1] - u_edgesOffset[i] == 0)
            continue;
        active_vertices++;
        direct_distance  = runRDMA(startVertex, G, false, new_vertex_list, new_offset, new_size,
             u_adjacencyList, u_edgesOffset, 6);

        printf("average time: %.2f pinning time: %.2f active_vertices: %d\n", time_total/active_vertices, time_readmostly_total/active_vertices, active_vertices);
    }
    printf("average time: %.2f pinning time: %.2f\n", time_total/active_vertices, time_readmostly_total/active_vertices);

    bool rdma_flag = false;
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

        s_ctx->gpu_buf_size = 20*1024*1024*1024llu; // N*sizeof(int)*3llu;
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

    printf("Number of vertices %lld tmp_edgesOffset[10]: %d\n", G.numVertices, G.edgesOffset_r[10]);

    free(G.adjacencyList_r);
    free(G.edgesOffset_r);
    printf("Number of edges %lld\n\n", G.numEdges);

    std::vector<int> distance(G.numVertices, 100000 /*std::numeric_limits<int>::max()*/);
    std::vector<int> parent(G.numVertices, std::numeric_limits<int>::max());
    std::vector<bool> visited(G.numVertices, false);
   
    printf("function: %s line: %d\n", __FILE__, __LINE__);

    if(rdma_flag){

        rdma_adjacencyList->start(G.numEdges *sizeof(unsigned int), GPU, NULL);

        for(size_t i = 0; i < G.numEdges; i++){
            rdma_adjacencyList->local_buffer[i] = u_adjacencyList[i];
        }
    }

    printf("function: %s line: %d\n", __FILE__, __LINE__);
    
    uint64_t max = u_edgesOffset[1] - u_edgesOffset[0], max_node;
    double avg = 0;

    for (size_t i = 0; i < G.numVertices; i++)
    {
        uint64_t degree = u_edgesOffset[i+1] - u_edgesOffset[i];
        avg += degree;
        if(max < degree) {
            max = degree;
            max_node = i;
        }
    }

    avg = avg / G.numVertices;
    printf("avg: %f max: %llu, node: %llu\n", avg, max, max_node);

    
    printf("function: %s line: %d\n", __FILE__, __LINE__);

    std::vector<int> expectedDistance(distance);
    std::vector<int> expectedParent(parent);
    start = std::chrono::steady_clock::now();
    
    int u_case = 7;
    uint *rdma_distance; //, *direct_distance;
    
    rdma_distance = runRDMA(startVertex, G, rdma_flag, new_vertex_list, new_offset, new_size,
                            u_adjacencyList, u_edgesOffset, u_case);
    
    rdma_adjacencyList->reset();

    number_of_vertices = 0;
    active_vertices = 0;
    time_total = 0;
    if(rdma_flag){
        // rdma_distance = runRDMA(startVertex, G, rdma_flag, new_vertex_list, new_offset, new_size,
        //         u_adjacencyList, u_edgesOffset, u_case);
        // cudaFree(s_ctx->gpu_buffer);

        for (size_t i = 0; i < number_of_vertices; i++)
        {
            startVertex = i;
            printf("vertex %d has degree of %d\n", startVertex, u_edgesOffset[i+1] - u_edgesOffset[i]);
            if(u_edgesOffset[i+1] - u_edgesOffset[i] == 0)
                continue;
            active_vertices++;
            rdma_distance = runRDMA(startVertex, G, rdma_flag, new_vertex_list, new_offset, new_size,
                u_adjacencyList, u_edgesOffset, u_case);
        
            rdma_adjacencyList->reset();

            printf("average time: %.2f active_vertices: %d\n", time_total/active_vertices, active_vertices);
        }
        
        printf("average time: %.2f\n", time_total/active_vertices);
        

        cudaFree(s_ctx->gpu_buffer);
    }

    printf("file: %s\n", argv[8]);

    // direct_distance = runRDMA(startVertex, G, rdma_flag, new_vertex_list, new_offset, new_size,
    //          u_adjacencyList, u_edgesOffset, 3);

    // direct_distance = runCudaSimpleBfs_emogi(startVertex, G);


    end = std::chrono::steady_clock::now();
    // u_distance->memcpyDtoH();
    for(size_t i = 0; i < 5; i++){
        printf("u_distance->size: %llu (*u_distance)[%d]: %d\n", u_distance, i, u_distance[i]);
        // printf("u_distance->size: %llu (*u_distance)[%d]: %d\n", u_distance->size, i, u_distance->local_buffer[i]);
        // printf("u_edgesOffset->size: %llu (*u_edgesOffset)[%d]: %llu\n", u_edgesOffset->size, i, u_edgesOffset->local_buffer[i]);
        // printf("u_edgesSize->size: %llu (*u_edgesSize)[%d]: %llu G.edgesSize_r[%d]: %llu\n", u_edgesSize->size, i, u_edgesSize->local_buffer[i], i, G.edgesSize_r[i]);
    }
    
    checkOutput_rdma(direct_distance, rdma_distance, G);
    // checkOutput_rdma(u_distance, expectedDistance, G);

    print_utilization<<<1,1>>>();
    cudaError_t ret2 = cudaDeviceSynchronize();
    if(cudaSuccess != ret1){  
        printf("cudaDeviceSynchronize error: %d\n", ret1);  
        exit(-1);
    }

    print_transferTime<<<1,1>>>();
    ret2 = cudaDeviceSynchronize();
    if(cudaSuccess != ret1){  
        printf("cudaDeviceSynchronize error: %d\n", ret1);  
        exit(-1);
    }

    // // //run CUDA queue parallel bfs
    // runCudaQueueBfs(startVertex, G, distance, parent);
    // checkOutput(distance, expectedDistance, G);

    // // //run CUDA scan parallel bfs
    // runCudaScanBfs(startVertex, G, distance, parent);
    // checkOutput(distance, expectedDistance, G);
    // finalizeCuda();
    
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    printf("Overall Elapsed time in milliseconds : %li ms. startVertex: %d\n", duration, startVertex);
    // printf("oversubscription: %d\n", oversubs_ratio_macro-1);

    

    return 0;

	return 0;
}

// __global__
// void simpleBfs_rdma(size_t n, unsigned int *level, rdma_buf<unsigned int> *d_adjacencyList, rdma_buf<unsigned int> *d_edgesOffset,
//                rdma_buf<unsigned int> *d_edgesSize, rdma_buf<unsigned int> *d_distance, rdma_buf<unsigned int> *d_parent, unsigned int *changed) {
//     size_t thid = blockIdx.x * blockDim.x + threadIdx.x;
//     int valueChange = 0;
//     if(thid < n /*d_distance->size/sizeof(uint)*/){
//         unsigned int k = (*d_distance)[thid];
        
//         if (/*thid < n && */k == *level) {
//             unsigned int u = thid;
//             for (unsigned int i = (*d_edgesOffset)[u]; i < (*d_edgesOffset)[u] + (*d_edgesSize)[u]; i++) {
                
//                 int v = (*d_adjacencyList)[i];
//                 unsigned int dist = (*d_distance)[v];
//                 if (*level + 1 < dist) {
                    
//                     unsigned int new_dist = *level + 1;
                   
//                     (*d_distance).rvalue(v, new_dist /*(int) level + 1*/);
                   
//                     valueChange = 1;
//                 }
//             }
//             // printf(" for finished\n");
//         }
//         // __syncthreads();
//         if (valueChange) {
//             *changed = valueChange;
//         }
//     }
//     // __syncthreads();
// }

// __global__
// void simpleBfs_rdma(size_t n, unsigned int level, rdma_buf<unsigned int> *d_adjacencyList, rdma_buf<unsigned int> *d_edgesOffset,
//                rdma_buf<unsigned int> *d_edgesSize, unsigned int *d_distance, unsigned int *changed)

__global__ void bfs_kernel_coalesce_chunk(unsigned int *label, unsigned int level, const uint64_t vertex_count, uint64_t *vertexList, unsigned int *edgeList, unsigned int *changed) {
    const uint64_t tid = blockDim.x * BLOCK_SIZE * blockIdx.y + blockDim.x * blockIdx.x + threadIdx.x;
    const uint64_t warpIdx = tid >> WARP_SHIFT;
    const uint64_t laneIdx = tid & ((1 << WARP_SHIFT) - 1);
    const uint64_t chunkIdx = warpIdx * CHUNK_SIZE;
    uint64_t chunk_size = CHUNK_SIZE;

    if((chunkIdx + CHUNK_SIZE) > vertex_count) {
        if ( vertex_count > chunkIdx )
            chunk_size = vertex_count - chunkIdx;
        else
            return;
    }

    for(uint32_t i = chunkIdx; i < chunk_size + chunkIdx; i++) {
        if(label[i] == level) {
            const uint64_t start = vertexList[i];
            const uint64_t shift_start = start & MEM_ALIGN;
            const uint64_t end = vertexList[i+1];

            for(uint64_t j = shift_start + laneIdx; j < end; j += WARP_SIZE) {
                if (j >= start) {
                    const uint next = edgeList[j];

                    // if(label[next] == MYINFINITY) {
                    if(label[next] > level + 1) {
                        label[next] = level + 1;
                        *changed = 1;
                    }
                }
            }
        }
    }
}

__global__
void simpleBfs_csr(size_t n, unsigned int level, unsigned int *d_adjacencyList,
                    uint64_t *vertex_list, unsigned int *d_distance, unsigned int *changed) {

    // size_t thid = blockIdx.x * blockDim.x + threadIdx.x;
    // const uint64_t laneIdx = thid & ((1 << 2) - 1);

    // // Warp ID within the block
    // size_t warpId = thid / (1 << 2);
    
    // if(warpId * 4 < n){ // Each warp now processes two vertices
        
    //     for (int vertexOffset = 0; vertexOffset < 4; vertexOffset++) {
    //         size_t currentVertex = warpId * 4 + vertexOffset;
            
    //         if (currentVertex < n) {
    //             size_t k = d_distance[currentVertex];
                
    //             if (k == level) {
    //                 for (size_t i = vertex_list[currentVertex] + laneIdx; 
    //                      i < vertex_list[currentVertex + 1]; 
    //                      i += (1 << 2)) {
                             
    //                     unsigned int v = d_adjacencyList[i];
    //                     uint64_t dist = d_distance[v];
                        
    //                     if (level + 1 < dist) {
    //                         d_distance[v] = level + 1;
    //                         *changed = 1;
    //                     }
    //                 }
    //             }
    //         }
    //     }
    // }

    const uint64_t tid = blockDim.x * BLOCK_SIZE * blockIdx.y + blockDim.x * blockIdx.x + threadIdx.x;
    const uint64_t warpIdx = tid >> WARP_SHIFT;
    const uint64_t laneIdx = tid & ((1 << WARP_SHIFT) - 1);
    const uint64_t chunkIdx = warpIdx * CHUNK_SIZE;
    uint64_t chunk_size = CHUNK_SIZE;

    if((chunkIdx + CHUNK_SIZE) > n) {
        if ( n > chunkIdx )
            chunk_size = n - chunkIdx;
        else
            return;
    }

    for(uint32_t i = chunkIdx; i < chunk_size + chunkIdx; i++) {
        if(d_distance[i] == level) {
            const uint64_t start = vertex_list[i];
            const uint64_t shift_start = start & MEM_ALIGN;
            const uint64_t end = vertex_list[i+1];

            for(uint64_t j = shift_start + laneIdx; j < end; j += WARP_SIZE) {
                if (j >= start) {
                    const uint next = d_adjacencyList[j];

                    // if(label[next] == MYINFINITY) {
                    if(d_distance[next] > level + 1) {
                        d_distance[next] = level + 1;
                        *changed = 1;
                    }
                }
            }
        }
    }
}

// __global__
// void simpleBfs_csr(size_t n, unsigned int level, unsigned int *d_adjacencyList,
//                     unsigned int *vertex_list, unsigned int *d_distance, unsigned int *changed) {

//     size_t thid = blockIdx.x * blockDim.x + threadIdx.x;
//     const uint64_t laneIdx = thid & ((1 << WARP_SHIFT) - 1);

//     // // Warp ID within the block
//     size_t warpId = thid / (1 << WARP_SHIFT);
    
//     // size_t iterations = 0;
//     if(warpId < n /*d_distance->size/sizeof(uint)*/){
//         unsigned int k = d_distance[warpId];
//         if (k == level) {
           
//             for (size_t i = vertex_list[warpId] + laneIdx; i < vertex_list[warpId+1]; i += (1 << WARP_SHIFT)) {
//                 unsigned int v;
               
//                 v = d_adjacencyList[i];
//                 unsigned int dist = d_distance[v];
//                 if (level + 1 < dist) {
                    
//                     d_distance[v] = level + 1; /*(int) level + 1*/
//                     *changed = 1;
//                     // valueChange = 1;
//                 }
//             }
//         }
//     }
// }

__global__
void simpleBfs_csr_warp_opt(size_t n, size_t numVertices, unsigned int level, unsigned int *d_adjacencyList, unsigned int *d_edgesOffset,
                    unsigned int *d_edgesSize, unsigned int *d_distance, unsigned int *changed) {
    // Page size in elements (64KB / 4 bytes per unsigned int)
    const size_t pageSize = 64*1024 / sizeof(unsigned int);
    // Elements per warp
    const size_t elementsPerWarp = pageSize / warpSize;

    // Global thread ID
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;

    // Warp ID within the block
    size_t warpId = tid / warpSize;

    // Thread lane within the warp
    size_t lane = threadIdx.x % warpSize;

    // Determine which page this warp will process
    size_t pageStart = warpId * pageSize;

    // Ensure we don't process out-of-bounds pages
    if (pageStart < n * pageSize) {
        bool localChanged = false;
        
        // Process elements within the page
        for (size_t i = 0; i < elementsPerWarp; ++i) {
            size_t elementIdx = pageStart + lane + i*warpSize;
            if (elementIdx < numVertices) {
                unsigned int k = d_distance[elementIdx];
                if (k == level) {
                    // printf("d_edgesOffset[%llu]: %u, d_distance[%llu]: %u\n", 
                    //         (long long int) elementIdx, d_edgesOffset[elementIdx], (long long int) elementIdx, d_distance[elementIdx]);
                    for(size_t j = d_edgesOffset[elementIdx]; j < d_edgesOffset[elementIdx+1] /*+ d_edgesSize[elementIdx]*/; j += 1) {
                        int v = d_adjacencyList[j];
                        unsigned int dist = d_distance[v];

                        if (level + 1 < dist) {
                            d_distance[v] = level + 1;
                            *changed = 1;
                        }
                    }
                }
            }
        }
    }
}

__global__
void simpleBfs_modVertexList_rdma(size_t n, unsigned int level, rdma_buf<unsigned int> *d_adjacencyList,
                    unsigned int *vertex_list, unsigned int *d_distance, unsigned int *changed) {

    // Thread index
    size_t thid = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int warpSize = 32;
    // Warp index and lane index
    size_t warpId = thid / warpSize;
    size_t laneId = thid % warpSize;
    // Warp size
    

    // Buffer for storing distances and change flag in shared memory
    unsigned int shared_distance;
    unsigned int warp_changed;

    if (warpId < n && level == d_distance[warpId]) {
        // Each warp processes one node
        // if (laneId == 0) {
            shared_distance = d_distance[warpId];
            warp_changed = 0;
        // }

        // __syncwarp(); // Synchronize within warp
        if (shared_distance == level) {
            unsigned int nodeStart = vertex_list[warpId];
            unsigned int nodeEnd = vertex_list[warpId + 1];

            for (size_t i = nodeStart + laneId; i < nodeEnd; i += warpSize) {
                unsigned int v = (*d_adjacencyList)[i];
                unsigned int dist = d_distance[v];
                
                if (level + 1 < dist) {
                    d_distance[v] = level + 1;
                    warp_changed = 1; // Mark distance change
                }
            }
        }

        // __syncwarp(); // Synchronize within warp

        // Use the first thread in the warp to set the changed flag
        if (warp_changed) {
            *changed = 1;
            // atomicExch(changed, 1);
        }
    }
}


__global__
void simpleBfs_normal_rdma(size_t n, size_t vertexCount, unsigned int level, rdma_buf<unsigned int> *d_edgeList, unsigned int *d_vertex_list,
                      unsigned int *d_distance, unsigned int *changed) {
    size_t thid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    int valueChange = 0;
    // size_t iterations = 0;
    // if(thid < n /*d_distance->size/sizeof(uint)*/){
        
            for (size_t i = thid; i < n; i += /*vertexCount*/ stride) {
                unsigned int v;
                unsigned int k = d_distance[d_vertex_list[i]];
                
                if(k == level){
                    // v = d_edgeList[i];
                    v = (*d_edgeList)[i];
                    unsigned int dist = d_distance[v];
                    if (level + 1 < dist) {
                        
                            d_distance[v] = level + 1; /*(int) level + 1*/
                        
                        valueChange = 1;
                    }
                }
                // iterations++;
            }
            
        // }
        
        

        // __syncthreads();
        if (valueChange) {
            *changed = valueChange;
        }
    // }
}

// __global__
// void simpleBfs_rdma(size_t n, unsigned int level, rdma_buf<unsigned int> *d_adjacencyList, unsigned int *d_edgesOffset,
//                     unsigned int *d_edgesSize, unsigned int *d_distance, unsigned int *changed) {
//     size_t thid = blockIdx.x * blockDim.x + threadIdx.x;
//     int valueChange = 0;

//     // Each warp works on 4KB of data
//     size_t warpSize = 32;
//     size_t warpId = thid / warpSize;
//     size_t laneId = thid % warpSize;

//     // Calculate the start and end of the 4KB region for this warp
//     size_t regionSize = 4*1024 / sizeof(unsigned int); // 4KB region size in terms of number of elements
//     size_t startIdx = warpId * regionSize;
//     size_t endIdx = (startIdx + regionSize) < n ? (startIdx + regionSize) : n;
//     // min(startIdx + regionSize, n);

//     for (size_t idx = startIdx + laneId; idx < endIdx; idx += warpSize) {
//         if (idx < n) {
//             unsigned int k = d_distance[idx];
//             if (k == level) {
//                 for (size_t i = d_edgesOffset[idx]; i < d_edgesOffset[idx + 1]; i++) {
//                     unsigned int v;
//                     v = (*d_adjacencyList)[i];
//                     unsigned int dist = d_distance[v];
//                     if (level + 1 < dist) {
//                         d_distance[v] = level + 1;
//                         valueChange = 1;
//                     }
//                 }
//             }
//         }
//     }

//     if (__syncthreads_or(valueChange)) {
//         *changed = 1;
//     }
// }


__global__
void simpleBfs_normal(size_t n, size_t vertexCount, unsigned int level, unsigned int *d_edgeList, unsigned int *d_vertex_list,
                      unsigned int *d_distance, unsigned int *changed) {
    size_t thid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    int valueChange = 0;
    // size_t iterations = 0;
    if(thid < n /*d_distance->size/sizeof(uint)*/){
        
            for (size_t i = thid; i < n; i += stride) {
                
                unsigned int v;
                unsigned int k = d_distance[d_vertex_list[i]];
                if(k == level){
                    v = d_edgeList[i];
                    
                    unsigned int dist = d_distance[v];
                    if (level + 1 < dist) {
                        
                            d_distance[v] = level + 1; /*(int) level + 1*/
                        
                        valueChange = 1;
                    }
                }
                // iterations++;
            }
        
        if (valueChange) {
            *changed = valueChange;
        }
    }
}

__global__ __launch_bounds__(1024,2)
void simpleBfs_rdma(size_t n, unsigned int level, rdma_buf<unsigned int> *d_adjacencyList, uint64_t *d_edgesOffset,
                    unsigned int *d_edgesSize, unsigned int *d_distance, unsigned int *changed) {

    // size_t thid = blockIdx.x * blockDim.x + threadIdx.x;
    // const uint64_t laneIdx = thid & ((1 << WARP_SHIFT) - 1);

    // // Warp ID within the block
    // size_t warpId = thid / (1 << WARP_SHIFT);
    
    // if(warpId * 4 < n){ // Each warp now processes two vertices
        
    //     for (int vertexOffset = 0; vertexOffset < 4; vertexOffset++) {
    //         unsigned int currentVertex = warpId * 4 + vertexOffset;
            
    //         if (currentVertex < n) {
    //             unsigned int k = d_distance[currentVertex];
                
    //             if (k == level) {
    //                 for (size_t i = d_edgesOffset[currentVertex] + laneIdx; 
    //                      i < d_edgesOffset[currentVertex + 1]; 
    //                      i += (1 << WARP_SHIFT)) {
                             
    //                     unsigned int v = (*d_adjacencyList)[i];
    //                     unsigned int dist = d_distance[v];
                        
    //                     if (level + 1 < dist) {
    //                         d_distance[v] = level + 1;
    //                         *changed = 1;
    //                     }
    //                 }
    //             }
    //         }
    //     }
    // }


    // size_t thid = blockIdx.x * blockDim.x + threadIdx.x;
    // const uint64_t laneIdx = thid & ((1 << WARP_SHIFT) - 1);

    // // // Warp ID within the block
    // size_t warpId = thid / (1 << WARP_SHIFT);
    
    // // size_t iterations = 0;
    // if(warpId < n /*d_distance->size/sizeof(uint)*/){
    //     unsigned int k = d_distance[warpId];
    //     if (k == level) {
    //         for (size_t i = d_edgesOffset[warpId] + laneIdx; i < d_edgesOffset[warpId+1]; i += (1 << WARP_SHIFT)) {
    //             unsigned int v;
               
    //             v = (*d_adjacencyList)[i];
    //             unsigned int dist = d_distance[v];
    //             if (level + 1 < dist) {
                    
    //                 d_distance[v] = level + 1; /*(int) level + 1*/
    //                 *changed = 1;
    //                 // valueChange = 1;
    //             }
    //         }
    //     }
    // }


    const uint64_t tid = blockDim.x * 512 * blockIdx.y + blockDim.x * blockIdx.x + threadIdx.x;
    const uint64_t warpIdx = tid >> WARP_SHIFT;
    const uint64_t laneIdx = tid & ((1 << WARP_SHIFT) - 1);
    const uint64_t chunkIdx = warpIdx * CHUNK_SIZE;
    uint64_t chunk_size = CHUNK_SIZE;

    if((chunkIdx + CHUNK_SIZE) > n) {
        if ( n > chunkIdx )
            chunk_size = n - chunkIdx;
        else
            return;
    }

    for(uint32_t i = chunkIdx; i < chunk_size + chunkIdx; i++) {
        if(d_distance[i] == level) {
            const uint64_t start = d_edgesOffset[i];
            const uint64_t shift_start = start & MEM_ALIGN;
            const uint64_t end = d_edgesOffset[i+1];

            for(uint64_t j = shift_start + laneIdx; j < end; j += WARP_SIZE) {
                if (j >= start) {
                    const uint next = (*d_adjacencyList)[j];

                    // if(label[next] == MYINFINITY) {
                    if(d_distance[next] > level + 1) {
                        d_distance[next] = level + 1;
                        *changed = 1;
                    }
                }
            }
        }
    }
}

__global__
void simpleBfs_uvm(size_t n, unsigned int level, unsigned int *d_adjacencyList, unsigned int *d_edgesOffset,
                    unsigned int *d_edgesSize, unsigned int *d_distance, unsigned int *changed) {
    size_t thid = blockIdx.x * blockDim.x + threadIdx.x;
    int valueChange = 0;
    if(thid < n /*d_distance->size/sizeof(uint)*/){
        unsigned int k = d_distance[thid];
        if (k == level) {
            // uint edgesOffset = d_edgesOffset[thid];
            // uint edgesSize = d_edgesSize[thid];
            for (size_t i = d_edgesOffset[thid]; i < d_edgesOffset[thid] + d_edgesSize[thid]; i += 1) {
                // double time1 = clock();
                // unsigned int *tmp = (unsigned int *) d_adjacencyList.d_TLB[i/1024].device_address;
                // unsigned int v = tmp[i%1024]; // (*d_adjacencyList)[i];
                unsigned int v = d_adjacencyList[i];
                // int v = d_adjacencyList[i];
                // double time2 = clock();
                // printf(" %f ", time2 - time1);
                unsigned int dist = d_distance[v];
                if (level + 1 < dist) {
                    
                        d_distance[v] = level + 1; /*(int) level + 1*/
                    
                    valueChange = 1;
                }
            }
        }
        // __syncthreads();
        if (valueChange) {
            *changed = valueChange;
        }
    }
}


__global__ void sssp_GPU_Kernel(int numEdges,
                                int numEdgesPerThread,
                                uint *dist,
                                uint *preNode,
                                uint *edgesSource,
                                uint *edgesEnd,
                                uint *edgesWeight,
                                bool *finished) {
    int threadId = blockDim.x * blockIdx.x + threadIdx.x;
    int startId = threadId * numEdgesPerThread;
    // if(threadId == 0) printf("hello from sssp\n"); 
    if (startId >= numEdges) {
        return;
    }
    
    int endId = (threadId + 1) * numEdgesPerThread;
    if (endId >= numEdges) {
        endId = numEdges;
    }

    for (int nodeId = startId; nodeId < endId; nodeId++) {
        uint source = edgesSource[nodeId];
        uint end = edgesEnd[nodeId]; // edgelist
        uint weight = 1; // edgesWeight[nodeId];
        
        if (dist[source] + weight < dist[end]) {
            atomicMin(&dist[end], dist[source] + weight);
            // dist[end] = dist[source] + weight;
            preNode[end] = source;
            *finished = false;
        }
    }
}

__global__
void simpleBfs_normal_rdma_optimized(int numEdgesPerThread, size_t numEdges, size_t vertexCount, unsigned int level, rdma_buf<unsigned int> *d_edgeList, unsigned int *d_vertex_list,
                      unsigned int *d_distance, unsigned int *changed) {
    // size_t thid = blockIdx.x * blockDim.x + threadIdx.x;
    // int stride = blockDim.x * gridDim.x;
    // int valueChange = 0;
    
    
    int threadId = blockDim.x * blockIdx.x + threadIdx.x;
    int startId = threadId * numEdgesPerThread;
    // if(threadId == 0) printf("hello from sssp\n"); 
    if (startId >= numEdges) {
        return;
    }
    
    int endId = (threadId + 1) * numEdgesPerThread;
    if (endId >= numEdges) {
        endId = numEdges;
    }

    for (int nodeId = startId; nodeId < endId; nodeId++) {

        unsigned int v;
        unsigned int k = d_distance[d_vertex_list[nodeId]];

        if(k == level){
            // v = d_edgeList[i];
            v = D_adjacencyList[nodeId];
            unsigned int dist = d_distance[v];
            if (level + 1 < dist) {
                
                d_distance[v] = level + 1; /*(int) level + 1*/
                *changed = 1;
                // valueChange = 1;
            }
        }

       
    }

    // // __syncthreads();
    // if (valueChange) {
    //     *changed = valueChange;
    // }
    // // }
}


__global__ __launch_bounds__(128,16)
void kernel_coalesce_ptr_pc(unsigned int *ptr, unsigned int *label, const uint32_t level, const uint64_t vertex_count, unsigned int *edgeOffset,
                            unsigned int * edgeSize, unsigned *edgeList, unsigned int *changed) {
    const uint64_t tid = blockDim.x * BLOCK_NUM * blockIdx.y + blockDim.x * blockIdx.x + threadIdx.x;
    const uint64_t warpIdx = tid >> WARP_SHIFT;
    const uint64_t laneIdx = tid & ((1 << WARP_SHIFT) - 1);
    //array_d_t<uint64_t> d_array = *da;
    if(warpIdx < vertex_count && label[warpIdx] == level) {
        // bam_ptr<uint64_t> ptr(da);
        const uint64_t start = edgeOffset[warpIdx];
        const uint64_t shift_start = start & 0xFFFFFFFFFFFFFFF0;
        const uint64_t end = edgeOffset[warpIdx] + edgeSize[warpIdx];

        for(uint64_t i = shift_start + laneIdx; i < end; i += WARP_SIZE) {
            if (i >= start) {
                //const EdgeT next = edgeList[i];
                //EdgeT next = da->seq_read(i);
                unsigned int next = ptr[i];
//                printf("tid: %llu, idx: %llu next: %llu\n", (unsigned long long) tid, (unsigned long long) i, (unsigned long long) next);
                unsigned int dist = label[next];
                if(/*label[next] == MYINFINITY*/level + 1 < dist) {
                //    if(level ==0)
                //            printf("tid:%llu, level:%llu, next: %llu\n", tid, (unsigned long long)level, (unsigned long long)next);
                    label[next] = level + 1;
                    *changed = true;
                }
            }
        }
    }
}


__global__ __launch_bounds__(128,32)
void kernel_coalesce_ptr_pc_rdma(rdma_buf<unsigned int> *ptr, unsigned int *label, const uint32_t level, const uint64_t vertex_count, unsigned int *edgeOffset,
                            unsigned int * edgeSize, unsigned int *changed) {
    const uint64_t tid = blockDim.x * BLOCK_NUM * blockIdx.y + blockDim.x * blockIdx.x + threadIdx.x;
    const uint64_t warpIdx = tid >> WARP_SHIFT;
    const uint64_t laneIdx = tid & ((1 << WARP_SHIFT) - 1);
    //array_d_t<uint64_t> d_array = *da;
    if(warpIdx < vertex_count && label[warpIdx] == level) {
        // bam_ptr<uint64_t> ptr(da);
        const uint64_t start = edgeOffset[warpIdx];
        const uint64_t shift_start = start & 0xFFFFFFFFFFFFFFF0;
        const uint64_t end = edgeOffset[warpIdx] + edgeSize[warpIdx];

        for(uint64_t i = shift_start + laneIdx; i < end; i += WARP_SIZE) {
            if (i >= start) {
                unsigned int next = (*ptr)[i];                
                unsigned int dist = label[next];
                if(level + 1 < dist) {
                    label[next] = level + 1;
                    *changed = true;
                }
            }
        }
    }
}

// normal kernel - with optimized warp access; one warp working on one page only
__global__
void simpleBfs_normal_rdma_optimized(size_t n, size_t vertexCount, unsigned int level, rdma_buf<unsigned int> *d_edgeList, unsigned int *d_vertex_list,
                      unsigned int *d_distance, unsigned int *changed) {
    size_t thid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    int valueChange = 0;
        
            for (size_t i = thid; i < n; i += stride) {
                unsigned int v;
                unsigned int k = d_distance[d_vertex_list[i]];
                
                if(k == level){
                    // v = d_edgeList[i];
                    // v = D_adjacencyList[i];
                    v = (*d_edgeList)[i];
                    unsigned int dist = d_distance[v];
                    if (level + 1 < dist) {
                        
                            d_distance[v] = level + 1; /*(int) level + 1*/
                        
                        valueChange = 1;
                    }
                }
            }
            
        if (valueChange) {
            *changed = valueChange;
        }
}

// __global__
// void simpleBfs_normal_rdma_optimized2(size_t n, size_t vertexCount, unsigned int level, rdma_buf<unsigned int> *d_edgeList, unsigned int *d_vertex_list,
//                                      unsigned int *d_distance, unsigned int *changed) {
//     const size_t page_size = 64 * 1024;  // 64KB in bytes
//     const size_t elements_per_page = page_size / sizeof(unsigned int);  // Elements per 64KB page
    
//     size_t thid = blockIdx.x * blockDim.x + threadIdx.x;
//     size_t stride = blockDim.x * gridDim.x;
//     size_t warp_id = thid / warpSize;
//     size_t warp_thread_id = thid % warpSize;
    
//     bool valueChanged = false; // Using a bool for local flag
    
//     for (size_t page_id = warp_id; page_id < n / elements_per_page; page_id += (stride / warpSize)) {
//         size_t start_idx = page_id * elements_per_page;
//         size_t end_idx = start_idx + elements_per_page < n ? start_idx + elements_per_page : n; // min(start_idx + elements_per_page, n);
        
//         for (size_t i = start_idx + warp_thread_id; i < end_idx; i += warpSize) {
//             unsigned int v;
//             unsigned int k = d_distance[d_vertex_list[i]];
            
//             if (k == level) {
//                 v = (*d_edgeList)[i];
//                 unsigned int dist = d_distance[v];
//                 if (level + 1 < dist) {
//                     d_distance[v] = level + 1;
//                     valueChanged = true;
//                 }
//             }
//         }
//     }

//     // Use atomic operation to set the changed flag
//     if (valueChanged) {
//         *changed = 1;
//         // atomicOr(changed, 1);
//     }
// }

__global__
void simpleBfs_normal_rdma_optimized2(size_t n, size_t vertexCount, unsigned int level, rdma_buf<unsigned int> *d_edgeList, unsigned int *d_vertex_list,
                                     unsigned int *d_distance, unsigned int *changed) {
    const size_t page_size = 64 * 1024;  // 64KB in bytes
    const size_t elements_per_page = page_size / sizeof(unsigned int);  // Elements per 64KB page
    
    size_t thid = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;
    size_t warp_id = thid / warpSize;
    size_t warp_thread_id = thid % warpSize;
    
    __shared__ bool shared_changed;
    if (threadIdx.x == 0) {
        shared_changed = false;
    }
    __syncthreads();

    for (size_t page_id = warp_id; page_id < (n + elements_per_page - 1) / elements_per_page; page_id += (stride / warpSize)) {
        size_t start_idx = page_id * elements_per_page;
        size_t end_idx = start_idx + elements_per_page < n ? start_idx + elements_per_page : n; // min(start_idx + elements_per_page, n);

        for (size_t i = start_idx + warp_thread_id; i < end_idx; i += warpSize) {
            unsigned int v;
            unsigned int k = d_distance[d_vertex_list[i]];
            
            if (k == level) {
                v = (*d_edgeList)[i];
                unsigned int dist = d_distance[v];
                if (level + 1 < dist) {
                    d_distance[v] = level + 1;
                    shared_changed = true;
                }
            }
        }
    }
    
    __syncthreads();
    if (threadIdx.x == 0 && shared_changed) {
        *changed = 1;
    }
}

__global__
void simpleBfs_rdma_optimized_thread_different_page(size_t n, size_t numVertices, unsigned int level, rdma_buf<unsigned int> *d_adjacencyList, unsigned int *d_edgesOffset,
                    unsigned int *d_edgesSize, unsigned int *d_distance, unsigned int *changed) {
    // Warp size
    const size_t warpSize = 32;

    // Global thread ID
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;

    // Warp ID within the grid
    size_t warpId = tid / warpSize;

    // Thread lane within the warp
    size_t lane = threadIdx.x % warpSize;

    // Ensure we don't process out-of-bounds vertices
    if (warpId < numVertices) {
        bool localChanged = false;

        // Fetch edges for this node
        // if (lane == 0) {
            unsigned int k = d_distance[warpId];
            if (k == level) {
                size_t edgeStart = d_edgesOffset[warpId];
                size_t edgeEnd = d_edgesOffset[warpId + 1];

                // Broadcast edgeStart and edgeEnd to the whole warp
                unsigned int edgeStartWarp = __shfl_sync(0xFFFFFFFF, edgeStart, 0);
                unsigned int edgeEndWarp = __shfl_sync(0xFFFFFFFF, edgeEnd, 0);

                // Process neighbors in parallel within the warp
                for (size_t j = edgeStartWarp + lane; j < edgeEndWarp; j += warpSize) {
                    unsigned int v = (*d_adjacencyList)[j];
                    unsigned int dist = d_distance[v];
                    if (level + 1 < dist) {
                        d_distance[v] = level + 1;
                        localChanged = true;
                    }
                }
            }
        // }

        // Use warp-wide OR to set the changed flag if needed
        if (__any_sync(0xFFFFFFFF, localChanged)) {
            atomicOr(changed, 1);
        }
    }
}

__global__ __launch_bounds__(1024,2)
void check_edgeList(rdma_buf<unsigned int> *a, unsigned int *b, size_t size){
    size_t tid = blockDim.x * blockIdx.x + threadIdx.x;
    if(tid == 0) printf("checking edgelist correctness\n");
    if(tid < size){
        unsigned int a_here = (*a)[tid];
        // __nanosleep(100000);
        if(a_here != b[tid]){
            printf("tid: %llu, a_here: %d b[tid]: %d\n", tid, a_here, b[tid]);
        } 
        else if(a_here == b[tid]){
            atomicAdd((unsigned long long *)&correct_results, 1);
        }
    }
}

__global__ void 
kernel_coalesce_new_repr(uint level, size_t n, size_t numVertex, const uint64_t new_size, unsigned int *d_distance, 
                        uint64_t *new_offset, unsigned int *new_vertex_list, unsigned int *edgeList, unsigned int *changed) {


    // Page size in elements (64KB / 4 bytes per unsigned int)
    const size_t pageSize = 8*1024 / sizeof(unsigned int);
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
        for (size_t i = 0; i < elementsPerWarp; ++i) {
            size_t elementIdx = pageStart + lane + i * warpSize;
            if(elementIdx < new_size){
                uint startVertex = new_vertex_list[elementIdx];
                unsigned int k = d_distance[startVertex];
                if (k == level) {
                    // Process adjacent nodes
                    // if(new_offset[elementIdx+1] - d_edgesOffset[elementIdx] >= 2*1024)
                    //     printf("elementx: %llu\n", elementIdx);
                    for(size_t j = new_offset[elementIdx]; j < new_offset[elementIdx+1]; ++j) {
                        uint v = edgeList[j]; // shared_data[j - pageStart];
                        
                        unsigned int dist = d_distance[v];
                        if (level + 1 < dist) {
                            d_distance[v] = level + 1;
                            *changed = 1;
                        }
                    }

                }
            }
        }
    }

}

__global__ // __launch_bounds__(1024,2)
void kernel_coalesce_new_repr_rdma(uint level, size_t n, size_t numVertex, const uint64_t new_size, unsigned int *d_distance, 
                                uint64_t *new_offset, unsigned int *new_vertex_list, rdma_buf<unsigned int> *edgeList, unsigned int *changed) {

    // const uint64_t tid = blockDim.x * 1024 * blockIdx.y + blockDim.x * blockIdx.x + threadIdx.x;
    // const uint64_t warpIdx = tid >> WARP_SHIFT;
    // const uint64_t laneIdx = tid & ((1 << WARP_SHIFT) - 1);
    // const uint64_t chunkIdx = warpIdx * CHUNK_SIZE;
    // uint64_t chunk_size = CHUNK_SIZE;

    // if((chunkIdx + CHUNK_SIZE) > new_size) {
    //     if ( new_size > chunkIdx )
    //         chunk_size = new_size - chunkIdx;
    //     else
    //         return;
    // }

    // for(uint32_t i = chunkIdx; i < chunk_size + chunkIdx; i++) {
    //     uint startVertex = new_vertex_list[i];
    //     if(d_distance[startVertex] == level) {
    //         const uint64_t start = new_offset[i];
    //         const uint64_t shift_start = start & MEM_ALIGN;
    //         const uint64_t end = new_offset[i+1];

    //         for(uint64_t j = shift_start + laneIdx; j < end; j += WARP_SIZE) {
    //             if (j >= start) {
    //                 const uint next = (*edgeList)[j];

    //                 // if(label[next] == MYINFINITY) {
    //                 if(d_distance[next] > level + 1) {
    //                     d_distance[next] = level + 1;
    //                     *changed = 1;
    //                 }
    //             }
    //         }
    //     }
    // }

    // size_t thid = blockIdx.x * blockDim.x + threadIdx.x;
    // const uint64_t laneIdx = thid & ((1 << WARP_SHIFT) - 1);

    // // // Warp ID within the block
    // size_t warpId = thid / (1 << WARP_SHIFT);
    
    // // size_t iterations = 0;
    // if(warpId < new_size /*d_distance->size/sizeof(uint)*/){
    //     uint startVertex = new_vertex_list[warpId];
    //     unsigned int k = d_distance[startVertex];
    //     if (k == level) {
    //         for (size_t i = new_offset[warpId] + laneIdx; i < new_offset[warpId+1]; i += (1 << WARP_SHIFT)) {
    //             unsigned int v;
               
    //             v = (*edgeList)[i];
    //             unsigned int dist = d_distance[v];
    //             if (level + 1 < dist) {
                    
    //                 d_distance[v] = level + 1; /*(int) level + 1*/
    //                 *changed = 1;
    //                 // valueChange = 1;
    //             }
    //         }
    //     }
    // }


    // Page size in elements (64KB / 4 bytes per unsigned int)
    const size_t pageSize = 8*1024 / sizeof(uint64_t);
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
        for (size_t i = 0; i < elementsPerWarp; ++i) {
            size_t elementIdx = pageStart + lane + i * warpSize;
            if(elementIdx < new_size){
                uint startVertex = new_vertex_list[elementIdx];
                unsigned int k = d_distance[startVertex];
                if (k == level) {
                    // Process adjacent nodes
                    // if(new_offset[elementIdx+1] - d_edgesOffset[elementIdx] >= 2*1024)
                    //     printf("elementx: %llu\n", elementIdx);
                    for(size_t j = new_offset[elementIdx]; j < new_offset[elementIdx+1]; ++j) {
                        uint v = (*edgeList)[j]; // shared_data[j - pageStart];
                        
                        unsigned int dist = d_distance[v];
                        if (level + 1 < dist) {
                            d_distance[v] = level + 1;
                            *changed = 1;
                        }
                    }
                }
            }
        }
    }
}

__global__
void simpleBfs_rdma_optimized_warp(size_t n, size_t numVertices, unsigned int level, rdma_buf<unsigned int> *d_adjacencyList, unsigned int *d_edgesOffset,
                    unsigned int *d_edgesSize, unsigned int *d_distance, unsigned int *changed) {
    // Page size in elements (64KB / 4 bytes per unsigned int)
    const size_t pageSize = 8*1024 / sizeof(unsigned int);
    // Elements per warp
    const size_t elementsPerWarp = pageSize / warpSize;

    // Global thread ID
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;

    // Warp ID within the block
    size_t warpId = tid / warpSize;

    // Thread lane within the warp
    size_t lane = threadIdx.x % warpSize;

    // Determine which page this warp will process
    size_t pageStart = warpId * pageSize;

    // Ensure we don't process out-of-bounds pages
    if (pageStart < n * pageSize) {
        bool localChanged = false;
        
        // Process elements within the page
        for (size_t i = 0; i < elementsPerWarp; ++i) {
            size_t elementIdx = pageStart + lane + i * warpSize;
            if (elementIdx < numVertices) {
                unsigned int k = d_distance[elementIdx];
                if (k == level) {
                    // printf("d_edgesOffset[%llu]: %u, d_distance[%llu]: %u\n", 
                    //         (long long int) elementIdx, d_edgesOffset[elementIdx], (long long int) elementIdx, d_distance[elementIdx]);
                    for(size_t j = d_edgesOffset[elementIdx]; j < d_edgesOffset[elementIdx+1] /*+ d_edgesSize[elementIdx]*/; ++j) {
                        int v = (*d_adjacencyList)[j];
                        // if(v >= numVertices || v < 0)
                            // printf("j: %llu V: %d numVertices: %lu\n", j, v, numVertices);
                        
                        unsigned int dist = d_distance[v];
                        if (level + 1 < dist) {
                            d_distance[v] = level + 1;
                            *changed = 1;
                        }
                    }
                }
            }
        }
    }
}

__global__
void simpleBfs_rdma_optimized_warp2(size_t n, unsigned int level, rdma_buf<unsigned int> *d_adjacencyList, unsigned int *d_edgesOffset,
                                   unsigned int *d_edgesSize, unsigned int *d_distance, unsigned int *changed) {
    // Page size in elements (64KB / 4 bytes per unsigned int)
    const size_t pageSize = 64*1024 / sizeof(unsigned int);
    // Elements per warp
    const size_t elementsPerWarp = pageSize / warpSize;

    // Global thread ID
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;

    // Warp ID within the block
    size_t warpId = tid / warpSize;

    // Thread lane within the warp
    size_t lane = threadIdx.x % warpSize;

    // Global warp ID
    size_t globalWarpId = tid / warpSize;

    // Number of warps in the grid
    size_t numWarps = (blockDim.x * gridDim.x) / warpSize;

    bool localChanged = false;

    for (size_t pageStart = globalWarpId * pageSize; pageStart < n * pageSize; pageStart += numWarps * pageSize) {
        // Process elements within the page
        for (size_t i = 0; i < elementsPerWarp; ++i) {
            size_t elementIdx = pageStart + lane + i * warpSize;
            if (elementIdx < n) {
                unsigned int k = d_distance[elementIdx];
                if (k == level) {
                    // printf("d_edgesOffset[%llu]: %u, d_edgesSize[%llu]: %u\n", 
                    //         (long long int) elementIdx, d_edgesOffset[elementIdx], (long long int) elementIdx, d_edgesSize[elementIdx]);
                    for (size_t j = d_edgesOffset[elementIdx]; j < d_edgesOffset[elementIdx] + d_edgesSize[elementIdx]; ++j) {
                        unsigned int v = (*d_adjacencyList)[j];
                        unsigned int dist = d_distance[v];
                        if (level + 1 < dist) {
                            d_distance[v] = level + 1;
                            localChanged = true;
                        }
                    }
                }
            }
        }
    }

    // Use atomic operation to set the changed flag if needed
    if (localChanged) {
        atomicOr(changed, 1);
    }
}

__global__
void simpleBfs_rdma_optimized_dynamic(size_t n, unsigned int level, rdma_buf<unsigned int> *d_adjacencyList, unsigned int *d_edgesOffset,
                    unsigned int *d_edgesSize, unsigned int *d_distance, unsigned int *changed){
    // Shared memory queue
    extern __shared__ unsigned int queue[];
    __shared__ int queueStart, queueEnd;
    
    if (threadIdx.x == 0) {
        queueStart = 0;
        queueEnd = 0;
    }
    __syncthreads();

    // Global thread ID
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;

    // Each thread initially processes its own node
    if (tid < n) {
        unsigned int k = d_distance[tid];
        if (k == level) {
            int pos = atomicAdd(&queueEnd, 1);
            if (pos < blockDim.x) {  // Ensure we do not exceed shared memory limits
                queue[pos] = tid;
            }
        }
    }
    __syncthreads();

    // Process the queue
    while (queueStart < queueEnd) {
        int node;
        if (threadIdx.x == 0 && queueStart < blockDim.x) {
            node = queue[queueStart];
            atomicAdd(&queueStart, 1);
        }
        node = __shfl_sync(0xFFFFFFFF, node, 0);

        bool localChanged = false;

        if (node < n) {  // Ensure the node index is within bounds
            unsigned int k = d_distance[node];

            if (k == level) {
                size_t edgeStart = d_edgesOffset[node];
                size_t edgeEnd = edgeStart + d_edgesSize[node];

                for (size_t j = edgeStart; j < edgeEnd; ++j) {
                    if (j >= n) continue;  // Ensure edge index is within bounds

                    unsigned int v = (*d_adjacencyList)[j];
                    unsigned int dist = d_distance[v];
                    if (level + 1 < dist) {
                        d_distance[v] = level + 1;
                        localChanged = true;

                        // Add the newly discovered node to the queue
                        int pos = atomicAdd(&queueEnd, 1);
                        if (pos < blockDim.x) {  // Ensure we do not exceed shared memory limits
                            queue[pos] = v;
                        }
                    }
                }
            }
        }
        __syncthreads();

        // Use atomic operation to set the changed flag if needed
        if (localChanged) {
            atomicOr(changed, 1);
        }
    }
}

__global__
void simpleBfs_rdma_dynamic_page(size_t n, unsigned int level, rdma_buf<unsigned int> *d_adjacencyList, unsigned int *d_edgesOffset,
                                 unsigned int *d_edgesSize, unsigned int *d_distance, unsigned int *changed) {
    // Page size in elements (64KB / 4 bytes per unsigned int)
    const size_t pageSize = 64 * 1024 / sizeof(unsigned int);
    // Number of pages
    const size_t numPages = (n * sizeof(unsigned int) + 64 * 1024 - 1) / (64 * 1024);
    // Elements per warp
    const size_t elementsPerWarp = pageSize / warpSize;

    // Warp ID within the grid
    size_t warpId = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;

    // Thread lane within the warp
    size_t lane = threadIdx.x % warpSize;

    __shared__ unsigned int currentPage;

    if (lane == 0) {
        currentPage = atomicAdd(&currentPage, 1);
    }
    __syncthreads();

    // Ensure we don't process out-of-bounds pages
    while (currentPage < numPages) {
        size_t pageStart = currentPage * pageSize;
        bool localChanged = false;

        // Process elements within the page
        for (size_t i = 0; i < elementsPerWarp; ++i) {
            size_t elementIdx = pageStart + lane + i * warpSize;
            if (elementIdx < n) {
                unsigned int k = d_distance[elementIdx];
                if (k == level) {
                    for (size_t j = d_edgesOffset[elementIdx]; j < d_edgesOffset[elementIdx] + d_edgesSize[elementIdx]; ++j) {
                        unsigned int v = (*d_adjacencyList)[j];
                        unsigned int dist = d_distance[v];
                        if (level + 1 < dist) {
                            d_distance[v] = level + 1;
                            localChanged = true;
                        }
                    }
                }
            }
        }

        // Use atomic operation to set the changed flag if needed
        if (localChanged) {
            atomicOr(changed, 1);
        }

        if (lane == 0) {
            currentPage = atomicAdd(&currentPage, 1);
        }
        __syncthreads();
    }
}

__global__ void 
kernel_baseline(uint32_t *label, const uint32_t level, const uint64_t vertex_count, const uint64_t *vertexList, 
                const uint64_t *edgeList, bool *changed) {
    const uint64_t tid = blockDim.x * BLOCK_SIZE * blockIdx.y + blockDim.x * blockIdx.x + threadIdx.x;

    if(tid < vertex_count && label[tid] == level) {
        const uint64_t start = vertexList[tid];
        const uint64_t end = vertexList[tid+1];
        // printf("level: %d label[%d]: %d start: %d end: %d\n", (int) level, (int) tid, (int) label[tid], (int) start, (int) end);
        for(uint64_t i = start; i < end; i++) {
            const uint64_t next = edgeList[i];

            // if(label[next] == MYINFINITY) {
            if(label[next] > level + 1) {
                label[next] = level + 1;
                *changed = true;
            }
        }
    }
}





































// #include <cstdio>
// #include <cuda.h>
// #include <cuda_runtime.h>
// #include <string>
// #include <cstring>
// #include <unistd.h>
// #include "graph.h"
// #include "bfsCPU.h"
// #include "bfs.cuh"

// using namespace std;
// // using namespace std;


// // extern "C"{
// //   #include "rdma_utils.h"
// // }

// // #include "../../src/rdma_utils.cuh"
// #include <time.h>
// // #include "../../include/runtime_eviction.h"
// // #include "../../include/runtime_prefetching_2nic.h"
// #include "../../include/runtime_prefetching_2nic.h"

// // Size of array
// #define N 1*1024*1024llu

// #define BLOCK_NUM 1024ULL
// #define MYINFINITY 2147483647llu

// #define BLOCK_SIZE 1024
// #define WARP_SHIFT 4
// #define WARP_SIZE 32

// #define MEM_ALIGN_64 (~(0xfULL))
// #define MEM_ALIGN MEM_ALIGN_64

// #define CHUNK_SHIFT 3
// #define CHUNK_SIZE (1 << CHUNK_SHIFT)

// #define GPU 0

// // __device__ size_t transfer_time;

// __device__ rdma_buf<unsigned int> D_adjacencyList;

// __global__ void test(rdma_buf<unsigned int> *a/*, rdma_buf<int> *b, rdma_buf<int> *c*/);

// __global__
// void simpleBfs_normal(size_t n, size_t vertexCount, unsigned int level, unsigned int *d_edgeList, unsigned int *d_vertex_list,
//                       unsigned int *d_distance, unsigned int *changed);

// __global__
// void simpleBfs_normal_rdma(size_t n, size_t vertexCount, unsigned int level, rdma_buf<unsigned int> *d_edgeList, unsigned int *d_vertex_list,
//                       unsigned int *d_distance, unsigned int *changed);

// __global__
// void simpleBfs_uvm(size_t n, unsigned int level, unsigned int *d_adjacencyList, unsigned int *d_edgesOffset,
//                     unsigned int *d_edgesSize, unsigned int *d_distance, unsigned int *changed);

// __global__
// void simpleBfs_rdma(size_t n, unsigned int level, rdma_buf<unsigned int> *d_adjacencyList, unsigned int *d_edgesOffset,
//                     unsigned int *d_edgesSize, unsigned int *d_distance, unsigned int *changed);

// __global__
// void simpleBfs_rdma_2(size_t n, unsigned int level, rdma_buf<unsigned int> *d_adjacencyList, unsigned int *d_edgesOffset,
//                       unsigned int *d_edgesSize, unsigned int *d_distance, unsigned int *changed);    

// __global__
// void simpleBfs_rdma_3(size_t n, unsigned int level, rdma_buf<unsigned int> *d_adjacencyList, unsigned int *vertexList, unsigned int *changed);               

// __global__ __launch_bounds__(128,16)
// void kernel_coalesce_ptr_pc(unsigned int *ptr, unsigned int *label, const uint32_t level, const uint64_t vertex_count, unsigned int *edgeOffset,
//                             unsigned int * edgeSize, unsigned *edgeList, unsigned int *changed);

// __global__
// void simpleBfs_normal_rdma_optimized(int numEdgesPerThread, size_t numEdges, size_t vertexCount, unsigned int level, rdma_buf<unsigned int> *d_edgeList, unsigned int *d_vertex_list,
//                       unsigned int *d_distance, unsigned int *changed);

// __global__ __launch_bounds__(128,32)
// void kernel_coalesce_ptr_pc_rdma(rdma_buf<unsigned int> *ptr, unsigned int *label, const uint32_t level, const uint64_t vertex_count, unsigned int *edgeOffset,
//                             unsigned int * edgeSize, unsigned int *changed);

// __global__
// void simpleBfs_normal_rdma_optimized2(size_t n, size_t vertexCount, unsigned int level, rdma_buf<unsigned int> *d_edgeList, unsigned int *d_vertex_list,
//                                      unsigned int *d_distance, unsigned int *changed);

// __global__
// void simpleBfs_rdma_optimized_warp(size_t n, size_t numVertices, unsigned int level, rdma_buf<unsigned int> *d_adjacencyList, unsigned int *d_edgesOffset,
//                     unsigned int *d_edgesSize, unsigned int *d_distance, unsigned int *changed);

// __global__
// void simpleBfs_rdma_dynamic_page(size_t n, unsigned int level, rdma_buf<unsigned int> *d_adjacencyList, unsigned int *d_edgesOffset,
//                                  unsigned int *d_edgesSize, unsigned int *d_distance, unsigned int *changed);

// __global__
// void simpleBfs_rdma_optimized_dynamic(size_t n, unsigned int level, rdma_buf<unsigned int> *d_adjacencyList, unsigned int *d_edgesOffset,
//                     unsigned int *d_edgesSize, unsigned int *d_distance, unsigned int *changed);

// __global__
// void simpleBfs_rdma_optimized_warp2(size_t n, unsigned int level, rdma_buf<unsigned int> *d_adjacencyList, unsigned int *d_edgesOffset,
//                                    unsigned int *d_edgesSize, unsigned int *d_distance, unsigned int *changed);

// __global__
// void simpleBfs_modVertexList(size_t n, unsigned int level, unsigned int *d_adjacencyList,
//                     unsigned int *vertex_list, unsigned int *d_distance, unsigned int *changed);

// __global__
// void simpleBfs_modVertexList_rdma(size_t n, unsigned int level, rdma_buf<unsigned int> *d_adjacencyList,
//                     unsigned int *vertex_list, unsigned int *d_distance, unsigned int *changed);

// __global__
// void simpleBfs_rdma_optimized_thread_different_page(size_t n, size_t numVertices, unsigned int level, rdma_buf<unsigned int> *d_adjacencyList, unsigned int *d_edgesOffset,
//                     unsigned int *d_edgesSize, unsigned int *d_distance, unsigned int *changed);

// __global__ void kernel_coalesce_chunk_rdma(unsigned int *label, unsigned int level, unsigned int vertex_count,
//                                            unsigned int *vertexList, rdma_buf<unsigned int> *edgeList, uint *changed);

// __global__ void 
// kernel_baseline(uint32_t *label, const uint32_t level, const uint64_t vertex_count, const uint64_t *vertexList, 
//                 const uint64_t *edgeList, bool *changed);

// __global__ void kernel_coalesce(uint32_t *label, const uint32_t level, const uint64_t vertex_count, 
//                                 const uint64_t *vertexList, const uint64_t *edgeList, bool *changed);

// __global__ //  __launch_bounds__(1024,2) 
// void  kernel_coalesce_new_repr_rdma(uint level, size_t n, size_t numVertex, const uint64_t new_size, unsigned int *d_distance, 
//                     unsigned int *new_offset, unsigned int *new_vertex_list, rdma_buf<unsigned int> *edgeList, unsigned int *changed);

// __global__ void 
// kernel_coalesce_new_repr(uint level, size_t n, size_t numVertex, const uint64_t new_size, unsigned int *d_distance, 
//                         unsigned int *new_offset, unsigned int *new_vertex_list, unsigned int *edgeList, unsigned int *changed);

// __global__ void 
// bfs_kernel_coalesce_chunk(unsigned int *label, unsigned int level, const uint64_t vertex_count, unsigned int *vertexList, unsigned int *edgeList, unsigned int *changed);

// __global__ void 
// check_edgeList(rdma_buf<unsigned int> *a, unsigned int *b, size_t size);

// // __global__ void kernel_coalesce_chunk(unsigned int *label, const uint level, const uint64_t vertex_count, const uint64_t *vertexList,
// //                                       const uint64_t *edgeList, uint *changed);

// __global__ void kernel_coalesce_chunk(uint32_t *label, const uint32_t level, const uint64_t vertex_count, const uint64_t *vertexList,
//                                       const uint64_t *edgeList, bool *changed) {
//     const uint64_t tid = blockDim.x * BLOCK_SIZE * blockIdx.y + blockDim.x * blockIdx.x + threadIdx.x;
//     const uint64_t warpIdx = tid >> WARP_SHIFT;
//     const uint64_t laneIdx = tid & ((1 << WARP_SHIFT) - 1);
//     const uint64_t chunkIdx = warpIdx * CHUNK_SIZE;
//     uint64_t chunk_size = CHUNK_SIZE;

//     if((chunkIdx + CHUNK_SIZE) > vertex_count) {
//         if ( vertex_count > chunkIdx )
//             chunk_size = vertex_count - chunkIdx;
//         else
//             return;
//     }

//     for(uint32_t i = chunkIdx; i < chunk_size + chunkIdx; i++) {
//         if(label[i] == level) {
//             const uint64_t start = vertexList[i];
//             const uint64_t shift_start = start & MEM_ALIGN;
//             const uint64_t end = vertexList[i+1];

//             for(uint64_t j = shift_start + laneIdx; j < end; j += WARP_SIZE) {
//                 if (j >= start) {
//                     const uint64_t next = edgeList[j];

//                     // if(label[next] == MYINFINITY) {
//                     if(label[next] > level + 1) {
//                         label[next] = level + 1;
//                         *changed = true;
//                     }
//                 }
//             }
//         }
//     }
// }


// // Kernel
// __global__ void add_vectors_uvm(int *a, int *b, int *c, int size)
// {
// 	int id = blockDim.x * blockIdx.x + threadIdx.x;
// 	// if(id < size) {
// 		c[id] = a[id] + b[id];
// 		// printf("c[%d]: %d\n", id, c[id]);
// 	// }
// }

// #define htonl(x)  ((((uint32_t)(x) & 0xff000000) >> 24) |\
//                    (((uint32_t)(x) & 0x00ff0000) >>  8) |\
//                    (((uint32_t)(x) & 0x0000ff00) <<  8) |\
//                    (((uint32_t)(x) & 0x000000ff) << 24))

// #define WARP_SIZE 32

// void delay(int number_of_seconds)
// {
//     // Converting time into milli_seconds
//     int milli_seconds = 1000000 * number_of_seconds;
 
//     // Storing start time
//     clock_t start_time = clock();
 
//     // looping till required time is not achieved
//     while (clock() < start_time + milli_seconds)
//         ;
// }

// enum { NS_PER_SECOND = 1000000000 };

// void sub_timespec(struct timespec t1, struct timespec t2, struct timespec *td)
// {
//     td->tv_nsec = t2.tv_nsec - t1.tv_nsec;
//     td->tv_sec  = t2.tv_sec - t1.tv_sec;
//     if (td->tv_sec > 0 && td->tv_nsec < 0)
//     {
//         td->tv_nsec += NS_PER_SECOND;
//         td->tv_sec--;
//     }
//     else if (td->tv_sec < 0 && td->tv_nsec > 0)
//     {
//         td->tv_nsec -= NS_PER_SECOND;
//         td->tv_sec++;
//     }
// }

// void usage(const char *argv0)
// {
//   fprintf(stderr, "usage: %s <mode> <server-address> <server-port>\n  mode = \"read\", \"write\"\n", argv0);
//   exit(1);
// }



// void runCpu(int startVertex, Graph &G, std::vector<int> &distance,
//             std::vector<int> &parent, std::vector<bool> &visited) {
//     printf("Starting sequential bfs.\n");
//     auto start = std::chrono::steady_clock::now();
//     bfsCPU(startVertex, G, distance, parent, visited);
//     auto end = std::chrono::steady_clock::now();
//     long duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
//     printf("Elapsed time in milliseconds : %li ms.\n\n", duration);
// }


// #define checkError(ans) { gpuAssert((ans), __FILE__, __LINE__); }
// inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
// {
//    if (code != cudaSuccess) 
//    {
//       fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
//       if (abort) exit(code);
//    }

// }

// rdma_buf<unsigned int> *rdma_adjacencyList;
// // rdma_buf<unsigned int> *u_edgesOffset;
// // rdma_buf<unsigned int> *u_edgesSize;
// // rdma_buf<unsigned int> *u_distance;
// unsigned int *u_adjacencyList;
// unsigned int *uvm_adjacencyList;
// unsigned int *u_edgesOffset;
// unsigned int *u_edgesSize;
// unsigned int *u_distance;
// unsigned int *u_startVertices;
// rdma_buf<unsigned int> *u_parent;
// rdma_buf<unsigned int> *u_currentQueue;
// rdma_buf<unsigned int> *u_nextQueue;
// rdma_buf<unsigned int> *u_degrees;

// // uint *u_adjacencyList;
// // uint *u_edgesOffset;
// // uint *u_edgesSize;
// // uint *u_distance;
// // uint *u_parent;
// // uint *u_currentQueue;
// // uint *u_nextQueue;
// // uint *u_degrees;

// uint *incrDegrees;


// void initCuda(Graph &G) {

//     checkError(cudaMallocManaged(&rdma_adjacencyList, sizeof(rdma_buf<unsigned int>)));
//     checkError(cudaMallocManaged(&uvm_adjacencyList, G.numEdges*sizeof(unsigned int)));
    
//     auto start = std::chrono::steady_clock::now();
//     // checkError(cudaMallocHost(&u_adjacencyList, G.numEdges*sizeof(unsigned int)));
//     u_adjacencyList = (unsigned int *) malloc(G.numEdges*sizeof(uint)); // new uint[G.numEdges];
//     // u_adjacencyList = new uint[G.numEdges];
//     auto end = std::chrono::steady_clock::now();
//     long duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
//     printf("Elapsed time in milliseconds for cudaMallocHost: %li ms.\n", duration);

    
    
//     // checkError(cudaMallocManaged(&u_edgesOffset, sizeof(rdma_buf<unsigned int>)));
//     // checkError(cudaMallocManaged(&u_edgesSize, sizeof(rdma_buf<unsigned int>)));
//     // checkError(cudaMallocManaged(&u_distance, sizeof(rdma_buf<unsigned int>)));
    
//     // checkError(cudaMallocHost(&u_startVertices, G.numEdges*sizeof(unsigned int)));
//     checkError(cudaMallocHost(&u_distance, G.numVertices*sizeof(unsigned int)));
//     checkError(cudaMallocHost(&u_edgesOffset, (G.numVertices+1)*sizeof(unsigned int)));
//     checkError(cudaMallocHost(&u_edgesSize, G.numVertices*sizeof(unsigned int)));

//     // rdma_adjacencyList->start(G.numEdges *sizeof(unsigned int));

//     // u_edgesOffset->start(G.numVertices *sizeof(unsigned int));
//     // u_edgesSize->start(G.numVertices *sizeof(unsigned int));
//     // u_distance->start(G.numVertices *sizeof(unsigned int));
//     // u_parent->start(G.numVertices *sizeof(unsigned int));
//     // u_currentQueue->start(G.numVertices *sizeof(unsigned int));
//     // u_nextQueue->start(G.numVertices *sizeof(unsigned int));
//     // u_degrees->start(G.numVertices *sizeof(unsigned int));

//     // checkError(cudaMallocManaged(&u_adjacencyList, G.numEdges * sizeof(int) ));
//     // checkError(cudaMallocManaged(&u_edgesOffset, G.numVertices * sizeof(int) ));
//     // checkError(cudaMallocManaged(&u_edgesSize, G.numVertices * sizeof(int)) );
//     // checkError(cudaMallocManaged(&u_distance, G.numVertices * sizeof(int) ));
//     // checkError(cudaMallocManaged(&u_parent, G.numVertices * sizeof(int) ));
//     // checkError(cudaMallocManaged(&u_currentQueue, G.numVertices * sizeof(int) ));
//     // checkError(cudaMallocManaged(&u_nextQueue, G.numVertices * sizeof(int) ));
//     // checkError(cudaMallocManaged(&u_degrees, G.numVertices * sizeof(int) ));



//     // checkError(cudaMallocHost((void **) &incrDegrees, sizeof(int) * G.numVertices));


//     // memcpy(u_adjacencyList, G.adjacencyList.data(), G.numEdges * sizeof(int));
//     // memcpy(u_edgesOffset, G.edgesOffset.data(), G.numVertices * sizeof(int));
//     // memcpy(u_edgesSize, G.edgesSize.data(), G.numVertices * sizeof(int));
//     // checkError(cudaMemcpy(d_adjacencyList, G.adjacencyList.data(), G.numEdges * sizeof(int), cudaMemcpyHostToDevice));
//     // checkError(cudaMemcpy(d_edgesOffset, G.edgesOffset.data(), G.numVertices * sizeof(int), cudaMemcpyHostToDevice));
//     // checkError(cudaMemcpy(d_edgesSize, G.edgesSize.data(), G.numVertices * sizeof(int), cudaMemcpyHostToDevice ));

// }

// void finalizeCuda() {

//     checkError(cudaFree(u_adjacencyList));
//     checkError(cudaFree(u_edgesOffset));
//     checkError(cudaFree(u_edgesSize));
//     checkError(cudaFree(u_distance));
//     checkError(cudaFree(u_parent));
//     checkError(cudaFree(u_currentQueue));
//     checkError(cudaFree(u_nextQueue));
//     checkError(cudaFree(u_degrees));
//     checkError(cudaFreeHost(incrDegrees));
// }

// // void checkOutput(std::vector<int> &distance, std::vector<int> &expectedDistance, Graph &G) {
// //     for (int i = 0; i < G.numVertices; i++) {
// //         if (expectedDistance[i] != *(u_distance+i) ) {
// //             printf("%d %d %d\n", i, distance[i], expectedDistance[i]);
// //             printf("Wrong output!\n");
// //             exit(1);
// //         }
// //     }

// //     printf("Output OK!\n\n");
// // }

// void checkOutput_rdma(uint *u_distance, uint *expectedDistance, Graph &G) {
//     size_t num_errors = 0;
//     for (int i = 0; i < G.numVertices; i++) {
//         if(i < 10){
//                 printf("%d ", i);
//                 printf("%llu ", u_distance[i]);
//                 printf("%d\n", expectedDistance[i]);
//             }
//         if (expectedDistance[i] != u_distance[i] ) {
//             // 
            
//             // printf("Wrong output!\n");
//             num_errors++;
//             // exit(1);
//         }
//     }

//     printf("num_errors: %llu Output OK!\n\n", num_errors);
// }


// void initializeCudaBfs(int startVertex, std::vector<int> &distance, std::vector<int> &parent, Graph &G) {
//     //initialize values
//     std::fill(distance.begin(), distance.end(), 100000/*std::numeric_limits<int>::max()*/);
//     std::fill(parent.begin(), parent.end(), std::numeric_limits<int>::max());
//     distance[startVertex] = 0;
//     parent[startVertex] = 0;

//     // checkError(cudaMemcpy(d_distance, distance.data(), G.numVertices * sizeof(int), cudaMemcpyHostToDevice));
//     // checkError(cudaMemcpy(d_parent, parent.data(), G.numVertices * sizeof(int), cudaMemcpyHostToDevice));
//     // memcpy(u_distance->local_buffer, distance.data(), G.numVertices * sizeof(int));
//     // memcpy(u_parent->local_buffer, parent.data(), G.numVertices * sizeof(int));
//     printf("printing samples from parent and distance vectors initializations...\n");
//     for (size_t i = 0; i < G.numVertices; i++)
//     {
//         u_distance[i] = 2147483647; // distance.data()[i];
//         // u_parent->local_buffer[i] = parent.data()[i];
//     }
//     u_distance[startVertex] = 0;

//     for (size_t i = 0; i < 5; i++)
//     {
//         printf("u_distance->local_buffer[%llu]: %llu; distance.data()[%llu]: %llu\n", i, u_distance[i], i, distance.data()[i]);
//         // printf("u_parent->local_buffer[%llu]: %llu; parent.data()[%llu]: %llu\n", i, u_parent->local_buffer[i], i, parent.data()[i]);
        
//     }
    

//     int firstElementQueue = startVertex;
//     // cudaMemcpy(d_currentQueue, &firstElementQueue, sizeof(int), cudaMemcpyHostToDevice);
//     // *u_currentQueue->local_buffer = firstElementQueue;
// }

// void finalizeCudaBfs(std::vector<int> &distance, std::vector<int> &parent, Graph &G) {
//     //copy memory from device
//     // checkError(cudaMemcpy(distance.data(), d_distance, G.numVertices * sizeof(int), cudaMemcpyDeviceToHost));
//     // checkError(cudaMemcpy(parent.data(), d_parent, G.numVertices * sizeof(int), cudaMemcpyDeviceToHost));
// }

// __global__ void transfer(size_t size, rdma_buf<unsigned int> *d_adjacencyList, unsigned int *changed)
// {
//     size_t id = blockDim.x * blockIdx.x + threadIdx.x;
//     size_t stride = blockDim.x * gridDim.x;
    
//         for (size_t i = id; i < size ; i += stride)
//         {
//             unsigned int y = (*d_adjacencyList)[i];
//             // y++;
//             // *changed += y;
//         }
// }

// __global__ void assign_array(rdma_buf<unsigned int> *adjacencyList){
//     D_adjacencyList = *adjacencyList;
//     printf("D_adjacencyList.d_TLB[0].state: %d\n", D_adjacencyList.d_TLB[0].state);
//     // printf("D_adjacencyList.d_TLB[0].device_address: %p\n", D_adjacencyList.d_TLB[0].page_number);
// }

// __global__ void test2(size_t size, int level, unsigned int *d_distance, rdma_buf<unsigned int> *d_edgesOffset,
//                       rdma_buf<unsigned int> *d_edgesSize, rdma_buf<unsigned int> *d_adjacencyList, unsigned int *changed)
// {
//     size_t id = blockDim.x * blockIdx.x + threadIdx.x;
//     int valueChange = 0;
//     size_t stride = blockDim.x * gridDim.x;
//     // size_t size = d_distance->size/sizeof(unsigned int);
    
//     if(id < size){
//         unsigned int k = d_distance[id];
//         uint edgesOffset = (*d_edgesOffset)[id];
//         uint edgesSize = (*d_edgesSize)[id];
//         for (size_t i = edgesOffset; i < edgesOffset + edgesSize /*d_adjacencyList->size/sizeof(unsigned int)*/; i += 1)
//         {
//             unsigned int y = (*d_adjacencyList)[i];
//             if(k == level){
//             //     if(i < edgesOffset + edgesSize && i >= edgesOffset){
//             //         unsigned int dist = (*d_distance)[y];
//                     if (level + 1 < d_distance[y]) {
                    
//                         unsigned int new_dist = level + 1;
//                         d_distance[i] = new_dist /*(int) level + 1*/;
//                         valueChange = 1;
//                     }
//             //     }
//             }
//         }
//     }
//         if (valueChange) {
//             *changed = valueChange;
//         }
//     // }
//     // a->rvalue(id, id);
//     // c->rvalue(id, (*a)[id] + (*b)[id]); 
//     // if(id == 0) printf("(*b)[%d]: %d\n", id, (*b)[id]);
// }

// void runCudaSimpleBfs(int startVertex, Graph &G, std::vector<int> &distance,
//                       std::vector<int> &parent) 
// {
//     initializeCudaBfs(startVertex, distance, parent, G);

//     for (size_t i = 0; i < G.numVertices; i++)
//     {
//         if(u_distance[i] == 0)
//             printf("%zu-%zu ", u_edgesOffset[i], u_edgesOffset[i] + u_edgesSize[i]);
//             // printf("%zu-%zu ", u_edgesOffset->local_buffer[i], u_edgesOffset->local_buffer[i] + u_edgesSize->local_buffer[i]);
//         // printf("%llu ", u_edgesOffset->local_buffer[i]);
//     }

//     for (size_t i = 0; i < G.numVertices; i++)
//     {
//         if(u_distance[i] == 0){
//             printf("u_distance->host_buffer[%llu]: %u\n", i, u_distance[i]);
//         }
//     }

//     unsigned int *d_distance, *d_edgesSize, *d_edgesOffset, *d_adjacencyList, *d_startVertices;
//     // rdma_buf<unsigned int> *d_adjacencyList;
//     // checkError(cudaMalloc((void **) &d_adjacencyList, sizeof(rdma_buf<unsigned int>)));
//     checkError(cudaMalloc((void **) &d_startVertices, G.numEdges*sizeof(unsigned int)));
//     checkError(cudaMemcpy(d_startVertices, u_startVertices, G.numEdges*sizeof(unsigned int), cudaMemcpyHostToDevice));

//     checkError(cudaMalloc((void **) &d_adjacencyList, G.numEdges*sizeof(unsigned int)));
//     checkError(cudaMemcpy(d_adjacencyList, u_adjacencyList, G.numEdges*sizeof(unsigned int), cudaMemcpyHostToDevice));
//     // checkError(cudaMemcpy(d_adjacencyList, u_adjacencyList, sizeof(rdma_buf<unsigned int>), cudaMemcpyHostToDevice));

//     checkError(cudaMalloc((void **) &d_distance, G.numVertices*sizeof(unsigned int)));
//     checkError(cudaMemset(d_distance, 100000, G.numVertices*sizeof(unsigned int)));
//     checkError(cudaMemset(&d_distance[startVertex], 0, 1*sizeof(unsigned int)));

//     checkError(cudaMalloc((void **) &d_edgesSize, G.numVertices*sizeof(unsigned int)));
//     checkError(cudaMalloc((void **) &d_edgesOffset, G.numVertices*sizeof(unsigned int)));
    
//     checkError(cudaMemcpy(d_edgesOffset, u_edgesOffset, G.numVertices*sizeof(unsigned int), cudaMemcpyHostToDevice));
//     checkError(cudaMemcpy(d_edgesSize, u_edgesSize, G.numVertices*sizeof(unsigned int), cudaMemcpyHostToDevice));

//     uint *changed;
//     checkError(cudaMallocHost((void **) &changed, sizeof(unsigned int)));


//     //launch kernel
//     printf("Starting simple parallel bfs.\n");

//     cudaError_t ret1 = cudaDeviceSynchronize();
//     if(cudaSuccess != ret1){  
//         printf("cudaDeviceSynchronize error: %d\n", ret1);  
//         exit(-1);
//     }
//     auto start = std::chrono::steady_clock::now();
    
//     cudaEvent_t event1, event2;
//     cudaEventCreate(&event1);
//     cudaEventCreate(&event2);

//     ret1 = cudaDeviceSynchronize();

//     cudaEventRecord(event1, (cudaStream_t)1);
    
//     unsigned int level;
//     // checkError(cudaMallocManaged((void **) &level, sizeof(unsigned int)));
//     level = 0;
//     // transfer<<<2048, 512>>>(rdma_adjacencyList->size/sizeof(unsigned int), rdma_adjacencyList, changed);
    
//     // printf("cudaDeviceSynchronize for transfer: %d\n", ret1); 
    
//     cudaEventRecord(event2, (cudaStream_t) 1);
//     ret1 = cudaDeviceSynchronize();
//     assign_array<<< 1 , 1 >>>(rdma_adjacencyList);
//     ret1 = cudaDeviceSynchronize();
//     printf("cudaDeviceSynchronize for transfer: %d\n", ret1);  
//     if(cudaSuccess != ret1){  
//         printf("cudaDeviceSynchronize error for transfer: %d\n", ret1);  
//         exit(-1);
//     }
//     cudaEventSynchronize(event1); //optional
//     cudaEventSynchronize(event2); //wait for the event to be executed!
//     float dt_ms;
//     cudaEventElapsedTime(&dt_ms, event1, event2);
//     printf("Elapsed time for transfer with cudaEvent: %f\n", dt_ms);
//     ret1 = cudaDeviceSynchronize();
//     int numEdgesPerThread = 8;
//     int numThreadsPerBlock = 512;
//     // int numBlock = (numNodes) / (numThreadsPerBlock) + 1;
//     int numBlock = (G.numEdges) / (numThreadsPerBlock * numEdgesPerThread) + 1;
//     cudaEventRecord(event1, (cudaStream_t)1);
//     *changed = 1;
//     while (*changed) {
//         *changed = 0;
//         // void *args[] = {&G.numVertices, &level, &d_adjacencyList, &d_edgesOffset, &d_edgesSize, &d_distance, &d_parent,
//         //                 &changed};
//         // checkError(cuLaunchKernel(cuSimpleBfs, G.numVertices / 1024 + 1, 1, 1,
//         //                           1024, 1, 1, 0, 0, args, 0));
//         // ret1 = cudaDeviceSynchronize();
//         // printf("cudaDeviceSynchronize: %d\n", ret1);  
//         // if(cudaSuccess != ret1){  
//         //     printf("cudaDeviceSynchronize error: %d\n", ret1);  
//         //     exit(-1);
//         // }
//         // printf("G.numVertices: %llu\n", G.numVertices); 
//         // simpleBfs_uvm<<< G.numVertices / 512 + 1, 512 >>>(G.numVertices, level, uvm_adjacencyList, d_edgesOffset, d_edgesSize, d_distance, changed);  
//         // simpleBfs_rdma<<< G.numVertices / 256 + 1, 256 >>>(G.numVertices, level, rdma_adjacencyList, d_edgesOffset, d_edgesSize, d_distance, changed);    
//         // kernel_coalesce_hash_ptr_pc<<< G.numVertices / 512 + 1, 512 >>>(u_adjacencyList, d_distance, level, G.numVertices); 
//         // simpleBfs_normal_rdma<<< /*G.numVertices / 512 + 1, 512*/ 2048, 512 >>>(G.numEdges, G.numVertices, level, rdma_adjacencyList, d_startVertices, d_distance, changed);
//         // simpleBfs_normal_rdma_optimized<<<numBlock, numThreadsPerBlock>>>(numEdgesPerThread, G.numEdges, G.numVertices, level, rdma_adjacencyList, d_startVertices, d_distance, changed);
//         // simpleBfs_normal<<< G.numVertices / 512 + 1, 512 >>>(G.numEdges, G.numVertices, level, uvm_adjacencyList/*d_adjacencyList u_adjacencyList*/, d_startVertices, d_distance, changed);
        
//         // for cudamemcpy 
//         // simpleBfs_normal<<< /*G.numVertices / 512 + 1, 512*/ 2048, 8 >>>(G.numEdges, G.numVertices, level, uvm_adjacencyList, d_startVertices, d_distance, changed);      
//         ret1 = cudaDeviceSynchronize();
//         // test2<<< /*G.numVertices/256+1, 256*/ G.numVertices/256+1, 256>>>(G.numVertices, level, u_distance, u_edgesOffset, u_edgesSize, u_adjacencyList, changed);
//         // test<<< 2, 1024>>>(u_adjacencyList);
        

//         level++;
//     }
//     cudaEventRecord(event2, (cudaStream_t) 1);
//     printf("cudaGetLastError(): %d level: %d\n", cudaGetLastError(), level);
//         ret1 = cudaDeviceSynchronize();
//         printf("cudaDeviceSynchronize: %d *changed: %d\n", ret1, *changed);  
//         if(cudaSuccess != ret1){  
//             printf("cudaDeviceSynchronize error: %d\n", ret1);  
//             exit(-1);
//         }


    
//     cudaEventSynchronize(event1); //optional
//     cudaEventSynchronize(event2); //wait for the event to be executed!

//     // calculate time
//     dt_ms;
//     cudaEventElapsedTime(&dt_ms, event1, event2);
//     printf("Elapsed time with cudaEvent: %f\n", dt_ms);

//     auto end = std::chrono::steady_clock::now();
//     long duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
//     printf("Elapsed time in milliseconds : %li ms.\n", duration);

//     checkError(cudaMemcpy(u_distance, d_distance, G.numVertices*sizeof(unsigned int), cudaMemcpyHostToDevice));

//     finalizeCudaBfs(distance, parent, G);
// }

// __global__ 
// void test_batch(rdma_buf<unsigned int> *a, unsigned int *b, size_t size){
//     size_t tid = blockDim.x * blockIdx.x + threadIdx.x;
//     size_t stride = blockDim.x * gridDim.x;
//     // size_t index = tid*16*1024;

//     for (size_t i = tid; i < size; i += stride)
//     {
//         unsigned int ab = (*a)[i];
//         if(ab != b[i]){
//             printf("unmatched: %llu expected: %d got: %d\n", i, b[i], ab);
//         }
//     }
    

//     // if(index < size){
//     //     unsigned int ab = (*a)[index];
//     //     if(ab != b[index]){
//     //         printf("unmatched: %llu expected: %d got: %d\n", index/16384, b[index], ab);
//     //     }
//     // }
// }

// unsigned int *runCudaSimpleBfs_emogi(int startVertex, Graph &G) 
// {
//     // initializeCudaBfs(startVertex, distance, parent, G);
    
//     printf("fuction: %s, line: %d\n", __func__, __LINE__);

//     unsigned int *d_distance, *d_edgesSize, *d_startVertices, *d_vertexList;
//     uint64_t *d_adjacencyList, *d_edgesOffset;
//     // rdma_buf<unsigned int> *d_adjacencyList;
//     // checkError(cudaMalloc((void **) &d_adjacencyList, sizeof(rdma_buf<unsigned int>)));

//     // checkError(cudaMalloc((void **) &d_startVertices, G.numEdges*sizeof(unsigned int)));
//     // checkError(cudaMemcpy(d_startVertices, u_startVertices, G.numEdges*sizeof(unsigned int), cudaMemcpyHostToDevice));

//     uint64_t *u_adjacencyList_64;
//     u_adjacencyList_64 = (uint64_t *) malloc(sizeof(uint64_t)*G.numEdges);  
//     // checkError(cudaMallocHost((void **) &u_adjacencyList_64, sizeof(uint64_t)*G.numEdges));
//     for (size_t i = 0; i < G.numEdges; i++)
//     {
//        u_adjacencyList_64[i] = u_adjacencyList[i];
//     }

//     int mem = 2;
//     size_t edge_size = G.numEdges*sizeof(uint64_t);
//     if(mem == 0){
//         checkError(cudaMalloc((void **) &d_adjacencyList, G.numEdges*sizeof(uint64_t)));
//         cudaDeviceSynchronize();
//         auto start = std::chrono::steady_clock::now();
//         checkError(cudaMemcpy(d_adjacencyList, u_adjacencyList_64, G.numEdges*sizeof(uint64_t), cudaMemcpyHostToDevice));
//         auto end = std::chrono::steady_clock::now();
//         long duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
//         printf("Elapsed time for tranfser of edge list in milliseconds : %li ms.\n\n", duration);
//     }
//     else if (mem == 1) {   
//         // case UVM_READONLY:
//             checkError(cudaMallocManaged((void**)&d_adjacencyList, edge_size));
//             // file.read((char*)edgeList_d, edge_size);
//             memcpy(d_adjacencyList, u_adjacencyList_64, edge_size);


//             auto start2 = std::chrono::steady_clock::now();
//             checkError(cudaMemAdvise(d_adjacencyList, edge_size, cudaMemAdviseSetReadMostly, 0));

//             auto end2 = std::chrono::steady_clock::now();
//             long duration2 = std::chrono::duration_cast<std::chrono::milliseconds>(end2 - start2).count();
//             printf("Elapsed time for memAdvise SetReadMostly in milliseconds : %li ms.\n\n", duration2);
//     }
     
//     else if(mem == 2){ 
//           // break;
//         // case UVM_DIRECT:
//             checkError(cudaMallocManaged((void**)&d_adjacencyList, edge_size));
//             // file.read((char*)edgeList_d, edge_size);
//             std::cout << "line: " << __LINE__  <<  " edge_size: " << edge_size << std::endl;
//             memcpy(d_adjacencyList, u_adjacencyList_64, edge_size);
//             // for (size_t i = 0; i < edge_count; i++)
//             // {
//             //     // std::cout << "i: " << i << " edgeList_d[i]: "<< edgeList_d[i] << std::endl;
//             //     edgeList_d[i] = temp_edgeList[i];
                
//             // }
//             std::cout << "line: " << __LINE__ << std::endl;

//             auto start1 = std::chrono::steady_clock::now();
//             // checkError(cudaMemAdvise(d_adjacencyList, edge_size, cudaMemAdviseSetAccessedBy, 0));
//             auto end1 = std::chrono::steady_clock::now();
//             long duration1 = std::chrono::duration_cast<std::chrono::milliseconds>(end1 - start1).count();
//             printf("Elapsed time for memAdvise SetAccessedBy in milliseconds : %li ms.\n\n", duration1);

//             cudaDeviceProp devProp;
//             cudaGetDeviceProperties(&devProp, 0);
//             // Calculate memory utilization
//             size_t totalMemory = devProp.totalGlobalMem;
//             size_t freeMemory;
//             size_t usedMemory;
//             float workload_size = (float) G.numEdges*sizeof(uint);
//             cudaMemGetInfo(&freeMemory, &totalMemory);
//             usedMemory = totalMemory - freeMemory;
//             printf("Total GPU Memory: %.2f MiB\n", (float) totalMemory / (1024 * 1024));
//             printf("Free GPU Memory: %.2f MiB\n", (float) freeMemory / (1024 * 1024));
//             printf("Used GPU Memory: %.2f MiB\n", (float) usedMemory / (1024 * 1024));

//             printf("Workload size: %.2f\n", workload_size/1024/1024);
//             float oversubs_ratio = 0;
//             void *tmp_ptr;
//             cudaMalloc(&tmp_ptr, (size_t) (freeMemory - workload_size));
//             cudaMemGetInfo(&freeMemory, &totalMemory);
//             printf("Free GPU Memory: %.2f MiB\n", (float) freeMemory / (1024 * 1024));
//             if(oversubs_ratio > 0){
//                 void *over_ptr;
//                 long long unsigned int os_size = freeMemory - workload_size /(1 + oversubs_ratio);
//                 printf("workload: %.2f\n",  workload_size);
//                 printf("workload: %llu\n",  os_size);
//                 cudaMalloc(&over_ptr, os_size); 
//                 printf("os_size: %u\n", os_size/1024/1024);
//             }
//             cudaMemGetInfo(&freeMemory, &totalMemory);
//             printf("Free GPU Memory: %.2f MiB\n", (float) freeMemory / (1024 * 1024));
//             // break;
//     }
    
//     // checkError(cudaMemcpy(d_adjacencyList, u_adjacencyList, sizeof(rdma_buf<unsigned int>), cudaMemcpyHostToDevice));

//     checkError(cudaMalloc((void **) &d_distance, G.numVertices*sizeof(unsigned int)));
//     uint *h_distance = new uint[G.numVertices];
//     for (size_t i = 0; i < G.numVertices; i++)
//     {
//         h_distance[i] = 100000;
//     }
//     h_distance[startVertex] = 0;
//     checkError(cudaMemcpy(d_distance, h_distance, G.numVertices*sizeof(unsigned int), cudaMemcpyHostToDevice));
//     // checkError(cudaMemset(d_distance, 100000 /*std::numeric_limits<int>::max()*/, G.numVertices*sizeof(unsigned int)));
//     // checkError(cudaMemset(&d_distance[startVertex], 0, sizeof(unsigned int)));

//     checkError(cudaMalloc((void **) &d_edgesSize, G.numVertices*sizeof(unsigned int)));
//     checkError(cudaMemcpy(d_edgesSize, u_edgesSize, G.numVertices*sizeof(unsigned int), cudaMemcpyHostToDevice));

//     checkError(cudaMalloc((void **) &d_edgesOffset, (G.numVertices+1)*sizeof(uint64_t)));

//     uint64_t *u_edgesOffset_64 = (uint64_t *) malloc(sizeof(uint64_t)*(G.numVertices+1)); 
//     for (size_t i = 0; i < G.numVertices+1; i++)
//     {
//        u_edgesOffset_64[i] = u_edgesOffset[i];
//     }
    
//     checkError(cudaMemcpy(d_edgesOffset, u_edgesOffset_64, (G.numVertices+1)*sizeof(uint64_t), cudaMemcpyHostToDevice));

//     printf("fuction: %s, line: %d\n", __func__, __LINE__);

//     auto start = std::chrono::steady_clock::now();
//     // cudaEventRecord(event1, (cudaStream_t)1);
    
//     //launch kernel
//     printf("Starting simple parallel bfs.\n");
//     cudaError_t ret1 = cudaDeviceSynchronize();
//     if(cudaSuccess != ret1){  
//         printf("cudaDeviceSynchronize error: %d\n", ret1);  
//         exit(-1);
//     }

//     ret1 = cudaDeviceSynchronize();
//     // ret1 = cudaDeviceSynchronize();
//     // cudaEventRecord(event2, (cudaStream_t) 1);
//     auto end = std::chrono::steady_clock::now();

//     // cudaEventSynchronize(event1); //optional
//     // cudaEventSynchronize(event2); //wait for the event to be executed!

//     float dt_ms;
//     // cudaEventElapsedTime(&dt_ms, event1, event2);
//     // printf("Elapsed time for transfer: %f\n", dt_ms);
//     long duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
//     printf("Elapsed time in milliseconds for transfer : %li ms.\n", duration);

//     uint64_t numblocks, numthreads;
//     numthreads = BLOCK_SIZE;
//     int type = 0;
//     uint64_t vertex_count = G.numVertices;
//     switch (type) {
//         case 0:
//             numblocks = ((vertex_count + numthreads) / numthreads);
//             break;
//         case 1:
//             numblocks = ((vertex_count * WARP_SIZE + numthreads) / numthreads);
//             break;
//         case 2:
//             numblocks = ((vertex_count * (WARP_SIZE / CHUNK_SIZE) + numthreads) / numthreads);
//             break;
//         default:
//             fprintf(stderr, "Invalid type\n");
//             exit(1);
//             break;
//     }
//     dim3 blockDim(BLOCK_SIZE, (numblocks+BLOCK_SIZE)/BLOCK_SIZE);

//     unsigned int level;
//     level = 0;
//     printf("fuction: %s, line: %d\n", __func__, __LINE__);
    
    

//     bool *changed_d, changed_h;
//     // checkError(cudaMallocHost((void **) &changed_h, sizeof(bool)));
//     checkError(cudaMalloc((void **) &changed_d, sizeof(bool)));

//     ret1 = cudaDeviceSynchronize();
//     cudaEvent_t event1, event2;
//     cudaEventCreate(&event1);
//     cudaEventCreate(&event2);
//     start = std::chrono::steady_clock::now();
//     cudaEventRecord(event1,  (cudaStream_t) 1);
//     changed_h = true;

//     do {
//             changed_h = false;
//             checkError(cudaMemcpy(changed_d, &changed_h, sizeof(bool), cudaMemcpyHostToDevice));

//             switch (type) {
//                 case 0:
//                     kernel_baseline<<<blockDim, numthreads>>>
//                     (d_distance, level, vertex_count, d_edgesOffset, d_adjacencyList, changed_d);
//                     break;
//                 case 1:
//                     kernel_coalesce<<<blockDim, numthreads>>>
//                     (d_distance, level, vertex_count, d_edgesOffset, d_adjacencyList, changed_d);
//                     break;
//                 case 2:
//                     kernel_coalesce_chunk<<<blockDim, numthreads>>>
//                     (d_distance, level, vertex_count, d_edgesOffset, d_adjacencyList, changed_d);
//                     break;
//                 default:
//                     fprintf(stderr, "Invalid type\n");
//                     exit(1);
//                     break;
//             }
//             // printf("cudaDeviceSynchronize(): %d\n", cudaDeviceSynchronize());
//             cudaDeviceSynchronize();
//             printf("level: %d\n", level);
//             level++;

//             checkError(cudaMemcpy(&changed_h, changed_d, sizeof(bool), cudaMemcpyDeviceToHost));
//         } while(changed_h);

//     cudaEventRecord(event2,  (cudaStream_t) 1);
//     cudaEventSynchronize(event1); //optional
//     cudaEventSynchronize(event2); //wait for the event to be executed!
//     cudaEventElapsedTime(&dt_ms, event1, event2);
//     printf("cudaGetLastError(): %d level: %d\n", cudaGetLastError(), level);
//         ret1 = cudaDeviceSynchronize();
//         printf("cudaDeviceSynchronize: %d *changed: %d\n", ret1, changed_h);  
//         if(cudaSuccess != ret1){  
//             printf("cudaDeviceSynchronize error: %d\n", ret1);  
//             exit(-1);
//         }

//     // calculate time
//     // float dt_ms;
    
//     printf("Elapsed time with cudaEvent: %f\n", dt_ms);

//     end = std::chrono::steady_clock::now();
//     duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
//     printf("Elapsed time in milliseconds : %li ms.\n", duration);

//     checkError(cudaMemcpy(h_distance, d_distance, G.numVertices*sizeof(uint), cudaMemcpyDeviceToHost));
//     return h_distance;
// }

// __device__ size_t sum_page_faults = 0;

// __global__ void
// print_retires(void){
//     // size_t max = cq_wait[0];
//     // for (size_t i = 0; i < 128; i++)
//     // {
//     //     if(max < cq_wait[i]) max = cq_wait[i];
//     // }
    
//     // printf("g_qp_index: %llu cq_wait: %llu\n", g_qp_index, max);
//     // g_qp_index = 0;
//     // for (size_t i = 0; i < 128; i++)
//     // {
//     //     max = 0;
//     // }
//     sum_page_faults += g_qp_index;
//     printf("g_qp_index: %llu sum page fault: %llu\n", g_qp_index, sum_page_faults);
//     // g_qp_index = 0;
// }

// void runCudaSimpleBfs_optimized(int startVertex, Graph &G, std::vector<int> &distance,
//                       std::vector<int> &parent, bool rdma, bool uvm) 
// {
//     // initializeCudaBfs(startVertex, distance, parent, G);

//     printf("fuction: %s, line: %d\n", __func__, __LINE__);

//     unsigned int *d_distance, *d_edgesSize, *d_edgesOffset, *d_adjacencyList, *d_startVertices, *d_vertexList;
//     // rdma_buf<unsigned int> *d_adjacencyList;
//     // checkError(cudaMalloc((void **) &d_adjacencyList, sizeof(rdma_buf<unsigned int>)));

//     // checkError(cudaMalloc((void **) &d_startVertices, G.numEdges*sizeof(unsigned int)));
//     // checkError(cudaMemcpy(d_startVertices, u_startVertices, G.numEdges*sizeof(unsigned int), cudaMemcpyHostToDevice));

//     if(!rdma)
//     checkError(cudaMalloc((void **) &d_adjacencyList, G.numEdges*sizeof(unsigned int)));
    
//     // checkError(cudaMemcpy(d_adjacencyList, u_adjacencyList, sizeof(rdma_buf<unsigned int>), cudaMemcpyHostToDevice));

//     checkError(cudaMalloc((void **) &d_distance, G.numVertices*sizeof(unsigned int)));
//     uint *h_distance = new uint[G.numVertices];
//     for (size_t i = 0; i < G.numVertices; i++)
//     {
//         h_distance[i] = 100000;
//     }
//     h_distance[startVertex] = 0;
//     checkError(cudaMemcpy(d_distance, h_distance, G.numVertices*sizeof(unsigned int), cudaMemcpyHostToDevice));
//     free(h_distance);
//     // checkError(cudaMemcpy(d_distance, u_distance, G.numVertices*sizeof(unsigned int), cudaMemcpyHostToDevice));
//     // checkError(cudaMemset(d_distance, 100000, G.numVertices*sizeof(unsigned int)));
//     // checkError(cudaMemset(&d_distance[startVertex], 0, 1*sizeof(unsigned int)));

//     checkError(cudaMalloc((void **) &d_edgesSize, G.numVertices*sizeof(unsigned int)));
//     // checkError(cudaMemcpy(d_edgesSize, u_edgesSize, G.numVertices*sizeof(unsigned int), cudaMemcpyHostToDevice));

//     checkError(cudaMalloc((void **) &d_edgesOffset, (G.numVertices+1)*sizeof(unsigned int)));
//     checkError(cudaMemcpy(d_edgesOffset, u_edgesOffset, (G.numVertices+1)*sizeof(unsigned int), cudaMemcpyHostToDevice));

//     // checkError(cudaMalloc((void **) &d_vertexList, (G.numVertices + 1)*sizeof(unsigned int)));
//     // checkError(cudaMemcpy(d_vertexList, u_edgesOffset, G.numVertices*sizeof(unsigned int), cudaMemcpyHostToDevice));
    
//     unsigned int changed_h, *d_changed;
//     checkError(cudaMalloc((void **) &d_changed, sizeof(unsigned int)));

//     printf("fuction: %s, line: %d\n", __func__, __LINE__);

//     //launch kernel
//     printf("Starting simple parallel bfs.\n");
//     cudaError_t ret1 = cudaDeviceSynchronize();
//     if(cudaSuccess != ret1){  
//         printf("cudaDeviceSynchronize error: %d\n", ret1);  
//         exit(-1);
//     }
    
    
//     cudaEvent_t event1, event2;
//     cudaEventCreate(&event1);
//     cudaEventCreate(&event2);

//     uint64_t threadsPerBlock = 256;
//     uint64_t numblocks = ((G.numVertices + threadsPerBlock - 1) / threadsPerBlock);
//     size_t sharedMemorySize = threadsPerBlock * sizeof(unsigned int);

//     ret1 = cudaDeviceSynchronize();
//     auto start = std::chrono::steady_clock::now();
//     cudaEventRecord(event1, (cudaStream_t)1);
//     if(!rdma)
//         checkError(cudaMemcpy(d_adjacencyList, u_adjacencyList, G.numEdges*sizeof(unsigned int), cudaMemcpyHostToDevice));
//     // ret1 = cudaDeviceSynchronize();
//     cudaEventRecord(event2, (cudaStream_t) 1);
//     auto end = std::chrono::steady_clock::now();

//     cudaEventSynchronize(event1); //optional
//     cudaEventSynchronize(event2); //wait for the event to be executed!

//     float dt_ms;
//     cudaEventElapsedTime(&dt_ms, event1, event2);
//     printf("Elapsed time for transfer: %f\n", dt_ms);
//     long duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
//     printf("Elapsed time in milliseconds for transfer : %li ms.\n", duration);

//     unsigned int level;
//     level = 0;
//     printf("fuction: %s, line: %d\n", __func__, __LINE__);

//     cudaDeviceProp devProp;
//     cudaGetDeviceProperties(&devProp, 0);
//     printf("Cuda device clock rate = %d\n", devProp.clockRate);

//     ret1 = cudaDeviceSynchronize();
//     cudaEventRecord(event1, (cudaStream_t)1);

//     if(uvm){
//         cudaDeviceProp devProp;
//         cudaGetDeviceProperties(&devProp, 0);
//         // Calculate memory utilization
//         size_t totalMemory = devProp.totalGlobalMem;
//         size_t freeMemory;
//         size_t usedMemory;
//         float workload_size = ((float) G.numEdges*sizeof(uint));
//         cudaMemGetInfo(&freeMemory, &totalMemory);
//         usedMemory = totalMemory - freeMemory;
//         printf("Total GPU Memory: %.2f MiB\n", (float) totalMemory / (1024 * 1024));
//         printf("Free GPU Memory: %.2f MiB\n", (float) freeMemory / (1024 * 1024));
//         printf("Used GPU Memory: %.2f MiB\n", (float) usedMemory / (1024 * 1024));

//         printf("Workload size: %.2f\n", workload_size/1024/1024);
//         float oversubs_ratio = 0;
//         void *tmp_ptr;
//         cudaMalloc(&tmp_ptr, (size_t) (freeMemory - workload_size));
//         cudaMemGetInfo(&freeMemory, &totalMemory);
//         printf("Free GPU Memory: %.2f MiB\n", (float) freeMemory / (1024 * 1024));
//         if(oversubs_ratio > 0){
//             void *over_ptr;
//             long long unsigned int os_size = freeMemory - workload_size /(1 + oversubs_ratio);
//             printf("workload: %.2f\n",  workload_size);
//             printf("workload: %llu\n",  os_size);
//             cudaMalloc(&over_ptr, os_size); 
//             printf("os_size: %u\n", os_size/1024/1024);
//         }
//         cudaMemGetInfo(&freeMemory, &totalMemory);
//         printf("Free GPU Memory: %.2f MiB\n", (float) freeMemory / (1024 * 1024));
//     }

//     uint64_t numthreads = BLOCK_SIZE;
//     numblocks = ((G.numVertices * (WARP_SIZE / CHUNK_SIZE) + numthreads) / numthreads);
//     dim3 blockDim(BLOCK_SIZE, (numblocks+BLOCK_SIZE)/BLOCK_SIZE);
//     start = std::chrono::steady_clock::now();
//     changed_h = 1;
//     while (changed_h) {
//         changed_h = 0;
//         checkError(cudaMemcpy(d_changed, &changed_h, sizeof(unsigned int), cudaMemcpyHostToDevice));
//         auto start1 = std::chrono::steady_clock::now();
//         if(!rdma){
//             // kernel_coalesce_ptr_pc<<< blockDim1, numthreads>>>(d_adjacencyList, d_distance, level, (const uint64_t) G.numVertices, d_edgesOffset,
//             //                     d_edgesSize, d_adjacencyList, changed);
            
//             // simpleBfs_normal<<< G.numVertices / 1024 + 1, 1024 >>>(G.numEdges, G.numVertices, level, uvm_adjacencyList /* d_adjacencyList u_adjacencyList */, d_startVertices, d_distance, changed);
//             // simpleBfs_uvm<<<G.numVertices / 1024 + 1, 1024>>>(G.numVertices, level, d_adjacencyList, d_edgesOffset, d_edgesSize, d_distance, changed); 
            
//             simpleBfs_modVertexList<<<G.numVertices / 1024 + 1, 1024>>>(G.numVertices, level, d_adjacencyList, d_edgesOffset, d_distance, d_changed); 
//             // kernel_coalesce_chunk<<<blockDim, numthreads>>>(d_distance, level, G.numVertices, d_edgesOffset,
//             //                       d_adjacencyList, d_changed);
            
//             // kernel_coalesce(unsigned int *d_distance, unsigned int level, size_t n, 
//             //                     unsigned int *vertexList, unsigned int *d_adjacencyList, unsigned int *changed) 
        
//         }
//         else{
//             // kernel_coalesce_ptr_pc_rdma<<< blockDim1, numthreads>>>(rdma_adjacencyList, d_distance, level, (const uint64_t) G.numVertices, d_edgesOffset,
//             //                     d_edgesSize, changed);
//             // simpleBfs_normal_rdma<<< /*G.numVertices / 512 + 1, 512*/ 2048, 512 >>>(G.numEdges, G.numVertices, level, rdma_adjacencyList, d_startVertices, d_distance, changed);
//             // simpleBfs_rdma<<< /*G.numVertices / 256 + 1, 512*/ 2048, 512 >>>(G.numVertices, level, rdma_adjacencyList, d_edgesOffset, d_edgesSize, d_distance, changed); 
//             // simpleBfs_normal_rdma_optimized2<<<1024, 512>>>(G.numEdges, G.numVertices, level, rdma_adjacencyList, d_startVertices, d_distance, changed);
            

//             // simpleBfs_rdma_optimized_dynamic<<< numblocks, threadsPerBlock, sharedMemorySize >>>(G.numVertices, level, rdma_adjacencyList, d_edgesOffset, d_edgesSize, d_distance, changed);

//             // check_edgeList<<< G.numEdges/512+1, 512>>>
//             // (rdma_adjacencyList, uvm_adjacencyList, G.numEdges);

//             // the next one for friendster:
//             size_t n_pages = G.numVertices*sizeof(unsigned int)/(4*1024)+1;
//             // if(level >= 2)
//                 simpleBfs_rdma_optimized_warp<<</*G.numVertices / 256 + 1, 256*/(n_pages*32)/512 + 1, 512>>>(
//                         n_pages, G.numVertices, level, rdma_adjacencyList, d_edgesOffset, d_edgesSize, d_distance, d_changed);
//             // else
//                 // simpleBfs_modVertexList_rdma<<< (G.numVertices*32)/512 + 1, 512>>>(
//                 //         G.numVertices, level, rdma_adjacencyList, d_edgesOffset, d_distance, d_changed); 

//                 // simpleBfs_rdma<<< G.numVertices/512 + 1, 512>>>
//                 // (G.numVertices, level, rdma_adjacencyList, d_edgesOffset, d_edgesSize, d_distance, d_changed);


//             // simpleBfs_rdma_optimized_thread_different_page<<</*G.numVertices / 256 + 1, 256*/1024, 512>>>
//             // (n_pages,  G.numVertices, level, rdma_adjacencyList, d_edgesOffset, d_edgesSize, d_distance, changed);
            
//             // test_batch <<< 2048*2, 512 >>> (rdma_adjacencyList, d_adjacencyList, rdma_adjacencyList->size/sizeof(unsigned int));

//             // kernel_coalesce_chunk_rdma<<<blockDim, numthreads>>>
//             // (d_distance, level, G.numVertices, d_edgesOffset, rdma_adjacencyList, d_changed);

//             // simpleBfs_rdma_optimized_warp2<<</*G.numVertices / 256 + 1, 256*/1024, 512>>>(G.numVertices, level, rdma_adjacencyList, d_edgesOffset, d_edgesSize, d_distance, changed);

//             // simpleBfs_rdma_dynamic_page<<<G.numVertices / 512 + 1, 512>>>(G.numVertices, level, rdma_adjacencyList, d_edgesOffset, d_edgesSize, d_distance, changed);
        
//             // print_retires<<<1,1>>>();
//             printf("cudaGetLastError(): %d level: %d\n", cudaGetLastError(), level);
            
//         }
//         ret1 = cudaDeviceSynchronize();
//         auto end1 = std::chrono::steady_clock::now();
//         long duration1 = std::chrono::duration_cast<std::chrono::milliseconds>(end1 - start1).count();
//         printf("Elapsed time in milliseconds : %li ms.\n\n", duration);
        
//         checkError(cudaMemcpy(&changed_h, d_changed, sizeof(unsigned int), cudaMemcpyDeviceToHost));
//         level++;
//     }
//     cudaEventRecord(event2, (cudaStream_t) 1);
//     printf("cudaGetLastError(): %d level: %d\n", cudaGetLastError(), level);
//         ret1 = cudaDeviceSynchronize();
//         printf("cudaDeviceSynchronize: %d *changed: %d\n", ret1, changed_h);  
//         if(cudaSuccess != ret1){  
//             printf("cudaDeviceSynchronize error: %d\n", ret1);  
//             exit(-1);
//         }


    
//     cudaEventSynchronize(event1); //optional
//     cudaEventSynchronize(event2); //wait for the event to be executed!
    

//     // calculate time
//     // float dt_ms;
//     cudaEventElapsedTime(&dt_ms, event1, event2);
//     printf("Elapsed time with cudaEvent: %f\n", dt_ms);

//     end = std::chrono::steady_clock::now();
//     duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
//     printf("Elapsed time in milliseconds : %li ms.\n", duration);

//     checkError(cudaMemcpy(u_distance, d_distance, G.numVertices*sizeof(unsigned int), cudaMemcpyDeviceToHost));

//     finalizeCudaBfs(distance, parent, G);
// }


// __global__ void test(rdma_buf<unsigned int> *a/*, rdma_buf<int> *b, rdma_buf<int> *c*/){
//     size_t id = blockDim.x * blockIdx.x + threadIdx.x;
//     int k = (*a)[id]; // + (*b)[id];
// }

// int alloc_global_cont(struct post_content *post_cont, struct poll_content *poll_cont, struct server_content_2nic *post_cont2, 
//                       struct gpu_memory_info gpu_mem){
//     struct post_content *d_post;
//     struct poll_content *d_poll;
//     struct server_content_2nic *d_post2;

//     cudaError_t ret0 = cudaMalloc((void **)&d_post, sizeof(struct post_content));
//     if(ret0 != cudaSuccess){
//         printf("Error on allocation post content!\n");
//         return -1;
//     }
//     ret0 = cudaMalloc((void **)&d_poll, sizeof(struct poll_content));
//     if(ret0 != cudaSuccess){
//         printf("Error on allocation poll content!\n");
//         return -1;
//     }
//     printf("sizeof(struct post_content): %d, sizeof(struct poll_content): %d\n", sizeof(struct post_content), sizeof(struct poll_content));
//     ret0 = cudaMemcpy(d_post, post_cont, sizeof(struct post_content), cudaMemcpyHostToDevice);
//     if(ret0 != cudaSuccess){
//         printf("Error on post copy!\n");
//         return -1;
//     }
//     ret0 = cudaMemcpy(d_poll, poll_cont, sizeof(struct poll_content), cudaMemcpyHostToDevice);
//     if(ret0 != cudaSuccess){
//         printf("Error on poll copy!\n");
//         return -1;
//     }

//     ret0 = cudaMalloc((void **)&d_post2, sizeof(struct server_content_2nic));
//     if(ret0 != cudaSuccess){
//         printf("Error on allocation post content!\n");
//         return -1;
//     }
//     ret0 = cudaMemcpy(d_post2, post_cont2, sizeof(struct server_content_2nic), cudaMemcpyHostToDevice);
//     if(ret0 != cudaSuccess){
//         printf("Error on poll copy!\n");
//         return -1;
//     }

//     // cudaSetDevice(0);
//     alloc_content<<<1,1>>>(d_post, d_poll);
//     alloc_global_content<<<1,1>>>(d_post, d_poll, d_post2, gpu_mem);
//     ret0 = cudaDeviceSynchronize();
//     if(ret0 != cudaSuccess){
//         printf("Error on alloc_content!\n");
//         return -1;
//     }
//     return 0;
// }


// __global__
// void print_utilization() {
//     printf("GPU_address_offset: %llu \n", GPU_address_offset);
// }

// __global__
// void print_transferTime(void) {
//     printf("transfer time: %llu \n", transfer_time);
// }

// float time_total = 0;
// float time_readmostly_total = 0;
// uint *runRDMA(int startVertex, Graph &G, bool rdma, unsigned int *new_vertex_list, unsigned int *new_offset, unsigned int new_size,
//              unsigned int *u_adjacencyList, unsigned int *u_edgesOffset, int u_case) 
// {
//     printf("fuction: %s, line: %d\n", __func__, __LINE__);
//     // u_case: 0: direct transfer; 1: rdma edgelist rep.; 2: new rep.

//     unsigned int *d_distance, *d_edgesSize, *d_edgesOffset, *d_adjacencyList, *d_startVertices, *d_vertexList;
//     unsigned int *d_new_vertexList, *d_new_offset;
//     unsigned int *return_distance;
//     // checkError(cudaMemcpy(d_adjacencyList, u_adjacencyList, sizeof(rdma_buf<unsigned int>), cudaMemcpyHostToDevice));
//     cudaSetDevice(GPU);
//     return_distance = new uint[G.numVertices];
//     checkError(cudaMalloc((void **) &d_distance, G.numVertices*sizeof(unsigned int)));
//     uint *h_distance = new uint[G.numVertices];
//     for (size_t i = 0; i < G.numVertices; i++)
//     {
//         h_distance[i] = 100000;
//     }
//     h_distance[startVertex] = 0;
//     checkError(cudaMemcpy(d_distance, h_distance, G.numVertices*sizeof(unsigned int), cudaMemcpyHostToDevice));
//     free(h_distance);
//     checkError(cudaMalloc((void **) &d_edgesSize, G.numVertices*sizeof(unsigned int)));
//     checkError(cudaMalloc((void **) &d_edgesOffset, (G.numVertices+1)*sizeof(unsigned int)));
//     checkError(cudaMemcpy(d_edgesOffset, u_edgesOffset, (G.numVertices+1)*sizeof(unsigned int), cudaMemcpyHostToDevice));

//     if(u_case == 2 || u_case == 3 || u_case == 4){
//         checkError(cudaMalloc((void **) &d_new_vertexList, new_size*sizeof(unsigned int)));
//         checkError(cudaMemcpy(d_new_vertexList, new_vertex_list, new_size*sizeof(unsigned int), cudaMemcpyHostToDevice));

//         checkError(cudaMalloc((void **) &d_new_offset, (new_size+1)*sizeof(unsigned int)));
//         checkError(cudaMemcpy(d_new_offset, new_offset, (new_size+1)*sizeof(unsigned int), cudaMemcpyHostToDevice));
//     }
    
//     unsigned int changed_h, *d_changed;
//     checkError(cudaMalloc((void **) &d_changed, sizeof(unsigned int)));

//     printf("fuction: %s, line: %d\n", __func__, __LINE__);

//     //launch kernel
//     printf("Starting simple parallel bfs.\n");
//     cudaError_t ret1 = cudaDeviceSynchronize();
//     if(cudaSuccess != ret1){  
//         printf("cudaDeviceSynchronize error: %d\n", ret1);  
//         exit(-1);
//     }
    
    
//     cudaEvent_t event1, event2;
//     cudaEventCreate(&event1);
//     cudaEventCreate(&event2);

//     uint64_t threadsPerBlock = 256;
//     uint64_t numblocks = ((G.numVertices + threadsPerBlock - 1) / threadsPerBlock);
//     size_t sharedMemorySize = threadsPerBlock * sizeof(unsigned int);

//     ret1 = cudaDeviceSynchronize();
//     if(u_case == 0 || u_case == 3){
//         checkError(cudaMalloc((void **) &d_adjacencyList, G.numEdges*sizeof(unsigned int)));

//         ret1 = cudaDeviceSynchronize();
//         auto start = std::chrono::steady_clock::now();
//         cudaEventRecord(event1, (cudaStream_t)1);
        
//         checkError(cudaMemcpy(d_adjacencyList, u_adjacencyList, G.numEdges*sizeof(unsigned int), cudaMemcpyHostToDevice));
//         // ret1 = cudaDeviceSynchronize();
//         cudaEventRecord(event2, (cudaStream_t) 1);
//         auto end = std::chrono::steady_clock::now();

//         cudaEventSynchronize(event1); //optional
//         cudaEventSynchronize(event2); //wait for the event to be executed!
//         float dt_ms;
//         cudaEventElapsedTime(&dt_ms, event1, event2);
//         printf("Elapsed time for transfer: %f\n", dt_ms);
//         long duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
//         printf("Elapsed time in milliseconds for transfer : %li ms. data: %f\n", duration, (double) G.numEdges*sizeof(unsigned int)/(1024*1024*1024llu));
//         // exit(0);
//     }

//     auto start1 = std::chrono::steady_clock::now();
//     // checkError(cudaMemAdvise(uvm_adjacencyList, G.numEdges*sizeof(unsigned int), cudaMemAdviseSetAccessedBy, 0));
//     auto end1 = std::chrono::steady_clock::now();
//     long duration1 = std::chrono::duration_cast<std::chrono::milliseconds>(end1 - start1).count();
//     printf("Elapsed time for memAdvise SetAccessedBy in milliseconds : %li ms.\n\n", duration1);

//     unsigned int level;
//     level = 0;
//     printf("fuction: %s, line: %d\n", __func__, __LINE__);

    
//     if(u_case == 6){
//         ret1 = cudaDeviceSynchronize();
//         auto start2 = std::chrono::steady_clock::now();
//         checkError(cudaMemAdvise(uvm_adjacencyList, G.numEdges*sizeof(uint), cudaMemAdviseSetReadMostly, 0));
//         ret1 = cudaDeviceSynchronize();
//         auto end2 = std::chrono::steady_clock::now();
//         long duration2 = std::chrono::duration_cast<std::chrono::milliseconds>(end2 - start2).count();
//         printf("Elapsed time for memAdvise SetAccessedBy in milliseconds : %li ms.\n\n", duration2);

//         cudaDeviceProp devProp;
//         cudaGetDeviceProperties(&devProp, 0);
//         // Calculate memory utilization
//         size_t totalMemory = devProp.totalGlobalMem;
//         size_t freeMemory;
//         size_t usedMemory;
//         float workload_size = (float) G.numEdges*sizeof(uint);
//         cudaMemGetInfo(&freeMemory, &totalMemory);
//         usedMemory = totalMemory - freeMemory;
//         printf("Total GPU Memory: %.2f MiB\n", (float) totalMemory / (1024 * 1024));
//         printf("Free GPU Memory: %.2f MiB\n", (float) freeMemory / (1024 * 1024));
//         printf("Used GPU Memory: %.2f MiB\n", (float) usedMemory / (1024 * 1024));

//         printf("Workload size: %.2f\n", workload_size/1024/1024);
//         float oversubs_ratio = 0;
//         void *tmp_ptr;
//         // cudaMalloc(&tmp_ptr, (size_t) (freeMemory - workload_size));
//         cudaMemGetInfo(&freeMemory, &totalMemory);
//         printf("Free GPU Memory: %.2f MiB\n", (float) freeMemory / (1024 * 1024));
        
//         if(oversubs_ratio > 0){
//             void *over_ptr;
//             long long unsigned int os_size = freeMemory - workload_size /(1 + oversubs_ratio);
//             printf("workload: %.2f\n",  workload_size);
//             printf("workload: %llu\n",  os_size);
//             cudaMalloc(&over_ptr, os_size); 
//             printf("os_size: %u\n", os_size/1024/1024);
//         }

//         cudaMemGetInfo(&freeMemory, &totalMemory);
//         printf("Free GPU Memory: %.2f MiB\n", (float) freeMemory / (1024 * 1024));
//     }

//     cudaDeviceProp devProp;
//     cudaGetDeviceProperties(&devProp, 0);
//     printf("Cuda device clock rate = %d\n", devProp.clockRate);

    

//     uint64_t numthreads = BLOCK_SIZE;
//     numblocks = ((G.numVertices * (WARP_SIZE / CHUNK_SIZE) + numthreads) / numthreads);
//     dim3 blockDim(BLOCK_SIZE, (numblocks+BLOCK_SIZE)/BLOCK_SIZE);
//     delay(10);
//     // transfer<<<2048, 256>>>(rdma_adjacencyList->size/sizeof(unsigned int), rdma_adjacencyList, NULL);
//     ret1 = cudaDeviceSynchronize();

//     auto start = std::chrono::steady_clock::now();
//     changed_h = 1;
//     ret1 = cudaDeviceSynchronize();
//     cudaEventRecord(event1, (cudaStream_t)1);

    
        
//     cudaSetDevice(GPU);
//     while (changed_h) {
//         changed_h = 0;
//         checkError(cudaMemcpy(d_changed, &changed_h, sizeof(unsigned int), cudaMemcpyHostToDevice));
//         auto start = std::chrono::steady_clock::now();
//         switch (u_case)
//         {

//         case 0:{
//             printf("direct transfer\n");
//             simpleBfs_modVertexList<<<G.numVertices / 1024 + 1, 1024>>>
//             (G.numVertices, level, d_adjacencyList, d_edgesOffset, d_distance, d_changed); 
//             ret1 = cudaDeviceSynchronize();
//             break;
//         }
//         case 1:{
//             printf("rdma edgelist representation\n");
//             size_t n_pages = G.numVertices*sizeof(unsigned int)/(4*1024)+1;
//             numthreads = 1024;
//             simpleBfs_rdma<<< (G.numVertices / numthreads)*(1 << WARP_SHIFT) + 1, numthreads >>>(G.numVertices, level, rdma_adjacencyList, d_edgesOffset,
//                     d_edgesSize, d_distance, d_changed);
//             // simpleBfs_rdma_optimized_warp<<</*G.numVertices / 256 + 1, 256*/(n_pages*32)/384 + 1, 384>>>(
//             //             n_pages, G.numVertices, level, rdma_adjacencyList, d_edgesOffset, d_edgesSize, d_distance, d_changed); 
//             ret1 = cudaDeviceSynchronize();
//             break;
//         }
//         case 2: {// new representation{
//             printf("new representation new_size: %d\n", new_size);
//             size_t n_pages = new_size*sizeof(unsigned int)/(8*1024)+1;
//             kernel_coalesce_new_repr_rdma<<<(n_pages*32)/256 + 1, 256>>>
//             (level, n_pages, G.numVertices, new_size, d_distance, d_new_offset, d_new_vertexList, rdma_adjacencyList,
//             d_changed);
//             ret1 = cudaDeviceSynchronize();
//             break;
//         }
//         case 3:{
//             printf("direct new representation\n");
//             size_t n_pages = new_size*sizeof(unsigned int)/(4*1024)+1;
//             kernel_coalesce_new_repr<<<(n_pages*32)/512 + 1, 512>>>
//             (level, n_pages, G.numVertices, new_size, d_distance, d_new_offset, d_new_vertexList, d_adjacencyList,
//             d_changed);
//             ret1 = cudaDeviceSynchronize();
//             break;
//         }
//         case 4: {
//             printf("uvm new representation\n");
//             size_t n_pages = new_size*sizeof(unsigned int)/(4*1024)+1;
//             kernel_coalesce_new_repr<<<(n_pages*32)/512 + 1, 512>>>
//             (level, n_pages, G.numVertices, new_size, d_distance, d_new_offset, d_new_vertexList, uvm_adjacencyList,
//             d_changed);
//             ret1 = cudaDeviceSynchronize();
//             break;
//         }
//         case 5:{
//             printf("uvm transfer edge list\n");
//             // simpleBfs_modVertexList<<<G.numVertices / 1024 + 1, 1024>>>
//             // (G.numVertices, level, uvm_adjacencyList, d_edgesOffset, d_distance, d_changed); 
//             numthreads = BLOCK_SIZE;
//             numblocks = ((G.numVertices * (WARP_SIZE / CHUNK_SIZE) + numthreads) / numthreads);
//             dim3 blockDim(BLOCK_SIZE, (numblocks+BLOCK_SIZE)/BLOCK_SIZE);
//             bfs_kernel_coalesce_chunk<<<blockDim, numthreads>>>
//             (d_distance, level, G.numVertices, d_edgesOffset, uvm_adjacencyList, d_changed);
//             ret1 = cudaDeviceSynchronize();
//             break;
//         }
//         case 6:{
//             printf("uvm transfer\n");
//             simpleBfs_modVertexList<<<G.numVertices / 1024 + 1, 1024>>>
//             (G.numVertices, level, /*d_adjacencyList*/uvm_adjacencyList, d_edgesOffset, d_distance, d_changed); 
//             ret1 = cudaDeviceSynchronize();
//             break;
//         }
//         default:
//             break;
//         }
//         if(u_case == 1 || u_case == 2){
//             print_retires<<<1,1>>>();
//             ret1 = cudaDeviceSynchronize();
//         }
        
//         printf("ret1: %d cudaGetLastError(): %d level: %d\n", ret1, cudaGetLastError(), level);
//         auto end = std::chrono::steady_clock::now();
//         long duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
//         printf("Elapsed time in milliseconds : %li ms.\n\n", duration);
        
//         checkError(cudaMemcpy(&changed_h, d_changed, sizeof(unsigned int), cudaMemcpyDeviceToHost));
//         level++;
//     }
//     cudaEventRecord(event2, (cudaStream_t) 1);
//     printf("cudaGetLastError(): %d level: %d\n", cudaGetLastError(), level);
//         ret1 = cudaDeviceSynchronize();
//         printf("cudaDeviceSynchronize: %d *changed: %d\n", ret1, changed_h);  
//         if(cudaSuccess != ret1){  
//             printf("cudaDeviceSynchronize error: %d\n", ret1);  
//             exit(-1);
//         }


    
//     cudaEventSynchronize(event1); //optional
//     cudaEventSynchronize(event2); //wait for the event to be executed!
    

//     // calculate time
//     float dt_ms;
//     cudaEventElapsedTime(&dt_ms, event1, event2);
//     printf("Elapsed time with cudaEvent: %f\n", dt_ms);

//     auto end = std::chrono::steady_clock::now();
//     long duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
//     printf("Elapsed time in milliseconds : %li ms.\n", duration);

//     time_total += dt_ms;
//     checkError(cudaMemcpy(return_distance, d_distance, G.numVertices*sizeof(unsigned int), cudaMemcpyDeviceToHost));

//     if(u_case == 2 || u_case == 3 || u_case == 4){
//         checkError(cudaFree(d_new_offset));
//         checkError(cudaFree(d_new_vertexList));
//     }

//     if(u_case == 0 || u_case == 3){ // direct transfer
//         checkError(cudaFree(d_adjacencyList));
//     }
//     if(u_case == 4 || u_case == 5){
//         cudaFree(uvm_adjacencyList);
//     }

//     checkError(cudaFree(d_distance));
//     checkError(cudaFree(d_changed));
//     checkError(cudaFree(d_edgesOffset));
//     checkError(cudaFree(d_edgesSize));

//     return return_distance;
//     // finalizeCudaBfs(distance, parent, G);
// }

// // Main program
// int main(int argc, char **argv)
// {   
//     if (argc != 9)
//         usage(argv[0]);

//     cudaSetDevice(GPU);
    
//     printf("Function: %s line number: %d 1024MB: %d bytes REQUEST_SIZE: %d\n",__func__, __LINE__, MB(1024), REQUEST_SIZE);
    
//     // Get the process ID
//     pid_t pid = getpid();
    
//     // Print the process ID
//     printf("The process ID is: %d\n", pid);

//     // read graph from standard input
//     Graph G;
//     Graph_m G_m;
//     unsigned int *tmp_edgesOffset, *tmp_edgesSize, *tmp_adjacencyList;
    
//     unsigned int startVertex = atoi(argv[7]);
//     // printf("function: %s line: %d u_edgesOffset->local_buffer: %p\n", __FILE__, __LINE__, u_edgesOffset->local_buffer);

//     // readGraph(G, argc, argv);
//     readfile(G, G_m, argc, argv, tmp_edgesOffset, tmp_edgesSize, tmp_adjacencyList);

//     initCuda(G);

//     for(size_t i = 0; i < G.numEdges; i++){
//         uvm_adjacencyList[i] = G.adjacencyList_r[i];
//         u_adjacencyList[i] = G.adjacencyList_r[i];
//     }
//     for(size_t i = 0; i < G.numVertices+1; i++){
//         u_edgesOffset[i] = G.edgesOffset_r[i];
//     }

//     // ------------------- new representation for rdma only: -----------------//
//     auto start = std::chrono::steady_clock::now();                
//     size_t new_size = 0, treshold = 128;
//     for (size_t i = 0; i < G.numVertices; i++)
//     {
//         uint64_t degree = u_edgesOffset[i+1] - u_edgesOffset[i];
        
//         if(degree <= treshold){
//             new_size++;
//         }
//         else{
//             size_t count = degree/treshold + 1;
//             new_size += count;
//         }
//     }
//     unsigned int *new_vertex_list, *new_offset;
//     size_t index_zero = 0;
//     new_vertex_list = new uint[new_size];
//     new_offset = new uint[new_size+1];
//     new_offset[0] = 0;
//     for (size_t i = 0; i < G.numVertices; i++)
//     {
//         uint64_t degree = u_edgesOffset[i+1] - u_edgesOffset[i];
        
//         if(degree <= treshold){
//             new_vertex_list[index_zero] = i;
//             new_offset[index_zero+1] = u_edgesOffset[i+1];
//             index_zero++;
//         }
//         else{
//             size_t count = degree/treshold + 1;
//             size_t total = degree;
//             for (size_t k = 0; k < count; k++)
//             {
//                 new_vertex_list[index_zero] = i;
//                 if(total > treshold) new_offset[index_zero+1] = new_offset[index_zero] + treshold;
//                 else new_offset[index_zero+1] = u_edgesOffset[i+1];
//                 index_zero++;
//                 total = total - treshold;
//             }
//         }
//     }
//     auto end = std::chrono::steady_clock::now();
//     long duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
//     printf("Elapsed time for preprocessing in milliseconds : %li ms.\n\n", duration);
//     /***********************************************************/

//     uint *direct_distance;
//     int number_of_vertices = 200;
//     int active_vertices = 0;
//     direct_distance  = runRDMA(startVertex, G, false, new_vertex_list, new_offset, new_size,
//              u_adjacencyList, u_edgesOffset, 6);

//     init_gpu(0);
//     int deviceCount;
//     cudaGetDeviceCount(&deviceCount);
//     printf("deviceCount: %d\n", deviceCount);

//     bool rdma_flag = true;
//     cudaError_t ret1;
//     struct context_2nic *s_ctx = (struct context_2nic *)malloc(sizeof(struct context_2nic));
//     if(rdma_flag){
//         s_ctx->gpu_cq = NULL;
//         s_ctx->wqbuf = NULL;
//         s_ctx->cqbuf = NULL;
//         s_ctx->gpu_qp = NULL;


//         int num_msg = (unsigned long) atoi(argv[4]);
//         int mesg_size = (unsigned long) atoi(argv[5]);
//         int num_bufs = (unsigned long) atoi(argv[6]);

        
//         struct post_content post_cont, *d_post, host_post;
//         struct poll_content poll_cont, *d_poll, host_poll;
//         // struct post_content2 /*post_cont2,*/ *d_post2;
//         struct server_content_2nic post_cont2, *d_post2;
//         struct host_keys keys;
//         struct gpu_memory_info gpu_mem;

//         int num_iteration = num_msg;
//         s_ctx->n_bufs = num_bufs;

//         s_ctx->gpu_buf_size = 16*1024*1024*1024llu; // N*sizeof(int)*3llu;
//         s_ctx->gpu_buffer = NULL;

//         // // remote connection:
//         // int ret = connect(argv[2], s_ctx);

//         // local connect
//         char *mlx_name = "mlx5_0";
//         // int ret = local_connect(mlx_name, s_ctx);
//         int ret = local_connect_2nic(mlx_name, s_ctx, 0, GPU);

//         mlx_name = "mlx5_2";
//         // int ret = local_connect(mlx_name, s_ctx);
//         ret = local_connect_2nic(mlx_name, s_ctx, 1, GPU);

//         ret = prepare_post_poll_content_2nic(s_ctx, &post_cont, &poll_cont, &post_cont2, \
//                                         &host_post, &host_poll, &keys, &gpu_mem);
//         if(ret == -1) {
//             printf("Post and poll contect creation failed\n");    
//             exit(-1);
//         }

//         printf("alloc synDev ret: %d\n", cudaDeviceSynchronize());
//         cudaSetDevice(GPU);
//         alloc_global_cont(&post_cont, &poll_cont, &post_cont2, gpu_mem);

//         // if(cudaSuccess != ){    
//         printf("alloc synDev ret1: %d\n", cudaDeviceSynchronize());
//             // return -1;
//         // }

//         ret1 = cudaDeviceSynchronize();
//         printf("ret: %d\n", ret1);
//         if(cudaSuccess != ret1){    
//             return -1;
//         }

//         size_t restricted_gpu_mem = 16*1024*1024*1024llu;
//         // restricted_gpu_mem = restricted_gpu_mem / 3;
//         const size_t page_size = REQUEST_SIZE;
//         // const size_t numPages = ceil((double)restricted_gpu_mem/page_size);

//         printf("function: %s line: %d\n", __FILE__, __LINE__);
//         alloc_global_host_content(host_post, host_poll, keys, gpu_mem);
//         printf("function: %s line: %d\n", __FILE__, __LINE__);

//         ret1 = cudaDeviceSynchronize();
//         printf("ret: %d\n", ret1);
//         if(cudaSuccess != ret1){    
//             return -1;
//         }

//         ret1 = cudaDeviceSynchronize();
//         printf("ret: %d\n", ret1);
//         if(cudaSuccess != ret1){    
//             return -1;
//         }
        
//         printf("restricted_gpu_mem: %zu\n", restricted_gpu_mem);
//         cudaSetDevice(GPU);
//         start_page_queue<<<1, 1>>>(/*s_ctx->gpu_buf_size*/restricted_gpu_mem, page_size);
//         ret1 = cudaDeviceSynchronize();
//         printf("ret: %d\n", ret1);
//         if(cudaSuccess != ret1){    
//             return -1;
//         }
//     }

//     printf("Number of vertices %lld tmp_edgesOffset[10]: %d\n", G.numVertices, G.edgesOffset_r[10]);
//     printf("Number of edges %lld\n\n", G.numEdges);

//     std::vector<int> distance(G.numVertices, 100000 /*std::numeric_limits<int>::max()*/);
//     std::vector<int> parent(G.numVertices, std::numeric_limits<int>::max());
//     std::vector<bool> visited(G.numVertices, false);
   
//     printf("function: %s line: %d\n", __FILE__, __LINE__);

//     if(rdma_flag){

//         rdma_adjacencyList->start(G.numEdges *sizeof(unsigned int), GPU, NULL);
//         for(size_t i = 0; i < G.numEdges; i++){
//             rdma_adjacencyList->local_buffer[i] = G.adjacencyList_r[i];
//         }
//     }

//     printf("function: %s line: %d\n", __FILE__, __LINE__);
    
//     uint64_t max = u_edgesOffset[1] - u_edgesOffset[0], max_node;
//     double avg = 0;
//     for (size_t i = 0; i < G.numVertices; i++)
//     {
//         uint64_t degree = u_edgesOffset[i+1] - u_edgesOffset[i];
//         avg += degree;
//         if(max < degree) {
//             max = degree;
//             max_node = i;
//         }
//     }
//     avg = avg / G.numVertices;
//     printf("avg: %f max: %llu, node: %llu\n", avg, max, max_node);

//     free(G.adjacencyList_r);
//     free(G.edgesOffset_r);
    
//     printf("function: %s line: %d\n", __FILE__, __LINE__);

//     std::vector<int> expectedDistance(distance);
//     std::vector<int> expectedParent(parent);
//     start = std::chrono::steady_clock::now();
    
//     int u_case = 2;
    
//     uint *rdma_distance; //, *direct_distance;
//     active_vertices = 0;
//     time_total = 0;
//     if(rdma_flag){
//         for (size_t i = 0; i < number_of_vertices; i++)
//         {
//             startVertex = i;
//             printf("vertex %d has degree of %d\n", startVertex, u_edgesOffset[i+1] - u_edgesOffset[i]);
//             if(u_edgesOffset[i+1] - u_edgesOffset[i] == 0)
//                 continue;
//             active_vertices++;
//             rdma_distance = runRDMA(startVertex, G, rdma_flag, new_vertex_list, new_offset, new_size,
//                 u_adjacencyList, u_edgesOffset, u_case);
//             printf("average time: %.2f pinning time: %.2f\n", time_total/active_vertices, time_readmostly_total/active_vertices);
//             rdma_adjacencyList->reset();

//         }
//         printf("average time: %.2f pinning time: %.2f\n", time_total/active_vertices, time_readmostly_total/active_vertices);

//         cudaFree(s_ctx->gpu_buffer);
//     }

//     // direct_distance = runRDMA(startVertex, G, rdma_flag, new_vertex_list, new_offset, new_size,
//     //          u_adjacencyList, u_edgesOffset, 3);

//     // direct_distance = runCudaSimpleBfs_emogi(startVertex, G);


//     end = std::chrono::steady_clock::now();
//     // u_distance->memcpyDtoH();
//     for(size_t i = 0; i < 5; i++){
//         printf("u_distance->size: %llu (*u_distance)[%d]: %d\n", u_distance, i, u_distance[i]);
//         // printf("u_distance->size: %llu (*u_distance)[%d]: %d\n", u_distance->size, i, u_distance->local_buffer[i]);
//         // printf("u_edgesOffset->size: %llu (*u_edgesOffset)[%d]: %llu\n", u_edgesOffset->size, i, u_edgesOffset->local_buffer[i]);
//         // printf("u_edgesSize->size: %llu (*u_edgesSize)[%d]: %llu G.edgesSize_r[%d]: %llu\n", u_edgesSize->size, i, u_edgesSize->local_buffer[i], i, G.edgesSize_r[i]);
//     }
    
//     checkOutput_rdma(direct_distance, rdma_distance, G);
//     // checkOutput_rdma(u_distance, expectedDistance, G);

//     print_utilization<<<1,1>>>();
//     cudaError_t ret2 = cudaDeviceSynchronize();
//     if(cudaSuccess != ret1){  
//         printf("cudaDeviceSynchronize error: %d\n", ret1);  
//         exit(-1);
//     }

//     print_transferTime<<<1,1>>>();
//     ret2 = cudaDeviceSynchronize();
//     if(cudaSuccess != ret1){  
//         printf("cudaDeviceSynchronize error: %d\n", ret1);  
//         exit(-1);
//     }

//     // // //run CUDA queue parallel bfs
//     // runCudaQueueBfs(startVertex, G, distance, parent);
//     // checkOutput(distance, expectedDistance, G);

//     // // //run CUDA scan parallel bfs
//     // runCudaScanBfs(startVertex, G, distance, parent);
//     // checkOutput(distance, expectedDistance, G);
//     finalizeCuda();
    
//     duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
//     printf("Overall Elapsed time in milliseconds : %li ms. startVertex: %d\n", duration, startVertex);

    

//     return 0;

// 	return 0;
// }

// // __global__
// // void simpleBfs_rdma(size_t n, unsigned int *level, rdma_buf<unsigned int> *d_adjacencyList, rdma_buf<unsigned int> *d_edgesOffset,
// //                rdma_buf<unsigned int> *d_edgesSize, rdma_buf<unsigned int> *d_distance, rdma_buf<unsigned int> *d_parent, unsigned int *changed) {
// //     size_t thid = blockIdx.x * blockDim.x + threadIdx.x;
// //     int valueChange = 0;
// //     if(thid < n /*d_distance->size/sizeof(uint)*/){
// //         unsigned int k = (*d_distance)[thid];
        
// //         if (/*thid < n && */k == *level) {
// //             unsigned int u = thid;
// //             for (unsigned int i = (*d_edgesOffset)[u]; i < (*d_edgesOffset)[u] + (*d_edgesSize)[u]; i++) {
                
// //                 int v = (*d_adjacencyList)[i];
// //                 unsigned int dist = (*d_distance)[v];
// //                 if (*level + 1 < dist) {
                    
// //                     unsigned int new_dist = *level + 1;
                   
// //                     (*d_distance).rvalue(v, new_dist /*(int) level + 1*/);
                   
// //                     valueChange = 1;
// //                 }
// //             }
// //             // printf(" for finished\n");
// //         }
// //         // __syncthreads();
// //         if (valueChange) {
// //             *changed = valueChange;
// //         }
// //     }
// //     // __syncthreads();
// // }

// // __global__
// // void simpleBfs_rdma(size_t n, unsigned int level, rdma_buf<unsigned int> *d_adjacencyList, rdma_buf<unsigned int> *d_edgesOffset,
// //                rdma_buf<unsigned int> *d_edgesSize, unsigned int *d_distance, unsigned int *changed)

// __global__ void bfs_kernel_coalesce_chunk(unsigned int *label, unsigned int level, const uint64_t vertex_count, unsigned int *vertexList, unsigned int *edgeList, unsigned int *changed) {
//     const uint64_t tid = blockDim.x * BLOCK_SIZE * blockIdx.y + blockDim.x * blockIdx.x + threadIdx.x;
//     const uint64_t warpIdx = tid >> WARP_SHIFT;
//     const uint64_t laneIdx = tid & ((1 << WARP_SHIFT) - 1);
//     const uint64_t chunkIdx = warpIdx * CHUNK_SIZE;
//     uint64_t chunk_size = CHUNK_SIZE;

//     if((chunkIdx + CHUNK_SIZE) > vertex_count) {
//         if ( vertex_count > chunkIdx )
//             chunk_size = vertex_count - chunkIdx;
//         else
//             return;
//     }

//     for(uint32_t i = chunkIdx; i < chunk_size + chunkIdx; i++) {
//         if(label[i] == level) {
//             const uint64_t start = vertexList[i];
//             const uint64_t shift_start = start & MEM_ALIGN;
//             const uint64_t end = vertexList[i+1];

//             for(uint64_t j = shift_start + laneIdx; j < end; j += WARP_SIZE) {
//                 if (j >= start) {
//                     const uint next = edgeList[j];

//                     // if(label[next] == MYINFINITY) {
//                     if(label[next] > level + 1) {
//                         label[next] = level + 1;
//                         *changed = 1;
//                     }
//                 }
//             }
//         }
//     }
// }

// __global__
// void simpleBfs_modVertexList(size_t n, unsigned int level, unsigned int *d_adjacencyList,
//                     unsigned int *vertex_list, unsigned int *d_distance, unsigned int *changed) {

//     size_t thid = blockIdx.x * blockDim.x + threadIdx.x;
//     int valueChange = 0;
//     unsigned int k = d_distance[thid];
//     if(thid < n && k == level) {
//         for (size_t i = vertex_list[thid]; i < vertex_list[thid+1]; i += 1) {
//             unsigned int v = d_adjacencyList[i];
//             unsigned int dist = d_distance[v];
//             if (level + 1 < dist) {
//                 d_distance[v] = level + 1; /*(int) level + 1*/
//                 valueChange = 1;
//             }
//         }
//     }
//     // __syncthreads();
//     if (valueChange) {
//         *changed = 1;
//     }
// }

// __global__
// void simpleBfs_modVertexList_rdma(size_t n, unsigned int level, rdma_buf<unsigned int> *d_adjacencyList,
//                     unsigned int *vertex_list, unsigned int *d_distance, unsigned int *changed) {

//     // Thread index
//     size_t thid = blockIdx.x * blockDim.x + threadIdx.x;
//     const unsigned int warpSize = 32;
//     // Warp index and lane index
//     size_t warpId = thid / warpSize;
//     size_t laneId = thid % warpSize;
//     // Warp size
    

//     // Buffer for storing distances and change flag in shared memory
//     unsigned int shared_distance;
//     unsigned int warp_changed;

//     if (warpId < n && level == d_distance[warpId]) {
//         // Each warp processes one node
//         // if (laneId == 0) {
//             shared_distance = d_distance[warpId];
//             warp_changed = 0;
//         // }

//         // __syncwarp(); // Synchronize within warp
//         if (shared_distance == level) {
//             unsigned int nodeStart = vertex_list[warpId];
//             unsigned int nodeEnd = vertex_list[warpId + 1];

//             for (size_t i = nodeStart + laneId; i < nodeEnd; i += warpSize) {
//                 unsigned int v = (*d_adjacencyList)[i];
//                 unsigned int dist = d_distance[v];
                
//                 if (level + 1 < dist) {
//                     d_distance[v] = level + 1;
//                     warp_changed = 1; // Mark distance change
//                 }
//             }
//         }

//         // __syncwarp(); // Synchronize within warp

//         // Use the first thread in the warp to set the changed flag
//         if (warp_changed) {
//             *changed = 1;
//             // atomicExch(changed, 1);
//         }
//     }
// }


// __global__
// void simpleBfs_normal_rdma(size_t n, size_t vertexCount, unsigned int level, rdma_buf<unsigned int> *d_edgeList, unsigned int *d_vertex_list,
//                       unsigned int *d_distance, unsigned int *changed) {
//     size_t thid = blockIdx.x * blockDim.x + threadIdx.x;
//     int stride = blockDim.x * gridDim.x;
//     int valueChange = 0;
//     // size_t iterations = 0;
//     // if(thid < n /*d_distance->size/sizeof(uint)*/){
        
//             for (size_t i = thid; i < n; i += /*vertexCount*/ stride) {
//                 unsigned int v;
//                 unsigned int k = d_distance[d_vertex_list[i]];
                
//                 if(k == level){
//                     // v = d_edgeList[i];
//                     v = (*d_edgeList)[i];
//                     unsigned int dist = d_distance[v];
//                     if (level + 1 < dist) {
                        
//                             d_distance[v] = level + 1; /*(int) level + 1*/
                        
//                         valueChange = 1;
//                     }
//                 }
//                 // iterations++;
//             }
            
//         // }
        
        

//         // __syncthreads();
//         if (valueChange) {
//             *changed = valueChange;
//         }
//     // }
// }

// // __global__
// // void simpleBfs_rdma(size_t n, unsigned int level, rdma_buf<unsigned int> *d_adjacencyList, unsigned int *d_edgesOffset,
// //                     unsigned int *d_edgesSize, unsigned int *d_distance, unsigned int *changed) {
// //     size_t thid = blockIdx.x * blockDim.x + threadIdx.x;
// //     int valueChange = 0;

// //     // Each warp works on 4KB of data
// //     size_t warpSize = 32;
// //     size_t warpId = thid / warpSize;
// //     size_t laneId = thid % warpSize;

// //     // Calculate the start and end of the 4KB region for this warp
// //     size_t regionSize = 4*1024 / sizeof(unsigned int); // 4KB region size in terms of number of elements
// //     size_t startIdx = warpId * regionSize;
// //     size_t endIdx = (startIdx + regionSize) < n ? (startIdx + regionSize) : n;
// //     // min(startIdx + regionSize, n);

// //     for (size_t idx = startIdx + laneId; idx < endIdx; idx += warpSize) {
// //         if (idx < n) {
// //             unsigned int k = d_distance[idx];
// //             if (k == level) {
// //                 for (size_t i = d_edgesOffset[idx]; i < d_edgesOffset[idx + 1]; i++) {
// //                     unsigned int v;
// //                     v = (*d_adjacencyList)[i];
// //                     unsigned int dist = d_distance[v];
// //                     if (level + 1 < dist) {
// //                         d_distance[v] = level + 1;
// //                         valueChange = 1;
// //                     }
// //                 }
// //             }
// //         }
// //     }

// //     if (__syncthreads_or(valueChange)) {
// //         *changed = 1;
// //     }
// // }


// __global__
// void simpleBfs_normal(size_t n, size_t vertexCount, unsigned int level, unsigned int *d_edgeList, unsigned int *d_vertex_list,
//                       unsigned int *d_distance, unsigned int *changed) {
//     size_t thid = blockIdx.x * blockDim.x + threadIdx.x;
//     int stride = blockDim.x * gridDim.x;
//     int valueChange = 0;
//     // size_t iterations = 0;
//     if(thid < n /*d_distance->size/sizeof(uint)*/){
        
//             for (size_t i = thid; i < n; i += stride) {
                
//                 unsigned int v;
//                 unsigned int k = d_distance[d_vertex_list[i]];
//                 if(k == level){
//                     v = d_edgeList[i];
                    
//                     unsigned int dist = d_distance[v];
//                     if (level + 1 < dist) {
                        
//                             d_distance[v] = level + 1; /*(int) level + 1*/
                        
//                         valueChange = 1;
//                     }
//                 }
//                 // iterations++;
//             }
        
//         if (valueChange) {
//             *changed = valueChange;
//         }
//     }
// }

// __global__ __launch_bounds__(1024,2)
// void simpleBfs_rdma(size_t n, unsigned int level, rdma_buf<unsigned int> *d_adjacencyList, unsigned int *d_edgesOffset,
//                     unsigned int *d_edgesSize, unsigned int *d_distance, unsigned int *changed) {
//     size_t thid = blockIdx.x * blockDim.x + threadIdx.x;
//     const uint64_t laneIdx = thid & ((1 << WARP_SHIFT) - 1);

//     // // Warp ID within the block
//     size_t warpId = thid / (1 << WARP_SHIFT);
    
//     // size_t iterations = 0;
//     if(warpId < n /*d_distance->size/sizeof(uint)*/){
//         unsigned int k = d_distance[warpId];
//         if (k == level) {
           
//             for (size_t i = d_edgesOffset[warpId] + laneIdx; i < d_edgesOffset[warpId+1]; i += (1 << WARP_SHIFT)) {
//                 unsigned int v;
               
//                 v = (*d_adjacencyList)[i];
//                 unsigned int dist = d_distance[v];
//                 if (level + 1 < dist) {
                    
//                     d_distance[v] = level + 1; /*(int) level + 1*/
//                     *changed = 1;
//                     // valueChange = 1;
//                 }
//             }
//         }
        
//         // if (valueChange) {
            
//         // }
//     }
// }

// __global__
// void simpleBfs_uvm(size_t n, unsigned int level, unsigned int *d_adjacencyList, unsigned int *d_edgesOffset,
//                     unsigned int *d_edgesSize, unsigned int *d_distance, unsigned int *changed) {
//     size_t thid = blockIdx.x * blockDim.x + threadIdx.x;
//     int valueChange = 0;
//     if(thid < n /*d_distance->size/sizeof(uint)*/){
//         unsigned int k = d_distance[thid];
//         if (k == level) {
//             // uint edgesOffset = d_edgesOffset[thid];
//             // uint edgesSize = d_edgesSize[thid];
//             for (size_t i = d_edgesOffset[thid]; i < d_edgesOffset[thid] + d_edgesSize[thid]; i += 1) {
//                 // double time1 = clock();
//                 // unsigned int *tmp = (unsigned int *) d_adjacencyList.d_TLB[i/1024].device_address;
//                 // unsigned int v = tmp[i%1024]; // (*d_adjacencyList)[i];
//                 unsigned int v = d_adjacencyList[i];
//                 // int v = d_adjacencyList[i];
//                 // double time2 = clock();
//                 // printf(" %f ", time2 - time1);
//                 unsigned int dist = d_distance[v];
//                 if (level + 1 < dist) {
                    
//                         d_distance[v] = level + 1; /*(int) level + 1*/
                    
//                     valueChange = 1;
//                 }
//             }
//         }
//         // __syncthreads();
//         if (valueChange) {
//             *changed = valueChange;
//         }
//     }
// }


// __global__ void sssp_GPU_Kernel(int numEdges,
//                                 int numEdgesPerThread,
//                                 uint *dist,
//                                 uint *preNode,
//                                 uint *edgesSource,
//                                 uint *edgesEnd,
//                                 uint *edgesWeight,
//                                 bool *finished) {
//     int threadId = blockDim.x * blockIdx.x + threadIdx.x;
//     int startId = threadId * numEdgesPerThread;
//     // if(threadId == 0) printf("hello from sssp\n"); 
//     if (startId >= numEdges) {
//         return;
//     }
    
//     int endId = (threadId + 1) * numEdgesPerThread;
//     if (endId >= numEdges) {
//         endId = numEdges;
//     }

//     for (int nodeId = startId; nodeId < endId; nodeId++) {
//         uint source = edgesSource[nodeId];
//         uint end = edgesEnd[nodeId]; // edgelist
//         uint weight = 1; // edgesWeight[nodeId];
        
//         if (dist[source] + weight < dist[end]) {
//             atomicMin(&dist[end], dist[source] + weight);
//             // dist[end] = dist[source] + weight;
//             preNode[end] = source;
//             *finished = false;
//         }
//     }
// }

// __global__
// void simpleBfs_normal_rdma_optimized(int numEdgesPerThread, size_t numEdges, size_t vertexCount, unsigned int level, rdma_buf<unsigned int> *d_edgeList, unsigned int *d_vertex_list,
//                       unsigned int *d_distance, unsigned int *changed) {
//     // size_t thid = blockIdx.x * blockDim.x + threadIdx.x;
//     // int stride = blockDim.x * gridDim.x;
//     // int valueChange = 0;
    
    
//     int threadId = blockDim.x * blockIdx.x + threadIdx.x;
//     int startId = threadId * numEdgesPerThread;
//     // if(threadId == 0) printf("hello from sssp\n"); 
//     if (startId >= numEdges) {
//         return;
//     }
    
//     int endId = (threadId + 1) * numEdgesPerThread;
//     if (endId >= numEdges) {
//         endId = numEdges;
//     }

//     for (int nodeId = startId; nodeId < endId; nodeId++) {

//         unsigned int v;
//         unsigned int k = d_distance[d_vertex_list[nodeId]];

//         if(k == level){
//             // v = d_edgeList[i];
//             v = D_adjacencyList[nodeId];
//             unsigned int dist = d_distance[v];
//             if (level + 1 < dist) {
                
//                 d_distance[v] = level + 1; /*(int) level + 1*/
//                 *changed = 1;
//                 // valueChange = 1;
//             }
//         }

       
//     }

//     // // __syncthreads();
//     // if (valueChange) {
//     //     *changed = valueChange;
//     // }
//     // // }
// }


// __global__ __launch_bounds__(128,16)
// void kernel_coalesce_ptr_pc(unsigned int *ptr, unsigned int *label, const uint32_t level, const uint64_t vertex_count, unsigned int *edgeOffset,
//                             unsigned int * edgeSize, unsigned *edgeList, unsigned int *changed) {
//     const uint64_t tid = blockDim.x * BLOCK_NUM * blockIdx.y + blockDim.x * blockIdx.x + threadIdx.x;
//     const uint64_t warpIdx = tid >> WARP_SHIFT;
//     const uint64_t laneIdx = tid & ((1 << WARP_SHIFT) - 1);
//     //array_d_t<uint64_t> d_array = *da;
//     if(warpIdx < vertex_count && label[warpIdx] == level) {
//         // bam_ptr<uint64_t> ptr(da);
//         const uint64_t start = edgeOffset[warpIdx];
//         const uint64_t shift_start = start & 0xFFFFFFFFFFFFFFF0;
//         const uint64_t end = edgeOffset[warpIdx] + edgeSize[warpIdx];

//         for(uint64_t i = shift_start + laneIdx; i < end; i += WARP_SIZE) {
//             if (i >= start) {
//                 //const EdgeT next = edgeList[i];
//                 //EdgeT next = da->seq_read(i);
//                 unsigned int next = ptr[i];
// //                printf("tid: %llu, idx: %llu next: %llu\n", (unsigned long long) tid, (unsigned long long) i, (unsigned long long) next);
//                 unsigned int dist = label[next];
//                 if(/*label[next] == MYINFINITY*/level + 1 < dist) {
//                 //    if(level ==0)
//                 //            printf("tid:%llu, level:%llu, next: %llu\n", tid, (unsigned long long)level, (unsigned long long)next);
//                     label[next] = level + 1;
//                     *changed = true;
//                 }
//             }
//         }
//     }
// }


// __global__ __launch_bounds__(128,32)
// void kernel_coalesce_ptr_pc_rdma(rdma_buf<unsigned int> *ptr, unsigned int *label, const uint32_t level, const uint64_t vertex_count, unsigned int *edgeOffset,
//                             unsigned int * edgeSize, unsigned int *changed) {
//     const uint64_t tid = blockDim.x * BLOCK_NUM * blockIdx.y + blockDim.x * blockIdx.x + threadIdx.x;
//     const uint64_t warpIdx = tid >> WARP_SHIFT;
//     const uint64_t laneIdx = tid & ((1 << WARP_SHIFT) - 1);
//     //array_d_t<uint64_t> d_array = *da;
//     if(warpIdx < vertex_count && label[warpIdx] == level) {
//         // bam_ptr<uint64_t> ptr(da);
//         const uint64_t start = edgeOffset[warpIdx];
//         const uint64_t shift_start = start & 0xFFFFFFFFFFFFFFF0;
//         const uint64_t end = edgeOffset[warpIdx] + edgeSize[warpIdx];

//         for(uint64_t i = shift_start + laneIdx; i < end; i += WARP_SIZE) {
//             if (i >= start) {
//                 unsigned int next = (*ptr)[i];                
//                 unsigned int dist = label[next];
//                 if(level + 1 < dist) {
//                     label[next] = level + 1;
//                     *changed = true;
//                 }
//             }
//         }
//     }
// }

// // normal kernel - with optimized warp access; one warp working on one page only
// __global__
// void simpleBfs_normal_rdma_optimized(size_t n, size_t vertexCount, unsigned int level, rdma_buf<unsigned int> *d_edgeList, unsigned int *d_vertex_list,
//                       unsigned int *d_distance, unsigned int *changed) {
//     size_t thid = blockIdx.x * blockDim.x + threadIdx.x;
//     int stride = blockDim.x * gridDim.x;
//     int valueChange = 0;
        
//             for (size_t i = thid; i < n; i += stride) {
//                 unsigned int v;
//                 unsigned int k = d_distance[d_vertex_list[i]];
                
//                 if(k == level){
//                     // v = d_edgeList[i];
//                     // v = D_adjacencyList[i];
//                     v = (*d_edgeList)[i];
//                     unsigned int dist = d_distance[v];
//                     if (level + 1 < dist) {
                        
//                             d_distance[v] = level + 1; /*(int) level + 1*/
                        
//                         valueChange = 1;
//                     }
//                 }
//             }
            
//         if (valueChange) {
//             *changed = valueChange;
//         }
// }

// // __global__
// // void simpleBfs_normal_rdma_optimized2(size_t n, size_t vertexCount, unsigned int level, rdma_buf<unsigned int> *d_edgeList, unsigned int *d_vertex_list,
// //                                      unsigned int *d_distance, unsigned int *changed) {
// //     const size_t page_size = 64 * 1024;  // 64KB in bytes
// //     const size_t elements_per_page = page_size / sizeof(unsigned int);  // Elements per 64KB page
    
// //     size_t thid = blockIdx.x * blockDim.x + threadIdx.x;
// //     size_t stride = blockDim.x * gridDim.x;
// //     size_t warp_id = thid / warpSize;
// //     size_t warp_thread_id = thid % warpSize;
    
// //     bool valueChanged = false; // Using a bool for local flag
    
// //     for (size_t page_id = warp_id; page_id < n / elements_per_page; page_id += (stride / warpSize)) {
// //         size_t start_idx = page_id * elements_per_page;
// //         size_t end_idx = start_idx + elements_per_page < n ? start_idx + elements_per_page : n; // min(start_idx + elements_per_page, n);
        
// //         for (size_t i = start_idx + warp_thread_id; i < end_idx; i += warpSize) {
// //             unsigned int v;
// //             unsigned int k = d_distance[d_vertex_list[i]];
            
// //             if (k == level) {
// //                 v = (*d_edgeList)[i];
// //                 unsigned int dist = d_distance[v];
// //                 if (level + 1 < dist) {
// //                     d_distance[v] = level + 1;
// //                     valueChanged = true;
// //                 }
// //             }
// //         }
// //     }

// //     // Use atomic operation to set the changed flag
// //     if (valueChanged) {
// //         *changed = 1;
// //         // atomicOr(changed, 1);
// //     }
// // }

// __global__
// void simpleBfs_normal_rdma_optimized2(size_t n, size_t vertexCount, unsigned int level, rdma_buf<unsigned int> *d_edgeList, unsigned int *d_vertex_list,
//                                      unsigned int *d_distance, unsigned int *changed) {
//     const size_t page_size = 64 * 1024;  // 64KB in bytes
//     const size_t elements_per_page = page_size / sizeof(unsigned int);  // Elements per 64KB page
    
//     size_t thid = blockIdx.x * blockDim.x + threadIdx.x;
//     size_t stride = blockDim.x * gridDim.x;
//     size_t warp_id = thid / warpSize;
//     size_t warp_thread_id = thid % warpSize;
    
//     __shared__ bool shared_changed;
//     if (threadIdx.x == 0) {
//         shared_changed = false;
//     }
//     __syncthreads();

//     for (size_t page_id = warp_id; page_id < (n + elements_per_page - 1) / elements_per_page; page_id += (stride / warpSize)) {
//         size_t start_idx = page_id * elements_per_page;
//         size_t end_idx = start_idx + elements_per_page < n ? start_idx + elements_per_page : n; // min(start_idx + elements_per_page, n);

//         for (size_t i = start_idx + warp_thread_id; i < end_idx; i += warpSize) {
//             unsigned int v;
//             unsigned int k = d_distance[d_vertex_list[i]];
            
//             if (k == level) {
//                 v = (*d_edgeList)[i];
//                 unsigned int dist = d_distance[v];
//                 if (level + 1 < dist) {
//                     d_distance[v] = level + 1;
//                     shared_changed = true;
//                 }
//             }
//         }
//     }
    
//     __syncthreads();
//     if (threadIdx.x == 0 && shared_changed) {
//         *changed = 1;
//     }
// }

// __global__
// void simpleBfs_rdma_optimized_thread_different_page(size_t n, size_t numVertices, unsigned int level, rdma_buf<unsigned int> *d_adjacencyList, unsigned int *d_edgesOffset,
//                     unsigned int *d_edgesSize, unsigned int *d_distance, unsigned int *changed) {
//     // Warp size
//     const size_t warpSize = 32;

//     // Global thread ID
//     size_t tid = blockIdx.x * blockDim.x + threadIdx.x;

//     // Warp ID within the grid
//     size_t warpId = tid / warpSize;

//     // Thread lane within the warp
//     size_t lane = threadIdx.x % warpSize;

//     // Ensure we don't process out-of-bounds vertices
//     if (warpId < numVertices) {
//         bool localChanged = false;

//         // Fetch edges for this node
//         // if (lane == 0) {
//             unsigned int k = d_distance[warpId];
//             if (k == level) {
//                 size_t edgeStart = d_edgesOffset[warpId];
//                 size_t edgeEnd = d_edgesOffset[warpId + 1];

//                 // Broadcast edgeStart and edgeEnd to the whole warp
//                 unsigned int edgeStartWarp = __shfl_sync(0xFFFFFFFF, edgeStart, 0);
//                 unsigned int edgeEndWarp = __shfl_sync(0xFFFFFFFF, edgeEnd, 0);

//                 // Process neighbors in parallel within the warp
//                 for (size_t j = edgeStartWarp + lane; j < edgeEndWarp; j += warpSize) {
//                     unsigned int v = (*d_adjacencyList)[j];
//                     unsigned int dist = d_distance[v];
//                     if (level + 1 < dist) {
//                         d_distance[v] = level + 1;
//                         localChanged = true;
//                     }
//                 }
//             }
//         // }

//         // Use warp-wide OR to set the changed flag if needed
//         if (__any_sync(0xFFFFFFFF, localChanged)) {
//             atomicOr(changed, 1);
//         }
//     }
// }

// __global__
// void check_edgeList(rdma_buf<unsigned int> *a, unsigned int *b, size_t size){
//     size_t tid = blockDim.x * blockIdx.x + threadIdx.x;
//     if(tid == 0) printf("checking edgelist correctness\n");
//     if(tid < size){
//         unsigned int a_here = (*a)[tid];
//         // __nanosleep(100000);
//         if(a_here != b[tid]){
//             printf("tid: %llu, a_here: %d b[tid]: %d\n", tid, a_here, b[tid]);
//         } 
//     }
// }

// __global__ void 
// kernel_coalesce_new_repr(uint level, size_t n, size_t numVertex, const uint64_t new_size, unsigned int *d_distance, 
//                         unsigned int *new_offset, unsigned int *new_vertex_list, unsigned int *edgeList, unsigned int *changed) {


//     // Page size in elements (64KB / 4 bytes per unsigned int)
//     const size_t pageSize = 4*1024 / sizeof(unsigned int);
//     // Elements per warp
//     const size_t elementsPerWarp = pageSize / warpSize;

//     // Global thread ID
//     size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
//     // if(tid == 0) printf("warpSize: %d\n", warpSize);
//     // Warp ID within the block
//     size_t warpId = tid / warpSize;

//     // Thread lane within the warp
//     size_t lane = threadIdx.x % warpSize;

//     // Determine which page this warp will process
//     size_t pageStart = warpId * pageSize;

//     // Ensure we don't process out-of-bounds pages
//     if (pageStart < n * pageSize) {
        
//         // Process elements within the page
//         for (size_t i = 0; i < elementsPerWarp; ++i) {
//             size_t elementIdx = pageStart + lane + i * warpSize;
//             if(elementIdx < new_size){
//                 uint startVertex = new_vertex_list[elementIdx];
//                 unsigned int k = d_distance[startVertex];
//                 if (k == level) {
//                     // Process adjacent nodes
//                     // if(new_offset[elementIdx+1] - d_edgesOffset[elementIdx] >= 2*1024)
//                     //     printf("elementx: %llu\n", elementIdx);
//                     for(size_t j = new_offset[elementIdx]; j < new_offset[elementIdx+1]; ++j) {
//                         uint v = edgeList[j]; // shared_data[j - pageStart];
                        
//                         unsigned int dist = d_distance[v];
//                         if (level + 1 < dist) {
//                             d_distance[v] = level + 1;
//                             *changed = 1;
//                         }
//                     }

//                 }
//             }
//         }
//     }

// }

// __global__ // __launch_bounds__(1024,2)
// void kernel_coalesce_new_repr_rdma(uint level, size_t n, size_t numVertex, const uint64_t new_size, unsigned int *d_distance, 
//                                 unsigned int *new_offset, unsigned int *new_vertex_list, rdma_buf<unsigned int> *edgeList, unsigned int *changed) {

    
//     // Page size in elements (64KB / 4 bytes per unsigned int)
//     const size_t pageSize = 8*1024 / sizeof(unsigned int);
//     // Elements per warp
//     const size_t elementsPerWarp = pageSize / warpSize;

//     // Global thread ID
//     size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
//     // if(tid == 0) printf("warpSize: %d\n", warpSize);
//     // Warp ID within the block
//     size_t warpId = tid / warpSize;

//     // Thread lane within the warp
//     size_t lane = threadIdx.x % warpSize;

//     // Determine which page this warp will process
//     size_t pageStart = warpId * pageSize;

//     // Ensure we don't process out-of-bounds pages
//     if (pageStart < n * pageSize) {
        
//         // Process elements within the page
//         for (size_t i = 0; i < elementsPerWarp; ++i) {
//             size_t elementIdx = pageStart + lane + i * warpSize;
//             if(elementIdx < new_size){
//                 uint startVertex = new_vertex_list[elementIdx];
//                 unsigned int k = d_distance[startVertex];
//                 if (k == level) {
//                     // Process adjacent nodes
//                     // if(new_offset[elementIdx+1] - d_edgesOffset[elementIdx] >= 2*1024)
//                     //     printf("elementx: %llu\n", elementIdx);
//                     for(size_t j = new_offset[elementIdx]; j < new_offset[elementIdx+1]; ++j) {
//                         uint v = (*edgeList)[j]; // shared_data[j - pageStart];
                        
//                         unsigned int dist = d_distance[v];
//                         if (level + 1 < dist) {
//                             d_distance[v] = level + 1;
//                             *changed = 1;
//                         }
//                     }

//                 }
//             }
//         }
//     }

// }

// __global__
// void simpleBfs_rdma_optimized_warp(size_t n, size_t numVertices, unsigned int level, rdma_buf<unsigned int> *d_adjacencyList, unsigned int *d_edgesOffset,
//                     unsigned int *d_edgesSize, unsigned int *d_distance, unsigned int *changed) {
//     // Page size in elements (64KB / 4 bytes per unsigned int)
//     const size_t pageSize = 4*1024 / sizeof(unsigned int);
//     // Elements per warp
//     const size_t elementsPerWarp = pageSize / warpSize;

//     // Global thread ID
//     size_t tid = blockIdx.x * blockDim.x + threadIdx.x;

//     // Warp ID within the block
//     size_t warpId = tid / warpSize;

//     // Thread lane within the warp
//     size_t lane = threadIdx.x % warpSize;

//     // Determine which page this warp will process
//     size_t pageStart = warpId * pageSize;

//     // Ensure we don't process out-of-bounds pages
//     if (pageStart < n * pageSize) {
//         bool localChanged = false;
        
//         // Process elements within the page
//         for (size_t i = 0; i < elementsPerWarp; ++i) {
//             size_t elementIdx = pageStart + lane + i * warpSize;
//             if (elementIdx < numVertices) {
//                 unsigned int k = d_distance[elementIdx];
//                 if (k == level) {
//                     // printf("d_edgesOffset[%llu]: %u, d_distance[%llu]: %u\n", 
//                     //         (long long int) elementIdx, d_edgesOffset[elementIdx], (long long int) elementIdx, d_distance[elementIdx]);
//                     for(size_t j = d_edgesOffset[elementIdx]; j < d_edgesOffset[elementIdx+1] /*+ d_edgesSize[elementIdx]*/; ++j) {
//                         int v = (*d_adjacencyList)[j];
//                         // if(v >= numVertices || v < 0)
//                             // printf("j: %llu V: %d numVertices: %lu\n", j, v, numVertices);
                        
//                         unsigned int dist = d_distance[v];
//                         if (level + 1 < dist) {
//                             d_distance[v] = level + 1;
//                             localChanged = true;
//                         }
//                     }
//                 }
//             }
//         }

//         // Use atomic operation to set the changed flag if needed
//         if (localChanged) {
//             // atomicOr(changed, 1);
//             *changed = 1;
//         }
//     }
// }

// __global__
// void simpleBfs_rdma_optimized_warp2(size_t n, unsigned int level, rdma_buf<unsigned int> *d_adjacencyList, unsigned int *d_edgesOffset,
//                                    unsigned int *d_edgesSize, unsigned int *d_distance, unsigned int *changed) {
//     // Page size in elements (64KB / 4 bytes per unsigned int)
//     const size_t pageSize = 64*1024 / sizeof(unsigned int);
//     // Elements per warp
//     const size_t elementsPerWarp = pageSize / warpSize;

//     // Global thread ID
//     size_t tid = blockIdx.x * blockDim.x + threadIdx.x;

//     // Warp ID within the block
//     size_t warpId = tid / warpSize;

//     // Thread lane within the warp
//     size_t lane = threadIdx.x % warpSize;

//     // Global warp ID
//     size_t globalWarpId = tid / warpSize;

//     // Number of warps in the grid
//     size_t numWarps = (blockDim.x * gridDim.x) / warpSize;

//     bool localChanged = false;

//     for (size_t pageStart = globalWarpId * pageSize; pageStart < n * pageSize; pageStart += numWarps * pageSize) {
//         // Process elements within the page
//         for (size_t i = 0; i < elementsPerWarp; ++i) {
//             size_t elementIdx = pageStart + lane + i * warpSize;
//             if (elementIdx < n) {
//                 unsigned int k = d_distance[elementIdx];
//                 if (k == level) {
//                     // printf("d_edgesOffset[%llu]: %u, d_edgesSize[%llu]: %u\n", 
//                     //         (long long int) elementIdx, d_edgesOffset[elementIdx], (long long int) elementIdx, d_edgesSize[elementIdx]);
//                     for (size_t j = d_edgesOffset[elementIdx]; j < d_edgesOffset[elementIdx] + d_edgesSize[elementIdx]; ++j) {
//                         unsigned int v = (*d_adjacencyList)[j];
//                         unsigned int dist = d_distance[v];
//                         if (level + 1 < dist) {
//                             d_distance[v] = level + 1;
//                             localChanged = true;
//                         }
//                     }
//                 }
//             }
//         }
//     }

//     // Use atomic operation to set the changed flag if needed
//     if (localChanged) {
//         atomicOr(changed, 1);
//     }
// }

// __global__
// void simpleBfs_rdma_optimized_dynamic(size_t n, unsigned int level, rdma_buf<unsigned int> *d_adjacencyList, unsigned int *d_edgesOffset,
//                     unsigned int *d_edgesSize, unsigned int *d_distance, unsigned int *changed){
//     // Shared memory queue
//     extern __shared__ unsigned int queue[];
//     __shared__ int queueStart, queueEnd;
    
//     if (threadIdx.x == 0) {
//         queueStart = 0;
//         queueEnd = 0;
//     }
//     __syncthreads();

//     // Global thread ID
//     size_t tid = blockIdx.x * blockDim.x + threadIdx.x;

//     // Each thread initially processes its own node
//     if (tid < n) {
//         unsigned int k = d_distance[tid];
//         if (k == level) {
//             int pos = atomicAdd(&queueEnd, 1);
//             if (pos < blockDim.x) {  // Ensure we do not exceed shared memory limits
//                 queue[pos] = tid;
//             }
//         }
//     }
//     __syncthreads();

//     // Process the queue
//     while (queueStart < queueEnd) {
//         int node;
//         if (threadIdx.x == 0 && queueStart < blockDim.x) {
//             node = queue[queueStart];
//             atomicAdd(&queueStart, 1);
//         }
//         node = __shfl_sync(0xFFFFFFFF, node, 0);

//         bool localChanged = false;

//         if (node < n) {  // Ensure the node index is within bounds
//             unsigned int k = d_distance[node];

//             if (k == level) {
//                 size_t edgeStart = d_edgesOffset[node];
//                 size_t edgeEnd = edgeStart + d_edgesSize[node];

//                 for (size_t j = edgeStart; j < edgeEnd; ++j) {
//                     if (j >= n) continue;  // Ensure edge index is within bounds

//                     unsigned int v = (*d_adjacencyList)[j];
//                     unsigned int dist = d_distance[v];
//                     if (level + 1 < dist) {
//                         d_distance[v] = level + 1;
//                         localChanged = true;

//                         // Add the newly discovered node to the queue
//                         int pos = atomicAdd(&queueEnd, 1);
//                         if (pos < blockDim.x) {  // Ensure we do not exceed shared memory limits
//                             queue[pos] = v;
//                         }
//                     }
//                 }
//             }
//         }
//         __syncthreads();

//         // Use atomic operation to set the changed flag if needed
//         if (localChanged) {
//             atomicOr(changed, 1);
//         }
//     }
// }

// __global__
// void simpleBfs_rdma_dynamic_page(size_t n, unsigned int level, rdma_buf<unsigned int> *d_adjacencyList, unsigned int *d_edgesOffset,
//                                  unsigned int *d_edgesSize, unsigned int *d_distance, unsigned int *changed) {
//     // Page size in elements (64KB / 4 bytes per unsigned int)
//     const size_t pageSize = 64 * 1024 / sizeof(unsigned int);
//     // Number of pages
//     const size_t numPages = (n * sizeof(unsigned int) + 64 * 1024 - 1) / (64 * 1024);
//     // Elements per warp
//     const size_t elementsPerWarp = pageSize / warpSize;

//     // Warp ID within the grid
//     size_t warpId = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;

//     // Thread lane within the warp
//     size_t lane = threadIdx.x % warpSize;

//     __shared__ unsigned int currentPage;

//     if (lane == 0) {
//         currentPage = atomicAdd(&currentPage, 1);
//     }
//     __syncthreads();

//     // Ensure we don't process out-of-bounds pages
//     while (currentPage < numPages) {
//         size_t pageStart = currentPage * pageSize;
//         bool localChanged = false;

//         // Process elements within the page
//         for (size_t i = 0; i < elementsPerWarp; ++i) {
//             size_t elementIdx = pageStart + lane + i * warpSize;
//             if (elementIdx < n) {
//                 unsigned int k = d_distance[elementIdx];
//                 if (k == level) {
//                     for (size_t j = d_edgesOffset[elementIdx]; j < d_edgesOffset[elementIdx] + d_edgesSize[elementIdx]; ++j) {
//                         unsigned int v = (*d_adjacencyList)[j];
//                         unsigned int dist = d_distance[v];
//                         if (level + 1 < dist) {
//                             d_distance[v] = level + 1;
//                             localChanged = true;
//                         }
//                     }
//                 }
//             }
//         }

//         // Use atomic operation to set the changed flag if needed
//         if (localChanged) {
//             atomicOr(changed, 1);
//         }

//         if (lane == 0) {
//             currentPage = atomicAdd(&currentPage, 1);
//         }
//         __syncthreads();
//     }
// }

// __global__ void 
// kernel_baseline(uint32_t *label, const uint32_t level, const uint64_t vertex_count, const uint64_t *vertexList, 
//                 const uint64_t *edgeList, bool *changed) {
//     const uint64_t tid = blockDim.x * BLOCK_SIZE * blockIdx.y + blockDim.x * blockIdx.x + threadIdx.x;

//     if(tid < vertex_count && label[tid] == level) {
//         const uint64_t start = vertexList[tid];
//         const uint64_t end = vertexList[tid+1];
//         // printf("level: %d label[%d]: %d start: %d end: %d\n", (int) level, (int) tid, (int) label[tid], (int) start, (int) end);
//         for(uint64_t i = start; i < end; i++) {
//             const uint64_t next = edgeList[i];

//             // if(label[next] == MYINFINITY) {
//             if(label[next] > level + 1) {
//                 label[next] = level + 1;
//                 *changed = true;
//             }
//         }
//     }
// }

// __global__ void kernel_coalesce(uint32_t *label, const uint32_t level, const uint64_t vertex_count, 
//                                 const uint64_t *vertexList, const uint64_t *edgeList, bool *changed) {
//     const uint64_t tid = blockDim.x * BLOCK_SIZE * blockIdx.y + blockDim.x * blockIdx.x + threadIdx.x;
//     const uint64_t warpIdx = tid >> WARP_SHIFT;
//     const uint64_t laneIdx = tid & ((1 << WARP_SHIFT) - 1);

//     if(warpIdx < vertex_count && label[warpIdx] == level) {
//         const uint64_t start = vertexList[warpIdx];
//         const uint64_t shift_start = start & MEM_ALIGN;
//         const uint64_t end = vertexList[warpIdx+1];

//         for(uint64_t i = shift_start + laneIdx; i < end; i += WARP_SIZE) {
//             if (i >= start) {
//                 const uint64_t next = edgeList[i];

//                 // if(label[next] == MYINFINITY) {
//                 if(label[next] > level + 1) {
//                     label[next] = level + 1;
//                     *changed = true;
//                 }
//             }
//         }
//     }
// }



// __global__ void kernel_coalesce_chunk_rdma(unsigned int *label, unsigned int level, unsigned int vertex_count, unsigned int *vertexList, rdma_buf<unsigned int> *edgeList, uint *changed) {
//     const uint64_t tid = blockDim.x * BLOCK_SIZE * blockIdx.y + blockDim.x * blockIdx.x + threadIdx.x;
//     const uint64_t warpIdx = tid >> WARP_SHIFT;
//     const uint64_t laneIdx = tid & ((1 << WARP_SHIFT) - 1);
//     const uint64_t chunkIdx = warpIdx * CHUNK_SIZE;
//     uint64_t chunk_size = CHUNK_SIZE;

//     if((chunkIdx + CHUNK_SIZE) > vertex_count) {
//         if ( vertex_count > chunkIdx )
//             chunk_size = vertex_count - chunkIdx;
//         else
//             return;
//     }

//     for(uint32_t i = chunkIdx; i < chunk_size + chunkIdx; i++) {
//         if(label[i] == level) {
//             const uint start = vertexList[i];
//             const uint shift_start = start & MEM_ALIGN;
//             const uint end = vertexList[i+1];

//             for(uint64_t j = shift_start + laneIdx; j < end; j += WARP_SIZE) {
//                 if (j >= start) {
//                     unsigned int next = (*edgeList)[j];

//                     // if(label[next] == MYINFINITY) {
//                     if(label[next] > level + 1) {
//                         label[next] = level + 1;
//                         *changed = 1;
//                     }
//                 }
//             }
//         }
//     }
// }