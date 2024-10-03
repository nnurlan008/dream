#include <cstdio>
#include <cuda.h>
#include <cuda_runtime.h>
#include <string>
#include <cstring>
#include "graph.h"
// #include "bfsCPU.h"
// #include "bfs.cuh"
#include <chrono>

using namespace std;
// using namespace std;


// extern "C"{
//   #include "rdma_utils.h"
// }

// #include "../../src/rdma_utils.cuh"
#include <time.h>
// #include "../../include/runtime_prefetching.h"
#include "../../include/runtime_prefetching_2nic.h"


// Size of array
#define N 1*1024*1024llu

#define BLOCK_NUM 1024ULL
#define MYINFINITY 2147483647llu

#define BLOCK_SIZE 1024
#define WARP_SHIFT 5
#define WARP_SIZE (1 << WARP_SHIFT)

#define CHUNK_SHIFT 3
#define CHUNK_SIZE (1 << CHUNK_SHIFT)

#define MEM_ALIGN_64 (~(0xfULL))
#define MEM_ALIGN_32 (~(0x1fULL))

#define MEM_ALIGN MEM_ALIGN_64

#define GPU 0

typedef uint64_t EdgeT;
typedef uint32_t WeightT;

__device__ rdma_buf<unsigned int> D_adjacencyList;

__global__ void test(rdma_buf<unsigned int> *a/*, rdma_buf<int> *b, rdma_buf<int> *c*/);

__global__ 
void kernel_baseline(bool *curr_visit, bool *next_visit, /*uint64_t*/ unsigned int vertex_count, /*uint64_t*/unsigned int *edgeSize_d, 
                     unsigned int *edgeOffset_d,
                     unsigned int *edgeList, unsigned int *comp, bool *changed, unsigned int *vertexVisitCount_d, unsigned int largebin,
                     unsigned int binelems, unsigned int *neigBin);

__global__ 
void kernel_rdma(bool *curr_visit, bool *next_visit, /*uint64_t*/ unsigned int vertex_count, /*uint64_t*/unsigned int *edgeSize_d, 
                     unsigned int *edgeOffset_d,
                     rdma_buf<unsigned int> *edgeList, unsigned int *comp, bool *changed, unsigned int *vertexVisitCount_d, unsigned int largebin,
                     unsigned int binelems, unsigned int *neigBin);

__global__ 
void kernel_baseline_normal(bool *curr_visit, bool *next_visit, /*uint64_t*/ unsigned int vertex_count, size_t edge_size,
                     unsigned int *endVertices, unsigned int *startVertices, unsigned int *comp, bool *changed, unsigned int *vertexVisitCount_d, unsigned int largebin,
                     unsigned int binelems, unsigned int *neigBin);

__global__ 
void kernel_baseline_normal2(bool *curr_visit, bool *next_visit, int numEdgesPerThread, unsigned int vertex_count, size_t edge_size,
                     unsigned int *endVertices, unsigned int *startVertices, unsigned int *comp, bool *changed, unsigned int *vertexVisitCount_d, unsigned int largebin,
                     unsigned int binelems, unsigned int *neigBin);

__global__ 
void kernel_rdma_normal(bool *curr_visit, bool *next_visit, /*uint64_t*/ unsigned int vertex_count, size_t edge_size,
                     rdma_buf<unsigned int> *endVertices, unsigned int *startVertices, unsigned int *comp, bool *changed, unsigned int *vertexVisitCount_d, unsigned int largebin,
                     unsigned int binelems, unsigned int *neigBin);

__global__ 
void kernel_rdma_normal2(bool *curr_visit, bool *next_visit, int numEdgesPerThread, unsigned int vertex_count, size_t edge_size,
                         rdma_buf<unsigned int> *endVertices, unsigned int *startVertices, unsigned int *comp, bool *changed, unsigned int *vertexVisitCount_d, unsigned int largebin,
                         unsigned int binelems, unsigned int *neigBin);

__global__ void kernel_coalesce(bool *curr_visit, bool *next_visit, uint64_t vertex_count, unsigned int *vertexList,
                                unsigned int *edgeList, unsigned long long *comp, bool *changed);

__global__ void cc_kernel_coalesce_new_repr(bool *curr_visit, size_t n, bool *next_visit, uint64_t new_size, unsigned int *new_vertex_list,
                                uint64_t *new_offset, unsigned int *edgeList, unsigned long long *comp, bool *changed);

__global__ void 
kernel_coalesce_rdma_warp(bool *curr_visit, bool *next_visit, size_t n, uint64_t vertex_count, uint64_t *vertexList,
                          rdma_buf<unsigned int> *edgeList, unsigned long long *comp, bool *changed);

__global__ void kernel_coalesce_rdma(bool *curr_visit, bool *next_visit, uint64_t vertex_count, uint64_t num_edges, unsigned int *vertexList,
                                rdma_buf<unsigned int> *edgeList, unsigned long long *comp, bool *changed);

__global__ void kernel_coalesce_new_repr_rdma(bool *curr_visit, size_t n, bool *next_visit, uint64_t new_size, unsigned int *new_vertex_list,
                                unsigned int *new_offset, rdma_buf<unsigned int> *edgeList, unsigned long long *comp, bool *changed);

__global__ void 
kernel_coalesce_new_repr(bool *curr_visit, bool *next_visit, uint64_t new_size, unsigned int *new_vertexList, unsigned int *new_offset,
                                unsigned int *edgeList, unsigned long long *comp, bool *changed);

__global__ void 
kernel_coalesce_chunk(bool *curr_visit, bool *next_visit, uint64_t vertex_count, uint64_t *vertexList, unsigned int *edgeList,
                      unsigned long long *comp, bool *changed);

__global__ void 
kernel_coalesce_chunk_rdma(bool *curr_visit, bool *next_visit, uint64_t vertex_count, unsigned int *vertexList, 
                           rdma_buf<unsigned int> *edgeList, unsigned long long *comp, bool *changed);

__global__ void 
kernel_coalesce_chunk_rdma_new_repr(bool *curr_visit, bool *next_visit, uint64_t new_size, unsigned int *new_vertexList, 
                           uint64_t *new_offset, rdma_buf<unsigned int> *edgeList, unsigned long long *comp, bool *changed);

__global__ void 
kernel_coalesce_new_repr_uvm(bool *curr_visit, size_t n, bool *next_visit, uint64_t new_size, unsigned int *new_vertex_list,
                            uint64_t *new_offset, unsigned int *edgeList, unsigned long long *comp, bool *changed);

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
    // bfsCPU(startVertex, G, distance, parent, visited);
    auto end = std::chrono::steady_clock::now();
    long duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    printf("Elapsed time in milliseconds : %li ms.\n\n", duration);
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
    printf("D_adjacencyList.d_TLB[0].device_address: %p\n", D_adjacencyList.d_TLB[0].device_address);
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

__global__ void transfer(size_t size, rdma_buf<unsigned int> *d_adjacencyList)
{
    size_t id = blockDim.x * blockIdx.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;
    
        for (size_t i = id; i < size ; i += stride)
        {
            unsigned int y = (*d_adjacencyList)[i];
        }
}

__global__ void transfer(size_t size, rdma_buf<unsigned int> *d_adjacencyList, int numEdgesPerThread)
{
    // size_t id = blockDim.x * blockIdx.x + threadIdx.x;
    // size_t stride = blockDim.x * gridDim.x;
    
    //     for (size_t i = id; i < size ; i += stride)
    //     {
    //         unsigned int y = (*d_adjacencyList)[i];
    //         // y++;
    //         // *changed += y;
    //     }

    int threadId = blockDim.x * blockIdx.x + threadIdx.x;
    int startId = threadId * numEdgesPerThread;
    // if(threadId == 0) printf("hello from sssp\n"); 
    if (startId >= size) {
        return;
    }
    
    int endId = (threadId + 1) * numEdgesPerThread;
    if (endId >= size) {
        endId = size;
    }

    for (int nodeId = startId; nodeId < endId; nodeId++) {
        // uint source = edgesSource[nodeId];
        uint source = (*d_adjacencyList)[nodeId];
    }
}

unsigned int *runCudaCC_normal(Graph G, bool rdma, bool transfer_early, bool uvm) 
{
    cudaError_t ret1;
    unsigned int *neighbin, *comp_d;
    unsigned int *comp_h;
    unsigned int *edgeList_d, *startVertices_d, *u_startVertices;
    bool *curr_visit_d, *next_visit_d, *comp_check, *changed_d;
    unsigned int *vertexVisitCount_d;

    unsigned int max_degree = 0;
    for (size_t i = 0; i < G.numVertices; i++)
    {
        if(max_degree < G.edgesSize_r[i]) max_degree = G.edgesSize_r[i];
    }
    
    rdma_buf<unsigned int> *rdma_edgeList;

    // if(uvm == rdma) {
    //     printf("Error look at uvm: %d rdma: %d\n", uvm, rdma);
    // }

    if(rdma){
        check_cuda_error(cudaMallocManaged((void **) &rdma_edgeList, sizeof(rdma_buf<unsigned int>)));
        rdma_edgeList->start(G.numEdges*sizeof(unsigned int), GPU, NULL);
        for (size_t i = 0; i < G.numEdges; i++)
        {
            rdma_edgeList->local_buffer[i] = G.adjacencyList_r[i];
        }
        rdma_edgeList->memcpyHostToServer();

        if(transfer_early){
            cudaEvent_t event_transfer1, event_transfer2;
            cudaEventCreate(&event_transfer1);
            cudaEventCreate(&event_transfer2);
            ret1 = cudaDeviceSynchronize();
            printf("cudaDeviceSynchronize for transfer: %d\n", ret1);
            cudaEventRecord(event_transfer1, (cudaStream_t)1);
            int numEdgesPerThread = 8;
            int numThreadsPerBlock = 512;
            // int numBlock = (numNodes) / (numThreadsPerBlock) + 1;
            int numBlock = (G.numEdges) / (numThreadsPerBlock * numEdgesPerThread) + 1;
            transfer<<</*2048, 512*/numBlock, numThreadsPerBlock>>>(rdma_edgeList->size/sizeof(unsigned int), rdma_edgeList, numEdgesPerThread);
            // transfer<<<2048, 512>>>(rdma_edgeList->size/sizeof(unsigned int), rdma_edgeList);
            cudaEventRecord(event_transfer2, (cudaStream_t) 1);
            ret1 = cudaDeviceSynchronize();
            printf("cudaDeviceSynchronize for transfer: %d\n", ret1);  
            if(cudaSuccess != ret1){  
                printf("cudaDeviceSynchronize error for transfer: %d\n", ret1);  
                exit(-1);
            }
            cudaEventSynchronize(event_transfer1); //optional
            cudaEventSynchronize(event_transfer2); //wait for the event to be executed!
            float dt_ms;
            cudaEventElapsedTime(&dt_ms, event_transfer1, event_transfer2);
            printf("Elapsed time for transfer with cudaEvent: %f\n", dt_ms);
        }

        
    }
    else if(!rdma && uvm){
        check_cuda_error(cudaMallocManaged((void **) &edgeList_d, G.numEdges * sizeof(unsigned int)));
        for (size_t i = 0; i < G.numEdges; i++)
        {
            edgeList_d[i] = G.adjacencyList_r[i];
        }
        printf("UVM in action");
    }
    else if(!rdma && !uvm){
        check_cuda_error(cudaMalloc((void **) &edgeList_d, G.numEdges * sizeof(unsigned int)));
        check_cuda_error(cudaMemcpy(edgeList_d, G.adjacencyList_r, G.numEdges * sizeof(unsigned int), cudaMemcpyHostToDevice));
        printf("cudamemcpy in action");
    }

    check_cuda_error(cudaMalloc((void**) &comp_d, G.numVertices * sizeof(unsigned int)));
    check_cuda_error(cudaMalloc((void **) &startVertices_d, G.numEdges * sizeof(unsigned int)));
    
    check_cuda_error(cudaMallocHost((void**) &changed_d, sizeof(bool)));
    
    check_cuda_error(cudaMalloc((void**)&vertexVisitCount_d, G.numVertices*sizeof(unsigned long long int)));
    check_cuda_error(cudaMemset(vertexVisitCount_d, 0, G.numVertices*sizeof(unsigned long long int)));

    check_cuda_error(cudaMalloc((void **) &neighbin, max_degree * sizeof(unsigned int)));
    check_cuda_error(cudaMemset(neighbin, 0, max_degree*sizeof(unsigned int)));

    check_cuda_error(cudaMalloc((void**) &curr_visit_d, G.numVertices * sizeof(bool)));
    check_cuda_error(cudaMemset(curr_visit_d, 0x01, G.numVertices * sizeof(bool)));

    check_cuda_error(cudaMalloc((void**) &next_visit_d, G.numVertices * sizeof(bool)));
    check_cuda_error(cudaMemset(next_visit_d, 0x00, G.numVertices * sizeof(bool)));

    comp_h = (unsigned int *) malloc(G.numVertices*sizeof(unsigned int));
    // Initialize values
    for (uint64_t i = 0; i < G.numVertices; i++)
        comp_h[i] = i;
    
    

    check_cuda_error(cudaMemcpy(comp_d, comp_h, G.numVertices * sizeof(unsigned int), cudaMemcpyHostToDevice));

    u_startVertices = (unsigned int *) malloc(G.numEdges*sizeof(unsigned int));
    for (size_t i = 0; i < G.numVertices; i++)
    {
        for(size_t k = G.edgesOffset_r[i]; k < G.edgesOffset_r[i] + G.edgesSize_r[i]; k++)
        {
            u_startVertices[k] = i;
            // index++;
        }
        
    }
    check_cuda_error(cudaMemcpy(startVertices_d, u_startVertices, G.numEdges*sizeof(unsigned int), cudaMemcpyHostToDevice));
    
    
    
    unsigned int binelems = 10000; 

    ret1 = cudaDeviceSynchronize();
    printf("ret1: %d\n", ret1);
    if(ret1 != cudaSuccess) exit(-1);
    int num_iter = 0;

    cudaEvent_t event1, event2;
    cudaEventCreate(&event1);
    cudaEventCreate(&event2);

    int numEdgesPerThread = 8;
    int numThreadsPerBlock = 512;
    // int numBlock = (numNodes) / (numThreadsPerBlock) + 1;
    int numBlock = (G.numEdges) / (numThreadsPerBlock * numEdgesPerThread) + 1;

    cudaEventRecord(event1, (cudaStream_t)1);

    do {

        *changed_d = false;
        // printf("*changed_d : %d\n", *changed_d);
        // printf("cudaGetLastError(): %d num_iter: %d\n", cudaGetLastError(), num_iter);
        if(!rdma){
            kernel_baseline_normal<<< /*G.numVertices/128 + 1, 128*/ 2048, 512 >>>(curr_visit_d, next_visit_d, G.numVertices, 
                                                            G.numEdges, edgeList_d, startVertices_d, comp_d, changed_d, 
                                                            vertexVisitCount_d, max_degree, binelems, neighbin);

            // kernel_baseline_normal2<<< /*G.numVertices/128 + 1, 128*/ numBlock, numThreadsPerBlock >>>(curr_visit_d, next_visit_d, numEdgesPerThread, G.numVertices, 
            //                                                 G.numEdges, edgeList_d, startVertices_d, comp_d, changed_d, 
            //                                                 vertexVisitCount_d, max_degree, binelems, neighbin);
        }
        else{
            // kernel_rdma_normal<<< /*G.numVertices/128 + 1, 128*/ 2048, 512 >>>(curr_visit_d, next_visit_d, G.numVertices, 
            //                                                 G.numEdges, rdma_edgeList, startVertices_d, comp_d, changed_d, 
            //                                                 vertexVisitCount_d, max_degree, binelems, neighbin);

            kernel_rdma_normal2<<< numBlock, numThreadsPerBlock >>>(curr_visit_d, next_visit_d, numEdgesPerThread, G.numVertices, 
                                                            G.numEdges, rdma_edgeList, startVertices_d, comp_d, changed_d, 
                                                            vertexVisitCount_d, max_degree, binelems, neighbin);
        }
        check_cuda_error(cudaMemset(curr_visit_d, 0x00, G.numVertices*sizeof(bool)));
        bool *temp1 = curr_visit_d;
        curr_visit_d = next_visit_d;
        next_visit_d = temp1;
        ret1 = cudaDeviceSynchronize();

        num_iter++;


    }while(*changed_d);

    cudaEventRecord(event2, (cudaStream_t) 1);

    check_cuda_error(cudaMemcpy(comp_h, comp_d, G.numVertices * sizeof(unsigned int), cudaMemcpyDeviceToHost));

    cudaEventSynchronize(event1); //optional
    cudaEventSynchronize(event2); //wait for the event to be executed!
    float dt_ms;
    cudaEventElapsedTime(&dt_ms, event1, event2);
    if(rdma)
        printf("Elapsed time for rdma normal with cudaEvent: %f\n", dt_ms);
    else printf("Elapsed time for baseline normal with cudaEvent: %f\n", dt_ms);

    ret1 = cudaDeviceSynchronize();
    printf("ret1: %d num_iter: %d\n", ret1, num_iter);
    if(ret1 != cudaSuccess) exit(-1);

    if(rdma){
        
        rdma_edgeList->memcpyDtoH();
        rdma_edgeList->memcpyServerToHost();
        size_t errs = 0;
        for (size_t i = 0; i < G.numEdges; i++)
        {
            if(rdma_edgeList->local_buffer[i] != G.adjacencyList_r[i])
                errs++;
        }
        printf("# of errors in tranfser: %zu\n", errs);
    }


    check_cuda_error(cudaFree(comp_d));
    check_cuda_error(cudaFree(curr_visit_d));
    check_cuda_error(cudaFree(vertexVisitCount_d));
    check_cuda_error(cudaFree(neighbin));
    check_cuda_error(cudaFree(startVertices_d));
    if(!rdma)
        check_cuda_error(cudaFree(edgeList_d));

    return comp_h;
}



unsigned int *runCudaCC(Graph G, bool rdma, bool transfer_early) 
{
    cudaError_t ret1;
    unsigned int *neighbin, *comp_d;
    unsigned int *comp_h;
    unsigned int *edgeList_d, *edgeSize_d, *edgeOffset_d;
    bool *curr_visit_d, *next_visit_d, *comp_check, *changed_d;
    unsigned int *vertexVisitCount_d;

    unsigned int max_degree = 0;
    for (size_t i = 0; i < G.numVertices; i++)
    {
        if(max_degree < G.edgesSize_r[i]) max_degree = G.edgesSize_r[i];
    }
    
    rdma_buf<unsigned int> *rdma_edgeList;
    
    if(rdma){
        check_cuda_error(cudaMallocHost((void **) &rdma_edgeList, sizeof(rdma_buf<unsigned int>)));
        rdma_edgeList->start(G.numEdges*sizeof(unsigned int), GPU, NULL);
        for (size_t i = 0; i < G.numEdges; i++)
        {
            rdma_edgeList->local_buffer[i] = G.adjacencyList_r[i];
        }
        rdma_edgeList->memcpyHostToServer();

        if(transfer_early){
            cudaEvent_t event_transfer1, event_transfer2;
            cudaEventCreate(&event_transfer1);
            cudaEventCreate(&event_transfer2);
            ret1 = cudaDeviceSynchronize();
            printf("cudaDeviceSynchronize for transfer: %d\n", ret1);
            cudaEventRecord(event_transfer1, (cudaStream_t)1);
            // transfer<<<2048, 512>>>(rdma_edgeList->size/sizeof(unsigned int), rdma_edgeList);
            cudaEventRecord(event_transfer2, (cudaStream_t) 1);
            ret1 = cudaDeviceSynchronize();
            printf("cudaDeviceSynchronize for transfer: %d\n", ret1);  
            if(cudaSuccess != ret1){  
                printf("cudaDeviceSynchronize error for transfer: %d\n", ret1);  
                exit(-1);
            }
            cudaEventSynchronize(event_transfer1); //optional
            cudaEventSynchronize(event_transfer2); //wait for the event to be executed!
            float dt_ms;
            cudaEventElapsedTime(&dt_ms, event_transfer1, event_transfer2);
            printf("Elapsed time for transfer with cudaEvent: %f\n", dt_ms);
        }

        
    }
    
    check_cuda_error(cudaMalloc((void**) &comp_d, G.numVertices * sizeof(unsigned int)));
    if(!rdma){
        check_cuda_error(cudaMalloc((void **) &edgeList_d, G.numEdges * sizeof(unsigned int)));
        check_cuda_error(cudaMemcpy(edgeList_d, G.adjacencyList_r, G.numEdges * sizeof(unsigned int), cudaMemcpyHostToDevice));
    }

    check_cuda_error(cudaMalloc((void **) &edgeSize_d, G.numVertices * sizeof(unsigned int)));
    check_cuda_error(cudaMalloc((void **) &edgeOffset_d, G.numVertices * sizeof(unsigned int)));
    check_cuda_error(cudaMallocHost((void**) &changed_d, sizeof(bool)));
    
    check_cuda_error(cudaMalloc((void**)&vertexVisitCount_d, G.numVertices*sizeof(unsigned long long int)));
    check_cuda_error(cudaMemset(vertexVisitCount_d, 0, G.numVertices*sizeof(unsigned long long int)));

    check_cuda_error(cudaMalloc((void **) &neighbin, max_degree * sizeof(unsigned int)));
    check_cuda_error(cudaMemset(neighbin, 0, max_degree*sizeof(unsigned int)));

    check_cuda_error(cudaMalloc((void**) &curr_visit_d, G.numVertices * sizeof(bool)));
    check_cuda_error(cudaMemset(curr_visit_d, 0x01, G.numVertices * sizeof(bool)));

    check_cuda_error(cudaMalloc((void**) &next_visit_d, G.numVertices * sizeof(bool)));
    check_cuda_error(cudaMemset(next_visit_d, 0x00, G.numVertices * sizeof(bool)));

    comp_h = (unsigned int *) malloc(G.numVertices*sizeof(unsigned int));
    // Initialize values
    for (uint64_t i = 0; i < G.numVertices; i++)
        comp_h[i] = i;
    
    

    check_cuda_error(cudaMemcpy(comp_d, comp_h, G.numVertices * sizeof(unsigned int), cudaMemcpyHostToDevice));
    check_cuda_error(cudaMemcpy(edgeSize_d, G.edgesSize_r, G.numVertices * sizeof(unsigned int), cudaMemcpyHostToDevice));
    
    check_cuda_error(cudaMemcpy(edgeOffset_d, G.edgesOffset_r, G.numVertices * sizeof(unsigned int), cudaMemcpyHostToDevice));
    
    unsigned int binelems = 10000; 

    ret1 = cudaDeviceSynchronize();
    printf("ret1: %d\n", ret1);
    if(ret1 != cudaSuccess) exit(-1);
    int num_iter = 0;

    cudaEvent_t event1, event2;
    cudaEventCreate(&event1);
    cudaEventCreate(&event2);

    cudaEventRecord(event1, (cudaStream_t)1);

    do {

        *changed_d = false;
        // printf("*changed_d : %d\n", *changed_d);
        // printf("cudaGetLastError(): %d num_iter: %d\n", cudaGetLastError(), num_iter);
        if(!rdma){
            kernel_baseline<<< G.numVertices/32 + 1, 32 >>>(curr_visit_d, next_visit_d, G.numVertices, 
                                                            edgeSize_d, edgeOffset_d, edgeList_d, comp_d, changed_d, 
                                                            vertexVisitCount_d, max_degree, binelems, neighbin);
        }
        else{
            kernel_rdma<<< G.numVertices/32 + 1, 32 >>>(curr_visit_d, next_visit_d, G.numVertices, 
                                                            edgeSize_d, edgeOffset_d, rdma_edgeList, comp_d, changed_d, 
                                                            vertexVisitCount_d, max_degree, binelems, neighbin);
        }
        check_cuda_error(cudaMemset(curr_visit_d, 0x00, G.numVertices*sizeof(bool)));
        bool *temp1 = curr_visit_d;
        curr_visit_d = next_visit_d;
        next_visit_d = temp1;
        ret1 = cudaDeviceSynchronize();

        num_iter++;


    }while(*changed_d);

    cudaEventRecord(event2, (cudaStream_t) 1);

    check_cuda_error(cudaMemcpy(comp_h, comp_d, G.numVertices * sizeof(unsigned int), cudaMemcpyDeviceToHost));

    cudaEventSynchronize(event1); //optional
    cudaEventSynchronize(event2); //wait for the event to be executed!
    float dt_ms;
    cudaEventElapsedTime(&dt_ms, event1, event2);
    printf("Elapsed time for transfer with cudaEvent: %f\n", dt_ms);

    ret1 = cudaDeviceSynchronize();
    printf("ret1: %d num_iter: %d\n", ret1, num_iter);
    if(ret1 != cudaSuccess) exit(-1);


    check_cuda_error(cudaFree(comp_d));
    check_cuda_error(cudaFree(curr_visit_d));
    check_cuda_error(cudaFree(vertexVisitCount_d));
    check_cuda_error(cudaFree(edgeOffset_d));
    check_cuda_error(cudaFree(neighbin));
    check_cuda_error(cudaFree(edgeSize_d));
    if(!rdma)
        check_cuda_error(cudaFree(edgeList_d));

    return comp_h;
}

__global__ void test(rdma_buf<unsigned int> *a/*, rdma_buf<int> *b, rdma_buf<int> *c*/){
    size_t id = blockDim.x * blockIdx.x + threadIdx.x;
    // rdma_buf<unsigned int> b(1024);
    // printf("(*a)[%d]: %d\n", 0, (*a)[0]);
    int k = (*a)[id]; // + (*b)[id];
    // printf("expected: rvalue\n");
    // printf("(*a)[%d]: %d\n", 1, (*a)[1]);
    // printf("(*a)[%d]: %d\n", 2, (*a)[2]);
    // printf("(*a)[%d]: %d\n", 3, (*a)[3]);
    // (*a)[0] = 80;
    
    // b[0] = 3;
    // printf("expected: lvalue b[0]: %d\n", b[0]);
    // k = (const unsigned int) b[0];
    // printf("expected: rvalue k: %d\n", k);
    // a->rvalue(0, 80);
    // a->rvalue(1, 81);
    // a->rvalue(2, 82);
    // a->rvalue(3, 83);

    // printf("(*a)[%d]: %d\n", 0, (*a)[0]);
    // printf("(*a)[%d]: %d\n", 1, (*a)[1]);
    // printf("(*a)[%d]: %d\n", 2, (*a)[2]);
    // printf("(*a)[%d]: %d\n", 3, (*a)[3]);

    // c->rvalue(id, (*a)[id] + (*b)[id]); 
    // if(id == 0) printf("(*b)[%d]: %d\n", id, (*b)[id]);
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

void oversubs(int os, size_t numEdges){
    cudaDeviceProp devProp;
    cudaGetDeviceProperties(&devProp, 0);
    // Calculate memory utilization
    size_t totalMemory = devProp.totalGlobalMem;
    size_t freeMemory;
    size_t usedMemory;
    float workload_size = ((float) numEdges*sizeof(uint));
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

float time_total = 0;
float time_pinning = 0;
rdma_buf<unsigned int> *rdma_edgeList = NULL;

unsigned long long *runCC_rdma(uint startVertex, size_t numVertices, size_t numEdges, unsigned int *edgeList, uint64_t *vertexList,
                uint new_size, unsigned int *new_vertexList, uint64_t *new_offset, int u_case){

    cudaError_t ret;
    uint iter, comp_total = 0;;
    unsigned int *d_edgeList, *uvm_edgeList;
    unsigned int *d_new_vertexList;
    uint64_t *d_new_offset, *d_vertexList;
    bool changed_h, *changed_d;
    bool *curr_visit_d, *next_visit_d, *comp_check;
    unsigned long long *comp_d, *comp_h;
    uint64_t numthreads, numblocks;
    cudaEvent_t start, end;
    float milliseconds;
    
    // u_case:
    // 0: direct transfer edgelist, 1: direct new representation
    // 2: rdma edgelist, 3: rdma new representation
    // 4: uvm edgeList, 5: uvm new representation

    if(u_case == 0 || u_case == 2 || u_case == 4){
        check_cuda_error(cudaMalloc((void **) &d_vertexList, sizeof(uint64_t)*(numVertices+1)));
        check_cuda_error(cudaMemcpy(d_vertexList, vertexList, sizeof(uint64_t)*(numVertices+1), cudaMemcpyHostToDevice));
    }

    if(u_case == 0 || u_case == 1){
        check_cuda_error(cudaMalloc((void **) &d_edgeList, sizeof(unsigned int)*numEdges));
        ret = cudaDeviceSynchronize();

        auto start = std::chrono::steady_clock::now();
        check_cuda_error(cudaMemcpy(d_edgeList, edgeList, sizeof(unsigned int)*numEdges, cudaMemcpyHostToDevice));
        auto end = std::chrono::steady_clock::now();
        long duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        printf("Elapsed time in milliseconds for transfer : %li ms. data: %f\n", 
                duration, (double) numEdges*sizeof(unsigned int)/(1024*1024*1024llu));
    }

    
    // printf("line: %d\n", __LINE__);
    if(u_case == 1 || u_case == 3 || u_case == 5){
        check_cuda_error(cudaMalloc((void **) &d_new_vertexList, sizeof(unsigned int)*new_size));
        check_cuda_error(cudaMemcpy(d_new_vertexList, new_vertexList, sizeof(unsigned int)*new_size, cudaMemcpyHostToDevice));

        check_cuda_error(cudaMalloc((void **) &d_new_offset, sizeof(uint64_t)*(new_size+1)));
        check_cuda_error(cudaMemcpy(d_new_offset, new_offset, sizeof(uint64_t)*(new_size+1), cudaMemcpyHostToDevice));
    }

    
    if(u_case == 2 || u_case == 3){

        if(rdma_edgeList == NULL){

            check_cuda_error(cudaMallocManaged((void **) &rdma_edgeList, sizeof(rdma_buf<unsigned int>)));
            rdma_edgeList->start(numEdges*sizeof(unsigned int), GPU, NULL);
            for(size_t i = 0; i < numEdges; i++){
                rdma_edgeList->local_buffer[i] = edgeList[i];
            }
            
        }
        // rdma_edgeList->memcpyHostToServer();
    }

    if(u_case == 4 || u_case == 5){
        ret = cudaDeviceSynchronize();
        check_cuda_error(cudaMallocManaged((void **) &d_edgeList, sizeof(unsigned int)*numEdges));
        memcpy(d_edgeList, edgeList, numEdges*sizeof(unsigned int));
        ret = cudaDeviceSynchronize();
        auto start = std::chrono::steady_clock::now();
        // check_cuda_error(cudaMemAdvise(d_edgeList, numEdges*sizeof(unsigned int), cudaMemAdviseSetAccessedBy, 0));
        check_cuda_error(cudaMemAdvise(d_edgeList, numEdges*sizeof(unsigned int), cudaMemAdviseSetReadMostly, 0));
        ret = cudaDeviceSynchronize();
        auto end = std::chrono::steady_clock::now();
        long duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        printf("Elapsed time in milliseconds for cudaMemAdviseSetReadMostly : %li ms. data: %f\n", 
                duration, (double) numEdges*sizeof(unsigned int)/(1024*1024*1024llu));
        time_pinning += duration;

    }

    comp_check = (bool*)malloc(numVertices * sizeof(bool));
    comp_h = (unsigned long long*)malloc(numVertices * sizeof(unsigned long long));
    check_cuda_error(cudaMalloc((void**)&curr_visit_d, numVertices * sizeof(bool)));
    check_cuda_error(cudaMalloc((void**)&next_visit_d, numVertices * sizeof(bool)));
    check_cuda_error(cudaMalloc((void**)&comp_d, numVertices * sizeof(unsigned long long)));
    // Initialize values
    for (uint64_t i = 0; i < numVertices; i++)
        comp_h[i] = i;

    memset(comp_check, 0, numVertices * sizeof(bool));
    check_cuda_error(cudaMemset(curr_visit_d, 0x01, numVertices * sizeof(bool)));
    check_cuda_error(cudaMemset(next_visit_d, 0x00, numVertices * sizeof(bool)));
    check_cuda_error(cudaMemcpy(comp_d, comp_h, numVertices * sizeof(uint64_t), cudaMemcpyHostToDevice));

    check_cuda_error(cudaMalloc((void **) &changed_d, sizeof(bool)));
    iter = 0;
    
    auto start2 = std::chrono::steady_clock::now();

    check_cuda_error(cudaEventCreate(&start));
    check_cuda_error(cudaEventCreate(&end));

    check_cuda_error(cudaEventRecord(start, 0));
    ret = cudaDeviceSynchronize();
    // oversubs(10, numEdges);
    do {
        changed_h = false;
        check_cuda_error(cudaMemcpy(changed_d, &changed_h, sizeof(bool), cudaMemcpyHostToDevice));

        auto start = std::chrono::steady_clock::now();
        switch (u_case)
        {
            case 0:{
                    printf("Direct transfer edgelist case: 0\n");
                    // numthreads = BLOCK_SIZE;
                    // numblocks = ((numVertices * WARP_SIZE + numthreads) / numthreads);
                    // dim3 blockDim(BLOCK_SIZE, (numblocks+BLOCK_SIZE)/BLOCK_SIZE);
                    // kernel_coalesce<<<blockDim, numthreads>>>
                    // (curr_visit_d, next_visit_d, numVertices, d_vertexList, d_edgeList, comp_d, changed_d);
                    
                    numthreads = BLOCK_SIZE/2;
                    numblocks = ((numVertices * (WARP_SIZE / CHUNK_SIZE) + numthreads) / numthreads);
                    dim3 blockDim(numthreads, (numblocks+numthreads)/numthreads);
                    kernel_coalesce_chunk<<<blockDim, numthreads>>>
                    (curr_visit_d, next_visit_d, numVertices, d_vertexList, d_edgeList, comp_d, changed_d);
                    ret = cudaDeviceSynchronize();
                    printf("ret: %d cudaGetLastError: %d iter: %d\n", ret, cudaGetLastError(), iter);
                break;
            }
            case 1:{
                    printf("Direct transfer new representation case: 1\n");
                    size_t n_pages = new_size*sizeof(unsigned int)/(4*1024)+1;
                    numthreads = BLOCK_SIZE/2;
                    numblocks = ((new_size * WARP_SIZE + numthreads) / numthreads);
                    dim3 blockDim(numthreads, (numblocks+numthreads)/numthreads);
                    cc_kernel_coalesce_new_repr<<<(new_size*32)/numthreads + 1, numthreads>>>
                    (curr_visit_d, n_pages, next_visit_d, new_size, d_new_vertexList, d_new_offset, d_edgeList, comp_d, changed_d);
                    
                    // numthreads = BLOCK_SIZE/1;
                    // numblocks = ((new_size * WARP_SIZE + numthreads) / numthreads);
                    // dim3 blockDim(numthreads, (numblocks+numthreads)/numthreads);
                    // kernel_coalesce_new_repr<<<blockDim, numthreads>>>
                    // (curr_visit_d, next_visit_d, new_size, d_new_vertexList, d_new_offset, d_edgeList, comp_d, changed_d);

                    ret = cudaDeviceSynchronize();
                    printf("ret: %d cudaGetLastError: %d iter: %d\n", ret, cudaGetLastError(), iter);
                break;
            }            
            case 2:{
                printf("RDMA edgelist case: 2\n");
                size_t n_pages = numVertices*sizeof(unsigned int)/(8*1024)+1;
                ret = cudaDeviceSynchronize(); 
                
                ret = cudaDeviceSynchronize();
                    // numthreads = BLOCK_SIZE/2;
                    // numblocks = ((numVertices * WARP_SIZE + numthreads) / numthreads);
                    // dim3 blockDim(numthreads, (numblocks+numthreads)/numthreads);
                    // kernel_coalesce_chunk_rdma<<<blockDim, numthreads>>>
                    // (curr_visit_d, next_visit_d, numVertices, d_vertexList, rdma_edgeList, comp_d, changed_d);
                ret = cudaDeviceSynchronize();
                print_retires<<<1,1>>>();
                kernel_coalesce_rdma_warp<<<(n_pages*32)/512+1, 512/*(numEdges)/512+1, 512*/>>>
                (curr_visit_d, next_visit_d, n_pages, numVertices, d_vertexList, rdma_edgeList, comp_d, changed_d);
                ret = cudaDeviceSynchronize();
                
                printf("ret: %d cudaGetLastError: %d iter: %d numVertices: %llu\n", ret, cudaGetLastError(), iter, numVertices);
                break;
            }
            case 3:{
                printf("RDMA new representation case: 3\n");
                ret = cudaDeviceSynchronize();
                // size_t n_pages = new_size*sizeof(unsigned int)/(8*1024)+1;
                // numthreads = BLOCK_SIZE/2;
                // numblocks = ((n_pages * WARP_SIZE + numthreads) / numthreads);
                // dim3 blockDim(numthreads, (numblocks+numthreads)/numthreads);
                // // for fs and gap-kron
                // kernel_coalesce_new_repr_rdma<<<(n_pages*32)/256+1, 256/*blockDim, numthreads*/>>>
                // (curr_visit_d, n_pages, next_visit_d, new_size, d_new_vertexList,  d_new_offset, rdma_edgeList, comp_d, changed_d);

                    // for gap-urand
                    numthreads = 1024/2; ///2;
                    numblocks = ((new_size * (WARP_SIZE / CHUNK_SIZE) + numthreads) / numthreads);
                    numblocks = ((new_size * 1 + numthreads) / numthreads);
                    dim3 blockDim(numthreads, (numblocks+numthreads)/numthreads);
                    kernel_coalesce_chunk_rdma_new_repr<<<blockDim, numthreads>>>
                    (curr_visit_d, next_visit_d, new_size, d_new_vertexList, d_new_offset, rdma_edgeList, comp_d, changed_d);
                ret = cudaDeviceSynchronize();
                print_retires<<<1,1>>>();
                printf("ret: %d cudaGetLastError: %d iter: %d numVertices: %llu\n", ret, cudaGetLastError(), iter, numVertices);
                break;
            }
            case 4:{
                printf("UVM  edgelist case: 4\n");
                
                numthreads = BLOCK_SIZE;
                numblocks = ((numVertices * (WARP_SIZE / CHUNK_SIZE) + numthreads) / numthreads);
                dim3 blockDim(numthreads, (numblocks+numthreads)/numthreads);
                kernel_coalesce_chunk<<<blockDim, numthreads>>>
                (curr_visit_d, next_visit_d, numVertices, d_vertexList, d_edgeList, comp_d, changed_d);
                ret = cudaDeviceSynchronize();
                printf("ret: %d cudaGetLastError: %d iter: %d\n", ret, cudaGetLastError(), iter);
                break;
            }
            case 5:{
                printf("UVM new representation case: 5\n");
                size_t n_pages = new_size*sizeof(unsigned int)/(4*1024)+1;
                // for fs and gap-kron
                numthreads = 512; ///2;
                numblocks = ((new_size * (WARP_SIZE / CHUNK_SIZE) + numthreads) / numthreads);
                dim3 blockDim(numthreads, (numblocks+numthreads)/numthreads);
                kernel_coalesce_new_repr_uvm<<</*(n_pages*32)/512+1, 512*/blockDim, numthreads>>>
                (curr_visit_d, n_pages, next_visit_d, new_size, d_new_vertexList,  d_new_offset, d_edgeList, comp_d, changed_d);
                ret = cudaDeviceSynchronize();
                printf("ret: %d cudaGetLastError: %d iter: %d\n", ret, cudaGetLastError(), iter);
                break;
            }
            default:
                break;
        }

        auto end = std::chrono::steady_clock::now();
        long duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        
        printf("Elapsed time in milliseconds: %li ms. for iteration: %d\n\n", duration, iter);
        
        check_cuda_error(cudaMemset(curr_visit_d, 0x00, numVertices * sizeof(bool)));

        bool *temp = curr_visit_d;
        curr_visit_d = next_visit_d;
        next_visit_d = temp;

        iter++;

        check_cuda_error(cudaMemcpy(&changed_h, changed_d, sizeof(bool), cudaMemcpyDeviceToHost));
    } while(changed_h);

    ret = cudaDeviceSynchronize(); 
    print_retires<<<1,1>>>();
    ret = cudaDeviceSynchronize();

    check_cuda_error(cudaEventRecord(end, 0));
    check_cuda_error(cudaEventSynchronize(end));
    check_cuda_error(cudaEventElapsedTime(&milliseconds, start, end));

    auto end2 = std::chrono::steady_clock::now();
    long duration2 = std::chrono::duration_cast<std::chrono::milliseconds>(end2 - start2).count();
    printf("total time by  by chrono: %li ms. data: %f\n", 
            duration2, (double) numEdges*sizeof(unsigned int)/(1024*1024*1024llu));

    printf("number of itereations: %d ", iter);
    time_total += milliseconds;

    check_cuda_error(cudaMemcpy(comp_h, comp_d, numVertices * sizeof(unsigned long long), cudaMemcpyDeviceToHost));

    for (uint64_t i = 0; i < numVertices; i++) {
        if (comp_check[comp_h[i]] == false) {
            comp_check[comp_h[i]] = true;
            comp_total++;
        }
    }

    printf("total components: %u ", comp_total);
    printf("total time bu cudaEvent: %f ms\n\n\n", milliseconds);
    
    if(u_case == 0 || u_case == 1 || u_case == 4 || u_case == 5){
        cudaFree(d_edgeList);
    }

    if(u_case == 1 || u_case == 3 || u_case == 5){
        cudaFree(d_new_vertexList);
        cudaFree(d_new_offset);
    }

    if(u_case == 0 || u_case == 2 || u_case == 4){
        cudaFree(d_vertexList);
    }

    check_cuda_error(cudaFree(curr_visit_d));
    check_cuda_error(cudaFree(next_visit_d));
    check_cuda_error(cudaFree(comp_d));
    check_cuda_error(cudaFree(changed_d));

    return comp_h;
}

// Main program
int main(int argc, char **argv)
{   
    // if (argc != 9)
    //     usage(argv[0]);
    
    // read graph from standard input
    Graph G;
    Graph_m G_m;
    unsigned int *tmp_edgesOffset, *tmp_edgesSize, *tmp_adjacencyList;
    
    int startVertex = atoi(argv[7]);
    // printf("function: %s line: %d u_edgesOffset->local_buffer: %p\n", __FILE__, __LINE__, u_edgesOffset->local_buffer);

    readfile(G, G_m, argc, argv, tmp_edgesOffset, tmp_edgesSize, tmp_adjacencyList);

    init_gpu(0);
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

        s_ctx->gpu_buf_size = 25*1024*1024*1024llu; // N*sizeof(int)*3llu;
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
    printf("Number of edges %lld\n\n", G.numEdges);


    printf("line: %d\n", __LINE__);
    printf("num edges: %llu num vertices: %llu\n", G.numEdges, G.numVertices);
    uint64_t *u_edgeoffset;
    unsigned int *u_edgeList;
    
    u_edgeList = (uint *) malloc(sizeof(uint)*G.numEdges); // new uint64_t[G.numEdges];
    // gpuErrorcheck(cudaMallocHost(&u_edgeList, sizeof(uint)*G.numEdges));
    u_edgeoffset = new uint64_t[G.numVertices + 1];
    printf("line: %d\n", __LINE__);
    for (size_t i = 0; i < G.numEdges; i++)
    {
        u_edgeList[i] = G.adjacencyList_r[i];
    }
    printf("line: %d\n", __LINE__);
    for (size_t i = 0; i < G.numVertices+1; i++)
    {
        u_edgeoffset[i] = G.edgesOffset_r[i];
    }
    printf("line: %d\n", __LINE__);
    u_edgeoffset[G.numVertices] = G.numEdges;
    free(G.edgesOffset_r);
    free(G.adjacencyList_r);

    uint64_t min = 0, max = u_edgeoffset[1] - u_edgeoffset[0], max_node;
    double avg = 0;
    
    for (size_t i = 0; i < G.numVertices; i++)
    {
        uint64_t degree = u_edgeoffset[i+1] - u_edgeoffset[i];
        
        if(max < degree) {
            max = degree;
            max_node = i;
        }
        // if(degree > 128){
        //     min++;
        //     avg += degree;
        //     // printf("degree: %llu\n", degree);
        // }
        // if(min > degree && degree != 0) min = degree;
    }
    avg = avg / G.numVertices;
    printf("avg: %f min: %llu max: : %llu, node: %llu\n", avg, min, max, max_node);
    auto start = std::chrono::steady_clock::now();                
    size_t new_size = 0, treshold = 32; // 16 for fs and gap-kron
    for (size_t i = 0; i < G.numVertices; i++)
    {
        uint64_t degree = u_edgeoffset[i+1] - u_edgeoffset[i];
        
        if(degree <= treshold){
            new_size++;
        }
        else{
            size_t count = degree/treshold + 1;
            new_size += count;
        }
    }
    unsigned int *new_vertex_list;
    uint64_t *new_offset;
    size_t index_zero = 0;
    new_vertex_list = new uint[new_size];
    new_offset = new uint64_t[new_size+1];
    new_offset[0] = 0;
    for (size_t i = 0; i < G.numVertices; i++)
    {
        uint64_t degree = u_edgeoffset[i+1] - u_edgeoffset[i];
        
        if(degree <= treshold){
            new_vertex_list[index_zero] = i;
            new_offset[index_zero+1] = u_edgeoffset[i+1];
            index_zero++;
        }
        else{
            size_t count = degree/treshold + 1;
            size_t total = degree;
            for (size_t k = 0; k < count; k++)
            {
                new_vertex_list[index_zero] = i;
                if(total > treshold) new_offset[index_zero+1] = new_offset[index_zero] + treshold;
                else new_offset[index_zero+1] = u_edgeoffset[i+1];
                index_zero++;
                total = total - treshold;
            }
        }
    }
    auto end = std::chrono::steady_clock::now();
    long duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    printf("Elapsed time for preprocessing in milliseconds : %li ms.\n\n", duration);
    
    printf("number of threads needed for my representation: %llu\n", new_size);
    printf("extra memory for Balanced CSR: %.2f MiB\n", (float) ((new_size*2-G.numVertices)*sizeof(unsigned int)/1024/1024));

    unsigned long long *direct_new_repr;
    unsigned long long *direct_edgelist;
    if(rdma_flag){
        direct_new_repr = runCC_rdma(startVertex, G.numVertices, G.numEdges, u_edgeList, u_edgeoffset, new_size, new_vertex_list, new_offset, 3);
        cudaFree(s_ctx->gpu_buffer);
    }
    else{
        direct_new_repr = runCC_rdma(startVertex, G.numVertices, G.numEdges, u_edgeList, u_edgeoffset, new_size, new_vertex_list, new_offset, 4);
    }

    direct_edgelist = runCC_rdma(startVertex, G.numVertices, G.numEdges, u_edgeList, u_edgeoffset, new_size, new_vertex_list, new_offset, 4);
    // unsigned long long *direct_new_repr = runCC_rdma(startVertex, G, u_edgeList, u_edgeoffset, new_size, new_vertex_list, new_offset, 1);

    // int active_vertices = 0;
    // time_total = 0;
    // int number_of_vertices = 200;
    // if(rdma_flag){
    //     for (size_t i = 0; i < number_of_vertices; i++)
    //     {
    //         startVertex = i;
    //         printf("vertex %d has degree of %d\n", startVertex, u_edgeoffset[i+1] - u_edgeoffset[i]);
    //         if(u_edgeoffset[i+1] - u_edgeoffset[i] == 0)
    //             continue;
    //         active_vertices++;
    //         direct_new_repr = runCC_rdma(startVertex, G.numVertices, G.numEdges, u_edgeList, u_edgeoffset, new_size, new_vertex_list, new_offset, 3);
    //         printf("average time: %.2f pinning time: %.2f\n", time_total/active_vertices, time_pinning/active_vertices);
    //         rdma_edgeList->reset();

    //     }
    //     printf("average time: %.2f pinning time: %.2f\n", time_total/active_vertices, time_pinning/active_vertices);

    //     cudaFree(s_ctx->gpu_buffer);
    // }
    // else {
    //     for (size_t i = 0; i < number_of_vertices; i++)
    //     {
    //         startVertex = i;
    //         printf("vertex %d has degree of %d\n", startVertex, u_edgeoffset[i+1] - u_edgeoffset[i]);
    //         if(u_edgeoffset[i+1] - u_edgeoffset[i] == 0)
    //             continue;
    //         active_vertices++;
    //         runCC_rdma(startVertex, G.numVertices, G.numEdges, u_edgeList, u_edgeoffset, new_size, new_vertex_list, new_offset, 4);
    //         printf("average time: %.2f pinning time: %.2f\n", time_total/active_vertices, time_pinning/active_vertices);

    //     }
    //     printf("average time: %.2f pinning time: %.2f\n", time_total/active_vertices, time_pinning/active_vertices);
    // }
    
    
    size_t unmatches = 0;
    for (size_t i = 0; i < G.numVertices; i++)
    {
        if(direct_edgelist[i] != direct_new_repr[i]){   
            unmatches++;
        }
    }
    

    printf("unmatches: %llu\n", unmatches);
    // runCudaCC_normal(Graph G, bool rdma, bool transfer_early, bool uvm) 
    // unsigned int *rdma_comp = runCudaCC_normal(G, true, false, false);
    // unsigned int *rdma_comp = runCudaCC(G, true, true);
    // unsigned int *cuda_comp = runCudaCC_normal(G, false, false, false);

    // size_t errors = 0;
    // for (size_t i = 0; i < G.numVertices; i++)
    // {
    //     if(rdma_comp[i] != cuda_comp[i]) errors++;
    //     // printf(" comp_h[%zu]: %llu ", comp_h);
    // }
    // printf(" errors: %zu\n", errors);
	return 0;
}

// from emogi
__global__ void kernel_coalesce(bool *curr_visit, bool *next_visit, uint64_t vertex_count, unsigned int *vertexList,
                                unsigned int *edgeList, unsigned long long *comp, bool *changed) {
    const uint64_t tid = blockDim.x * BLOCK_SIZE * blockIdx.y + blockDim.x * blockIdx.x + threadIdx.x;
    const uint64_t warpIdx = tid >> WARP_SHIFT;
    const uint64_t laneIdx = tid & ((1 << WARP_SHIFT) - 1);

    // if(tid == 0) printf("hello from tid: %d\n", tid);

    if (warpIdx < vertex_count && curr_visit[warpIdx] == true) {
        const uint64_t start = vertexList[warpIdx];
        const uint64_t shift_start = start & MEM_ALIGN;
        const uint64_t end = vertexList[warpIdx+1];

        for(uint64_t i = /*shift_*/start + laneIdx; i < end; i += WARP_SIZE) {
            // if (i >= start) {
                unsigned long long comp_src = comp[warpIdx];
                const EdgeT next = edgeList[i];

                unsigned long long comp_next = comp[next];
                unsigned long long comp_target;
                EdgeT next_target;

                if (comp_next != comp_src) {
                    if (comp_src < comp_next) {
                        next_target = next;
                        comp_target = comp_src;
                    }
                    else {
                        next_target = warpIdx;
                        comp_target = comp_next;
                    }

                    atomicMin(&comp[next_target], comp_target);
                    next_visit[next_target] = true;
                    *changed = true;
                }
            // }
        }
    }
}

__global__ void 
kernel_coalesce_new_repr(bool *curr_visit, bool *next_visit, uint64_t new_size, unsigned int *new_vertexList, unsigned int *new_offset,
                                unsigned int *edgeList, unsigned long long *comp, bool *changed) {
    const uint64_t tid = blockDim.x * BLOCK_SIZE * blockIdx.y + blockDim.x * blockIdx.x + threadIdx.x;
    const uint64_t warpIdx = tid >> WARP_SHIFT;
    const uint64_t laneIdx = tid & ((1 << WARP_SHIFT) - 1);

    // if(tid == 0) printf("hello from tid: %d\n", tid);

    if (warpIdx < new_size && curr_visit[new_vertexList[warpIdx]] == true) {
        unsigned int start_vertex = new_vertexList[warpIdx];

        const uint64_t start = new_offset[warpIdx];
        const uint64_t shift_start = start & MEM_ALIGN;
        const uint64_t end = new_offset[warpIdx+1];

        for(uint64_t i = /*shift_*/start + laneIdx; i < end; i += WARP_SIZE) {
            // if (i >= start) {
                unsigned long long comp_src = comp[start_vertex];
                const uint next = edgeList[i];

                unsigned long long comp_next = comp[next];
                unsigned long long comp_target;
                uint next_target;

                if (comp_next != comp_src) {
                    if (comp_src < comp_next) {
                        next_target = next;
                        comp_target = comp_src;
                    }
                    else {
                        next_target = start_vertex;
                        comp_target = comp_next;
                    }

                    atomicMin(&comp[next_target], comp_target);
                    next_visit[next_target] = true;
                    *changed = true;
                }
            // }
        }
    }
}

__global__ void cc_kernel_coalesce_new_repr(bool *curr_visit, size_t n, bool *next_visit, uint64_t new_size, unsigned int *new_vertex_list,
                                uint64_t *new_offset, unsigned int *edgeList, unsigned long long *comp, bool *changed) {
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
                // Process adjacent nodes
                // if(d_edgesOffset[elementIdx+1] - d_edgesOffset[elementIdx] >= 2*1024)
                    // printf("elementx: %llu\n", elementIdx);
                for(size_t j = new_offset[elementIdx]; j < new_offset[elementIdx+1]; ++j) {
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
                }
            }
        }
    }
}

__global__ void kernel_coalesce_rdma_warp(bool *curr_visit, bool *next_visit, size_t n, uint64_t vertex_count, uint64_t *vertexList,
                                rdma_buf<unsigned int> *edgeList, unsigned long long *comp, bool *changed) {
    size_t warp_size = 32;
    // Page size in elements (64KB / 4 bytes per unsigned int)
    const size_t pageSize = 8*1024 / sizeof(unsigned int);
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
            if (elementIdx < vertex_count && curr_visit[elementIdx] == true) {
                // Process adjacent nodes
                // if(d_edgesOffset[elementIdx+1] - d_edgesOffset[elementIdx] >= 2*1024)
                    // printf("elementx: %llu\n", elementIdx);
                for(size_t j = vertexList[elementIdx]; j < vertexList[elementIdx+1]; ++j) {
                    uint next = (*edgeList)[j];
                    if (comp[next] != comp[elementIdx]) {
                        if (comp[elementIdx] < comp[next]) {
                            // next_target = next;
                            // comp_target = comp[elementIdx];
                            atomicMin(&comp[next], comp[elementIdx]);
                            next_visit[next] = true;
                        }
                        else {
                            // next_target = elementIdx;
                            // comp_target = comp[next];
                            atomicMin(&comp[elementIdx], comp[next]);
                            next_visit[elementIdx] = true;   
                        }
                        *changed = true;
                    }
                }
            }
        }
    }
}

__global__ void kernel_coalesce_new_repr_rdma(bool *curr_visit, size_t n, bool *next_visit, uint64_t new_size, unsigned int *new_vertex_list,
                                unsigned int *new_offset, rdma_buf<unsigned int> *edgeList, unsigned long long *comp, bool *changed) {
    size_t warp_size = 32;
    // Page size in elements (64KB / 4 bytes per unsigned int)
    const size_t pageSize = 8*1024 / sizeof(unsigned int);
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
                                uint64_t *new_offset, unsigned int *edgeList, unsigned long long *comp, bool *changed) {
    
    const uint64_t tid = blockDim.x * 512 * blockIdx.y + blockDim.x * blockIdx.x + threadIdx.x;
    const uint64_t warpIdx = tid >> WARP_SHIFT;
    const uint64_t laneIdx = tid & ((1 << WARP_SHIFT) - 1);
    const uint64_t chunkIdx = warpIdx * CHUNK_SIZE;
    uint64_t chunk_size = CHUNK_SIZE;

    if((chunkIdx + CHUNK_SIZE) > new_size) {
        if ( new_size > chunkIdx )
            chunk_size = new_size - chunkIdx;
        else
            return;
    }

    for(uint32_t i = chunkIdx; i < chunk_size + chunkIdx; i++) {
        unsigned int start_vertex = new_vertex_list[i];
        if(curr_visit[start_vertex]) {
            const uint64_t start = new_offset[i];
            // const uint64_t shift_start = start & MEM_ALIGN;
            const uint64_t end = new_offset[i+1];

            for(uint64_t j = /*shift_*/start + laneIdx; j < end; j += WARP_SIZE) {
                // if (j >= start) {
                    // unsigned long long comp_src = comp[i];
                    const uint next = edgeList[j];

                    // unsigned long long comp_next = comp[next];
                    // unsigned long long comp_target;
                    uint next_target;

                    if (comp[next] != comp[start_vertex]) {
                        if (comp[start_vertex] < comp[next]) {
                            // next_target = next;
                            // comp_target = comp[i];
                            atomicMin(&comp[next], comp[start_vertex]);
                            next_visit[next] = true;
                        }
                        else {
                            // next_target = i;
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

    // size_t warp_size = 32;
    // // Page size in elements (64KB / 4 bytes per unsigned int)
    // const size_t pageSize = 4*1024 / sizeof(unsigned int);
    // // Elements per warp
    // const size_t elementsPerWarp = pageSize / warp_size; // warpSize;
    // // Global thread ID
    // size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // // Warp ID within the block
    // size_t warpId = tid / warp_size; // warpSize;

    // // Thread lane within the warp
    // size_t lane = threadIdx.x % warp_size; // warpSize;

    // // Determine which page this warp will process
    // size_t pageStart = warpId * pageSize;

    // // Ensure we don't process out-of-bounds pages
    // if (pageStart < n * pageSize) {
    //     // Process elements within the page
    //     for (size_t i = 0; i < elementsPerWarp; ++i) {
    //         size_t elementIdx = pageStart + lane + i * warp_size; // warpSize;
    //         if (elementIdx < new_size && curr_visit[new_vertex_list[elementIdx]] == true) {
    //             unsigned int start_vertex = new_vertex_list[elementIdx];
                
    //             // const uint64_t start = new_offset[elementIdx];
    //             // const uint64_t shift_start = start & MEM_ALIGN;
    //             // const uint64_t end = new_offset[elementIdx+1];

    //             // Process adjacent nodes
    //             for(size_t j = new_offset[elementIdx]/*&MEM_ALIGN*/; j < new_offset[elementIdx+1]; j += 1) {
    //                 // if(j >= new_offset[elementIdx]){
    //                     uint next = edgeList[j];
    //                     if (comp[next] != comp[start_vertex]) {
    //                         if (comp[start_vertex] < comp[next]) {
    //                             // next_target = next;
    //                             // comp_target = comp[elementIdx];
    //                             atomicMin(&comp[next], comp[start_vertex]);
    //                             next_visit[next] = true;
    //                         }
    //                         else {
    //                             // next_target = elementIdx;
    //                             // comp_target = comp[next];
    //                             atomicMin(&comp[start_vertex], comp[next]);
    //                             next_visit[start_vertex] = true;   
    //                         }
    //                         *changed = true;
    //                     }
    //                 // }
    //             }
    //         }
    //     }
    // }
}

__global__ void kernel_coalesce_rdma(bool *curr_visit, bool *next_visit, uint64_t vertex_count, uint64_t num_edges, unsigned int *vertexList,
                                rdma_buf<unsigned int> *edgeList, unsigned long long *comp, bool *changed) {
    
    const uint64_t tid = blockDim.x * 512 * blockIdx.y + blockDim.x * blockIdx.x + threadIdx.x;
    // blockIdx.x * blockDim.x + threadIdx.x;
    const uint64_t warpIdx = tid >> WARP_SHIFT;
    const uint64_t laneIdx = tid & ((1 << WARP_SHIFT) - 1);

    // if(tid == 0) printf("hello from tid: %d\n", tid);
    // uint next = (*edgeList)[tid];
    if (warpIdx < vertex_count && curr_visit[warpIdx] == true) {
        const uint64_t start = vertexList[warpIdx];
        const uint64_t shift_start = start & MEM_ALIGN;
        const uint64_t end = vertexList[warpIdx+1];

        for(uint64_t i = shift_start + laneIdx; i < end; i += WARP_SIZE) {
            if (i >= start) {
                // if(i >= num_edges || i < 0) 
                //     printf("i: %llu\n", i);
                const uint next = (*edgeList)[i];

                if (comp[next] != comp[warpIdx]) {
                    if (comp[warpIdx] < comp[next]) {
                        atomicMin(&comp[next], comp[warpIdx]);
                        next_visit[next] = true;
                    }
                    else {
                        atomicMin(&comp[warpIdx], comp[next]);
                        next_visit[warpIdx] = true;
                    }
                    *changed = true;
                }
            }
        }
    }
}

__global__ __launch_bounds__(1024,2)
void kernel_coalesce_chunk(bool *curr_visit, bool *next_visit, uint64_t vertex_count, uint64_t *vertexList, unsigned int *edgeList,
                     unsigned long long *comp, bool *changed) {
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
        if(curr_visit[i]) {
            const uint64_t start = vertexList[i];
            const uint64_t shift_start = start & MEM_ALIGN;
            const uint64_t end = vertexList[i+1];

            for(uint64_t j = shift_start + laneIdx; j < end; j += WARP_SIZE) {
                if (j >= start) {
                    // unsigned long long comp_src = comp[i];
                    const uint next = edgeList[j];

                    // unsigned long long comp_next = comp[next];
                    // unsigned long long comp_target;
                    uint next_target;

                    if (comp[next] != comp[i]) {
                        if (comp[i] < comp[next]) {
                            // next_target = next;
                            // comp_target = comp[i];
                            atomicMin(&comp[next], comp[i]);
                            next_visit[next] = true;
                        }
                        else {
                            // next_target = i;
                            // comp_target = comp[next];
                            atomicMin(&comp[i], comp[next]);
                            next_visit[i] = true;
                        }

                        
                        *changed = true;
                    }
                }
            }
        }
    }
}

__global__ void 
kernel_coalesce_chunk_rdma(bool *curr_visit, bool *next_visit, uint64_t vertex_count, unsigned int *vertexList, 
                           rdma_buf<unsigned int> *edgeList, unsigned long long *comp, bool *changed) {

    const uint64_t tid = blockDim.x * 512 * blockIdx.y + blockDim.x * blockIdx.x + threadIdx.x;
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
        if(curr_visit[i]) {
            const uint64_t start = vertexList[i];
            const uint64_t shift_start = start & MEM_ALIGN;
            const uint64_t end = vertexList[i+1];

            for(uint64_t j = shift_start + laneIdx; j < end; j += WARP_SIZE) {
                if (j >= start) {
                    // unsigned long long comp_src = comp[i];
                    const uint next = (*edgeList)[j];

                    // unsigned long long comp_next = comp[next];
                    // unsigned long long comp_target;
                    uint next_target;

                    if (comp[next] != comp[i]) {
                        if (comp[i] < comp[next]) {
                            // next_target = next;
                            // comp_target = comp[i];
                            atomicMin(&comp[next], comp[i]);
                            next_visit[next] = true;
                        }
                        else {
                            // next_target = i;
                            // comp_target = comp[next];
                            atomicMin(&comp[i], comp[next]);
                            next_visit[i] = true;
                        }

                        
                        *changed = true;
                    }
                }
            }
        }
    }
}

__global__ // __launch_bounds__(1024,2)  
void kernel_coalesce_chunk_rdma_new_repr(bool *curr_visit, bool *next_visit, uint64_t new_size, unsigned int *new_vertexList, 
                           uint64_t *new_offset, rdma_buf<unsigned int> *edgeList, unsigned long long *comp, bool *changed) {

    const uint64_t tid = blockDim.x * 512 * blockIdx.y + blockDim.x * blockIdx.x + threadIdx.x;
    const uint64_t warpIdx = tid >> 0; // WARP_SHIFT;
    // const uint64_t warpIdx = tid >> WARP_SHIFT;
    const uint64_t laneIdx = tid & ((1 << 0) - 1); //((1 << WARP_SHIFT) - 1);
    const uint64_t chunkIdx = warpIdx * 1; //CHUNK_SIZE;
    uint64_t chunk_size = 1; // CHUNK_SIZE;

    if((chunkIdx + CHUNK_SIZE) > new_size) {
        if ( new_size > chunkIdx )
            chunk_size = new_size - chunkIdx;
        else
            return;
    }

    for(uint32_t i = chunkIdx; i < chunk_size + chunkIdx; i++) {
        unsigned int start_vertex = new_vertexList[i];
        if(curr_visit[start_vertex]) {
            const uint64_t start = new_offset[i];
            // const uint64_t shift_start = start & MEM_ALIGN;
            const uint64_t end = new_offset[i+1];

            for(uint64_t j = /*shift_*/start + laneIdx; j < end; j += 1) {
                // if (j >= start) {
                    // unsigned long long comp_src = comp[i];
                    const uint next = (*edgeList)[j];

                    // unsigned long long comp_next = comp[next];
                    // unsigned long long comp_target;
                    uint next_target;

                    if (comp[next] != comp[start_vertex]) {
                        if (comp[start_vertex] < comp[next]) {
                            // next_target = next;
                            // comp_target = comp[i];
                            atomicMin(&comp[next], comp[start_vertex]);
                            next_visit[next] = true;
                        }
                        else {
                            // next_target = i;
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

__global__ void 
kernel_coalesce_chunk_rdma_new_repr_warp(bool *curr_visit, bool *next_visit, uint64_t new_size, unsigned int *new_vertexList, 
                           unsigned int *new_offset, rdma_buf<unsigned int> *edgeList, unsigned long long *comp, bool *changed) {

    const uint64_t tid = blockDim.x * 512 * blockIdx.y + blockDim.x * blockIdx.x + threadIdx.x;
    const uint64_t warpIdx = tid >> WARP_SHIFT;
    const uint64_t laneIdx = tid & ((1 << WARP_SHIFT) - 1);
    const uint64_t chunkIdx = warpIdx * CHUNK_SIZE;
    uint64_t chunk_size = CHUNK_SIZE;

    if((chunkIdx + CHUNK_SIZE) > new_size) {
        if ( new_size > chunkIdx )
            chunk_size = new_size - chunkIdx;
        else
            return;
    }

    for(uint32_t i = chunkIdx; i < chunk_size + chunkIdx; i++) {
        unsigned int start_vertex = new_vertexList[i];
        if(curr_visit[start_vertex]) {
            const uint64_t start = new_offset[i];
            const uint64_t shift_start = start & MEM_ALIGN;
            const uint64_t end = new_offset[i+1];

            for(uint64_t j = shift_start + laneIdx; j < end; j += WARP_SIZE) {
                if (j >= start) {
                    // unsigned long long comp_src = comp[i];
                    const uint next = (*edgeList)[j];

                    // unsigned long long comp_next = comp[next];
                    // unsigned long long comp_target;
                    uint next_target;

                    if (comp[next] != comp[start_vertex]) {
                        if (comp[start_vertex] < comp[next]) {
                            // next_target = next;
                            // comp_target = comp[i];
                            atomicMin(&comp[next], comp[start_vertex]);
                            next_visit[next] = true;
                        }
                        else {
                            // next_target = i;
                            // comp_target = comp[next];
                            atomicMin(&comp[start_vertex], comp[next]);
                            next_visit[start_vertex] = true;
                        }

                        
                        *changed = true;
                    }
                }
            }
        }
    }
}

__device__ void cc_compute(unsigned int cid, unsigned int *comp, unsigned int next, bool *next_visit, bool *changed){

    unsigned long long comp_src = comp[cid];
    unsigned long long comp_next = comp[next];
    unsigned long long comp_target;
    unsigned int next_target;

    if (comp_next != comp_src) {
       if (comp_src < comp_next) {
          next_target = next;
          comp_target = comp_src;
       }
       else {
          next_target = cid;
          comp_target = comp_next;
       }
       
       atomicMin(&comp[next_target], comp_target);
       next_visit[next_target] = true;
       *changed = true;
    }
}




__global__ 
void kernel_baseline(bool *curr_visit, bool *next_visit, /*uint64_t*/ unsigned int vertex_count, /*uint64_t*/unsigned int *edgeSize_d, 
                     unsigned int *edgeOffset_d,
                     unsigned int *edgeList, unsigned int *comp, bool *changed, unsigned int *vertexVisitCount_d, unsigned int largebin,
                     unsigned int binelems, unsigned int *neigBin) {
    // const uint64_t tid = blockDim.x * BLOCK_NUM * blockIdx.y + blockDim.x * blockIdx.x + threadIdx.x;
    size_t tid = blockDim.x * blockIdx.x + threadIdx.x;
 
    if(tid < vertex_count && curr_visit[tid] == true){
        const uint64_t start = edgeOffset_d[tid]; // vertexList[tid];
        const uint64_t shift_start = start & 0xFFFFFFFFFFFFFF00;
        const uint64_t end = edgeOffset_d[tid] + edgeSize_d[tid]; // vertexList[tid+1];

        // atomicAdd(&vertexVisitCount_d[tid], (end-start));

        // uint64_t diff =  end - start; 
        // uint64_t diffidx = diff/ binelems; //64 means 512B.
        // if(diffidx>largebin){
        //         diffidx = largebin;
        // }    
        // atomicAdd(&neigBin[diffidx], 1);
              

        for(uint64_t i = start; i < end; i++){
        //for(uint64_t i = start; i < end; i++){
            // if(i >= start){
                const unsigned int next = edgeList[i];
                cc_compute(tid, comp, next, next_visit, changed);

                // unsigned long long comp_src = comp[tid];
                // unsigned long long comp_next = comp[next];
                // unsigned long long comp_target;
                // unsigned int next_target;

                // if (comp_next != comp_src) {
                //     if (comp_src < comp_next) {
                //         next_target = next;
                //         comp_target = comp_src;
                //     }
                //     else {
                //         next_target = tid;
                //         comp_target = comp_next;
                //     }
                    
                //     atomicMin(&comp[next_target], comp_target);
                //     next_visit[next_target] = true;
                //     *changed = true;
                // }

            // }
        }
    }
}

__global__ 
void kernel_baseline_normal(bool *curr_visit, bool *next_visit, /*uint64_t*/ unsigned int vertex_count, size_t edge_size,
                     unsigned int *endVertices, unsigned int *startVertices, unsigned int *comp, bool *changed, unsigned int *vertexVisitCount_d, unsigned int largebin,
                     unsigned int binelems, unsigned int *neigBin) {
    // const uint64_t tid = blockDim.x * BLOCK_NUM * blockIdx.y + blockDim.x * blockIdx.x + threadIdx.x;
    size_t tid = blockDim.x * blockIdx.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;
 
    // if(tid < vertex_count && curr_visit[tid] == true){
        // const uint64_t start = edgeOffset_d[tid]; // vertexList[tid];
        // const uint64_t shift_start = start & 0xFFFFFFFFFFFFFF00;
        // const uint64_t end = edgeOffset_d[tid] + edgeSize_d[tid]; // vertexList[tid+1];

        // atomicAdd(&vertexVisitCount_d[tid], (end-start));

        // uint64_t diff =  end - start; 
        // uint64_t diffidx = diff/ binelems; //64 means 512B.
        // if(diffidx>largebin){
        //         diffidx = largebin;
        // }    
        // atomicAdd(&neigBin[diffidx], 1);
              

        for(uint64_t i = tid; i < edge_size; i += stride){
        //for(uint64_t i = start; i < end; i++){
            // if(i >= start){
                const unsigned int start = startVertices[i];
                if(start < vertex_count && curr_visit[start] == true){
                    const unsigned int next = endVertices[i];
                    cc_compute(start, comp, next, next_visit, changed);
                }

                // unsigned long long comp_src = comp[curr];
                // unsigned long long comp_next = comp[next];
                // unsigned long long comp_target;
                // unsigned int next_target;

                // if (comp_next != comp_src) {
                //     if (comp_src < comp_next) {
                //         next_target = next;
                //         comp_target = comp_src;
                //     }
                //     else {
                //         next_target = curr;
                //         comp_target = comp_next;
                //     }
                    
                //     atomicMin(&comp[next_target], comp_target);
                //     next_visit[next_target] = true;
                //     *changed = true;
                // }

            // }
        }
    // }
}

__global__ 
void kernel_baseline_normal2(bool *curr_visit, bool *next_visit, int numEdgesPerThread, unsigned int vertex_count, size_t edge_size,
                     unsigned int *endVertices, unsigned int *startVertices, unsigned int *comp, bool *changed, unsigned int *vertexVisitCount_d, unsigned int largebin,
                     unsigned int binelems, unsigned int *neigBin){

    int threadId = blockDim.x * blockIdx.x + threadIdx.x;
    int startId = threadId * numEdgesPerThread;
    // if(threadId == 0) printf("hello from sssp\n"); 
    if (startId >= edge_size) {
        return;
    }
    
    int endId = (threadId + 1) * numEdgesPerThread;
    if (endId >= edge_size) {
        endId = edge_size;
    }

    for(int nodeId = startId; nodeId < endId; nodeId++){
        //for(uint64_t i = start; i < end; i++){
            // if(i >= start){
                const unsigned int start = startVertices[nodeId];
                if(start < vertex_count && curr_visit[start] == true){
                    const unsigned int next = endVertices[nodeId];
                    cc_compute(start, comp, next, next_visit, changed);
                }

        }

}

__global__ 
void kernel_rdma(bool *curr_visit, bool *next_visit, /*uint64_t*/ unsigned int vertex_count, /*uint64_t*/unsigned int *edgeSize_d, 
                     unsigned int *edgeOffset_d,
                     rdma_buf<unsigned int> *edgeList, unsigned int *comp, bool *changed, unsigned int *vertexVisitCount_d, unsigned int largebin,
                     unsigned int binelems, unsigned int *neigBin) {
    // const uint64_t tid = blockDim.x * BLOCK_NUM * blockIdx.y + blockDim.x * blockIdx.x + threadIdx.x;
    size_t tid = blockDim.x * blockIdx.x + threadIdx.x;    
 
    if(tid < vertex_count && curr_visit[tid] == true){
        const uint64_t start = edgeOffset_d[tid]; // vertexList[tid];
        const uint64_t shift_start = start & 0xFFFFFFFFFFFFFF00;
        const uint64_t end = edgeOffset_d[tid] + edgeSize_d[tid]; // vertexList[tid+1];

        // atomicAdd(&vertexVisitCount_d[tid], (end-start));

        // uint64_t diff =  end - start; 
        // uint64_t diffidx = diff/ binelems; //64 means 512B.
        // if(diffidx>largebin){
        //         diffidx = largebin;
        // }    
        // atomicAdd(&neigBin[diffidx], 1);
              

        for(uint64_t i = start; i < end; i++){
        //for(uint64_t i = start; i < end; i++){
            // if(i >= start){
                const unsigned int next = (*edgeList)[i];
                
                // cc_compute(tid, comp, next, next_visit, changed);

                unsigned long long comp_src = comp[tid];
                unsigned long long comp_next = comp[next];
                unsigned long long comp_target;
                unsigned int next_target;

                if (comp_next != comp_src) {
                    if (comp_src < comp_next) {
                        next_target = next;
                        comp_target = comp_src;
                    }
                    else {
                        next_target = tid;
                        comp_target = comp_next;
                    }
                    
                    atomicMin(&comp[next_target], comp_target);
                    next_visit[next_target] = true;
                    *changed = true;
                }

            }
        // }
    }
}

__global__ 
void kernel_rdma_normal(bool *curr_visit, bool *next_visit, /*uint64_t*/ unsigned int vertex_count, size_t edge_size,
                     rdma_buf<unsigned int> *endVertices, unsigned int *startVertices, unsigned int *comp, bool *changed, unsigned int *vertexVisitCount_d, unsigned int largebin,
                     unsigned int binelems, unsigned int *neigBin) {
    // const uint64_t tid = blockDim.x * BLOCK_NUM * blockIdx.y + blockDim.x * blockIdx.x + threadIdx.x;
    size_t tid = blockDim.x * blockIdx.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;
 
    // if(tid < vertex_count && curr_visit[tid] == true){
        // const uint64_t start = edgeOffset_d[tid]; // vertexList[tid];
        // const uint64_t shift_start = start & 0xFFFFFFFFFFFFFF00;
        // const uint64_t end = edgeOffset_d[tid] + edgeSize_d[tid]; // vertexList[tid+1];

        // atomicAdd(&vertexVisitCount_d[tid], (end-start));

        // uint64_t diff =  end - start; 
        // uint64_t diffidx = diff/ binelems; //64 means 512B.
        // if(diffidx>largebin){
        //         diffidx = largebin;
        // }    
        // atomicAdd(&neigBin[diffidx], 1);
              

        for(uint64_t i = tid; i < edge_size; i += stride){
        //for(uint64_t i = start; i < end; i++){
            // if(i >= start){
                const unsigned int next = (*endVertices)[i];
                // __nanosleep(1000);
                const unsigned int start = startVertices[i];
                if(curr_visit[start] == true){
                    
                    cc_compute(start, comp, next, next_visit, changed);
                }

                // unsigned long long comp_src = comp[curr];
                // unsigned long long comp_next = comp[next];
                // unsigned long long comp_target;
                // unsigned int next_target;

                // if (comp_next != comp_src) {
                //     if (comp_src < comp_next) {
                //         next_target = next;
                //         comp_target = comp_src;
                //     }
                //     else {
                //         next_target = curr;
                //         comp_target = comp_next;
                //     }
                    
                //     atomicMin(&comp[next_target], comp_target);
                //     next_visit[next_target] = true;
                //     *changed = true;
                // }

            // }
        }
    // }
}

__global__ 
void kernel_rdma_normal2(bool *curr_visit, bool *next_visit, int numEdgesPerThread, unsigned int vertex_count, size_t edge_size,
                     rdma_buf<unsigned int> *endVertices, unsigned int *startVertices, unsigned int *comp, bool *changed, unsigned int *vertexVisitCount_d, unsigned int largebin,
                     unsigned int binelems, unsigned int *neigBin){

    int threadId = blockDim.x * blockIdx.x + threadIdx.x;
    int startId = threadId * numEdgesPerThread;
    // if(threadId == 0) printf("hello from sssp\n"); 
    if (startId >= edge_size) {
        return;
    }
    
    int endId = (threadId + 1) * numEdgesPerThread;
    if (endId >= edge_size) {
        endId = edge_size;
    }

    for(int nodeId = startId; nodeId < endId; nodeId++){
        //for(uint64_t i = start; i < end; i++){
            // if(i >= start){
                const unsigned int start = startVertices[nodeId];
                if(start < vertex_count && curr_visit[start] == true){
                    const unsigned int next = (*endVertices)[nodeId];
                    // __nanosleep(1000);
                    cc_compute(start, comp, next, next_visit, changed);

                    // unsigned long long comp_src = comp[start];
                    // unsigned long long comp_next = comp[next];
                    // unsigned long long comp_target;
                    // unsigned int next_target;

                    // if (comp_next != comp_src) {
                    // if (comp_src < comp_next) {
                    //     next_target = next;
                    //     comp_target = comp_src;
                    // }
                    // else {
                    //     next_target = start;
                    //     comp_target = comp_next;
                    // }
                    
                    // atomicMin(&comp[next_target], comp_target);
                    // next_visit[next_target] = true;
                    // *changed = true;
                    // }

                }

        }

}