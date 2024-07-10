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
#include "../../include/runtime.h"


// Size of array
#define N 1*1024*1024llu

#define BLOCK_NUM 1024ULL
#define MYINFINITY 2147483647llu

#define WARP_SHIFT 5
#define WARP_SIZE 32

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

// rdma_buf<unsigned int> *rdma_adjacencyList;
// // rdma_buf<unsigned int> *u_edgesOffset;
// // rdma_buf<unsigned int> *u_edgesSize;
// // rdma_buf<unsigned int> *u_distance;
// unsigned int *u_adjacencyList;
// unsigned int *uvm_adjacencyList;
// unsigned int *u_edgesOffset;
// unsigned int *u_edgesSize;
// unsigned int *u_distance;

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

//     check_cuda_error(cudaMallocManaged(&rdma_adjacencyList, sizeof(rdma_buf<unsigned int>)));
//     check_cuda_error(cudaMallocManaged(&uvm_adjacencyList, G.numEdges*sizeof(unsigned int)));
    
//     check_cuda_error(cudaMallocHost(&u_adjacencyList, G.numEdges*sizeof(unsigned int)));
//     // checkError(cudaMallocManaged(&u_adjacencyList, G.numEdges*sizeof(unsigned int)));
    
//     // checkError(cudaMallocManaged(&u_edgesOffset, sizeof(rdma_buf<unsigned int>)));
//     // checkError(cudaMallocManaged(&u_edgesSize, sizeof(rdma_buf<unsigned int>)));
//     // checkError(cudaMallocManaged(&u_distance, sizeof(rdma_buf<unsigned int>)));
    
//     check_cuda_error(cudaMallocHost(&u_startVertices, G.numEdges*sizeof(unsigned int)));
//     check_cuda_error(cudaMallocHost(&u_distance, G.numVertices*sizeof(unsigned int)));
//     check_cuda_error(cudaMallocHost(&u_edgesOffset, G.numVertices*sizeof(unsigned int)));
//     check_cuda_error(cudaMallocHost(&u_edgesSize, G.numVertices*sizeof(unsigned int)));

//     check_cuda_error(cudaMallocManaged(&u_parent, sizeof(rdma_buf<unsigned int>)));
//     check_cuda_error(cudaMallocManaged(&u_currentQueue, sizeof(rdma_buf<unsigned int>)));
//     check_cuda_error(cudaMallocManaged(&u_nextQueue, sizeof(rdma_buf<unsigned int>)));
//     check_cuda_error(cudaMallocManaged(&u_degrees, sizeof(rdma_buf<unsigned int>)));

//     rdma_adjacencyList->start(G.numEdges *sizeof(unsigned int));
//     // u_edgesOffset->start(G.numVertices *sizeof(unsigned int));
//     // u_edgesSize->start(G.numVertices *sizeof(unsigned int));
//     // u_distance->start(G.numVertices *sizeof(unsigned int));
//     u_parent->start(G.numVertices *sizeof(unsigned int));
//     u_currentQueue->start(G.numVertices *sizeof(unsigned int));
//     u_nextQueue->start(G.numVertices *sizeof(unsigned int));
//     u_degrees->start(G.numVertices *sizeof(unsigned int));

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

//     check_cuda_error(cudaFree(u_adjacencyList));
//     check_cuda_error(cudaFree(u_edgesOffset));
//     check_cuda_error(cudaFree(u_edgesSize));
//     check_cuda_error(cudaFree(u_distance));
//     check_cuda_error(cudaFree(u_parent));
//     check_cuda_error(cudaFree(u_currentQueue));
//     check_cuda_error(cudaFree(u_nextQueue));
//     check_cuda_error(cudaFree(u_degrees));
//     check_cuda_error(cudaFreeHost(incrDegrees));
// }

// void verify_output(std::vector<int> &distance, std::vector<int> &expectedDistance, Graph &G) {
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


// void initializeCudaCC(int startVertex, std::vector<int> &distance, std::vector<int> &parent, Graph &G) {
//     //initialize values
//     // std::fill(distance.begin(), distance.end(), 10000000 /*std::numeric_limits<int>::max()*/);
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
//         u_parent->local_buffer[i] = parent.data()[i];
//     }
//     u_distance[startVertex] = 0;

//     for (size_t i = 0; i < 5; i++)
//     {
//         printf("u_distance->local_buffer[%llu]: %llu; distance.data()[%llu]: %llu\n", i, u_distance[i], i, distance.data()[i]);
//         printf("u_parent->local_buffer[%llu]: %llu; parent.data()[%llu]: %llu\n", i, u_parent->local_buffer[i], i, parent.data()[i]);
        
//     }
    

//     int firstElementQueue = startVertex;
//     // cudaMemcpy(d_currentQueue, &firstElementQueue, sizeof(int), cudaMemcpyHostToDevice);
//     *u_currentQueue->local_buffer = firstElementQueue;
// }



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
        rdma_edgeList->start(G.numEdges*sizeof(unsigned int));
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
        rdma_edgeList->start(G.numEdges*sizeof(unsigned int));
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



__global__
void print_utilization() {
    printf("GPU_address_offset: %llu \n", GPU_address_offset);
}

// Main program
int main(int argc, char **argv)
{   
    // if (argc != 9)
    //     usage(argv[0]);
    
    init_gpu(0);
    printf("Function: %s line number: %d 1024MB: %d bytes REQUEST_SIZE: %d\n",__func__, __LINE__, MB(1024), REQUEST_SIZE);
    int num_msg = (unsigned long) atoi(argv[4]);
    int mesg_size = (unsigned long) atoi(argv[5]);
    int num_bufs = (unsigned long) atoi(argv[6]);

    cudaError_t ret1;

    struct context *s_ctx = (struct context *)malloc(sizeof(struct context));
    struct post_content post_cont, *d_post, host_post;
    struct poll_content poll_cont, *d_poll, host_poll;
    struct post_content2 post_cont2, *d_post2;
    struct host_keys keys;

    int num_iteration = num_msg;
    s_ctx->n_bufs = num_bufs;
    s_ctx->gpu_buf_size = 10*1024*1024*1024llu; // N*sizeof(int)*3llu;

    // remote connection:
    int ret = connect(argv[2], s_ctx);

    // // local connect
    // char *mlx_name = "mlx5_0";
    // int ret = local_connect(mlx_name, s_ctx);

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
    
    
    

    // read graph from standard input
    Graph G;
    Graph_m G_m;
    unsigned int *tmp_edgesOffset, *tmp_edgesSize, *tmp_adjacencyList;
    
    int startVertex = atoi(argv[7]);
    // printf("function: %s line: %d u_edgesOffset->local_buffer: %p\n", __FILE__, __LINE__, u_edgesOffset->local_buffer);

    // readGraph(G, argc, argv);
    readfile(G, G_m, argc, argv, tmp_edgesOffset, tmp_edgesSize, tmp_adjacencyList);

    printf("Number of vertices %lld tmp_edgesOffset[10]: %d\n", G.numVertices, G.edgesOffset_r[10]);
    printf("Number of edges %lld\n\n", G.numEdges);

    // runCudaCC_normal(Graph G, bool rdma, bool transfer_early, bool uvm) 
    unsigned int *rdma_comp = runCudaCC_normal(G, true, false, false);
    // unsigned int *rdma_comp = runCudaCC(G, true, true);
    unsigned int *cuda_comp = runCudaCC_normal(G, false, false, false);

    size_t errors = 0;
    for (size_t i = 0; i < G.numVertices; i++)
    {
        if(rdma_comp[i] != cuda_comp[i]) errors++;
        // printf(" comp_h[%zu]: %llu ", comp_h);
    }
    printf(" errors: %zu\n", errors);
	return 0;
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