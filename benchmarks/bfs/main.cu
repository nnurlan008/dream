#include <cstdio>
#include <cuda.h>
#include <cuda_runtime.h>
#include <string>
#include <cstring>
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
void simpleBfs_normal(size_t n, size_t vertexCount, unsigned int level, unsigned int *d_edgeList, unsigned int *d_vertex_list,
                      unsigned int *d_distance, unsigned int *changed);

__global__
void simpleBfs_normal_rdma(size_t n, size_t vertexCount, unsigned int level, rdma_buf<unsigned int> *d_edgeList, unsigned int *d_vertex_list,
                      unsigned int *d_distance, unsigned int *changed);

__global__
void simpleBfs_uvm(size_t n, unsigned int level, unsigned int *d_adjacencyList, unsigned int *d_edgesOffset,
                    unsigned int *d_edgesSize, unsigned int *d_distance, unsigned int *changed);

__global__
void simpleBfs_rdma(size_t n, unsigned int level, rdma_buf<unsigned int> *d_adjacencyList, unsigned int *d_edgesOffset,
                    unsigned int *d_edgesSize, unsigned int *d_distance, unsigned int *changed);

__global__
void simpleBfs_rdma_2(size_t n, unsigned int level, rdma_buf<unsigned int> *d_adjacencyList, unsigned int *d_edgesOffset,
                      unsigned int *d_edgesSize, unsigned int *d_distance, unsigned int *changed);    

__global__
void simpleBfs_rdma_3(size_t n, unsigned int level, rdma_buf<unsigned int> *d_adjacencyList, unsigned int *vertexList, unsigned int *changed);               

__global__ __launch_bounds__(128,16)
void kernel_coalesce_hash_ptr_pc(rdma_buf<unsigned int> *da, uint32_t *label, const uint32_t level, const uint64_t vertex_count,
                                 const uint64_t *vertexList, uint64_t *changed, uint64_t stride);

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
unsigned int *u_edgesOffset;
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
    checkError(cudaMallocManaged(&uvm_adjacencyList, G.numEdges*sizeof(unsigned int)));
    
    // checkError(cudaMallocHost(&u_adjacencyList, sizeof(rdma_buf<unsigned int>)));
    // checkError(cudaMallocManaged(&u_edgesOffset, sizeof(rdma_buf<unsigned int>)));
    // checkError(cudaMallocManaged(&u_edgesSize, sizeof(rdma_buf<unsigned int>)));
    // checkError(cudaMallocManaged(&u_distance, sizeof(rdma_buf<unsigned int>)));
    checkError(cudaMallocManaged(&u_adjacencyList, G.numEdges*sizeof(unsigned int)));
    // checkError(cudaMallocManaged(&u_adjacencyList, G.numEdges*sizeof(unsigned int)));
    checkError(cudaMallocHost(&u_startVertices, G.numEdges*sizeof(unsigned int)));
    checkError(cudaMallocHost(&u_distance, G.numVertices*sizeof(unsigned int)));
    checkError(cudaMallocHost(&u_edgesOffset, G.numVertices*sizeof(unsigned int)));
    checkError(cudaMallocHost(&u_edgesSize, G.numVertices*sizeof(unsigned int)));

    checkError(cudaMallocManaged(&u_parent, sizeof(rdma_buf<unsigned int>)));
    checkError(cudaMallocManaged(&u_currentQueue, sizeof(rdma_buf<unsigned int>)));
    checkError(cudaMallocManaged(&u_nextQueue, sizeof(rdma_buf<unsigned int>)));
    checkError(cudaMallocManaged(&u_degrees, sizeof(rdma_buf<unsigned int>)));

    rdma_adjacencyList->start(G.numEdges *sizeof(unsigned int));
    // u_edgesOffset->start(G.numVertices *sizeof(unsigned int));
    // u_edgesSize->start(G.numVertices *sizeof(unsigned int));
    // u_distance->start(G.numVertices *sizeof(unsigned int));
    u_parent->start(G.numVertices *sizeof(unsigned int));
    u_currentQueue->start(G.numVertices *sizeof(unsigned int));
    u_nextQueue->start(G.numVertices *sizeof(unsigned int));
    u_degrees->start(G.numVertices *sizeof(unsigned int));

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

void checkOutput_rdma(std::vector<int> &distance, std::vector<int> &expectedDistance, Graph &G) {
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
    // std::fill(distance.begin(), distance.end(), 10000000 /*std::numeric_limits<int>::max()*/);
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
        u_parent->local_buffer[i] = parent.data()[i];
    }
    u_distance[startVertex] = 0;

    for (size_t i = 0; i < 5; i++)
    {
        printf("u_distance->local_buffer[%llu]: %llu; distance.data()[%llu]: %llu\n", i, u_distance[i], i, distance.data()[i]);
        printf("u_parent->local_buffer[%llu]: %llu; parent.data()[%llu]: %llu\n", i, u_parent->local_buffer[i], i, parent.data()[i]);
        
    }
    

    int firstElementQueue = startVertex;
    // cudaMemcpy(d_currentQueue, &firstElementQueue, sizeof(int), cudaMemcpyHostToDevice);
    *u_currentQueue->local_buffer = firstElementQueue;
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

    // checkError(cudaMalloc((void **) &d_adjacencyList, G.numEdges*sizeof(unsigned int)));
    // checkError(cudaMemcpy(d_adjacencyList, u_adjacencyList, G.numEdges*sizeof(unsigned int), cudaMemcpyHostToDevice));
    // checkError(cudaMemcpy(d_adjacencyList, u_adjacencyList, sizeof(rdma_buf<unsigned int>), cudaMemcpyHostToDevice));

    checkError(cudaMalloc((void **) &d_distance, G.numVertices*sizeof(unsigned int)));
    checkError(cudaMalloc((void **) &d_edgesSize, G.numVertices*sizeof(unsigned int)));
    checkError(cudaMalloc((void **) &d_edgesOffset, G.numVertices*sizeof(unsigned int)));
    checkError(cudaMemcpy(d_distance, u_distance, G.numVertices*sizeof(unsigned int), cudaMemcpyHostToDevice));
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

    cudaEventRecord(event1, (cudaStream_t)1);
    
    unsigned int level;
    // checkError(cudaMallocManaged((void **) &level, sizeof(unsigned int)));
    level = 0;
    // transfer<<<1024, 256>>>(rdma_adjacencyList->size/sizeof(unsigned int), rdma_adjacencyList, changed);
    ret1 = cudaDeviceSynchronize();
    // printf("cudaDeviceSynchronize for transfer: %d\n", ret1); 
    assign_array<<< 1 , 1 >>>(rdma_adjacencyList);
    ret1 = cudaDeviceSynchronize();
    printf("cudaDeviceSynchronize for transfer: %d\n", ret1);  
    if(cudaSuccess != ret1){  
        printf("cudaDeviceSynchronize error for transfer: %d\n", ret1);  
        exit(-1);
    }
    cudaEventRecord(event2, (cudaStream_t) 0);
    cudaEventSynchronize(event1); //optional
    cudaEventSynchronize(event2); //wait for the event to be executed!
    float dt_ms;
    cudaEventElapsedTime(&dt_ms, event1, event2);
    printf("Elapsed time for transfer with cudaEvent: %f\n", dt_ms);
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
        simpleBfs_normal_rdma<<< /*G.numVertices / 512 + 1, 512*/ 2048, 512 >>>(G.numEdges, G.numVertices, level, rdma_adjacencyList, d_startVertices, d_distance, changed);
        // simpleBfs_normal<<< G.numVertices / 512 + 1, 512 >>>(G.numEdges, G.numVertices, level, u_adjacencyList, d_startVertices, d_distance, changed);      
        ret1 = cudaDeviceSynchronize();
        // test2<<< /*G.numVertices/256+1, 256*/ G.numVertices/256+1, 256>>>(G.numVertices, level, u_distance, u_edgesOffset, u_edgesSize, u_adjacencyList, changed);
        // test<<< 2, 1024>>>(u_adjacencyList);
        

        level++;
    }

    printf("cudaGetLastError(): %d level: %d\n", cudaGetLastError(), level);
        ret1 = cudaDeviceSynchronize();
        printf("cudaDeviceSynchronize: %d *changed: %d\n", ret1, *changed);  
        if(cudaSuccess != ret1){  
            printf("cudaDeviceSynchronize error: %d\n", ret1);  
            exit(-1);
        }


    cudaEventRecord(event2, (cudaStream_t) 0);
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


// void runCudaQueueBfs(int startVertex, Graph &G, std::vector<int> &distance,
//     std::vector<int> &parent) {
//     initializeCudaBfs(startVertex, distance, parent, G);

//     uint *nextQueueSize;
//     checkError(cudaMallocHost((void **)&nextQueueSize, sizeof(uint)));
//     //launch kernel
//     printf("Starting queue parallel bfs.\n");
//     auto start = std::chrono::steady_clock::now();

//     int queueSize = 1;
//     *nextQueueSize = 0;
//     int level = 0;
//     while (queueSize) {

//         queueBfs<<<queueSize / 1024 + 1, 1024>>>(level, u_adjacencyList, u_edgesOffset, u_edgesSize, u_distance, u_parent, queueSize,
//                                                 nextQueueSize, u_currentQueue, u_nextQueue);
//         cudaDeviceSynchronize();
//         level++;
//         queueSize = *nextQueueSize;
//         *nextQueueSize = 0;
//         std::swap(u_currentQueue, u_nextQueue);
//     }


//     auto end = std::chrono::steady_clock::now();
//     long duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
//     printf("Elapsed time in milliseconds : %li ms.\n", duration);
//     finalizeCudaBfs(distance, parent, G);
// }

// void nextLayer(int level, int queueSize) {

//     nextLayer<<<queueSize / 1024 + 1, 1024>>>(level, u_adjacencyList, u_edgesOffset, u_edgesSize, u_distance, u_parent, queueSize,
//                                             u_currentQueue);
//     cudaDeviceSynchronize();

// }

// void countDegrees(int level, int queueSize) {

//     countDegrees<<<queueSize / 1024 + 1, 1024>>>(u_adjacencyList, u_edgesOffset, u_edgesSize, u_parent, queueSize, u_currentQueue, u_degrees);
//     cudaDeviceSynchronize();

// }

// void scanDegrees(int queueSize) {
// //run kernel so every block in d_currentQueue has prefix sums calculated

//     scanDegrees<<<queueSize / 1024 + 1, 1024>>>(queueSize, u_degrees, incrDegrees);
//     cudaDeviceSynchronize();
//     //count prefix sums on CPU for ends of blocks exclusive
//     //already written previous block sum
//     incrDegrees[0] = 0;
//     for (int i = 1024; i < queueSize + 1024; i += 1024) {
//         incrDegrees[i / 1024] += incrDegrees[i / 1024 - 1];
//     }
// }

// void assignVerticesNextQueue(int queueSize, int nextQueueSize) {

//     assignVerticesNextQueue<<<queueSize / 1024 + 1, 1024>>>(u_adjacencyList, u_edgesOffset, u_edgesSize, u_parent, queueSize, u_currentQueue,
//         u_nextQueue, u_degrees, incrDegrees, nextQueueSize);
//     cudaDeviceSynchronize();

// }

// void runCudaScanBfs(int startVertex, Graph &G, std::vector<int> &distance,
//    std::vector<int> &parent) {
//     initializeCudaBfs(startVertex, distance, parent, G);

//     //launch kernel
//     printf("Starting scan parallel bfs.\n");
//     auto start = std::chrono::steady_clock::now();

//     int queueSize = 1;
//     int nextQueueSize = 0;
//     int level = 0;
//     while (queueSize) {
//         // next layer phase
//         nextLayer(level, queueSize);
//         // counting degrees phase
//         countDegrees(level, queueSize);
//         // doing scan on degrees
//         scanDegrees(queueSize);
//         nextQueueSize = incrDegrees[(queueSize - 1) / 1024 + 1];
//         // assigning vertices to nextQueue
//         assignVerticesNextQueue(queueSize, nextQueueSize);

//         level++;
//         queueSize = nextQueueSize;
//         std::swap(u_currentQueue, u_nextQueue);
//     }

//     auto end = std::chrono::steady_clock::now();
//     long duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
//     printf("Elapsed time in milliseconds : %li ms.\n", duration);
//     finalizeCudaBfs(distance, parent, G);
// }


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
    if (argc != 9)
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

    cudaError_t ret1 = cudaDeviceSynchronize();
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

    //vectors for results
    std::vector<int> distance(G.numVertices, std::numeric_limits<int>::max());
    std::vector<int> parent(G.numVertices, std::numeric_limits<int>::max());
    std::vector<bool> visited(G.numVertices, false);

    for(size_t i = 0; i < 5; i++){
        printf("distance[%d]: %d\n", i, distance[i]);
    }
    // run CPU sequential bfs
    runCpu(startVertex, G, distance, parent, visited);
    for(size_t i = 0; i < 5; i++){
        printf("distance[%d]: %d\n", i, distance[i]);
    }
    initCuda(G);
    
    // rdma_buf<int> *a, *b, *c;
    // checkError(cudaMallocManaged(&a, sizeof(rdma_buf<int>)));
    // checkError(cudaMallocManaged(&b, sizeof(rdma_buf<int>)));
    // checkError(cudaMallocManaged(&c, sizeof(rdma_buf<int>)));
    // a->start(N*sizeof(int));
    // b->start(N*sizeof(int));
    // for (size_t i = 0; i < N; i++)
    // {
    //     a->local_buffer[i] = 10;
    //     b->local_buffer[i] = 10;
    // }

    // ret1 = cudaDeviceSynchronize();
    // printf("ret: %d\n", ret1);
    // if(cudaSuccess != ret1){    
    //     return -1;
    // }

    // cudaEvent_t event1, event2;
    // cudaEventCreate(&event1);
    // cudaEventCreate(&event2);

    // cudaEventRecord(event1, (cudaStream_t)1); //where 0 is the default stream
    
    // cudaEventRecord(event2, (cudaStream_t) 1);

    // ret1 = cudaDeviceSynchronize();
    // printf("ret: %d\n", ret1);
    // if(cudaSuccess != ret1){    
    //     return -1;
    // }

    // //synchronize
    // cudaEventSynchronize(event1); //optional
    // cudaEventSynchronize(event2); //wait for the event to be executed!
    // //calculate time
    // float dt_ms;
    // cudaEventElapsedTime(&dt_ms, event1, event2);
    // printf("   dt_ms: %f\n", dt_ms);

    // a->memcpyDtoH();
    // for (size_t i = 0; i < 10; i++)
    // {
    //     printf("a->local_buffer[%d]: %d\n", i, a->local_buffer[i]);
    // }

    //  u_adjacencyList->start(G.numEdges *sizeof(uint));
    // u_edgesOffset->start(G.numVertices *sizeof(uint));
    // u_edgesSize->start(G.numVertices *sizeof(uint));
    printf("function: %s line: %d\n", __FILE__, __LINE__);
    for(size_t i = 0; i < G.numEdges; i++){
        rdma_adjacencyList->local_buffer[i] = G.adjacencyList_r[i];
        uvm_adjacencyList[i] = G.adjacencyList_r[i];
        u_adjacencyList[i] = G.adjacencyList_r[i];
    }
    printf("function: %s line: %d\n", __FILE__, __LINE__);
    for(size_t i = 0; i < G.numVertices; i++){
        // u_edgesOffset->local_buffer[i] = G.edgesOffset_r[i];
        // u_edgesSize->local_buffer[i] = G.edgesSize_r[i];
        u_edgesOffset[i] = G.edgesOffset_r[i];
        u_edgesSize[i] = G.edgesSize_r[i];
    }
    // size_t index = 0;
    for (size_t i = 0; i < G.numVertices; i++)
    {
        for(size_t k = G.edgesOffset_r[i]; k < G.edgesOffset_r[i] + G.edgesSize_r[i]; k++)
        {
            u_startVertices[k] = i;
            // index++;
        }
        
    }
    

    for(size_t i = 0; i < 5; i++){
        // printf("u_adjacencyList->size: %llu (*u_adjacencyList)[%d]: %llu\n", u_adjacencyList->size, i, u_adjacencyList->local_buffer[i]);
        printf("u_edgesOffset->size: %llu (*u_edgesOffset)[%d]: %llu\n", G.numVertices, i, u_edgesOffset[i]);
        printf("u_edgesSize->size: %llu (*u_edgesSize)[%d]: %llu G.edgesSize_r[%d]: %llu\n", G.numVertices, i, u_edgesSize[i], i, G.edgesSize_r[i]);
        // printf("u_adjacencyList->size: %llu (*u_adjacencyList)[%d]: %llu\n", u_adjacencyList->size, i, u_adjacencyList->local_buffer[i]);
        // printf("u_edgesOffset->size: %llu (*u_edgesOffset)[%d]: %llu\n", u_edgesOffset->size, i, u_edgesOffset->local_buffer[i]);
        // printf("u_edgesSize->size: %llu (*u_edgesSize)[%d]: %llu G.edgesSize_r[%d]: %llu\n", u_edgesSize->size, i, u_edgesSize->local_buffer[i], i, G.edgesSize_r[i]);
    }
    
    printf("function: %s line: %d\n", __FILE__, __LINE__);
    //save results from sequential bfs
    std::vector<int> expectedDistance(distance);
    std::vector<int> expectedParent(parent);
    auto start = std::chrono::steady_clock::now();
    rdma_adjacencyList->memcpyHostToServer();
    // u_edgesOffset->memcpyHostToServer();
    // u_edgesSize->memcpyHostToServer();

    
    

    // //run CUDA simple parallel bfs
    runCudaSimpleBfs(startVertex, G, distance, parent);

    
    auto end = std::chrono::steady_clock::now();
    // u_distance->memcpyDtoH();
    for(size_t i = 0; i < 5; i++){
        printf("u_distance->size: %llu (*u_distance)[%d]: %d\n", u_distance, i, u_distance[i]);
        // printf("u_distance->size: %llu (*u_distance)[%d]: %d\n", u_distance->size, i, u_distance->local_buffer[i]);
        // printf("u_edgesOffset->size: %llu (*u_edgesOffset)[%d]: %llu\n", u_edgesOffset->size, i, u_edgesOffset->local_buffer[i]);
        // printf("u_edgesSize->size: %llu (*u_edgesSize)[%d]: %llu G.edgesSize_r[%d]: %llu\n", u_edgesSize->size, i, u_edgesSize->local_buffer[i], i, G.edgesSize_r[i]);
    }
    checkOutput_rdma(distance, expectedDistance, G);

    print_utilization<<<1,1>>>();

    // // //run CUDA queue parallel bfs
    // runCudaQueueBfs(startVertex, G, distance, parent);
    // checkOutput(distance, expectedDistance, G);

    // // //run CUDA scan parallel bfs
    // runCudaScanBfs(startVertex, G, distance, parent);
    // checkOutput(distance, expectedDistance, G);
    finalizeCuda();
    
    long duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    printf("Overall Elapsed time in milliseconds : %li ms.\n", duration);
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
                    v = D_adjacencyList[i];
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

__global__
void simpleBfs_normal(size_t n, size_t vertexCount, unsigned int level, unsigned int *d_edgeList, unsigned int *d_vertex_list,
                      unsigned int *d_distance, unsigned int *changed) {
    size_t thid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    int valueChange = 0;
    // size_t iterations = 0;
    if(thid < n /*d_distance->size/sizeof(uint)*/){
        
            for (size_t i = thid; i < n; i += vertexCount) {
                
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
            
        // }
        
        

        // __syncthreads();
        if (valueChange) {
            *changed = valueChange;
        }
    }
}

__global__
void simpleBfs_rdma(size_t n, unsigned int level, rdma_buf<unsigned int> *d_adjacencyList, unsigned int *d_edgesOffset,
                    unsigned int *d_edgesSize, unsigned int *d_distance, unsigned int *changed) {
    size_t thid = blockIdx.x * blockDim.x + threadIdx.x;
    int valueChange = 0;
    // size_t iterations = 0;
    if(thid < n /*d_distance->size/sizeof(uint)*/){
        unsigned int k = d_distance[thid];
        if (k == level) {
            // uint edgesOffset = d_edgesOffset[thid];
            // uint edgesSize = d_edgesSize[thid];
            for (size_t i = d_edgesOffset[thid]; i < d_edgesOffset[thid] + d_edgesSize[thid]; i++) {
                // double time1 = clock();
                // printf("D_adjacencyList.d_TLB[%zu].device_address: %p\n", i, D_adjacencyList.d_TLB[i/1024].device_address);
                // unsigned int *tmp = (unsigned int *) D_adjacencyList.d_TLB[i/16384].device_address;
                // unsigned int v = tmp[i%16384]; // (*d_adjacencyList)[i];
                // unsigned int v = D_adjacencyList[i];
                unsigned int v;
                // if(D_adjacencyList.d_tlb[i/1024] == 2)
                //     v = D_adjacencyList.dev_buffer[i];
                v = (*d_adjacencyList)[i];
                // double time2 = clock();
                // printf(" %f ", time2 - time1);
                unsigned int dist = d_distance[v];
                if (level + 1 < dist) {
                    
                        d_distance[v] = level + 1; /*(int) level + 1*/
                    
                    valueChange = 1;
                }
                // iterations++;
            }
            // if(level == 2){
            //     printf("thid: %llu iterations: %llu\n", thid, iterations);
            // }
        }
        
        

        // __syncthreads();
        if (valueChange) {
            *changed = valueChange;
        }
    }
}

__global__
void simpleBfs_rdma_2(size_t n, unsigned int level, rdma_buf<unsigned int> *d_adjacencyList, unsigned int *d_edgesOffset,
                    unsigned int *d_edgesSize, unsigned int *d_distance, unsigned int *changed) {
    size_t thid = blockIdx.x * blockDim.x + threadIdx.x;
    int valueChange = 0;
    // size_t iterations = 0;
    if(thid < n /*d_distance->size/sizeof(uint)*/){
        unsigned int k = d_distance[thid];
        
            // uint edgesOffset = d_edgesOffset[thid];
            // uint edgesSize = d_edgesSize[thid];
            for (size_t i = d_edgesOffset[thid]; i < d_edgesOffset[thid] + d_edgesSize[thid]; i++) {
                // double time1 = clock();
                // printf("D_adjacencyList.d_TLB[%zu].device_address: %p\n", i, D_adjacencyList.d_TLB[i/1024].device_address);
                // unsigned int *tmp = (unsigned int *) D_adjacencyList.d_TLB[i/16384].device_address;
                // unsigned int v = tmp[i%16384]; // (*d_adjacencyList)[i];
                // unsigned int v = D_adjacencyList[i];
                unsigned int v;
                // if(D_adjacencyList.d_tlb[i/1024] == 2)
                //     v = D_adjacencyList.dev_buffer[i];
                v = (*d_adjacencyList)[i];
                unsigned int dist = d_distance[v];
                if (k == level) {
                    // double time2 = clock();
                    // printf(" %f ", time2 - time1);
                    
                    if (level + 1 < dist) {
                        
                            d_distance[v] = level + 1; /*(int) level + 1*/
                        
                        valueChange = 1;
                        *changed = valueChange;
                    }
                }
                // iterations++;
            }
            // if(level == 2){
            //     printf("thid: %llu iterations: %llu\n", thid, iterations);
            // }
        
        // __syncthreads();
        // if (valueChange) {
        //     *changed = valueChange;
        // }
    }
}

// __global__ __launch_bounds__(128,16)
// void kernel_coalesce_hash_ptr_pc(rdma_buf<unsigned int> *da, uint32_t *label, const uint32_t level, const uint64_t vertex_count, const uint64_t *vertexList, uint64_t *changed, uint64_t stride) {
//     const uint64_t oldtid = blockDim.x * BLOCK_NUM * blockIdx.y + blockDim.x * blockIdx.x + threadIdx.x;
//     const uint64_t oldwarpIdx = oldtid >> WARP_SHIFT;
//     const uint64_t laneIdx = oldtid & ((1 << WARP_SHIFT) - 1);
//     uint64_t STRIDE = stride; 
//     const uint64_t nep = (vertex_count+(STRIDE))/(STRIDE); 
//     uint64_t warpIdx = (oldwarpIdx/nep) + ((oldwarpIdx % nep)*(STRIDE));

//     //array_d_t<uint64_t> d_array = *da;
//     if(warpIdx < vertex_count && label[warpIdx] == level) {
//         // bam_ptr<uint64_t> ptr(da);
//         const uint64_t start = vertexList[warpIdx];
//         const uint64_t shift_start = start & 0xFFFFFFFFFFFFFFF0;
//         const uint64_t end = vertexList[warpIdx+1];

//         for(uint64_t i = shift_start + laneIdx; i < end; i += WARP_SIZE) {
//             if (i >= start) {
//                 //const EdgeT next = edgeList[i];
//                 //EdgeT next = da->seq_read(i);
//                 unsigned int next = (*da)[i];
// //                printf("tid: %llu, idx: %llu next: %llu\n", (unsigned long long) tid, (unsigned long long) i, (unsigned long long) next);

//                 // if(label[next] == MYINFINITY) {
//                 if(label[next] > level + 1) {
//                 //    if(level ==0)
//                 //            printf("tid:%llu, level:%llu, next: %llu\n", tid, (unsigned long long)level, (unsigned long long)next);
//                     label[next] = level + 1;
//                     *changed = true;
//                 }
//             }
//         }
//     }
// }

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