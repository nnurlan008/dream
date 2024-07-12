#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
// #include "./utilities/timer.hpp"
#include "graph.hpp"
// #include "./utilities/gpu_error_check.cuh"
#include "global.hpp"
#include "argument_parser.hpp"
#include <iostream> 
#include <fstream> 
#include <chrono>


using namespace std;
// using namespace std;

#define gpuErrorcheck(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

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



// Kernel
__global__ void add_vectors_uvm(int *a, int *b, int *c, int size)
{
	int id = blockDim.x * blockIdx.x + threadIdx.x;
	// if(id < size) {
		c[id] = a[id] + b[id];
		// printf("c[%d]: %d\n", id, c[id]);
	// }
}

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

#define PINNED 0
#define UVM 0

typedef struct OutEdge{
    uint end;
} E;

uint* sssp_CPU(Graph* graph, int source){
    int numNodes = graph->numNodes;
    int numEdges = graph->numEdges;
    uint *dist = new uint[numNodes];
    uint *preNode = new uint[numNodes];
    bool *processed = new bool[numNodes];

    for (int i = 0; i < numNodes; i++) {
        dist[i] = MAX_DIST;
        preNode[i] = uint(-1);
        processed[i] = false;
    }


    for (int i = 0; i < numEdges; i++) {
        Edge edge = graph->edges.at(i);
        if (edge.source == source){
            if (edge.weight < dist[edge.end]){
                dist[edge.end] = edge.weight;
                preNode[edge.end] = source;
            }
        } else {
            // Case: edge.source != source
            continue;
        }
    }

    
    bool finished = false;
    uint numIteration = 0;

    dist[source] = 0;
    preNode[source] = 0;
    processed[source] = true;

    auto start = std::chrono::steady_clock::now();
    while (!finished) {
        // uint minDist = MAX_DIST;
        finished = true;
        numIteration++;

        for (int i = 0; i < numEdges; i++){
            Edge edge = graph->edges.at(i);
            // Update its neighbor
            uint source = edge.source;
            uint end = edge.end;
            uint weight = edge.weight;

            if (dist[source] + weight < dist[end]) {
                dist[end] = dist[source] + weight;
                preNode[end] = source;
                finished = false;
            }
        }
        
    }
    auto end = std::chrono::steady_clock::now();
    

    printf("Process Done!\n");
    printf("Number of Iteration: %d\n", numIteration);
    long duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    printf("Overall Elapsed time in milliseconds : %li ms.\n", duration);
    // printf("The execution time of SSSP on CPU: %f ms\n", timer.elapsedTime());

    return dist;
}

__global__ void sssp_GPU_Kernel_rdma(int numEdges,
                                int numEdgesPerThread,
                                uint *dist,
                                uint *preNode,
                                rdma_buf<unsigned int> *edgesSource,
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
        // uint source = edgesSource[nodeId];
        uint source = (*edgesSource)[nodeId];
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

__global__ void sssp_GPU_Kernel_2(int numEdges,
                                int numEdgesPerThread,
                                uint *dist,
                                uint *preNode,
                                E *edgeList,
                                uint *nodePointer,
                                bool *finished) {
    int threadId = blockDim.x * blockIdx.x + threadIdx.x;
    int startIndex = nodePointer[threadId];
    
    if (startIndex >= numEdges) {
        return;
    }
    
    int endIndex = (threadId + 1) * numEdgesPerThread;
    if (endIndex >= numEdges) {
        endIndex = numEdges;
    }

    for (int nodeId = startIndex; nodeId < endIndex; nodeId++) {
        uint source = threadId; // 
        uint end = edgeList[nodeId].end; // 
        uint weight = 1; // edgesWeight[nodeId];
        
        if (dist[source] + weight < dist[end]) {
            atomicMin(&dist[end], dist[source] + weight);
            // dist[end] = dist[source] + weight;
            preNode[end] = source;
            *finished = false;
        }
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

uint* sssp_GPU(Graph *graph, int source) {
    
    unsigned int numNodes;
    unsigned int numEdges;

    unsigned int *dist ;
    unsigned int *preNode;
    unsigned int *edgesSource;
    unsigned int *edgesEnd;
    unsigned int *nodePointer;
    E *edgeList;
    
    rdma_buf<unsigned int> *rdma_edgesSource;

    // write these array to binary file:
    bool write = false;
    if(write){
        
        numNodes = graph->numNodes;
        numEdges = graph->numEdges;
        dist = new uint[numNodes];
        preNode = new uint[numNodes];
        edgesSource = new uint[numEdges];
        edgesEnd = new uint[numEdges];
        // uint *edgesWeight = new uint[numEdges];

        for (int i = 0; i < numNodes; i++) {
            dist[i] = MAX_DIST;
            preNode[i] = uint(-1);
        }


        for (int i = 0; i < numEdges; i++) {
            Edge edge = graph->edges.at(i);
            
            // Transfer the vector to the following three arrays
            edgesSource[i] = edge.source;
            edgesEnd[i] = edge.end;
            // edgesWeight[i] = edge.weight;

            if (edge.source == source){
                if (edge.weight < dist[edge.end]){
                    dist[edge.end] = edge.weight;
                    preNode[edge.end] = source;
                }
            } else {
                // Case: edge.source != source
                continue;
            }
        }

        string input = string(graph->graphFilePath);
        std::ofstream outfile(/*input.substr(0, input.length()-2)+*/"sssp_tw.bcsr", std::ofstream::binary);
        std::cout << /*input.substr(0, input.length()-2)+*/"sssp_tw.bcsr" << std::endl;

        outfile.write((char*)&numNodes, sizeof(unsigned int));
		outfile.write((char*)&numEdges, sizeof(unsigned int));
		outfile.write ((char*)dist, sizeof(unsigned int)*numNodes);
		outfile.write ((char*)preNode, sizeof(unsigned int)*numNodes);
        outfile.write ((char*)edgesSource, sizeof(unsigned int)*numEdges);
        outfile.write ((char*)edgesEnd, sizeof(unsigned int)*numEdges);
        // outfile.write ((char*)edgesWeight, sizeof(unsigned int)*numEdges);

        outfile.close();
        exit(0);
    }
    else{ // read from binary file
        bool subway = false;
        if(subway) // subway's style binary
        {
            printf("Reading from Subway's binary file...\n");
            ifstream infile (graph->graphFilePath, ios::in | ios::binary);
	
            infile.read ((char*)&numNodes, sizeof(uint));
            infile.read ((char*)&numEdges, sizeof(uint));
            
            nodePointer = new uint[numNodes+1];
            gpuErrorcheck(cudaMallocHost(&edgeList, (numEdges) * sizeof(E)));
            
            infile.read ((char*)nodePointer, sizeof(uint)*numNodes);
            infile.read ((char*)edgeList, sizeof(E)*numEdges);
            nodePointer[numNodes] = numEdges;
            printf("Read from Subway's binary file...\n");
            infile.close();

            graph->numNodes = numNodes;
            graph->numEdges = numEdges;
            dist = new uint[numNodes];
            preNode = new uint[numNodes];
            edgesSource = new uint[numEdges];
            edgesEnd = new uint[numEdges];
            // uint *edgesWeight = new uint[numEdges];

            for (int i = 0; i < numNodes; i++) {
                dist[i] = MAX_DIST;
                preNode[i] = uint(-1);
            }



            for (int i = nodePointer[source]; i < nodePointer[source+1]; i++) {
                // Edge edge = graph->edges.at(i);
                
                // Transfer the vector to the following three arrays
                // edgesSource[i] = edge.source;
                // edgesEnd[i] = edge.end;
                // edgesWeight[i] = edge.weight;

                // if (edge.source == source){
                    if (1 < dist[edgeList[i].end]){
                        dist[edgeList[i].end] = 1;
                        preNode[edgeList[i].end] = source;
                    }
                // } else {
                //     // Case: edge.source != source
                //     continue;
                // }
            }

            dist[source] = 0;
            preNode[source] = 0;


            uint *d_dist;
            uint *d_preNode;
            bool *d_finished;
            E *d_edgeList;
            uint *d_edgesSource;
            uint *d_edgesEnd;
            uint *d_nodePointer;
            // uint *d_edgesWeight;

            gpuErrorcheck(cudaMalloc(&d_dist, numNodes * sizeof(uint)));
            gpuErrorcheck(cudaMalloc(&d_preNode, numNodes * sizeof(uint)));
            gpuErrorcheck(cudaMalloc(&d_finished, sizeof(bool)));
            gpuErrorcheck(cudaMalloc(&d_edgeList, numEdges * sizeof(E)));
            gpuErrorcheck(cudaMalloc(&d_nodePointer, (numNodes+1) * sizeof(uint)));
            // gpuErrorcheck(cudaMalloc(&d_edgesSource, numEdges * sizeof(uint)));
            // gpuErrorcheck(cudaMalloc(&d_edgesEnd, numEdges * sizeof(uint)));
            // gpuErrorcheck(cudaMalloc(&d_edgesWeight, numEdges * sizeof(uint)));

            gpuErrorcheck(cudaMemcpy(d_dist, dist, numNodes * sizeof(uint), cudaMemcpyHostToDevice));
            gpuErrorcheck(cudaMemcpy(d_preNode, preNode, numNodes * sizeof(uint), cudaMemcpyHostToDevice));
            gpuErrorcheck(cudaMemcpy(d_edgeList, edgeList, numEdges * sizeof(E), cudaMemcpyHostToDevice));
            gpuErrorcheck(cudaMemcpy(d_nodePointer, nodePointer, (numNodes+1) * sizeof(uint), cudaMemcpyHostToDevice));
            // gpuErrorcheck(cudaMemcpy(d_edgesSource, edgesSource, numEdges * sizeof(uint), cudaMemcpyHostToDevice));
            // gpuErrorcheck(cudaMemcpy(d_edgesEnd, edgesEnd, numEdges * sizeof(uint), cudaMemcpyHostToDevice));
            // gpuErrorcheck(cudaMemcpy(d_edgesWeight, edgesWeight, numEdges * sizeof(uint), cudaMemcpyHostToDevice));
            
            
            cudaEvent_t event1, event2;
            cudaEventCreate(&event1);
            cudaEventCreate(&event2);

            cudaError_t ret1 = cudaDeviceSynchronize();
            if(cudaSuccess != ret1){  
                printf("cudaDeviceSynchronize error: %d\n", ret1);  
                exit(-1);
            }

            

            int numIteration = 0;
            int numEdgesPerThread = 8;
            int numThreadsPerBlock = 512;
            int numBlock = (numNodes) / (numThreadsPerBlock) + 1;
            // int numBlock = (numEdges) / (numThreadsPerBlock * numEdgesPerThread) + 1;
            bool *finished;

            gpuErrorcheck(cudaMallocHost(&finished, sizeof(bool)));
            *finished = true;

            
            cudaEventRecord(event1, (cudaStream_t)1);
            do {
                numIteration++;
                *finished = true;

                // gpuErrorcheck(cudaMemcpy(d_finished, &finished, sizeof(bool), cudaMemcpyHostToDevice));

                // TO-DO PARALLEL
                // sssp_GPU_Kernel_2<<<numBlock, numThreadsPerBlock>>>(numEdges,
                //                         numEdgesPerThread,
                //                         d_dist,
                //                         d_preNode,
                //                         d_edgeList,
                //                         d_nodePointer,
                //                         d_finished);
                // sssp_GPU_Kernel<<< numBlock, numThreadsPerBlock >>> (numEdges,
                //                                                     numEdgesPerThread,
                //                                                     d_dist,
                //                                                     d_preNode,
                //                                                     d_edgesSource,
                //                                                     d_edgesEnd,
                //                                                     NULL,
                //                                                     d_finished);

                sssp_GPU_Kernel_rdma<<< numBlock, numThreadsPerBlock >>> (numEdges,
                                                                    numEdgesPerThread,
                                                                    d_dist,
                                                                    d_preNode,
                                                                    rdma_edgesSource,
                                                                    d_edgesEnd,
                                                                    NULL,
                                                                    finished);

                // gpuErrorcheck(cudaPeekAtLastError());
                gpuErrorcheck(cudaDeviceSynchronize()); 
                // gpuErrorcheck(cudaMemcpy(&finished, d_finished, sizeof(bool), cudaMemcpyDeviceToHost));
            } while(!finished);
            cudaEventRecord(event2, (cudaStream_t) 1);
            
            cudaEventSynchronize(event1); //optional
            cudaEventSynchronize(event2); //wait for the event to be executed!
            
            float dt_ms;
            cudaEventElapsedTime(&dt_ms, event1, event2);
            printf("The execution time of SSSP on GPU: %f ms\n", dt_ms);

            printf("Process Done!\n");
            printf("Number of Iteration: %d\n", numIteration);
            // printf("The execution time of SSSP on GPU: %f ms\n", timer.elapsedTime());
                
            gpuErrorcheck(cudaMemcpy(dist, d_dist, numNodes * sizeof(uint), cudaMemcpyDeviceToHost));

            gpuErrorcheck(cudaFree(d_dist));
            gpuErrorcheck(cudaFree(d_preNode));
            gpuErrorcheck(cudaFree(d_finished));
            gpuErrorcheck(cudaFree(d_edgesSource));
            gpuErrorcheck(cudaFree(d_edgesEnd));
            // gpuErrorcheck(cudaFree(d_edgesWeight));
            
            return dist;

        }
        else{ // read from sssp_tw.bcsr

            uint *d_dist;
            uint *d_preNode;
            bool *d_finished;
            // E *d_edgeList;
            uint *d_edgesSource;
            uint *d_edgesEnd;
            // uint *d_nodePointer;
            // uint *d_edgesWeight;
            std::cout << "graph->graphFilePath:" << graph->graphFilePath << endl;
            ifstream infile1 (graph->graphFilePath, ios::in | ios::binary);
            printf("Reading from binary file...\n");
            infile1.read ((char*)&numNodes, sizeof(unsigned int));
            
            infile1.read ((char*)&numEdges, sizeof(unsigned int));
            printf("Read from binary file numNodes: %llu numEdges: %llu\n", numNodes, numEdges);

            if(UVM){
                printf("UVM...\n");

                cudaDeviceProp devProp;
                cudaGetDeviceProperties(&devProp, 0);
                // Calculate memory utilization
                size_t totalMemory = devProp.totalGlobalMem;
                size_t freeMemory;
                size_t usedMemory;
                float workload_size = ((float) 2*numNodes*sizeof(uint) + 2*numEdges*sizeof(uint));
                cudaMemGetInfo(&freeMemory, &totalMemory);
                usedMemory = totalMemory - freeMemory;
                printf("Total GPU Memory: %.2f MiB\n", (float) totalMemory / (1024 * 1024));
                printf("Free GPU Memory: %.2f MiB\n", (float) freeMemory / (1024 * 1024));
                printf("Used GPU Memory: %.2f MiB\n", (float) usedMemory / (1024 * 1024));

                printf("Workload size: %.2f\n", workload_size/1024/1024);
                float oversubs_ratio = 1;
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

                gpuErrorcheck(cudaMallocManaged(&d_dist, numNodes * sizeof(uint)));
                gpuErrorcheck(cudaMallocManaged(&d_preNode, numNodes * sizeof(uint)));
                
                // gpuErrorcheck(cudaMalloc(&d_edgeList, numEdges * sizeof(E)));
                // gpuErrorcheck(cudaMalloc(&d_nodePointer, (numNodes+1) * sizeof(uint)));
                gpuErrorcheck(cudaMallocManaged(&d_edgesSource, numEdges * sizeof(uint)));
                gpuErrorcheck(cudaMallocManaged(&d_edgesEnd, numEdges * sizeof(uint)));

                printf("Read from binary file numEdges: %llu\n", numEdges);
                infile1.read ((char*)d_dist, sizeof(unsigned int)*numNodes);
                printf("Read from binary file...\n");
                infile1.read ((char*)d_preNode, sizeof(unsigned int)*numNodes);
                infile1.read ((char*)d_edgesSource, sizeof(unsigned int)*numEdges);
                infile1.read ((char*)d_edgesEnd, sizeof(unsigned int)*numEdges);
                printf("Read from binary file...\n");
                infile1.close();

                d_dist[source] = 0;
                d_preNode[source] = 0;

            }
            else{
                bool rdma = false;
                bool uvm_flag = true;
                if(PINNED){
                    gpuErrorcheck(cudaMallocHost(&dist, (numNodes) * sizeof(unsigned int)));
                    gpuErrorcheck(cudaMallocHost(&preNode, (numNodes) * sizeof(unsigned int)));
                    gpuErrorcheck(cudaMallocHost(&edgesSource, (numEdges) * sizeof(unsigned int)));
                    gpuErrorcheck(cudaMallocHost(&edgesEnd, (numEdges) * sizeof(unsigned int)));
                }
                else {
                    dist = new uint[numNodes];
                    preNode = new uint[numNodes];
                    // uvm_flag = false;
                    if(uvm_flag){
                        gpuErrorcheck(cudaMallocManaged(&edgesSource, (numEdges) * sizeof(unsigned int)));
                        printf("UVM in action\n");
                    }else{
                        edgesSource = new uint[numEdges];
                    }
                    edgesEnd = new uint[numEdges];
                }

                unsigned int *edgesOffset = new uint[numNodes];
                unsigned int *edgesSize = new uint[numNodes];

                printf("Read from binary file numEdges: %llu\n", numEdges);
                infile1.read ((char*)edgesOffset, sizeof(unsigned int)*numNodes);
                infile1.read ((char*)edgesSize, sizeof(unsigned int)*numNodes);

                printf("Reading edgesEnd from binary file numNodes: %llu\n", numNodes);
                infile1.read ((char*)edgesEnd, sizeof(unsigned int)*numEdges);

                printf("Read edgesSource from binary file numNodes: %llu finished\n", numNodes);
                for (size_t i = 0; i < numNodes; i++)
                {
                    for(size_t k = edgesOffset[i]; k < edgesOffset[i] + edgesSize[i]; k++)
                    {
                        edgesSource[k] = i;
                        // index++;
                    }
                    
                }

                delete edgesOffset;
                delete edgesSize;

                for (int i = 0; i < numNodes; i++) {
                    dist[i] = MAX_DIST;
                    preNode[i] = uint(-1);
                }


                for (int i = 0; i < numEdges; i++) {
                    // Edge edge = graph->edges.at(i);
                    
                    // Transfer the vector to the following three arrays
                    // edgesSource[i] = edgesSource[i]; // edge.source;
                    // edgesEnd[i] = edgesEnd[i]; // edge.end;
                    // edgesWeight[i] = edge.weight;

                    if (edgesSource[i] == source){
                        if (1 < dist[edgesEnd[i]]){
                            dist[edgesEnd[i]] = 1;
                            preNode[edgesEnd[i]] = source;
                        }
                    } else {
                        // Case: edge.source != source
                        continue;
                    }
                }

                
                printf("Read from binary file...\n");
                // infile1.read ((char*)preNode, sizeof(unsigned int)*numNodes);
                // infile1.read ((char*)edgesSource, sizeof(unsigned int)*numEdges);
                // infile1.read ((char*)edgesEnd, sizeof(unsigned int)*numEdges);
                printf("Read from binary file...\n");
                infile1.close();

                dist[source] = 0;
                preNode[source] = 0;

                
                
                if(rdma){
                    gpuErrorcheck(cudaMallocManaged(&rdma_edgesSource, sizeof(rdma_buf<unsigned int>)));
                    rdma_edgesSource->start(numEdges*sizeof(unsigned int));
                    for (size_t i = 0; i < numEdges; i++)
                    {
                        rdma_edgesSource->local_buffer[i] = edgesSource[i];
                    }
                    rdma_edgesSource->memcpyHostToServer();
                    printf("edge source is rdma\n");
                    bool transfer_early = false;
                    if(transfer_early){
                        cudaEvent_t event_transfer1, event_transfer2;
                        cudaEventCreate(&event_transfer1);
                        cudaEventCreate(&event_transfer2);
                        cudaError_t ret1 = cudaDeviceSynchronize();
                        printf("cudaDeviceSynchronize for transfer: %d\n", ret1);
                        cudaEventRecord(event_transfer1, (cudaStream_t)1);
                        int numEdgesPerThread = 8;
                        int numThreadsPerBlock = 512;
                        // int numBlock = (numNodes) / (numThreadsPerBlock) + 1;
                        int numBlock = (numEdges) / (numThreadsPerBlock * numEdgesPerThread) + 1;
                        transfer<<</*2048, 512*/numBlock, numThreadsPerBlock>>>(rdma_edgesSource->size/sizeof(unsigned int), rdma_edgesSource, numEdgesPerThread);
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
                else if (!uvm_flag){
                    printf("edge source is not uvm but normal cudamemcpy\n");
                    gpuErrorcheck(cudaMalloc(&d_edgesSource, numEdges * sizeof(uint)));
                    gpuErrorcheck(cudaMemcpy(d_edgesSource, edgesSource, numEdges * sizeof(uint), cudaMemcpyHostToDevice));
                }

                gpuErrorcheck(cudaMalloc(&d_dist, numNodes * sizeof(uint)));
                gpuErrorcheck(cudaMalloc(&d_preNode, numNodes * sizeof(uint)));
                // gpuErrorcheck(cudaMalloc(&d_edgeList, numEdges * sizeof(E)));
                // gpuErrorcheck(cudaMalloc(&d_nodePointer, (numNodes+1) * sizeof(uint)));
                
                gpuErrorcheck(cudaMalloc(&d_edgesEnd, numEdges * sizeof(uint)));
                // gpuErrorcheck(cudaMalloc(&d_edgesWeight, numEdges * sizeof(uint)));

                gpuErrorcheck(cudaMemcpy(d_dist, dist, numNodes * sizeof(uint), cudaMemcpyHostToDevice));
                gpuErrorcheck(cudaMemcpy(d_preNode, preNode, numNodes * sizeof(uint), cudaMemcpyHostToDevice));
                // gpuErrorcheck(cudaMemcpy(d_edgeList, edgeList, numEdges * sizeof(E), cudaMemcpyHostToDevice));
                // gpuErrorcheck(cudaMemcpy(d_nodePointer, nodePointer, (numNodes+1) * sizeof(uint), cudaMemcpyHostToDevice));
                
                gpuErrorcheck(cudaMemcpy(d_edgesEnd, edgesEnd, numEdges * sizeof(uint), cudaMemcpyHostToDevice));
                // gpuErrorcheck(cudaMemcpy(d_edgesWeight, edgesWeight, numEdges * sizeof(uint), cudaMemcpyHostToDevice));

            }
            
            gpuErrorcheck(cudaMalloc(&d_finished, sizeof(bool)));
            
            cudaEvent_t event1, event2;
            cudaEventCreate(&event1);
            cudaEventCreate(&event2);

            cudaError_t ret1 = cudaDeviceSynchronize();
            if(cudaSuccess != ret1){  
                printf("cudaDeviceSynchronize error: %d\n", ret1);  
                exit(-1);
            }

            int numIteration = 0;
            int numEdgesPerThread = 8;
            int numThreadsPerBlock = 512;
            // int numBlock = (numNodes) / (numThreadsPerBlock) + 1;
            int numBlock = (numEdges) / (numThreadsPerBlock * numEdgesPerThread) + 1;
            bool finished = true;

            printf("Process Starts here are line: %d!\n", __LINE__);
            cudaEventRecord(event1, (cudaStream_t)1);
            do {
                numIteration++;
                finished = true;

                gpuErrorcheck(cudaMemcpy(d_finished, &finished, sizeof(bool), cudaMemcpyHostToDevice));

                // TO-DO PARALLEL
                //direct transfer
                // sssp_GPU_Kernel<<< numBlock, numThreadsPerBlock >>> (numEdges,
                //                                                     numEdgesPerThread,
                //                                                     d_dist,
                //                                                     d_preNode,
                //                                                     d_edgesSource,
                //                                                     d_edgesEnd,
                //                                                     NULL,
                //                                                     d_finished);

                // uvm:
                sssp_GPU_Kernel<<< numBlock, numThreadsPerBlock >>> (numEdges,
                                                                    numEdgesPerThread,
                                                                    d_dist,
                                                                    d_preNode,
                                                                    edgesSource,
                                                                    d_edgesEnd,
                                                                    NULL,
                                                                    d_finished);

                // rdma:
                // sssp_GPU_Kernel_rdma<<< numBlock, numThreadsPerBlock >>> (numEdges,
                //                                                     numEdgesPerThread,
                //                                                     d_dist,
                //                                                     d_preNode,
                //                                                     rdma_edgesSource,
                //                                                     d_edgesEnd,
                //                                                     NULL,
                //                                                     d_finished);


                // printf("cudaGetLastError(): %d\n", cudaGetLastError());
                // gpuErrorcheck(cudaPeekAtLastError());
                gpuErrorcheck(cudaDeviceSynchronize()); 
                gpuErrorcheck(cudaMemcpy(&finished, d_finished, sizeof(bool), cudaMemcpyDeviceToHost));
            } while(/*!finished*/numIteration<7);
            cudaEventRecord(event2, (cudaStream_t) 1);
            cudaEventSynchronize(event1); //optional
            cudaEventSynchronize(event2); //wait for the event to be executed!

            // calculate time
            float dt_ms;
            cudaEventElapsedTime(&dt_ms, event1, event2);
            printf("The execution time of SSSP on GPU: %f ms\n", dt_ms);


            printf("Process Done!\n");
            printf("Number of Iteration: %d\n", numIteration);
            // printf("The execution time of SSSP on GPU: %f ms\n", timer.elapsedTime());
                
            // gpuErrorcheck(cudaMemcpy(dist, d_dist, numNodes * sizeof(uint), cudaMemcpyDeviceToHost));

            
            // gpuErrorcheck(cudaFree(d_edgesWeight));
            gpuErrorcheck(cudaMemcpy(dist, d_dist, numNodes * sizeof(uint), cudaMemcpyDeviceToHost));

            gpuErrorcheck(cudaFree(d_dist));
            gpuErrorcheck(cudaFree(d_preNode));
            gpuErrorcheck(cudaFree(d_finished));
            gpuErrorcheck(cudaFree(d_edgesSource));
            gpuErrorcheck(cudaFree(d_edgesEnd));

            Edge newEdge;
            for (size_t i = 0; i < numEdges; i++)
            {
                newEdge.end = edgesEnd[i];
                newEdge.source = edgesSource[i];
                newEdge.weight = 1;

                graph->edges.push_back(newEdge);
            }
            
            graph->numNodes = numNodes;
            graph->numEdges = numEdges;
            
            return dist;


        }
    }

    dist[source] = 0;
    preNode[source] = 0;


    uint *d_dist;
    uint *d_preNode;
    bool *d_finished;
    E *d_edgeList;
    uint *d_edgesSource;
    uint *d_edgesEnd;
    uint *d_nodePointer;
    // uint *d_edgesWeight;

    gpuErrorcheck(cudaMalloc(&d_dist, numNodes * sizeof(uint)));
    gpuErrorcheck(cudaMalloc(&d_preNode, numNodes * sizeof(uint)));
    gpuErrorcheck(cudaMalloc(&d_finished, sizeof(bool)));
    gpuErrorcheck(cudaMalloc(&d_edgeList, numEdges * sizeof(E)));
    gpuErrorcheck(cudaMalloc(&d_nodePointer, (numNodes+1) * sizeof(uint)));
    // gpuErrorcheck(cudaMalloc(&d_edgesSource, numEdges * sizeof(uint)));
    // gpuErrorcheck(cudaMalloc(&d_edgesEnd, numEdges * sizeof(uint)));
    // gpuErrorcheck(cudaMalloc(&d_edgesWeight, numEdges * sizeof(uint)));

    gpuErrorcheck(cudaMemcpy(d_dist, dist, numNodes * sizeof(uint), cudaMemcpyHostToDevice));
    gpuErrorcheck(cudaMemcpy(d_preNode, preNode, numNodes * sizeof(uint), cudaMemcpyHostToDevice));
    gpuErrorcheck(cudaMemcpy(d_edgeList, edgeList, numEdges * sizeof(E), cudaMemcpyHostToDevice));
    gpuErrorcheck(cudaMemcpy(d_nodePointer, nodePointer, (numNodes+1) * sizeof(uint), cudaMemcpyHostToDevice));
    // gpuErrorcheck(cudaMemcpy(d_edgesSource, edgesSource, numEdges * sizeof(uint), cudaMemcpyHostToDevice));
    // gpuErrorcheck(cudaMemcpy(d_edgesEnd, edgesEnd, numEdges * sizeof(uint), cudaMemcpyHostToDevice));
    // gpuErrorcheck(cudaMemcpy(d_edgesWeight, edgesWeight, numEdges * sizeof(uint), cudaMemcpyHostToDevice));
    
    cudaEvent_t event1, event2;
    cudaEventCreate(&event1);
    cudaEventCreate(&event2);

    cudaError_t ret1 = cudaDeviceSynchronize();
    if(cudaSuccess != ret1){  
        printf("cudaDeviceSynchronize error: %d\n", ret1);  
        exit(-1);
    }

    int numIteration = 0;
    int numEdgesPerThread = 8;
    int numThreadsPerBlock = 512;
    int numBlock = (numNodes) / (numThreadsPerBlock) + 1;
    // int numBlock = (numEdges) / (numThreadsPerBlock * numEdgesPerThread) + 1;
    bool finished = true;

    
    cudaEventRecord(event1, (cudaStream_t)1);
    do {
        numIteration++;
        finished = true;

        gpuErrorcheck(cudaMemcpy(d_finished, &finished, sizeof(bool), cudaMemcpyHostToDevice));

        // TO-DO PARALLEL
        sssp_GPU_Kernel_2<<<numBlock, numThreadsPerBlock>>>(numEdges,
                                numEdgesPerThread,
                                d_dist,
                                d_preNode,
                                d_edgeList,
                                d_nodePointer,
                                d_finished);
        

        gpuErrorcheck(cudaPeekAtLastError());
        gpuErrorcheck(cudaDeviceSynchronize()); 
        gpuErrorcheck(cudaMemcpy(&finished, d_finished, sizeof(bool), cudaMemcpyDeviceToHost));
    } while(!finished);
    cudaEventRecord(event2, (cudaStream_t) 1);
    cudaEventSynchronize(event1); //optional
    cudaEventSynchronize(event2); //wait for the event to be executed!

    // calculate time
    float dt_ms;
    cudaEventElapsedTime(&dt_ms, event1, event2);
    printf("The execution time of SSSP on GPU: %f ms\n", dt_ms);


    printf("Process Done!\n");
    printf("Number of Iteration: %d\n", numIteration);
    // printf("The execution time of SSSP on GPU: %f ms\n", timer.elapsedTime());
        
    gpuErrorcheck(cudaMemcpy(dist, d_dist, numNodes * sizeof(uint), cudaMemcpyDeviceToHost));

    gpuErrorcheck(cudaFree(d_dist));
    gpuErrorcheck(cudaFree(d_preNode));
    gpuErrorcheck(cudaFree(d_finished));
    gpuErrorcheck(cudaFree(d_edgesSource));
    gpuErrorcheck(cudaFree(d_edgesEnd));
    // gpuErrorcheck(cudaFree(d_edgesWeight));
    
    return dist;
}

// Main program
int main(int argc, char **argv)
{   
    if (argc < 9)
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
    
    ArgumentParser args(argc, argv);
    // cout << "Input file : " << args.inputFilePath << endl;
    Graph graph(args.inputFilePath);
    //  Graph graph("datasets/simpleGraph.txt");
    printf("main starts...\n");
    graph.readGraph();
    
    int sourceNode;

    if (args.hasSourceNode) {
        sourceNode = args.sourceNode;
    } else {
        // Use graph default source 
        sourceNode = graph.defaultSource;
    }

    uint *dist_gpu = sssp_GPU(&graph, sourceNode);
    args.runOnCPU = true;
    if (args.runOnCPU) {
        uint *dist_cpu = sssp_CPU(&graph, sourceNode);
        compareResult(dist_cpu, dist_gpu, graph.numNodes);
    }
     
    
	return 0;
}

