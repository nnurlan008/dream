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

#include "graph.h"


using namespace std;
// using namespace std;

<<<<<<< HEAD
#define WEIGHT_ON_GPU 0

=======
>>>>>>> origin/cloudlab
#define BLOCK_SIZE 1024
#define WARP_SHIFT 5
#define WARP_SIZE (1 << WARP_SHIFT)

#define CHUNK_SHIFT 3 // WARP_SHIFT
#define CHUNK_SIZE (1 << CHUNK_SHIFT)

#define MEM_ALIGN_64 (~(0xfULL))
#define MEM_ALIGN_32 (~(0x1fULL))

#define MEM_ALIGN MEM_ALIGN_64

typedef uint64_t EdgeT;
typedef float WeightT;


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
// #include "../../include/runtime_prefetching.h"
#include "../../include/runtime_eviction.h"


// Size of array
#define N 1*1024*1024llu

#define BLOCK_NUM 1024ULL
#define MYINFINITY 2147483647llu

__device__ rdma_buf<unsigned int> D_adjacencyList;

__global__ void update(bool *label, WeightT *costList, WeightT *newCostList, const uint32_t vertex_count, bool *changed);

__global__ void kernel_baseline(uint64_t numEdges, uint64_t numVertex, uint64_t *edgeOffset, uint64_t * edgeList,
                                uint *dist, bool *finished);

__global__ void emogi_new_repr(bool *label, const WeightT *costList, WeightT *newCostList, 
            const uint64_t vertex_count, unsigned int *vertexList, /*EdgeT*/ unsigned int *edgeList, uint64_t *weightList);

__global__ void
kernel_coalesce_rdma(bool *label, const WeightT *costList, WeightT *newCostList, const uint64_t vertex_count, uint64_t *vertexList, 
                                /*EdgeT*/ rdma_buf<unsigned int> *edgeList, WeightT *weightList);
                    
__global__ void 
kernel_baseline_rdma(size_t n, bool *label, const WeightT *costList, WeightT *newCostList, const uint64_t vertex_count,
                     const uint64_t *vertexList, rdma_buf<uint64_t> *edgeList, const WeightT *weightList);

__global__ void 
kernel_baseline_rdma_1(size_t n, uint64_t numEdges, uint64_t numVertex, uint64_t *edgeOffset, rdma_buf<uint64_t> *edgeList,
                                uint *dist, bool *changed);

__global__ void 
kernel_coalesce_rdma_opt(bool *label, size_t n, const WeightT *costList, WeightT *newCostList, const uint64_t vertex_count,
                     unsigned int *vertexList, rdma_buf<unsigned int> *edgeList, const WeightT *weightList);

__global__
void kernel_coalesce_rdma_opt_1(bool *label, size_t n, size_t numVertices, size_t numEdges, unsigned int level, rdma_buf<unsigned int> *d_adjacencyList, 
                    uint64_t *d_edgesOffset, const WeightT *costList, WeightT *newCostList, unsigned int *d_distance);

__global__
void kernel_coalesce_rdma_opt_warp(bool *label, size_t n, size_t numVertices, unsigned int level, rdma_buf<unsigned int> *d_adjacencyList, 
                    uint64_t *d_edgesOffset, const WeightT *costList, WeightT *newCostList, unsigned int *d_distance);


__global__ void kernel_coalesce_new_repr(bool *label, const WeightT *costList, WeightT *newCostList, int n, const uint64_t new_size, uint64_t *new_offset, 
                                unsigned int *new_vertex_list, const /*EdgeT*/ unsigned int *edgeList, const WeightT *weightList);

__global__ void kernel_coalesce_new_repr_rdma(bool *label, size_t n, const WeightT *costList, WeightT *newCostList, const uint64_t new_size, uint64_t *new_offset, 
                                unsigned int *new_vertex_list, rdma_buf<unsigned int> *edgeList, const WeightT *weightList);

__global__ void emogi_csr_repr(bool *label, const WeightT *costList, WeightT *newCostList, const uint64_t vertex_count, uint64_t *vertexList, 
                                /*EdgeT*/ unsigned int *edgeList, WeightT *weightList);

__global__ void 
emogi_csr_repr_opt(bool *label, size_t n, const WeightT *costList, WeightT *newCostList, const uint64_t vertex_count,
                     uint64_t *vertexList, unsigned int *edgeList, const WeightT *weightList);

__global__ void 
kernel_coalesce_rdma_bfs_new_repr(bool *label, size_t n, const WeightT *costList, WeightT *newCostList, const uint64_t new_size, uint64_t *new_offset, 
                                  unsigned int *new_vertex_list, rdma_buf<unsigned int> *edgeList, float *weightList, bool *d_changed);

__global__ void
emogi_rdma_bfs_new_repr(bool *label, size_t n, const WeightT *costList, WeightT *newCostList, const uint64_t new_size, uint64_t *new_offset, 
                                  unsigned int *new_vertex_list, unsigned int *edgeList, float *weightList, bool *d_changed);

<<<<<<< HEAD
__global__ // __launch_bounds__(1024,2) 
void kernel_coalesce_new_repr_rdma_weight(bool *label, size_t n, const WeightT *costList, WeightT *newCostList, const uint64_t new_size, uint64_t *new_offset, 
                                unsigned int *new_vertex_list, rdma_buf<unsigned int> *edgeList, rdma_buf<float> *weightList);

__global__ void __launch_bounds__(1024,2) 
kernel_coalesce_rdma_weights(bool *label, const WeightT *costList, WeightT *newCostList, const uint64_t vertex_count, uint64_t *vertexList, 
                                /*EdgeT*/ rdma_buf<unsigned int> *edgeList, rdma_buf<WeightT> *weightList);

=======
>>>>>>> origin/cloudlab
__global__
void check_edgeList(rdma_buf<unsigned int> *a, unsigned int *b, size_t size);

// Kernel
__global__ void add_vectors_uvm(int *a, int *b, int *c, int size)
{
	int id = blockDim.x * blockIdx.x + threadIdx.x;
	// if(id < size) {
		c[id] = a[id] + b[id];
		// printf("c[%d]: %d\n", id, c[id]);
	// }
}

__device__ void AtomicMin(float * const address, const float value)
{
	if (*address <= value)
		return;

	uint32_t * const address_as_i = (uint32_t*)address;
    uint32_t old = *address_as_i, assumed;

	do {
        assumed = old;
		if (__int_as_float(assumed) <= value)
			break;               

        old = atomicCAS(address_as_i, assumed, __int_as_float(value));
    } while (assumed != old);
}

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
    // g_qp_index = 0;
    // for (size_t i = 0; i < 128; i++)
    // {
    //     max = 0;
    // }
}

__global__ void
copy_page_fault_number(size_t *pf_min, uint *node, uint startVertex){
    
    if(*pf_min > g_qp_index){
        *pf_min = g_qp_index;
        *node = startVertex;
    }
    printf("page fault min: %llu startVertex: %d\n", *pf_min, *node);    
    g_qp_index = 0;
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

// typedef struct OutEdge{
//     uint end;
// } E;

uint* sssp_CPU(uint source, uint64_t numVertex, uint64_t numEdges, uint64_t *vertexList, uint64_t *edgeList){
    int numNodes = numVertex;
    // int numEdges = numEdges;
    uint *dist = new uint[numNodes];
    uint *preNode = new uint[numNodes];
    bool *processed = new bool[numNodes];

    for (int i = 0; i < numNodes; i++) {
        dist[i] = MAX_DIST;
        preNode[i] = uint(-1);
        processed[i] = false;
    }


    for (int i = 0; i < numEdges; i++) {
        for (size_t k = vertexList[i]; k < vertexList[i+1]; k++)
        {
            if (i == source){
                uint64_t end = edgeList[i];
                if (1 < dist[end]){
                    dist[end] = 1;
                    preNode[end] = source;
                }
            } else {
                // Case: edge.source != source
                continue;
            }
        }

        // Edge edge = graph->edges.at(i);
        // if (i == source){
        //     if (edge.weight < dist[edge.end]){
        //         dist[edge.end] = edge.weight;
        //         preNode[edge.end] = source;
        //     }
        // } else {
        //     // Case: edge.source != source
        //     continue;
        // }
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


        for (size_t i = 0; i < numVertex; i++)
        {
            for (size_t k = vertexList[i]; k < vertexList[i+1]; k++)
            {
                uint64_t end = edgeList[i];
                uint weight = 1;

                if (dist[source] + weight < dist[end]) {
                    dist[end] = dist[source] + weight;
                    preNode[end] = source;
                    finished = false;
                }
                
            }
            
        }
        
        // for (int i = 0; i < numEdges; i++){
        //     Edge edge = graph->edges.at(i);
        //     // Update its neighbor
        //     uint source = edge.source;
        //     uint end = edge.end;
        //     uint weight = edge.weight;

        //     if (dist[source] + weight < dist[end]) {
        //         dist[end] = dist[source] + weight;
        //         preNode[end] = source;
        //         finished = false;
        //     }
        // }
        
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

__device__ size_t diff = 0;
<<<<<<< HEAD
void *tmp_ptr = NULL;
void *over_ptr = NULL;
=======
>>>>>>> origin/cloudlab

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
<<<<<<< HEAD
    
=======
    void *tmp_ptr;
>>>>>>> origin/cloudlab
    cudaMalloc(&tmp_ptr, (size_t) (freeMemory - workload_size));
    cudaMemGetInfo(&freeMemory, &totalMemory);
    printf("Free GPU Memory: %.2f MiB\n", (float) freeMemory / (1024 * 1024));
    if(oversubs_ratio > 0){
        
<<<<<<< HEAD
=======
        void *over_ptr;
>>>>>>> origin/cloudlab
        long long unsigned int os_size = freeMemory - workload_size /(1 + oversubs_ratio);
        printf("workload: %.2f\n",  workload_size);
        printf("workload: %llu\n",  os_size);
        cudaMalloc(&over_ptr, os_size); 
        printf("os_size: %u\n", os_size/1024/1024);
    }
    cudaMemGetInfo(&freeMemory, &totalMemory);
    printf("Free GPU Memory: %.2f MiB\n", (float) freeMemory / (1024 * 1024));
}

float time_total= 0;
float time_total_pinning= 0;
WeightT* runEmogi(uint source, uint64_t numEdges, uint64_t numVertex, uint64_t *edgeOffset,
                unsigned int *edgeList, int representation, int mem, size_t new_size, uint64_t *new_offset, 
                unsigned int *new_vertex_list, WeightT *u_weights, int u_case){

    cudaError_t ret;
    unsigned int *h_dist, *d_dist, *d_new_vertexList;
    uint64_t *d_vertexList, *h_vertexList, *d_new_offset;
    unsigned int *d_edgeList;
    bool *finished, *label_d, *changed_d, changed_h;
    WeightT *costList_d, *newCostList_d, *weightList_h, *weightList_d, *h_distance, *costList_h;
    uint64_t numblocks_update, numthreads, numblocks_kernel;
    double avg_milliseconds;
    float milliseconds;
    uint32_t one, iter;
    WeightT offset = 0;
    WeightT zero;
    cudaEvent_t start, end;
    int type = 0;
    size_t weight_size = numEdges*sizeof(WeightT);

    h_distance = (WeightT *) malloc(sizeof(WeightT)*numVertex);
    // h_vertexList = (unsigned int *) malloc(sizeof(unsigned int)*numVertex);
    size_t edge_size = numEdges*sizeof(unsigned int);
    size_t vertex_size = (numVertex + 1)*sizeof(uint64_t);

    costList_h = (WeightT *)malloc(numVertex * sizeof(WeightT));

    for (uint64_t i = 0; i < numVertex; i++) {
        costList_h[i] = 1000000000.0f;
    }

    // for (size_t i = 0; i < numVertex; i++)
    // {
    //     h_vertexList[i] = edgeOffset[i];
    // }
    

    // int mem = 0;
    switch (mem) {
        case 0:{
            
            // weightList_h = (WeightT*)malloc(weight_size);
            
            gpuErrorcheck(cudaMalloc((void**)&d_edgeList, edge_size));
            // checkCudaErrors(cudaMalloc((void**)&weightList_d, weight_size));

            // for (uint64_t i = 0; i < weight_count; i++)
            //     weightList_h[i] += offset;

            break;
        }
        case 1:{
            gpuErrorcheck(cudaMallocManaged((void**)&d_edgeList, edge_size));
            // checkCudaErrors(cudaMallocManaged((void**)&weightList_d, weight_size));
            memcpy(d_edgeList, edgeList, edge_size);
            // for (uint64_t i = 0; i < weight_count; i++)
            //     weightList_d[i] += offset;
            cudaDeviceSynchronize();
            auto start = std::chrono::steady_clock::now();  
            // gpuErrorcheck(cudaMemAdvise(d_edgeList, edge_size, cudaMemAdviseSetReadMostly, 0));
            cudaDeviceSynchronize();
            auto end = std::chrono::steady_clock::now();
            long duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
            printf("Elapsed time for cudaMemAdviseSetReadMostly in milliseconds : %li ms.\n\n", duration);
            time_total_pinning = time_total_pinning + duration;
            // gpuErrorcheck(cudaMemAdvise(d_edgeList, edge_size, cudaMemAdviseSetAccessedBy, 0));
            break;}
        case 2:{
            gpuErrorcheck(cudaMallocManaged((void**)&d_edgeList, edge_size));
            // checkCudaErrors(cudaMallocManaged((void**)&weightList_d, weight_size));
            memcpy(d_edgeList, edgeList, edge_size);

            // for (uint64_t i = 0; i < weight_count; i++)
            //     weightList_d[i] += offset;
            auto start = std::chrono::steady_clock::now();                
            // gpuErrorcheck(cudaMemAdvise(d_edgeList, edge_size, cudaMemAdviseSetAccessedBy, 0));
            auto end = std::chrono::steady_clock::now();
            long duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
            printf("Elapsed time for setAccessedBy in milliseconds : %li ms.\n\n", duration);
            // gpuErrorcheck(cudaMemAdvise(weightList_d, weight_size, cudaMemAdviseSetAccessedBy, device));
            break;
            }
        case 3: 
        {
            gpuErrorcheck(cudaMallocManaged((void**)&d_edgeList, edge_size));
            // checkCudaErrors(cudaMallocManaged((void**)&weightList_d, weight_size));
            memcpy(d_edgeList, edgeList, edge_size);
            // for (uint64_t i = 0; i < weight_count; i++)
            //     weightList_d[i] += offset;

            auto start = std::chrono::steady_clock::now(); 
            // gpuErrorcheck(cudaMemAdvise(d_edgeList, edge_size, cudaMemAdviseSetReadMostly, 0));
            // gpuErrorcheck(cudaMemAdvise(d_edgeList, edge_size, cudaMemAdviseSetAccessedBy, 0));
            auto end = std::chrono::steady_clock::now();
            long duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
<<<<<<< HEAD
            printf("Elapsed time for cudaMemAdviseSetReadMostly in milliseconds : %li ms.\n\n", duration);
=======
            printf("Elapsed time for setAccessedBy in milliseconds : %li ms.\n\n", duration);
>>>>>>> origin/cloudlab

            cudaDeviceProp devProp;
            cudaGetDeviceProperties(&devProp, 0);
            // Calculate memory utilization
            size_t totalMemory = devProp.totalGlobalMem;
            size_t freeMemory;
            size_t usedMemory;
<<<<<<< HEAD
            float workload_size = (float) numEdges*sizeof(uint) + numEdges*sizeof(float);
=======
            float workload_size = (float) numEdges*sizeof(uint);
>>>>>>> origin/cloudlab
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

            // gpuErrorcheck(cudaMemAdvise(d_edgeList, edge_size, cudaMemAdviseSetReadMostly, 0));
            // checkCudaErrors(cudaMemAdvise(weightList_d, weight_size, cudaMemAdviseSetReadMostly, 0));
            break;
        }
        
    }


    // Allocate memory for GPU
    if(u_case == 0){
        gpuErrorcheck(cudaMalloc((void**)&d_vertexList, vertex_size));
        gpuErrorcheck(cudaMemcpy(d_vertexList, edgeOffset, vertex_size, cudaMemcpyHostToDevice));
    }

    gpuErrorcheck(cudaMalloc((void**)&label_d, numVertex * sizeof(bool)));
    gpuErrorcheck(cudaMalloc((void**)&changed_d, sizeof(bool)));
    gpuErrorcheck(cudaMalloc((void**)&costList_d, numVertex * sizeof(WeightT)));
    gpuErrorcheck(cudaMalloc((void**)&newCostList_d, numVertex * sizeof(WeightT)));

    if(u_case == 1){
        gpuErrorcheck(cudaMalloc((void**)&d_new_vertexList, sizeof(unsigned int)*new_size));
        gpuErrorcheck(cudaMalloc((void**)&d_new_offset, sizeof(uint64_t)*(new_size+1)));
        gpuErrorcheck(cudaMemcpy(d_new_vertexList, new_vertex_list, sizeof(unsigned int)*new_size, cudaMemcpyHostToDevice));
        gpuErrorcheck(cudaMemcpy(d_new_offset, new_offset, sizeof(uint64_t)*(new_size+1), cudaMemcpyHostToDevice));
    }

    gpuErrorcheck(cudaEventCreate(&start));
    gpuErrorcheck(cudaEventCreate(&end));

    printf("Allocation finished\n");
    fflush(stdout);

    // Initialize values
    // gpuErrorcheck(cudaMemcpy(d_vertexList, h_vertexList, vertex_size, cudaMemcpyHostToDevice));
<<<<<<< HEAD

    if(WEIGHT_ON_GPU){
        gpuErrorcheck(cudaMalloc((void **) &weightList_d, weight_size));
        gpuErrorcheck(cudaMemcpy(weightList_d, u_weights, weight_size, cudaMemcpyHostToDevice));
    }
    else{
        gpuErrorcheck(cudaMallocManaged((void **) &weightList_d, weight_size));
        memcpy(weightList_d, u_weights, weight_size);
    }
=======
    gpuErrorcheck(cudaMalloc((void **) &weightList_d, weight_size));
    
    gpuErrorcheck(cudaMemcpy(weightList_d, u_weights, weight_size, cudaMemcpyHostToDevice));
>>>>>>> origin/cloudlab

    cudaDeviceSynchronize();
    if (mem == 0) {
        cudaEvent_t t_start, t_end;
        float millisecond = 0;
        gpuErrorcheck(cudaEventCreate(&t_start));
        gpuErrorcheck(cudaEventCreate(&t_end));
        auto start = std::chrono::steady_clock::now(); 
        
        gpuErrorcheck(cudaEventRecord(t_start, (cudaStream_t) 1));
        gpuErrorcheck(cudaMemcpy(d_edgeList, edgeList, edge_size, cudaMemcpyHostToDevice));
        
        gpuErrorcheck(cudaEventRecord(t_end, (cudaStream_t) 1));
        gpuErrorcheck(cudaEventSynchronize(t_end));
        gpuErrorcheck(cudaEventElapsedTime(&milliseconds, t_start, t_end));
        auto end = std::chrono::steady_clock::now();
        long duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        printf("Elapsed time %*f ms\n", 12, milliseconds);
        printf("Elapsed time for Edge list direct transfer in milliseconds : %li ms.\n", duration);
        
        printf("transferred amount: %f GB\n", (double) edge_size/(1024*1024*1024llu));
        // checkCudaErrors(cudaMemcpy(weightList_d, weightList_h, weight_size, cudaMemcpyHostToDevice));
    }

    numthreads = BLOCK_SIZE;
    uint64_t numblocks_kernel_new;

    switch (type) {
        case 0:
            numblocks_kernel = ((numVertex * WARP_SIZE + numthreads) / numthreads);
            numblocks_kernel_new = ((new_size * WARP_SIZE + numthreads) / numthreads);
            break;
        case 1:
            numblocks_kernel = ((numVertex * (WARP_SIZE / CHUNK_SIZE) + numthreads) / numthreads);
            numblocks_kernel_new = ((new_size * (WARP_SIZE / CHUNK_SIZE) + numthreads) / numthreads);
            break;
        default:
            fprintf(stderr, "Invalid type\n");
            exit(1);
            break;
    }

    numblocks_update = ((numVertex + numthreads) / numthreads);

    dim3 blockDim_kernel(numthreads, (numblocks_kernel+numthreads)/numthreads);
    dim3 blockDim_kernel_new(numthreads, (numblocks_kernel_new+numthreads)/numthreads);
    dim3 blockDim_update(BLOCK_SIZE, (numblocks_update+BLOCK_SIZE)/BLOCK_SIZE);



    avg_milliseconds = 0.0f;
<<<<<<< HEAD
    int oversubs_uvm = 1;
    printf("Initialization done oversubscription: %d\n", oversubs_uvm);
    oversubs(oversubs_uvm, 32*1024*1024*1024llu);
=======
    int oversubs_uvm = 7;
    printf("Initialization done oversubscription: %d\n", oversubs_uvm);
    // oversubs(oversubs_uvm, edge_size);
>>>>>>> origin/cloudlab
    fflush(stdout);

    // Set root
    // for (int i = 0; i < num_run; i++) {
        zero = 0;
        one = 1;
        // gpuErrorcheck(cudaMemset(costList_d, 0xFF, numVertex * sizeof(WeightT)));
        // gpuErrorcheck(cudaMemset(newCostList_d, 0xFF, numVertex * sizeof(WeightT)));
        gpuErrorcheck(cudaMemcpy(costList_d, costList_h, numVertex*sizeof(WeightT), cudaMemcpyHostToDevice));
        gpuErrorcheck(cudaMemcpy(newCostList_d, costList_h, numVertex*sizeof(WeightT), cudaMemcpyHostToDevice));
        gpuErrorcheck(cudaMemset(label_d, 0x0, numVertex * sizeof(bool)));
        gpuErrorcheck(cudaMemcpy(&label_d[source], &one, sizeof(bool), cudaMemcpyHostToDevice));
        gpuErrorcheck(cudaMemcpy(&costList_d[source], &zero, sizeof(WeightT), cudaMemcpyHostToDevice));
        gpuErrorcheck(cudaMemcpy(&newCostList_d[source], &zero, sizeof(WeightT), cudaMemcpyHostToDevice));

        iter = 0;

        gpuErrorcheck(cudaEventRecord(start, (cudaStream_t) 1));

        // Run SSSP
        do {
            changed_h = false;
            gpuErrorcheck(cudaMemcpy(changed_d, &changed_h, sizeof(bool), cudaMemcpyHostToDevice));

            switch (u_case) {
                
                case 0:{
                    printf("emogi csr repr optimized for warp, iter: %d\n", iter);
                    // emogi_new_repr<<<blockDim_kernel, numthreads>>>
                    // (label_d, costList_d, newCostList_d, numVertex, (unsigned int *) d_vertexList, d_edgeList, weightList_d);
                    numthreads = BLOCK_SIZE;
                    numblocks_kernel = ((numVertex * (WARP_SIZE / CHUNK_SIZE) + numthreads) / numthreads);
                    dim3 blockDim_kernel1(numthreads, (numblocks_kernel+numthreads)/numthreads);

                    emogi_csr_repr<<< blockDim_kernel1, numthreads /*(numVertex/512)*(1 << WARP_SHIFT)+1, 512*/>>>
                    (label_d, costList_d, newCostList_d, numVertex, d_vertexList, d_edgeList, weightList_d);
                    update<<<blockDim_update, BLOCK_SIZE>>>(label_d, costList_d, newCostList_d, numVertex, changed_d);
                    break;
                }
                case 1:
                {
                    printf("emogi new repr optimized for warp\n");
                    size_t n_pages = new_size*sizeof(uint64_t)/(8*1024);
                    numthreads = BLOCK_SIZE;
                    numblocks_kernel = ((new_size * (WARP_SIZE / CHUNK_SIZE) + numthreads) / numthreads);
                    dim3 blockDim_kernel1(numthreads, (numblocks_kernel+numthreads)/numthreads);

                    kernel_coalesce_new_repr<<< blockDim_kernel1, numthreads /*n_pages*32/512+1, 512*/>>>
                    (label_d, costList_d, newCostList_d, n_pages, new_size, d_new_offset, d_new_vertexList, d_edgeList, weightList_d);
                    update<<<blockDim_update, BLOCK_SIZE>>>(label_d, costList_d, newCostList_d, numVertex, changed_d);

                    // emogi_new_repr<<<(new_size/512)*(1 << WARP_SHIFT)+1, 512 /*blockDim_kernel, numthreads*/>>>
                    // (label_d, costList_d, newCostList_d, new_size, d_new_vertexList, d_edgeList, d_new_offset);

                    // emogi_rdma_bfs_new_repr<<</*blockDim_kernel1, numthreads*/ (n_pages*32)/numthreads+1, numthreads >>>
                    //     (label_d, n_pages, costList_d, newCostList_d, new_size, d_new_offset, d_new_vertexList, d_edgeList, weightList_d, changed_d);
                     
                    break;
                }
                case 2:{
                    printf("emogi csr repr optimized for warp");
                    size_t n_pages = new_size*sizeof(unsigned int)/(8*1024);

                        emogi_csr_repr_opt<<< (n_pages*32)/1024+1, 1024 >>>
                            (label_d, n_pages, costList_d, newCostList_d, numVertex, d_vertexList, d_edgeList, weightList_d);
                        break;
                }
                default:
                    fprintf(stderr, "Invalid type\n");
                    exit(1);
                    break;

            }
            ret = cudaDeviceSynchronize();
            // 
            ret = cudaDeviceSynchronize();
            printf("ret: %d cudaGetLastError: %d\n", ret, cudaGetLastError());

            iter++;

            gpuErrorcheck(cudaMemcpy(&changed_h, changed_d, sizeof(bool), cudaMemcpyDeviceToHost));
        } while(changed_h);

        gpuErrorcheck(cudaEventRecord(end, (cudaStream_t) 1));
        gpuErrorcheck(cudaEventSynchronize(end));
        gpuErrorcheck(cudaEventElapsedTime(&milliseconds, start, end));

        printf("Elapsed time %.2f ms iter: %d\n", milliseconds, iter);
        time_total = time_total + milliseconds;
        
        if(u_case){
            gpuErrorcheck(cudaMemcpy(h_distance, newCostList_d, sizeof(WeightT)*numVertex, cudaMemcpyDeviceToHost));
        }
        else{
            gpuErrorcheck(cudaMemcpy(h_distance, costList_d, sizeof(WeightT)*numVertex, cudaMemcpyDeviceToHost));
        }
        
        if(u_case == 1){
            gpuErrorcheck(cudaFree(d_new_vertexList));
            gpuErrorcheck(cudaFree(d_new_offset));
        }

        free(costList_h);
        gpuErrorcheck(cudaFree(weightList_d));
        gpuErrorcheck(cudaFree(d_edgeList));

        if(u_case == 0){
            gpuErrorcheck(cudaFree(d_vertexList));
        }
<<<<<<< HEAD

        

        if(over_ptr != NULL){
            gpuErrorcheck(cudaFree(over_ptr));
        }

        if(tmp_ptr != NULL){
            gpuErrorcheck(cudaFree(tmp_ptr));
        }

=======
>>>>>>> origin/cloudlab
        gpuErrorcheck(cudaFree(label_d));
        gpuErrorcheck(cudaFree(changed_d));
        gpuErrorcheck(cudaFree(costList_d));
        gpuErrorcheck(cudaFree(newCostList_d));
    
    return h_distance;
}

void runBaseline(uint source, uint64_t numEdges, uint64_t numVertex, uint64_t *edgeOffset,
                 uint64_t *edgeList, unsigned int *&h_distance){
    cudaError_t ret;
    unsigned int *h_dist, *d_dist;
    uint64_t *d_edgeOffset, *d_edgeList;
    bool *finished, h_finished = false;
    h_dist = new uint[numVertex];
    cudaEvent_t start, end;
    float milliseconds;
    int iter = 0;

    gpuErrorcheck(cudaEventCreate(&start));
    gpuErrorcheck(cudaEventCreate(&end));

    gpuErrorcheck(cudaMalloc((void **) &d_dist, numVertex*sizeof(unsigned int)));
    gpuErrorcheck(cudaMemset(d_dist, 100000, numVertex*sizeof(unsigned int)));
    gpuErrorcheck(cudaMemset(&d_dist[source], 0, sizeof(unsigned int)));
    
    
    gpuErrorcheck(cudaMalloc((void **) &finished, sizeof(bool)));
    

    gpuErrorcheck(cudaMalloc((void **) &d_edgeOffset, (numVertex + 1)*sizeof(uint64_t)));
    gpuErrorcheck(cudaMemcpy(d_edgeOffset, edgeOffset, (numVertex + 1)*sizeof(uint64_t), cudaMemcpyHostToDevice));

    gpuErrorcheck(cudaMalloc((void **) &d_edgeList, numEdges*sizeof(uint64_t)));
    gpuErrorcheck(cudaMemcpy(d_edgeList, edgeList, numEdges*sizeof(uint64_t), cudaMemcpyHostToDevice));


    ret = cudaDeviceSynchronize();
    printf("ret: %d cudaGetLastError: %d\n", ret, cudaGetLastError());
    gpuErrorcheck(cudaEventRecord(start, (cudaStream_t) 1));

    do{
        h_finished = false;
        gpuErrorcheck(cudaMemcpy(finished, &h_finished, sizeof(bool), cudaMemcpyHostToDevice));

        kernel_baseline<<< numVertex/1024 + 1, 1024 >>>(numEdges, numVertex, d_edgeOffset, d_edgeList,
                                d_dist, finished);

        ret = cudaDeviceSynchronize();
        // printf("ret: %d cudaGetLastError: %d \n", ret, cudaGetLastError());

        gpuErrorcheck(cudaMemcpy(&h_finished, finished, sizeof(bool), cudaMemcpyDeviceToHost));
        iter++;            
    }
    while (h_finished);

    gpuErrorcheck(cudaEventRecord(end, (cudaStream_t) 1));
    gpuErrorcheck(cudaEventSynchronize(end));
    gpuErrorcheck(cudaEventElapsedTime(&milliseconds, start, end));

    printf("Baseline kernel elapsed time %*f ms iter: %d\n", 12, milliseconds, iter);  
    
    gpuErrorcheck(cudaMemcpy(&h_distance, d_dist, sizeof(unsigned int), cudaMemcpyDeviceToHost));   
}

__device__ uint64_t edgeCounter = 0;
<<<<<<< HEAD
rdma_buf<unsigned int> *rdma_edgeList = NULL;
rdma_buf<WeightT> *rdma_weightList = NULL;

=======
rdma_buf<unsigned int> *rdma_edgeList;
>>>>>>> origin/cloudlab
WeightT* runRDMA(uint source, uint64_t numEdges, uint64_t numVertex, uint64_t *edgeOffset,
                 unsigned int *edgeList, int representation, size_t new_size, uint64_t *new_offset, 
                unsigned int *new_vertex_list, WeightT *u_weights, int u_case){
    cudaError_t ret;
    unsigned int *h_dist, *d_dist, *vertexList_uint, *d_new_vertexList;
    uint64_t *d_new_offset, *d_vertexList;
    unsigned int *d_edgeList;
    bool *finished, *label_d, *changed_d, changed_h;
    WeightT *costList_d, *newCostList_d, *weightList_h, *weightList_d, *h_distance, *costList_h;
    uint64_t numblocks_update, numthreads, numblocks_kernel;
    double avg_milliseconds;
    float milliseconds;
    uint32_t one, iter;
    WeightT offset = 0;
    WeightT zero;
    cudaEvent_t start, end;
    size_t weight_size = numEdges*sizeof(WeightT);

    // vertexList_uint = new uint[numVertex + 1];
    h_distance = new WeightT[numVertex];
    size_t edge_size = numEdges*sizeof(unsigned int);
    size_t vertex_size = (numVertex + 1)*sizeof(uint64_t);
    printf("edge_size: %.2f GB vertex_size: %.2f GB\n", 
            (float) edge_size/(1024*1024*1024), (float) vertex_size/(1024*1024*1024));

    costList_h = (WeightT *)malloc(numVertex * sizeof(WeightT));

    for (uint64_t i = 0; i < numVertex; i++) {
        costList_h[i] = 1000000000.0f;
    }

    // for (size_t i = 0; i < numVertex+1; i++)
    // {
    //     vertexList_uint[i] = edgeOffset[i];
    // }
    

    if(rdma_edgeList == NULL){
<<<<<<< HEAD

        if(WEIGHT_ON_GPU){
            gpuErrorcheck(cudaMallocManaged((void **) &rdma_edgeList, sizeof(rdma_buf<unsigned int>)));
            rdma_edgeList->start(numEdges *sizeof(unsigned int));

            for(size_t i = 0; i < numEdges; i++){
                rdma_edgeList->local_buffer[i] = edgeList[i];
            }
        }

        if(WEIGHT_ON_GPU == 0){
            gpuErrorcheck(cudaMallocManaged((void **) &rdma_weightList, sizeof(rdma_buf<WeightT>)));
            rdma_weightList->start(numEdges * sizeof(WeightT) * 2);

            for(size_t i = 0; i < numEdges; i++){
                rdma_weightList->local_buffer[2*i] = (float) edgeList[i];
                rdma_weightList->local_buffer[2*i+1] = (float) u_weights[i];
            }
=======
        gpuErrorcheck(cudaMallocManaged((void **) &rdma_edgeList, sizeof(rdma_buf<unsigned int>)));

        rdma_edgeList->start(numEdges *sizeof(unsigned int));

        for(size_t i = 0; i < numEdges; i++){
            rdma_edgeList->local_buffer[i] = edgeList[i];
>>>>>>> origin/cloudlab
        }
    }
    // rdma_edgeList->memcpyHostToServer();
    int mem = 0;
    switch (mem) {
        case 0:
            
            // weightList_h = (WeightT*)malloc(weight_size);
            
            // gpuErrorcheck(cudaMalloc((void**)&d_edgeList, edge_size));
            // checkCudaErrors(cudaMalloc((void**)&weightList_d, weight_size));

            // for (uint64_t i = 0; i < weight_count; i++)
            //     weightList_h[i] += offset;

            break;
        case 1:
            gpuErrorcheck(cudaMallocManaged((void**)&d_edgeList, edge_size));
            // checkCudaErrors(cudaMallocManaged((void**)&weightList_d, weight_size));
            memcpy(d_edgeList, edgeList, edge_size);
            // for (uint64_t i = 0; i < weight_count; i++)
            //     weightList_d[i] += offset;

            gpuErrorcheck(cudaMemAdvise(d_edgeList, edge_size, cudaMemAdviseSetReadMostly, 0));
            // checkCudaErrors(cudaMemAdvise(weightList_d, weight_size, cudaMemAdviseSetReadMostly, 0));
            break;
        case 2:
            printf("Allocating uvm memory for edgelist\n");
            gpuErrorcheck(cudaMallocManaged((void**)&d_edgeList, edge_size));
            printf("Copying uvm memory for edgelist\n");
            // checkCudaErrors(cudaMallocManaged((void**)&weightList_d, weight_size));
            memcpy(d_edgeList, edgeList, edge_size);
            printf("Copying done uvm memory for edgelist");
            // for (uint64_t i = 0; i < weight_count; i++)
            //     weightList_d[i] += offset;

            gpuErrorcheck(cudaMemAdvise(d_edgeList, edge_size, cudaMemAdviseSetAccessedBy, 0));
            // gpuErrorcheck(cudaMemAdvise(weightList_d, weight_size, cudaMemAdviseSetAccessedBy, device));
            break;

        case 3: // uvm with oversubsciption
            gpuErrorcheck(cudaMallocManaged((void**)&d_edgeList, edge_size));
            // checkCudaErrors(cudaMallocManaged((void**)&weightList_d, weight_size));
            memcpy(d_edgeList, edgeList, edge_size);
            // for (uint64_t i = 0; i < weight_count; i++)
            //     weightList_d[i] += offset;

            cudaDeviceProp devProp;
            cudaGetDeviceProperties(&devProp, 0);
            // Calculate memory utilization
            size_t totalMemory = devProp.totalGlobalMem;
            size_t freeMemory;
            size_t usedMemory;
            float workload_size = (float) numEdges*sizeof(uint);
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
            cudaMemGetInfo(&freeMemory, &totalMemory);
            printf("Free GPU Memory: %.2f MiB\n", (float) freeMemory / (1024 * 1024));

            // gpuErrorcheck(cudaMemAdvise(d_edgeList, edge_size, cudaMemAdviseSetReadMostly, 0));
            // checkCudaErrors(cudaMemAdvise(weightList_d, weight_size, cudaMemAdviseSetReadMostly, 0));
            break;
    }

    // Allocate memory for GPU
    
    gpuErrorcheck(cudaMalloc((void**)&label_d, numVertex * sizeof(bool)));
    gpuErrorcheck(cudaMalloc((void**)&changed_d, sizeof(bool)));
    gpuErrorcheck(cudaMalloc((void**)&costList_d, numVertex * sizeof(WeightT)));
    gpuErrorcheck(cudaMalloc((void**)&newCostList_d, numVertex * sizeof(WeightT)));

    if(representation){
        gpuErrorcheck(cudaMalloc((void**)&d_new_vertexList, sizeof(unsigned int)*new_size));
        gpuErrorcheck(cudaMalloc((void**)&d_new_offset, sizeof(uint64_t)*(new_size+1)));
        gpuErrorcheck(cudaMemcpy(d_new_vertexList, new_vertex_list, sizeof(unsigned int)*new_size, cudaMemcpyHostToDevice));
        gpuErrorcheck(cudaMemcpy(d_new_offset, new_offset, sizeof(uint64_t)*(new_size+1), cudaMemcpyHostToDevice));
    }else{
        gpuErrorcheck(cudaMalloc((void**)&d_vertexList, vertex_size));
        gpuErrorcheck(cudaMemcpy(d_vertexList, edgeOffset, vertex_size, cudaMemcpyHostToDevice));
    }

    gpuErrorcheck(cudaEventCreate(&start));
    gpuErrorcheck(cudaEventCreate(&end));

    printf("Allocation finished\n");
    fflush(stdout);

<<<<<<< HEAD
    if(WEIGHT_ON_GPU){
        // Initialize values
        gpuErrorcheck(cudaMalloc((void**) &weightList_d, weight_size));
        gpuErrorcheck(cudaMemcpy(weightList_d, u_weights, weight_size, cudaMemcpyHostToDevice));
    }
=======
    // Initialize values
    gpuErrorcheck(cudaMalloc((void**) &weightList_d, weight_size));
    gpuErrorcheck(cudaMemcpy(weightList_d, u_weights, weight_size, cudaMemcpyHostToDevice));
>>>>>>> origin/cloudlab
    

    if (mem == 0) {
        // gpuErrorcheck(cudaMemcpy(d_edgeList, edgeList, edge_size, cudaMemcpyHostToDevice));
        // checkCudaErrors(cudaMemcpy(weightList_d, weightList_h, weight_size, cudaMemcpyHostToDevice));
    }

    numthreads = BLOCK_SIZE/2;

    int type = 0;
    switch (type) {
        case 0:
            numblocks_kernel = ((numVertex * WARP_SIZE + numthreads) / numthreads);
            break;
        case 1:
            numblocks_kernel = ((numVertex * (WARP_SIZE / CHUNK_SIZE) + numthreads) / numthreads);
            break;
        default:
            fprintf(stderr, "Invalid type\n");
            exit(1);
            break;
    }

    numblocks_update = ((numVertex + numthreads) / numthreads);

    dim3 blockDim_kernel(numthreads, (numblocks_kernel+numthreads)/numthreads);
    // dim3 blockDim_kernel(BLOCK_SIZE, (numblocks_kernel+BLOCK_SIZE)/BLOCK_SIZE);
    numthreads = BLOCK_SIZE;
    numblocks_update = ((numVertex + BLOCK_SIZE) / BLOCK_SIZE);
    dim3 blockDim_update(BLOCK_SIZE, (numblocks_update+BLOCK_SIZE)/BLOCK_SIZE);

    avg_milliseconds = 0.0f;

    printf("Initialization done\n");
    fflush(stdout);

    // Set root
    // for (int i = 0; i < num_run; i++) {
        zero = 0;
        one = 1;
        // gpuErrorcheck(cudaMemset(costList_d, 0xFF, numVertex * sizeof(WeightT)));
        // gpuErrorcheck(cudaMemset(newCostList_d, 0xFF, numVertex * sizeof(WeightT)));
        gpuErrorcheck(cudaMemcpy(costList_d, costList_h, numVertex*sizeof(WeightT), cudaMemcpyHostToDevice));
        gpuErrorcheck(cudaMemcpy(newCostList_d, costList_h, numVertex*sizeof(WeightT), cudaMemcpyHostToDevice));
        gpuErrorcheck(cudaMemset(label_d, 0x0, numVertex * sizeof(bool)));
        gpuErrorcheck(cudaMemcpy(&label_d[source], &one, sizeof(bool), cudaMemcpyHostToDevice));
        gpuErrorcheck(cudaMemcpy(&costList_d[source], &zero, sizeof(WeightT), cudaMemcpyHostToDevice));
        gpuErrorcheck(cudaMemcpy(&newCostList_d[source], &zero, sizeof(WeightT), cudaMemcpyHostToDevice));

        iter = 0;
        ret = cudaDeviceSynchronize();
        gpuErrorcheck(cudaEventRecord(start, (cudaStream_t) 1));

        // Run SSSP
        do {
            changed_h = false;
            gpuErrorcheck(cudaMemcpy(changed_d, &changed_h, sizeof(bool), cudaMemcpyHostToDevice));
            auto start = std::chrono::steady_clock::now();
    
            switch (u_case) {
                case 0:{
                    kernel_coalesce_rdma<<<blockDim_kernel, numthreads/2>>>
                    (label_d, costList_d, newCostList_d, numVertex, d_vertexList, rdma_edgeList, weightList_d);
                    ret = cudaDeviceSynchronize();
                    update<<<blockDim_update, numthreads>>>(label_d, costList_d, newCostList_d, numVertex, changed_d);
                    break;
                }
                

                case 1:{
                    printf("new representation\n");
<<<<<<< HEAD
                    printf("representation is %d WEIGHT_ON_GPU: %d\n", representation, WEIGHT_ON_GPU);
=======
                    printf("representation is %d\n", representation);
>>>>>>> origin/cloudlab
                    size_t n_pages; // = numVertex*sizeof(uint64_t)/(8*1024);
                    
                    if(representation){

                        n_pages = new_size*sizeof(uint64_t)/(8*1024);
<<<<<<< HEAD
                        numthreads = BLOCK_SIZE;
                        numblocks_kernel = ((new_size * (WARP_SIZE / CHUNK_SIZE) + numthreads) / numthreads);
                        dim3 blockDim_kernel1(numthreads, (numblocks_kernel+numthreads)/numthreads);

                        if(WEIGHT_ON_GPU){
                            kernel_coalesce_new_repr_rdma<<<blockDim_kernel1, numthreads/*(new_size/512)*(1 << WARP_SHIFT)+1, 512*/>>>
                            (label_d, n_pages, costList_d, newCostList_d, new_size, d_new_offset, d_new_vertexList, rdma_edgeList, weightList_d);
                        }
                        else{
                            kernel_coalesce_new_repr_rdma_weight<<<blockDim_kernel1, numthreads/*(new_size/512)*(1 << WARP_SHIFT)+1, 512*/>>>
                            (label_d, n_pages, costList_d, newCostList_d, new_size, d_new_offset, d_new_vertexList, rdma_edgeList, rdma_weightList);
                        }
=======
                        numthreads = BLOCK_SIZE/2;
                        numblocks_kernel = ((new_size * (WARP_SIZE / CHUNK_SIZE) + numthreads) / numthreads);
                        dim3 blockDim_kernel1(numthreads, (numblocks_kernel+numthreads)/numthreads);

                        kernel_coalesce_new_repr_rdma<<<blockDim_kernel1, numthreads/*(new_size/512)*(1 << WARP_SHIFT)+1, 512*/>>>
                        (label_d, n_pages, costList_d, newCostList_d, new_size, d_new_offset, d_new_vertexList, rdma_edgeList, weightList_d);
>>>>>>> origin/cloudlab

                        // kernel_coalesce_rdma_bfs_new_repr<<</*blockDim_kernel1, numthreads*/ (n_pages*32)/numthreads+1, numthreads>>>
                        // (label_d, n_pages, costList_d, newCostList_d, new_size, d_new_offset, d_new_vertexList, rdma_edgeList, weightList_d, changed_d);
                    }
                    else {
                        n_pages = numVertex*sizeof(uint64_t)/(8*1024);
<<<<<<< HEAD
                        numthreads = BLOCK_SIZE/2;
                        numblocks_kernel = ((numVertex * (WARP_SIZE / CHUNK_SIZE) + numthreads) / numthreads);
                        dim3 blockDim_kernel1(numthreads, (numblocks_kernel+numthreads)/numthreads);

                        if(WEIGHT_ON_GPU){
                            kernel_coalesce_rdma<<<blockDim_kernel1, numthreads/*(numVertex/512)*(1 << WARP_SHIFT)+1, 512*/>>>
                                (label_d, costList_d, newCostList_d, numVertex, d_vertexList, rdma_edgeList, weightList_d);
                        }
                        else{
                            kernel_coalesce_rdma_weights<<<blockDim_kernel1, numthreads/*(numVertex/512)*(1 << WARP_SHIFT)+1, 512*/>>>
                                (label_d, costList_d, newCostList_d, numVertex, d_vertexList, rdma_edgeList, rdma_weightList);
                        }
=======
                        numthreads = BLOCK_SIZE;
                        numblocks_kernel = ((numVertex * (WARP_SIZE / CHUNK_SIZE) + numthreads) / numthreads);
                        dim3 blockDim_kernel1(numthreads, (numblocks_kernel+numthreads)/numthreads);
                        kernel_coalesce_rdma<<<blockDim_kernel1, numthreads/*(numVertex/512)*(1 << WARP_SHIFT)+1, 512*/>>>
                            (label_d, costList_d, newCostList_d, numVertex, d_vertexList, rdma_edgeList, weightList_d);

>>>>>>> origin/cloudlab
                        // kernel_coalesce_rdma_opt<<< (n_pages*32)/512+1, 512 >>>
                        // (label_d, n_pages, costList_d, newCostList_d, numVertex, d_vertexList, rdma_edgeList, weightList_d);
                        
                    }
<<<<<<< HEAD

=======
>>>>>>> origin/cloudlab
                    ret = cudaDeviceSynchronize();
                    update<<<blockDim_update, BLOCK_SIZE>>>(label_d, costList_d, newCostList_d, numVertex, changed_d);
                    break;
                }
                case 2:{
                    check_edgeList<<< numEdges/384+1, 384 >>>
                    (rdma_edgeList, d_edgeList, numEdges);
                    ret = cudaDeviceSynchronize();
                }
                
                case 3:{
                    size_t n_pages = numEdges*sizeof(uint64_t)/(64*1024);

                        // kernel_coalesce_rdma_opt<<< 2048*16, 512 /*blockDim_kernel, numthreads*/ >>>
                        // (label_d, n_pages, costList_d, newCostList_d, numVertex, d_vertexList, rdma_edgeList, weightList_d);
                    
                    ret = cudaDeviceSynchronize();
                    update<<<blockDim_update, numthreads>>>(label_d, costList_d, newCostList_d, numVertex, changed_d);
                    break;
                }
                case 4:{
                    size_t n_pages = numVertex*sizeof(unsigned int)/(4*1024);

                        // kernel_coalesce_rdma_opt_1<<< 512*32, 512 /*blockDim_kernel, numthreads*/ >>>
                        // (label_d, n_pages, numVertex, numEdges, 0, rdma_edgeList, d_vertexList, costList_d, newCostList_d, weightList_d);
                        
                    
                    ret = cudaDeviceSynchronize();
                    update<<<blockDim_update, numthreads>>>(label_d, costList_d, newCostList_d, numVertex, changed_d);
                    ret = cudaDeviceSynchronize();
                    break;
                }
                case 5:{
                    size_t n_pages = numEdges*sizeof(uint64_t)/(64*1024);

                    // kernel_coalesce_rdma_opt_1(bool *label, size_t n, size_t numVertices, unsigned int level, rdma_buf<unsigned int> *d_adjacencyList, unsigned int *d_edgesOffset,
                    // const WeightT *costList, WeightT *newCostList, unsigned int *d_distance, unsigned int *changed)

                        // kernel_coalesce_rdma_opt_warp<<< (numVertex*32)/512 + 1, 512 /*blockDim_kernel, numthreads*/ >>>
                        // (label_d, numVertex, numVertex, 0, rdma_edgeList, d_vertexList, costList_d, newCostList_d, weightList_d);
                    
                    ret = cudaDeviceSynchronize();
                    update<<<blockDim_update, numthreads>>>(label_d, costList_d, newCostList_d, numVertex, changed_d);
                    ret = cudaDeviceSynchronize();
                    break;
                }
                default:
                    fprintf(stderr, "Invalid type\n");
                    exit(1);
                    break;
            }
            auto end = std::chrono::steady_clock::now();
            long duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
            print_retires<<<1,1>>>();
            printf("ret: %d cudaGetLastError: %d\n", ret, cudaGetLastError());
            printf("Elapsed time in milliseconds : %li ms for iteration: %d\n\n", duration, iter);
            // printf("ret: %d cudaGetLastError: %d\n", ret, cudaGetLastError());

            
            
            // ret = cudaDeviceSynchronize();
            

            iter++;

            gpuErrorcheck(cudaMemcpy(&changed_h, changed_d, sizeof(bool), cudaMemcpyDeviceToHost));
        } while(changed_h);

        gpuErrorcheck(cudaEventRecord(end, (cudaStream_t) 1));
        gpuErrorcheck(cudaEventSynchronize(start));
        gpuErrorcheck(cudaEventSynchronize(end));
        gpuErrorcheck(cudaEventElapsedTime(&milliseconds, start, end));

        gpuErrorcheck(cudaMemcpy(h_distance, newCostList_d, sizeof(WeightT)*numVertex, cudaMemcpyDeviceToHost));

        printf("RDMA Elapsed time %*f ms iter: %d\n", 12, milliseconds, iter);
        time_total = time_total + milliseconds;

        // free(vertexList_uint);
        gpuErrorcheck(cudaFree(weightList_d));
        if(representation){
            gpuErrorcheck(cudaFree(d_new_vertexList));
            gpuErrorcheck(cudaFree(d_new_offset));
        }
        else{
            gpuErrorcheck(cudaFree(d_vertexList));
        }
        
        gpuErrorcheck(cudaFree(label_d));
        gpuErrorcheck(cudaFree(changed_d));
        gpuErrorcheck(cudaFree(costList_d));
        gpuErrorcheck(cudaFree(newCostList_d));
    return h_distance;
}


// Main program
int main(int argc, char **argv)
{   
    if (argc < 9)
        usage(argv[0]);
    
    init_gpu(0);

    Graph_x G;
    Graph_m G_m;
    
    unsigned int *tmp_edgesOffset, *tmp_edgesSize, *tmp_adjacencyList;
    float *u_weights;
    
    // readGraph(G, argc, argv);
    readfile(G, G_m, argc, argv, tmp_edgesOffset, tmp_edgesSize, tmp_adjacencyList, u_weights);

    for (size_t i = 0; i < 100; i++)
    {
        printf("u_weights[%d]: %f\n", i, u_weights[i]);
    }
    

    Graph graph(argv[8]);

    printf("main starts...\n");
    
    unsigned int sourceNode = atoi(argv[7]);

    // if (args.hasSourceNode) {
    //     sourceNode = args.sourceNode;
    // } else {
    //     // Use graph default source 
    //     sourceNode = graph.defaultSource;
    // }

    // uint *dist_gpu = sssp_GPU(&graph, sourceNode);
    printf("line: %d\n", __LINE__);
    printf("num edges: %llu num vertices: %llu\n", G.numEdges, G.numVertices);
    uint64_t *u_edgeoffset;
    unsigned int *u_edgeList;
    unsigned int *res_distance;
    res_distance = new uint[G.numVertices];
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
        if(degree > 128){
            min++;
            avg += degree;
            // printf("degree: %llu\n", degree);
        }
        // if(min > degree && degree != 0) min = degree;
    }
    avg = avg / min;
    printf("avg: %f min: %llu max: %llu, node: %llu\n", avg, min, max, max_node);
    auto start = std::chrono::steady_clock::now();                
    size_t new_size = 0, treshold = 128;
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

    int mem = 1;
    WeightT *emogi_result;
<<<<<<< HEAD
    emogi_result = runEmogi(sourceNode, G.numEdges, G.numVertices, u_edgeoffset, u_edgeList, 0, mem, new_size,
                                    new_offset, new_vertex_list, u_weights, 1);
=======
    // emogi_result = runEmogi(sourceNode, G.numEdges, G.numVertices, u_edgeoffset, u_edgeList, 0, mem, new_size,
    //                                 new_offset, new_vertex_list, u_weights, 1);
>>>>>>> origin/cloudlab
    
    // emogi_result = (float *) malloc(sizeof(float)*G.numVertices);

    // for (size_t i = 0; i < G.numVertices; i++)
    // {
    //     emogi_result[i] = 0;
    // }

<<<<<<< HEAD
    int number_of_vertices = 0;
=======
    int number_of_vertices = 200;
>>>>>>> origin/cloudlab
    int active_vertices = 0;
    time_total = 0;
    time_total_pinning = 0;
    printf("UVM Cuda Starts here..\n");
    for (size_t i = 0; i < number_of_vertices; i++)
    {
        sourceNode = i;
        printf("vertex %d has degree of %d\n", sourceNode, u_edgeoffset[i+1] - u_edgeoffset[i]);
        if(u_edgeoffset[i+1] - u_edgeoffset[i] == 0)
            continue;
        active_vertices++;
        emogi_result = runEmogi(sourceNode, G.numEdges, G.numVertices, u_edgeoffset, u_edgeList, 0, mem, new_size,
<<<<<<< HEAD
                                    new_offset, new_vertex_list, u_weights, 0);
=======
                                    new_offset, new_vertex_list, u_weights, 1);
>>>>>>> origin/cloudlab

        printf("average time: %.2f pinning time: %.2f active_vertices: %d\n", time_total/active_vertices, time_total_pinning/active_vertices, active_vertices);
    }
    
    printf("end: average time: %.2f pinning time: %.2f active_vertices: %d\n", time_total/active_vertices, time_total_pinning/active_vertices, active_vertices);
    
    
<<<<<<< HEAD
    bool rdma_flag = true;
=======
    bool rdma_flag = false;
>>>>>>> origin/cloudlab
    struct context *s_ctx = (struct context *)malloc(sizeof(struct context));
    if(rdma_flag){
        
        printf("Function: %s line number: %d 1024MB: %d bytes REQUEST_SIZE: %d\n",__func__, __LINE__, MB(1024), REQUEST_SIZE);
        int num_msg = (unsigned long) atoi(argv[4]);
        int mesg_size = (unsigned long) atoi(argv[5]);
        int num_bufs = (unsigned long) atoi(argv[6]);

        
        struct post_content post_cont, *d_post, host_post;
        struct poll_content poll_cont, *d_poll, host_poll;
        struct post_content2 post_cont2, *d_post2;
        struct host_keys keys;

        int num_iteration = num_msg;
        s_ctx->n_bufs = num_bufs;
<<<<<<< HEAD
        s_ctx->gpu_buf_size = 16*1024*1024*1024llu; // N*sizeof(int)*3llu;
=======
        s_ctx->gpu_buf_size = 5*1024*1024*1024llu; // N*sizeof(int)*3llu;
>>>>>>> origin/cloudlab

        // // remote connection:
        // int ret = connect(argv[2], s_ctx);

        // local connect
        char *mlx_name = "mlx5_0";
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

        size_t restricted_gpu_mem = 3612134270*sizeof(uint)/1; // 2*1024*1024*1024llu;
        // sizeof(uint)*4294966740llu/10; // 18*1024*1024*1024llu; // sizeof(unsigned int)*G.numEdges;
        // allowed_size = restricted_gpu_mem;
        const size_t page_size = REQUEST_SIZE;
        const size_t numPages = restricted_gpu_mem/page_size;

        printf("restricted_gpu_mem: %zu\n", restricted_gpu_mem);
        start_page_queue<<<1, 1>>>(/*s_ctx->gpu_buf_size*/restricted_gpu_mem, page_size);
        ret1 = cudaDeviceSynchronize();
        printf("ret: %d\n", ret1);
        if(cudaSuccess != ret1){    
            return -1;
        }
    }

    WeightT *rdma_result;
<<<<<<< HEAD
    number_of_vertices = 0;
=======
    number_of_vertices = 200;
>>>>>>> origin/cloudlab
    active_vertices = 0;
    time_total = 0;
    size_t min_page_fault = 10000000, *d_pf;
    unsigned int *min_source, h_min_source = 0;
    gpuErrorcheck(cudaMalloc((void **) &d_pf, sizeof(size_t)));
    gpuErrorcheck(cudaMalloc((void **) &min_source, sizeof(uint)));
    gpuErrorcheck(cudaMemset(d_pf, 10000000, sizeof(size_t)));
    gpuErrorcheck(cudaMemset(min_source, 0, sizeof(uint)));
    if(rdma_flag){
        printf("RDMA Cuda Starts here..\n");
<<<<<<< HEAD
        rdma_result = runRDMA(sourceNode, G.numEdges, G.numVertices, u_edgeoffset, u_edgeList, 1, new_size,
                                    new_offset, new_vertex_list, u_weights, 1);
        cudaFree(s_ctx->gpu_buffer);
=======
        // rdma_result = runRDMA(sourceNode, G.numEdges, G.numVertices, u_edgeoffset, u_edgeList, 1, new_size,
        //                             new_offset, new_vertex_list, u_weights, 1);
        // cudaFree(s_ctx->gpu_buffer);
>>>>>>> origin/cloudlab

        for (size_t i = 0; i < number_of_vertices; i++)
        {
            sourceNode = i;
            printf("vertex %d has degree of %d\n", sourceNode, u_edgeoffset[i+1] - u_edgeoffset[i]);
            if(u_edgeoffset[i+1] - u_edgeoffset[i] == 0)
                continue;
            active_vertices++;
            rdma_result = runRDMA(sourceNode, G.numEdges, G.numVertices, u_edgeoffset, u_edgeList, 1, new_size,
                                    new_offset, new_vertex_list, u_weights, 1);
        
            rdma_edgeList->reset();

            gpuErrorcheck(cudaMemcpy(d_pf, &min_page_fault, sizeof(size_t), cudaMemcpyHostToDevice));
            // gpuErrorcheck(cudaMemcpy(min_source, &startVertex, sizeof(uint), cudaMemcpyHostToDevice));
            gpuErrorcheck(cudaDeviceSynchronize());
            copy_page_fault_number<<< 1,1 >>>(d_pf, min_source, sourceNode);
            gpuErrorcheck(cudaMemcpy(&min_page_fault, d_pf, sizeof(size_t), cudaMemcpyDeviceToHost));
            gpuErrorcheck(cudaMemcpy(&h_min_source, min_source, sizeof(uint), cudaMemcpyDeviceToHost));

            printf("average time: %.2f active_vertices: %d\n", time_total/active_vertices, active_vertices);
        }
        
        printf("average time: %.2f\n", time_total/active_vertices);
    }
    // else
    //     rdma_result = runEmogi(sourceNode, G.numEdges, G.numVertices, u_edgeoffset, u_edgeList, 0, mem, new_size,
    //                                 new_offset, new_vertex_list, u_weights, 0);


    printf("Emogi Cuda Starts here..\n");
    // unsigned int *emogi_result = (unsigned int *) malloc(sizeof(unsigned int)*G.numVertices);
    // runEmogi(uint source, uint64_t numEdges, uint64_t numVertex, uint64_t *edgeOffset,
                // unsigned int *edgeList, int representation, size_t new_size, unsigned int *new_offset, 
                // unsigned int *new_vertex_list
    // int mem = 0;
    // WeightT *emogi_result = runEmogi(sourceNode, G.numEdges, G.numVertices, u_edgeoffset, u_edgeList, 1, mem, new_size,
    //                                 new_offset, new_vertex_list);

    printf("Cuda Starts ended..\n");

    printf("Comparing Emogi with RDMA\n");

    // if(rdma_flag)
    compareResult(emogi_result, rdma_result, G.numVertices);

    // // args.runOnCPU = true;
    // if (true) {
    //     uint *dist_cpu = sssp_CPU(sourceNode, G.numVertices, G.numEdges, u_edgeoffset, u_edgeList);
    //     compareResult(dist_cpu, res_distance, graph.numNodes);
    // }
     
    
	return 0;
}

__global__
void check_edgeList(rdma_buf<unsigned int> *a, unsigned int *b, size_t size){
    size_t tid = blockDim.x * blockIdx.x + threadIdx.x;
    if(tid == 0) printf("checking edgelist correctness\n");
    if(tid < size){
        unsigned int a_here = (*a)[tid];
        // __nanosleep(100000);
        if(a_here != b[tid]){
            printf("tid: %llu, a_here: %d b[tid]: %d\n", tid, a_here, b[tid]);
        } 
    }
}

__global__ 
void update(bool *label, WeightT *costList, WeightT *newCostList, const uint32_t vertex_count, bool *changed) {
	uint64_t tid = blockDim.x * BLOCK_SIZE * blockIdx.y + blockDim.x * blockIdx.x + threadIdx.x;

    if (tid < vertex_count) {
        if (newCostList[tid] < costList[tid]) {
            costList[tid] = newCostList[tid];
            label[tid] = true;
            *changed = true;
        }
    }
}


__global__ void kernel_baseline(uint64_t numEdges, uint64_t numVertex, uint64_t *edgeOffset, uint64_t * edgeList,
                                uint *dist, bool *changed) {
    uint64_t threadId = blockDim.x * blockIdx.x + threadIdx.x;
    // if(threadId == 0) printf("hello from thread 0\n");

    if (threadId < numVertex){

        for (uint64_t nodeId = edgeOffset[threadId]; nodeId < edgeOffset[threadId+1]; nodeId++) {
            uint64_t source = threadId;
            uint64_t end = edgeList[nodeId]; // edgelist
            uint weight = 1; // edgesWeight[nodeId];
            
            if (dist[source] + weight < dist[end]) {
                atomicMin(&dist[end], dist[source] + weight);
                // dist[end] = dist[source] + weight;
                // preNode[end] = source;
                *changed = true;
            }
        }
    }
}

__global__ __launch_bounds__(1024,2)
void kernel_coalesce_new_repr(bool *label, const WeightT *costList, WeightT *newCostList, int n, const uint64_t new_size, uint64_t *new_offset, 
                                unsigned int *new_vertex_list, const /*EdgeT*/ unsigned int *edgeList, const WeightT *weightList) {

    //  // Page size in elements (64KB / 4 bytes per unsigned int)
    // const size_t pageSize = 8*1024 / sizeof(unsigned int);
    // // Elements per warp
    // const size_t elementsPerWarp = pageSize / warpSize;

    // // Global thread ID
    // size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    // // if(tid == 0) printf("warpSize: %d\n", warpSize);
    // // Warp ID within the block
    // size_t warpId = tid / warpSize;

    // // Thread lane within the warp
    // size_t lane = threadIdx.x % warpSize;

    // // Determine which page this warp will process
    // size_t pageStart = warpId * pageSize;

    // // Ensure we don't process out-of-bounds pages
    // if (pageStart < n * pageSize) {
        

    //     // Process elements within the page
    //     for (size_t i = 0; i < elementsPerWarp; ++i) {
    //         size_t elementIdx = pageStart + lane + i * warpSize;
    //         if (elementIdx < new_size){
    //             uint startVertex = new_vertex_list[elementIdx];
    //             if (/*elementIdx < new_size && */label[startVertex]) {
    //                 WeightT cost = newCostList[startVertex];

    //                 // Process adjacent nodes
    //                 // if(new_offset[elementIdx+1] - d_edgesOffset[elementIdx] >= 2*1024)
    //                 //     printf("elementx: %llu\n", elementIdx);
    //                 for(size_t j = new_offset[elementIdx]; j < new_offset[elementIdx+1]; ++j) {
    //                     uint end_edge = edgeList[j]; // shared_data[j - pageStart];
    //                     if (newCostList[startVertex] != cost)
    //                         break;
    //                     if (newCostList[end_edge] > cost + weightList[j]) {
    //                         AtomicMin(&(newCostList[end_edge]), cost + weightList[j]);
    //                     }
    //                 }

    //                 // Mark node as processed
    //                 label[startVertex] = false;
    //             }
    //         }
    //     }
    // }



    // const uint64_t tid = blockDim.x * blockIdx.x + threadIdx.x;;
    // // blockDim.x * BLOCK_SIZE * blockIdx.y + blockDim.x * blockIdx.x + threadIdx.x;
    // // const uint64_t warpIdx = tid >> WARP_SHIFT;
    // // const uint64_t laneIdx = tid & ((1 << WARP_SHIFT) - 1);
    // uint start_vertex;
    // if(tid < new_size)
    //     start_vertex = new_vertex_list[tid];
    // if (tid < new_size /*vertex_count*/ && label[start_vertex]) {
    //     uint64_t start = new_offset[tid];
    //     // const uint64_t shift_start = start & MEM_ALIGN;
    //     uint64_t end = new_offset[tid+1];

    //     WeightT cost = newCostList[start_vertex];

    //     for(uint64_t i = start; i < end; i += 1) {
    //         if (newCostList[start_vertex] != cost)
    //             break;
    //         if (newCostList[edgeList[i]] > cost + weightList[i] && i >= start)
    //             atomicMin(&(newCostList[edgeList[i]]), cost + weightList[i]1);
    //     }

    //     label[start_vertex] = false;
    // }

    const uint64_t tid = blockDim.x * 1024 * blockIdx.y + blockDim.x * blockIdx.x + threadIdx.x;
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
        if (label[start_vertex]) {
            uint64_t start = new_offset[i];
            const uint64_t shift_start = start & MEM_ALIGN;
            uint64_t end = new_offset[i+1];

            WeightT cost = newCostList[start_vertex];

            for(uint64_t j = shift_start + laneIdx; j < end; j += WARP_SIZE) {
                if (newCostList[start_vertex] != cost)
                    break;
                uint end_node = edgeList[j];
                if (newCostList[end_node] > cost + weightList[j] && j >= start){
                    AtomicMin(&(newCostList[end_node]), cost + weightList[j]);
                    // newCostList[end_node] = cost + weightList[j];
                }
            }

            label[start_vertex] = false;
        }
    }
}

__global__ __launch_bounds__(1024,2) 
void emogi_csr_repr(bool *label, const WeightT *costList, WeightT *newCostList, const uint64_t vertex_count, uint64_t *vertexList, 
                                /*EdgeT*/ unsigned int *edgeList, WeightT *weightList) { // weightlist represents new offset
  
    // //  // Page size in elements (64KB / 4 bytes per unsigned int)
    // // const size_t pageSize = 8*1024 / sizeof(unsigned int);
    // // // Elements per warp
    // // const size_t elementsPerWarp = pageSize / warpSize;

    // const uint64_t tid = blockIdx.x * blockDim.x + threadIdx.x; // 
    // // const uint64_t tid = blockDim.x * BLOCK_SIZE * blockIdx.y + blockDim.x * blockIdx.x + threadIdx.x;
    
    // const uint64_t laneIdx = tid & ((1 << WARP_SHIFT) - 1);

    // // // Warp ID within the block
    // size_t warpId = tid / (1 << WARP_SHIFT);

    // // // Determine which page this warp will process
    // // size_t pageStart = warpId * pageSize;

    // if(warpId < vertex_count && label[warpId]) {
    //     uint64_t start = vertexList[warpId];
    //     const uint64_t shift_start = start & MEM_ALIGN;
    //     uint64_t end = vertexList[warpId+1];

    //     WeightT cost = newCostList[warpId];

    //     for(uint64_t i = shift_start + laneIdx; i < end; i += (1 << WARP_SHIFT)) {
    //         if (newCostList[warpId] != cost)
    //             break;
    //         if (newCostList[edgeList[i]] > cost + /*weightList[i]*/1 && i >= start)
    //             atomicMin(&(newCostList[edgeList[i]]), cost + /*weightList[i]*/1);
    //     }

    //     label[warpId] = false;
    // }

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
        if (label[i]) {
            uint64_t start = vertexList[i];
            const uint64_t shift_start = start & MEM_ALIGN;
            uint64_t end = vertexList[i+1];

            WeightT cost = newCostList[i];

            // printf("i: %d label[%d] shift_start: %llu end: %llu\n", i, label[i], shift_start, end);

            for (uint64_t j = shift_start + laneIdx; j < end; j += WARP_SIZE) {
                if (newCostList[i] != cost)
                    break;

                // printf("j: %llu, i: %d laneIdx: %d\n",j, i, laneIdx);

                if (newCostList[edgeList[j]] > cost + weightList[j] && j >= start){
                    // printf("newCostList[edgeList[j]]: %f, j: %llu weightList[j]: %f\n", newCostList[edgeList[j]], j, weightList[j]);
                    AtomicMin(&(newCostList[edgeList[j]]), cost + weightList[j]);
                }
            }

            label[i] = false;
        }
    }
}

__global__ void __launch_bounds__(1024,2) 
emogi_csr_repr_opt(bool *label, size_t n, const WeightT *costList, WeightT *newCostList, const uint64_t vertex_count,
                    uint64_t *vertexList, unsigned int *edgeList, const WeightT *weightList) { // weightlist represents new offset
   
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
            
                
                if (elementIdx < vertex_count && label[elementIdx]) {
                    WeightT cost = newCostList[elementIdx];

                    // Process adjacent nodes
                    // if(new_offset[elementIdx+1] - d_edgesOffset[elementIdx] >= 2*1024)
                    //     printf("elementx: %llu\n", elementIdx);
                    for(size_t j = vertexList[elementIdx]; j < vertexList[elementIdx+1]; ++j) {
                        uint end_edge = edgeList[j]; // shared_data[j - pageStart];
                        // if (newCostList[startVertex] != cost)
                        //     break;
                        if (newCostList[end_edge] > cost + 1) {
                            AtomicMin(&(newCostList[end_edge]), cost + 1);
                        }
                    }

                    // Mark node as processed
                    label[elementIdx] = false;
                }
            }
    }

}

__global__ void emogi_new_repr(bool *label, const WeightT *costList, WeightT *newCostList, const uint64_t vertex_count, unsigned int *vertexList, 
                                /*EdgeT*/ unsigned int *edgeList, uint64_t *weightList) { // weightlist represents new offset
    //  // Page size in elements (64KB / 4 bytes per unsigned int)
    // const size_t pageSize = 8*1024 / sizeof(unsigned int);
    // // Elements per warp
    // const size_t elementsPerWarp = pageSize / warpSize;

    const uint64_t tid = blockIdx.x * blockDim.x + threadIdx.x; // 
    // const uint64_t tid = blockDim.x * BLOCK_SIZE * blockIdx.y + blockDim.x * blockIdx.x + threadIdx.x;
    
    // const uint64_t laneIdx = tid & ((1 << WARP_SHIFT) - 1);

    // // Warp ID within the block
    size_t warpId = tid / warpSize;

    // // Determine which page this warp will process
    // size_t pageStart = warpId * pageSize;

    if(warpId < vertex_count){
        uint startVertex = vertexList[warpId];

        if (/*warpIdx < vertex_count && */label[startVertex]) {
            uint64_t start = weightList[warpId];
            // const uint64_t shift_start = start & MEM_ALIGN;
            uint64_t end = weightList[warpId+1];

            WeightT cost = newCostList[startVertex];

            for(uint64_t i = start; i < end; i += 1) {
                // if (newCostList[startVertex] != cost)
                //     break;
                if (newCostList[edgeList[i]] > cost + /*weightList[i]*/1 && i >= start)
                    AtomicMin(&(newCostList[edgeList[i]]), cost + /*weightList[i]*/1);
            }

            label[startVertex] = false;
        }

    }


    // const uint64_t tid = blockDim.x * BLOCK_SIZE * blockIdx.y + blockDim.x * blockIdx.x + threadIdx.x;
    // const uint64_t warpIdx = tid >> WARP_SHIFT;
    // const uint64_t laneIdx = tid & ((1 << WARP_SHIFT) - 1);

    // if (warpIdx < vertex_count && label[warpIdx]) {
    //     uint64_t start = vertexList[warpIdx];
    //     const uint64_t shift_start = start & MEM_ALIGN;
    //     uint64_t end = vertexList[warpIdx+1];

    //     WeightT cost = newCostList[warpIdx];

    //     for(uint64_t i = shift_start + laneIdx; i < end; i += WARP_SIZE) {
    //         if (newCostList[warpIdx] != cost)
    //             break;
    //         if (newCostList[edgeList[i]] > cost + /*weightList[i]*/1 && i >= start)
    //             atomicMin(&(newCostList[edgeList[i]]), cost + /*weightList[i]*/1);
    //     }

    //     label[warpIdx] = false;
    // }
}

__global__ void __launch_bounds__(1024,2) 
kernel_coalesce_rdma_bfs_new_repr(bool *label, size_t n, const WeightT *costList, WeightT *newCostList, const uint64_t new_size, uint64_t *new_offset, 
                                  unsigned int *new_vertex_list, rdma_buf<unsigned int> *edgeList, float *weightList, bool *d_changed) {

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
    //     if (newCostList[startVertex] != 4294967295U) {
    //         uint64_t start = new_offset[i];
    //         const uint64_t shift_start = start & MEM_ALIGN;
    //         uint64_t end = new_offset[i+1];

    //         WeightT cost = newCostList[startVertex];

    //         // #pragma unroll
    //         for(uint64_t j = shift_start + laneIdx; j < end; j += WARP_SIZE) {
    //             if (newCostList[startVertex] != cost)
    //                 break;
    //             uint end_node = (*edgeList)[j];
    //             uint newDist = cost + weightList[j];

    //             if (newCostList[end_node] > newDist /*1*/ && j >= start){

    //                 atomicMin(&(newCostList[end_node]), newDist);
    //                 *d_changed = true;
    //                 // label[end_node] = true;
    //             }
    //         }

    //         // label[i] = false;
    //     }
    // }


    // Page size in elements (64KB / 4 bytes per unsigned int)
    const size_t pageSize = (8*1024) / sizeof(uint64_t);
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
            if (elementIdx < new_size){
                uint startVertex = new_vertex_list[elementIdx];
                if (label[startVertex] && newCostList[startVertex] != 1000000000.0f) {
            
                    WeightT cost = newCostList[startVertex];

                    // Process adjacent nodes
                    // if(new_offset[elementIdx+1] - d_edgesOffset[elementIdx] >= 2*1024)
                    //     printf("elementx: %llu\n", elementIdx);
                    for(size_t j = new_offset[elementIdx]; j < new_offset[elementIdx+1]; ++j) {
                        
                        if (newCostList[startVertex] != cost)
                            break;
                        uint end_edge = (*edgeList)[j]; // shared_data[j - pageStart];
                        float newDist = cost + weightList[j];

                        if (newCostList[end_edge] > newDist /*1*/){

                            AtomicMin(&(newCostList[end_edge]), newDist);
                            *d_changed = true;
                            label[end_edge] = true;
                        }
                    }

                    // Mark node as processed
                    label[startVertex] = false;
                }
            }
        }
    }
}

__global__ __launch_bounds__(1024,2) 
void kernel_coalesce_new_repr_rdma(bool *label, size_t n, const WeightT *costList, WeightT *newCostList, const uint64_t new_size, uint64_t *new_offset, 
                                unsigned int *new_vertex_list, rdma_buf<unsigned int> *edgeList, const WeightT *weightList) {

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
        if (label[start_vertex]) {
            uint64_t start = new_offset[i];
            const uint64_t shift_start = start & MEM_ALIGN;
            uint64_t end = new_offset[i+1];

            WeightT cost = newCostList[start_vertex];

            for(uint64_t j = shift_start + laneIdx; j < end; j += WARP_SIZE) {
                if (newCostList[start_vertex] != cost)
                    break;
                uint end_node = (*edgeList)[j];
                if (newCostList[end_node] > cost + weightList[j] && j >= start){
                    AtomicMin(&(newCostList[end_node]), cost + weightList[j]);
                    // newCostList[end_node] = cost + weightList[j];
                }
            }

            label[start_vertex] = false;
        }
    }

    // uint64_t tid = blockIdx.x * blockDim.x + threadIdx.x; // 
    // size_t warpId = tid >> WARP_SHIFT;
    // size_t lane = tid & ((1 << WARP_SHIFT) - 1);

    // if(warpId < new_size){
    //     uint startVertex = new_vertex_list[warpId];

    //     if (label[startVertex]) {
    //         uint64_t start = new_offset[warpId];
    //         // const uint64_t shift_start = start & MEM_ALIGN;
    //         uint64_t end = new_offset[warpId+1];

    //         WeightT cost = newCostList[startVertex];

    //         for(uint64_t i = start + lane; i < end; i += (1 << WARP_SHIFT)) {
    //             // if (newCostList[startVertex] != cost)
    //             //     break;
    //             uint end_edge = (*edgeList)[i];
    //             if (newCostList[end_edge] > cost + /*weightList[i]*/1 /*&& i >= start*/)
    //                 atomicMin(&(newCostList[end_edge]), cost + /*weightList[i]*/1);
    //         }

    //         label[startVertex] = false;
    //     }

    // }


    // // Page size in elements (64KB / 4 bytes per unsigned int)
    // const size_t pageSize = 8*1024 / sizeof(unsigned int);
    // // Elements per warp
    // const size_t elementsPerWarp = pageSize / warpSize;

    // // Global thread ID
    // size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    // // if(tid == 0) printf("warpSize: %d\n", warpSize);
    // // Warp ID within the block
    // size_t warpId = tid / warpSize;

    // // Thread lane within the warp
    // size_t lane = threadIdx.x % warpSize;

    // // Determine which page this warp will process
    // size_t pageStart = warpId * pageSize;

    // // Ensure we don't process out-of-bounds pages
    // if (pageStart < n * pageSize) {
        

    //     // Process elements within the page
    //     for (size_t i = 0; i < elementsPerWarp; ++i) {
    //         size_t elementIdx = pageStart + lane + i * warpSize;
    //         if (elementIdx < new_size){
    //             uint startVertex = new_vertex_list[elementIdx];
    //             if (/*elementIdx < new_size && */label[startVertex]) {
    //                 WeightT cost = newCostList[startVertex];

    //                 // Process adjacent nodes
    //                 // if(new_offset[elementIdx+1] - d_edgesOffset[elementIdx] >= 2*1024)
    //                 //     printf("elementx: %llu\n", elementIdx);
    //                 for(size_t j = new_offset[elementIdx]; j < new_offset[elementIdx+1]; ++j) {
    //                     uint end_edge = (*edgeList)[j]; // shared_data[j - pageStart];
    //                     // if (newCostList[startVertex] != cost)
    //                     //     break;
    //                     if (newCostList[end_edge] > cost + 1) {
    //                         atomicMin(&(newCostList[end_edge]), cost + 1);
    //                     }
    //                 }

    //                 // Mark node as processed
    //                 label[startVertex] = false;
    //             }
    //         }
    //     }
    // }
}

<<<<<<< HEAD

__global__ // __launch_bounds__(1024,2) 
void kernel_coalesce_new_repr_rdma_weight(bool *label, size_t n, const WeightT *costList, WeightT *newCostList, const uint64_t new_size, uint64_t *new_offset, 
                                unsigned int *new_vertex_list, rdma_buf<unsigned int> *edgeList, rdma_buf<float> *weightList) {

    const uint64_t tid = blockDim.x * 1024 * blockIdx.y + blockDim.x * blockIdx.x + threadIdx.x;
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
        if (label[start_vertex]) {
            uint64_t start = new_offset[i];
            const uint64_t shift_start = start & MEM_ALIGN;
            uint64_t end = new_offset[i+1];

            WeightT cost = newCostList[start_vertex];

            for(uint64_t j = shift_start + laneIdx; j < end; j += WARP_SIZE) {
                if (newCostList[start_vertex] != cost)
                    break;
                uint end_node = (uint) (*weightList)[2*j];
                float w = (*weightList)[2*j + 1];
                if (newCostList[end_node] > cost + w && j >= start){
                    AtomicMin(&(newCostList[end_node]), cost + w);
                    // newCostList[end_node] = cost + weightList[j];
                }
            }

            label[start_vertex] = false;
        }
    }
}


=======
>>>>>>> origin/cloudlab
__global__ void __launch_bounds__(1024,2) 
emogi_rdma_bfs_new_repr(bool *label, size_t n, const WeightT *costList, WeightT *newCostList, const uint64_t new_size, uint64_t *new_offset, 
                                  unsigned int *new_vertex_list, unsigned int *edgeList, float *weightList, bool *d_changed) {

    // Page size in elements (64KB / 4 bytes per unsigned int)
    const size_t pageSize = (8*1024) / sizeof(uint64_t);
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
            if (elementIdx < new_size){
                uint startVertex = new_vertex_list[elementIdx];
                if (label[startVertex] && newCostList[startVertex] != 1000000000.0f) {
            
                    WeightT cost = newCostList[startVertex];

                    // Process adjacent nodes
                    // if(new_offset[elementIdx+1] - d_edgesOffset[elementIdx] >= 2*1024)
                    //     printf("elementx: %llu\n", elementIdx);
                    for(size_t j = new_offset[elementIdx]; j < new_offset[elementIdx+1]; ++j) {
                        
                        if (newCostList[startVertex] != cost)
                            break;
                        uint end_edge = edgeList[j]; // shared_data[j - pageStart];
                        float newDist = cost + weightList[j];

                        if (newCostList[end_edge] > newDist /*1*/){

                            AtomicMin(&(newCostList[end_edge]), newDist);
                            *d_changed = true;
                            label[end_edge] = true;
                        }
                    }

                    // Mark node as processed
                    label[startVertex] = false;
                }
            }
        }
    }
}

__global__ void __launch_bounds__(1024,2) 
kernel_coalesce_rdma(bool *label, const WeightT *costList, WeightT *newCostList, const uint64_t vertex_count, uint64_t *vertexList, 
                                /*EdgeT*/ rdma_buf<unsigned int> *edgeList, WeightT *weightList) {

    // const uint64_t tid = blockIdx.x * blockDim.x + threadIdx.x; // 
    // // const uint64_t tid = blockDim.x * BLOCK_SIZE * blockIdx.y + blockDim.x * blockIdx.x + threadIdx.x;
    
    // const uint64_t laneIdx = tid & ((1 << WARP_SHIFT) - 1);

    // // // Warp ID within the block
    // size_t warpId = tid / (1 << WARP_SHIFT);

    // // // Determine which page this warp will process
    // // size_t pageStart = warpId * pageSize;

    // if(warpId < vertex_count && label[warpId]) {
    //     uint64_t start = vertexList[warpId];
    //     uint64_t end = vertexList[warpId+1];

    //     WeightT cost = newCostList[warpId];

    //     for(uint64_t i = start + laneIdx; i < end; i += (1 << WARP_SHIFT)) {
    //         if (newCostList[warpId] != cost)
    //             break;
    //         unsigned int end_edge = (*edgeList)[i];
    //         if (newCostList[end_edge] > cost + /*weightList[i]*/1 && i >= start)
    //             atomicMin(&(newCostList[end_edge]), cost + /*weightList[i]*/1);
    //     }

    //     label[warpId] = false;
    // }

    const uint64_t tid = blockDim.x * 1024 * blockIdx.y + blockDim.x * blockIdx.x + threadIdx.x;
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
        if (label[i]) {
            uint64_t start = vertexList[i];
            const uint64_t shift_start = start & MEM_ALIGN;
            uint64_t end = vertexList[i+1];

            WeightT cost = newCostList[i];

            for(uint64_t j = shift_start + laneIdx; j < end; j += WARP_SIZE) {
                if (newCostList[i] != cost)
                    break;
                uint end_node = (*edgeList)[j];
                if (newCostList[end_node] > cost + weightList[j] /*1*/ && j >= start)
                    AtomicMin(&(newCostList[end_node]), cost + /*1*/ weightList[j]);
            }

            label[i] = false;
        }
    }
}


<<<<<<< HEAD
__global__ // __launch_bounds__(1024,2)
void kernel_coalesce_rdma_weights(bool *label, const WeightT *costList, WeightT *newCostList, const uint64_t vertex_count, uint64_t *vertexList, 
                                /*EdgeT*/ rdma_buf<unsigned int> *edgeList, rdma_buf<WeightT> *weightList) {

    // const uint64_t tid = blockIdx.x * blockDim.x + threadIdx.x; // 
    // // const uint64_t tid = blockDim.x * BLOCK_SIZE * blockIdx.y + blockDim.x * blockIdx.x + threadIdx.x;
    
    // const uint64_t laneIdx = tid & ((1 << WARP_SHIFT) - 1);

    // // // Warp ID within the block
    // size_t warpId = tid / (1 << WARP_SHIFT);

    // // // Determine which page this warp will process
    // // size_t pageStart = warpId * pageSize;

    // if(warpId < vertex_count && label[warpId]) {
    //     uint64_t start = vertexList[warpId];
    //     uint64_t end = vertexList[warpId+1];

    //     WeightT cost = newCostList[warpId];

    //     for(uint64_t i = start + laneIdx; i < end; i += (1 << WARP_SHIFT)) {
    //         if (newCostList[warpId] != cost)
    //             break;
    //         unsigned int end_edge = (*edgeList)[i];
    //         if (newCostList[end_edge] > cost + /*weightList[i]*/1 && i >= start)
    //             atomicMin(&(newCostList[end_edge]), cost + /*weightList[i]*/1);
    //     }

    //     label[warpId] = false;
    // }

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
        if (label[i]) {
            uint64_t start = vertexList[i];
            const uint64_t shift_start = start & MEM_ALIGN;
            uint64_t end = vertexList[i+1];

            WeightT cost = newCostList[i];

            for(uint64_t j = shift_start + laneIdx; j < end; j += WARP_SIZE) {
                if (newCostList[i] != cost)
                    break;
                uint end_node = (uint) (*weightList)[2*j];
                float w = (*weightList)[2*j+1];
                // printf("end_node; %d, w: %f\n", end_node, w);
                if (newCostList[end_node] > cost + w /*1*/ && j >= start)
                    AtomicMin(&(newCostList[end_node]), cost + /*1*/ w);
            }

            label[i] = false;
        }
    }
}

=======
>>>>>>> origin/cloudlab
#define PAGE_SIZE (64 * 1024)  // 64KB page size
#define PAGE_MASK (PAGE_SIZE - 1)

// __global__ void 
// kernel_coalesce_rdma_opt(bool *label, size_t numEdges, const WeightT *costList, WeightT *newCostList, const uint64_t vertex_count,
//                      const uint64_t *vertexList, rdma_buf<unsigned int> *edgeList, const WeightT *weightList)

__global__ void 
kernel_coalesce_rdma_opt(bool *label, size_t n, const WeightT *costList, WeightT *newCostList, const uint64_t vertex_count,
                     unsigned int *vertexList, rdma_buf<unsigned int> *edgeList, const WeightT *weightList) {

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
            
                
                if (elementIdx < vertex_count && label[elementIdx]) {
                    WeightT cost = newCostList[elementIdx];

                    // Process adjacent nodes
                    // if(new_offset[elementIdx+1] - d_edgesOffset[elementIdx] >= 2*1024)
                    //     printf("elementx: %llu\n", elementIdx);
                    for(size_t j = vertexList[elementIdx]; j < vertexList[elementIdx+1]; ++j) {
                        uint end_edge = (*edgeList)[j]; // shared_data[j - pageStart];
                        // if (newCostList[startVertex] != cost)
                        //     break;
                        if (newCostList[end_edge] > cost + 1) {
                            AtomicMin(&(newCostList[end_edge]), cost + 1);
                        }
                    }

                    // Mark node as processed
                    label[elementIdx] = false;
                }
            }
        
    }
}

__global__
void kernel_coalesce_rdma_opt_1(bool *label, size_t n, size_t numVertices, size_t numEdges, unsigned int level, rdma_buf<unsigned int> *d_adjacencyList, 
                        uint64_t *d_edgesOffset, const WeightT *costList, WeightT *newCostList, unsigned int *d_distance) {
    // extern __shared__ unsigned int shared_data[];
     
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
        // Calculate the number of elements in the current page
        // uint64_t limit = (n*pageSize) < numEdges ? (n*pageSize) : numEdges; 
        // size_t pageEnd = (pageStart + pageSize) < limit ? (pageStart + pageSize) : limit;
        // min(pageStart + pageSize, n * pageSize);

        // Load data into shared memory with boundary checks
        // if (threadIdx.x < (pageEnd - pageStart)) {
        //     /*shared_data[threadIdx.x] = */ uint k = (*d_adjacencyList)[pageStart];
        // }
        // __syncthreads();  // Synchronize threads to ensure shared memory is fully loaded

        // Process elements within the page
        for (size_t i = 0; i < elementsPerWarp; ++i) {
            size_t elementIdx = pageStart + lane + i * warpSize;
            if (elementIdx < numVertices && label[elementIdx]) {
                WeightT cost = newCostList[elementIdx];

                // Process adjacent nodes
                // if(d_edgesOffset[elementIdx+1] - d_edgesOffset[elementIdx] >= 2*1024)
                //     printf("elementx: %llu\n", elementIdx);
                for(size_t j = d_edgesOffset[elementIdx]; j < d_edgesOffset[elementIdx+1]; ++j) {
                    uint end_edge = (*d_adjacencyList)[j]; // shared_data[j - pageStart];
                    if (newCostList[elementIdx] != cost)
                        break;
                    if (newCostList[end_edge] > cost + 1) {
                        AtomicMin(&(newCostList[end_edge]), cost + 1);
                    }
                }

                // Mark node as processed
                label[elementIdx] = false;
            }
        }
    }
}

// __global__
// void kernel_coalesce_rdma_opt_1(bool *label, size_t n, size_t numVertices, unsigned int level, rdma_buf<unsigned int> *d_adjacencyList, uint64_t *d_edgesOffset,
//                     const WeightT *costList, WeightT *newCostList, unsigned int *d_distance) {
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

//     // Determine which page this warp will process
//     size_t pageStart = warpId * pageSize;

//     // Ensure we don't process out-of-bounds pages
//     if (pageStart < n * pageSize) {
//         // bool localChanged = false;
        
//         // Process elements within the page
//         for (size_t i = 0; i < elementsPerWarp; ++i) {
//             size_t elementIdx = pageStart + lane + i * warpSize;
//             if (elementIdx < numVertices && label[elementIdx]) {
//                 WeightT cost = newCostList[elementIdx];

//                 // Process adjacent nodes
//                 for(size_t j = d_edgesOffset[elementIdx]; j < d_edgesOffset[elementIdx+1]; ++j) {
//                     uint end_edge = (*d_adjacencyList)[j];
//                     if (newCostList[end_edge] > cost + 1) {
//                         atomicMin(&(newCostList[end_edge]), cost + 1);
//                         // localChanged = true;
//                     }
//                 }

//                 // Mark node as processed
//                 label[elementIdx] = false;
//             }
//         }

//         // Optionally use shared memory for local data if appropriate
//         // Shared memory should be declared as __shared__ type at the kernel level

//     }
// }


// __global__
// void kernel_coalesce_rdma_opt_1(bool *label, size_t n, size_t numVertices, size_t numEdges, unsigned int level, rdma_buf<unsigned int> *d_adjacencyList, uint64_t *d_edgesOffset,
//                     const WeightT *costList, WeightT *newCostList, unsigned int *d_distance) {
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

//     // Determine which page this warp will process
//     size_t pageStart = warpId * pageSize;

//     // Ensure we don't process out-of-bounds pages
//     if (pageStart < n * pageSize) {
//         bool localChanged = false;
        
//         // Process elements within the page
//         for (size_t i = 0; i < elementsPerWarp; ++i) {
//             size_t elementIdx = pageStart + lane + i * warpSize;
//             if (elementIdx < numVertices) {
//                 // unsigned int k = d_distance[elementIdx];
//                 if (label[elementIdx]) {
//                     WeightT cost = newCostList[elementIdx];

//                     // printf("d_edgesOffset[%llu]: %lu, label[%llu]: %d cost: %d\n", 
//                     //     (long long int) elementIdx, d_edgesOffset[elementIdx], (long long int) elementIdx, label[elementIdx], (int) cost);

//                     for(size_t j = d_edgesOffset[elementIdx]; j < d_edgesOffset[elementIdx+1] /*+ d_edgesSize[elementIdx]*/; ++j) {
//                         if (newCostList[elementIdx] != cost)
//                             break;
//                         uint end_edge = (*d_adjacencyList)[j];
//                         if (newCostList[end_edge] > cost + /*weightList[i]*/1 /*&& i >= d_edgesOffset[elementIdx]*/)
//                             atomicMin(&(newCostList[end_edge]), cost + /*weightList[i]*/1);


//                         // unsigned int v = (*d_adjacencyList)[j];
//                         // // printf(" %d ", v );
//                         // unsigned int dist = d_distance[v];
//                         // if (level + 1 < dist) {
//                         //     d_distance[v] = level + 1;
//                         //     localChanged = true;
//                         // }
//                     }

//                     label[elementIdx] = false;
//                 }
//             }
//         }

        
//     }
// }



__global__
void kernel_coalesce_rdma_opt_warp(bool *label, size_t n, size_t numVertices, unsigned int level, rdma_buf<unsigned int> *d_adjacencyList, 
                    uint64_t *d_edgesOffset, const WeightT *costList, WeightT *newCostList, unsigned int *d_distance) {

    // Thread index
    size_t thid = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int warpSize = 4;
    // Warp index and lane index
    size_t warpId = thid / warpSize;
    size_t laneId = thid % warpSize;
    // Warp size
    

    // Buffer for storing distances and change flag in shared memory
    unsigned int shared_distance;
    unsigned int warp_changed;

    if (warpId < n && label[warpId]) {
        // Each warp processes one node
        // if (laneId == 0) {
            WeightT cost = newCostList[warpId];
            // warp_changed = 0;
        // }

        // __syncwarp(); // Synchronize within warp
        // if (shared_distance == level) {
            uint64_t nodeStart = d_edgesOffset[warpId];
            uint64_t nodeEnd = d_edgesOffset[warpId + 1];

            for (size_t i = nodeStart + laneId; i < nodeEnd; i += warpSize) {
            
                if (newCostList[warpId] != cost)
                    break;
                uint end_edge = (*d_adjacencyList)[i];
                if (newCostList[end_edge] > cost + /*weightList[i]*/1 /*&& i >= nodeStart*/)
                    AtomicMin(&(newCostList[end_edge]), cost + /*weightList[i]*/1);
                
            }
        

        label[warpId] = false;
    }
}


// __global__ void 
// kernel_baseline_rdma(bool *label, const WeightT *costList, WeightT *newCostList, const uint64_t vertex_count,
//                      const uint64_t *vertexList, rdma_buf<uint64_t> *edgeList, const WeightT *weightList) {

//     const uint64_t tid = blockDim.x * blockIdx.x + threadIdx.x;

//     if (tid < vertex_count && label[tid]) {
//         uint64_t start = vertexList[tid];
//         uint64_t end = vertexList[tid+1];

//         WeightT cost = newCostList[tid];

//         for(uint64_t i = start; i < end; i += 1) {
//             if (newCostList[tid] != cost)
//                 break;
//             uint64_t end_edge = (*edgeList)[i];
//             if (newCostList[end_edge] > cost + /*weightList[i]*/1 && i >= start)
//                 atomicMin(&(newCostList[end_edge]), cost + /*weightList[i]*/1);
//         }

//         label[tid] = false;
//     }
// }

// __global__ void kernel_baseline_rdma(size_t n, bool *label, const WeightT *costList, WeightT *newCostList, 
//                                      const uint64_t vertex_count, const uint64_t *vertexList, 
//                                      rdma_buf<uint64_t> *edgeList, const WeightT *weightList) {
//     const uint64_t warpSize = 32; // Assuming warp size is 32
//     const uint64_t pageSize = 64 * 1024 / sizeof(uint64_t); // Number of edges per page
//     const uint64_t tid = blockDim.x * blockIdx.x + threadIdx.x;
//     const uint64_t warpId = threadIdx.x / warpSize; // Unique warp ID
//     const uint64_t numWarps = (vertex_count + warpSize - 1) / warpSize; // Number of warps
    
    

//     uint64_t pageStart = (vertexList[tid] / pageSize) * pageSize;
//     uint64_t pageEnd = ((vertexList[tid+1] + pageSize - 1) / pageSize) * pageSize;

//     if (pageStart >= n * pageSize) return;

//     if (label[tid]) {
//         WeightT cost = newCostList[tid];

//         // Each warp handles one page
//         for (uint64_t i = pageStart + warpId * (pageSize / warpSize); i < pageEnd; i += pageSize / numWarps) {
//             if (newCostList[tid] != cost) 
//                 break;
                
//             uint64_t end_edge = (*edgeList)[i];
//             if (newCostList[end_edge] > cost + /*weightList[i]*/1) {
//                 atomicMin(&(newCostList[end_edge]), cost + /*weightList[i]*/1);
//             }
//         }
        
//         label[tid] = false;
//     }
// }

// __global__ void kernel_baseline_rdma_1(size_t n, uint64_t numEdges, uint64_t numVertex, uint64_t *edgeOffset, rdma_buf<uint64_t> *edgeList,
//                                 uint *dist, bool *changed) {
//     uint64_t threadId = blockDim.x * blockIdx.x + threadIdx.x;
//     // if(threadId == 0) printf("hello from thread 0\n");

//     if (threadId < numVertex){

//         for (uint64_t nodeId = edgeOffset[threadId]; nodeId < edgeOffset[threadId+1]; nodeId++) {
//             uint64_t source = threadId;
//             uint64_t end = (*edgeList)[nodeId]; // edgelist
//             uint weight = 1; // edgesWeight[nodeId];
            
//             if (dist[source] + weight < dist[end]) {
//                 atomicMin(&dist[end], dist[source] + weight);
//                 // dist[end] = dist[source] + weight;
//                 // preNode[end] = source;
//                 *changed = true;
//             }
//         }
//     }
// }


