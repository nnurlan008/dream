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


__global__
void simpleBfs_rdma(int n, int level, rdma_buf<uint> *d_adjacencyList, rdma_buf<uint> *d_edgesOffset,
               rdma_buf<uint> *d_edgesSize, rdma_buf<uint> *d_distance, rdma_buf<uint> *d_parent, uint *changed);

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

rdma_buf<unsigned int> *u_adjacencyList;
rdma_buf<unsigned int> *u_edgesOffset;
rdma_buf<unsigned int> *u_edgesSize;
rdma_buf<unsigned int> *u_distance;
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

    checkError(cudaMallocManaged(&u_adjacencyList, sizeof(rdma_buf<uint>)));
    checkError(cudaMallocManaged(&u_edgesOffset, sizeof(rdma_buf<uint>)));
    checkError(cudaMallocManaged(&u_edgesSize, sizeof(rdma_buf<uint>)));
    checkError(cudaMallocManaged(&u_distance, sizeof(rdma_buf<uint>)));
    checkError(cudaMallocManaged(&u_parent, sizeof(rdma_buf<uint>)));
    checkError(cudaMallocManaged(&u_currentQueue, sizeof(rdma_buf<uint>)));
    checkError(cudaMallocManaged(&u_nextQueue, sizeof(rdma_buf<uint>)));
    checkError(cudaMallocManaged(&u_degrees, sizeof(rdma_buf<uint>)));

    u_adjacencyList->start(G.numEdges *sizeof(uint));
    u_edgesOffset->start(G.numVertices *sizeof(uint));
    u_edgesSize->start(G.numVertices *sizeof(uint));
    u_distance->start(G.numVertices *sizeof(uint));
    u_parent->start(G.numVertices *sizeof(uint));
    u_currentQueue->start(G.numVertices *sizeof(uint));
    u_nextQueue->start(G.numVertices *sizeof(uint));
    u_degrees->start(G.numVertices *sizeof(uint));

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
    for (int i = 0; i < G.numVertices; i++) {
        if (expectedDistance[i] != u_distance->host_buffer[i] ) {
            printf("%d %d %d\n", i, distance[i], expectedDistance[i]);
            printf("Wrong output!\n");
            exit(1);
        }
    }

    printf("Output OK!\n\n");
}


void initializeCudaBfs(int startVertex, std::vector<int> &distance, std::vector<int> &parent, Graph &G) {
    //initialize values
    std::fill(distance.begin(), distance.end(), std::numeric_limits<int>::max());
    std::fill(parent.begin(), parent.end(), std::numeric_limits<int>::max());
    distance[startVertex] = 0;
    parent[startVertex] = 0;

    // checkError(cudaMemcpy(d_distance, distance.data(), G.numVertices * sizeof(int), cudaMemcpyHostToDevice));
    // checkError(cudaMemcpy(d_parent, parent.data(), G.numVertices * sizeof(int), cudaMemcpyHostToDevice));
    // memcpy(u_distance->host_buffer, distance.data(), G.numVertices * sizeof(int));
    // memcpy(u_parent->host_buffer, parent.data(), G.numVertices * sizeof(int));
    printf("printing samples from parent and distance vectors initializations...\n");
    for (size_t i = 0; i < G.numVertices; i++)
    {
        u_distance->host_buffer[i] = distance.data()[i];
        u_parent->host_buffer[i] = parent.data()[i];
    }

    for (size_t i = 0; i < 5; i++)
    {
        printf("u_distance->host_buffer[%llu]: %llu; distance.data()[%llu]: %llu\n", i, u_distance->host_buffer[i], i, distance.data()[i]);
        printf("u_parent->host_buffer[%llu]: %llu; parent.data()[%llu]: %llu\n", i, u_parent->host_buffer[i], i, parent.data()[i]);
        
    }
    

    int firstElementQueue = startVertex;
    // cudaMemcpy(d_currentQueue, &firstElementQueue, sizeof(int), cudaMemcpyHostToDevice);
    *u_currentQueue->host_buffer = firstElementQueue;
}

void finalizeCudaBfs(std::vector<int> &distance, std::vector<int> &parent, Graph &G) {
    //copy memory from device
    // checkError(cudaMemcpy(distance.data(), d_distance, G.numVertices * sizeof(int), cudaMemcpyDeviceToHost));
    // checkError(cudaMemcpy(parent.data(), d_parent, G.numVertices * sizeof(int), cudaMemcpyDeviceToHost));
}

void runCudaSimpleBfs(int startVertex, Graph &G, std::vector<int> &distance,
                      std::vector<int> &parent) {
    initializeCudaBfs(startVertex, distance, parent, G);

    uint *changed;
    checkError(cudaMallocHost((void **) &changed, sizeof(uint)));

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

    cudaEventRecord(event1, (cudaStream_t)0);
    *changed = 1;
    int level = 0;
    while (*changed) {
        *changed = 0;
        // void *args[] = {&G.numVertices, &level, &d_adjacencyList, &d_edgesOffset, &d_edgesSize, &d_distance, &d_parent,
        //                 &changed};
        // checkError(cuLaunchKernel(cuSimpleBfs, G.numVertices / 1024 + 1, 1, 1,
        //                           1024, 1, 1, 0, 0, args, 0));
        ret1 = cudaDeviceSynchronize();
        printf("cudaDeviceSynchronize: %d\n", ret1);  
        if(cudaSuccess != ret1){  
            printf("cudaDeviceSynchronize error: %d\n", ret1);  
            exit(-1);
        }
        printf("G.numVertices: %llu\n", G.numVertices); 
        simpleBfs_rdma<<<G.numVertices / 512 + 1, 512>>>(G.numVertices, level, u_adjacencyList, u_edgesOffset, u_edgesSize, u_distance, u_parent, changed);                 
        printf("cudaGetLastError(): %d\n", cudaGetLastError());
        ret1 = cudaDeviceSynchronize();
        printf("cudaDeviceSynchronize: %d\n", ret1);  
        if(cudaSuccess != ret1){  
            printf("cudaDeviceSynchronize error: %d\n", ret1);  
            exit(-1);
        }

        level++;
    }
    cudaEventRecord(event2, (cudaStream_t) 1);
    cudaEventSynchronize(event1); //optional
    cudaEventSynchronize(event2); //wait for the event to be executed!

    // calculate time
    float dt_ms;
    cudaEventElapsedTime(&dt_ms, event1, event2);
    printf("Elapsed time with cudaEvent: %f\n", dt_ms);

    auto end = std::chrono::steady_clock::now();
    long duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    printf("Elapsed time in milliseconds : %li ms.\n", duration);

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


__global__ void test2(rdma_buf<int> *a/*, rdma_buf<int> *b, rdma_buf<int> *c*/){
    size_t id = blockDim.x * blockIdx.x + threadIdx.x;
    // printf("(*a)[%d]: %d\n", 0, (*a)[0]);
    int k = (*a)[1]; // + (*b)[id];
    // printf("(*a)[%d]: %d\n", 1, (*a)[1]);
    // printf("(*a)[%d]: %d\n", 2, (*a)[2]);
    // printf("(*a)[%d]: %d\n", 3, (*a)[3]);
    a->rvalue(0, 80);
    a->rvalue(1, 81);
    a->rvalue(2, 82);
    a->rvalue(3, 83);

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

__global__ void test2(rdma_buf<int> *a, rdma_buf<int> *b, rdma_buf<int> *c){
    size_t id = blockDim.x * blockIdx.x + threadIdx.x;

    // int k = (*a)[id] + (*b)[id];
    a->rvalue(id, id);
    // c->rvalue(id, (*a)[id] + (*b)[id]); 
    // if(id == 0) printf("(*b)[%d]: %d\n", id, (*b)[id]);
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
    s_ctx->gpu_buf_size = 28*1024*1024*1024llu; // N*sizeof(int)*3llu;

    // remote connection:
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
    // printf("function: %s line: %d u_edgesOffset->host_buffer: %p\n", __FILE__, __LINE__, u_edgesOffset->host_buffer);

    // readGraph(G, argc, argv);
    readfile(G, G_m, argc, argv, tmp_edgesOffset, tmp_edgesSize, tmp_adjacencyList);

    printf("Number of vertices %lld tmp_edgesOffset[10]: %d\n", G.numVertices, G.edgesOffset_r[10]);
    printf("Number of edges %lld\n\n", G.numEdges);

    //vectors for results
    std::vector<int> distance(G.numVertices, std::numeric_limits<int>::max());
    std::vector<int> parent(G.numVertices, std::numeric_limits<int>::max());
    std::vector<bool> visited(G.numVertices, false);

    // run CPU sequential bfs
    runCpu(startVertex, G, distance, parent, visited);
    initCuda(G);
    
    // rdma_buf<int> *a, *b, *c;
    // checkError(cudaMallocManaged(&a, sizeof(rdma_buf<int>)));
    // checkError(cudaMallocManaged(&b, sizeof(rdma_buf<int>)));
    // checkError(cudaMallocManaged(&c, sizeof(rdma_buf<int>)));
    // a->start(N*sizeof(int));
    // b->start(N*sizeof(int));
    // for (size_t i = 0; i < N; i++)
    // {
    //     a->host_buffer[i] = 10;
    //     b->host_buffer[i] = 10;
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
    // // test2<<<1,1>>>(a);
    // test2<<<N/1024,1024>>>(a, b, c);
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
    //     printf("a->host_buffer[%d]: %d\n", i, a->host_buffer[i]);
    // }

    //  u_adjacencyList->start(G.numEdges *sizeof(uint));
    // u_edgesOffset->start(G.numVertices *sizeof(uint));
    // u_edgesSize->start(G.numVertices *sizeof(uint));
    printf("function: %s line: %d\n", __FILE__, __LINE__);
    for(size_t i = 0; i < G.numEdges; i++){
        u_adjacencyList->host_buffer[i] = G.adjacencyList_r[i];
    }
    printf("function: %s line: %d\n", __FILE__, __LINE__);
    for(size_t i = 0; i < G.numVertices; i++){
        u_edgesOffset->host_buffer[i] = G.edgesOffset_r[i];
        u_edgesSize->host_buffer[i] = G.edgesSize_r[i];
    }
    for(size_t i = 0; i < 5; i++){
        printf("u_adjacencyList->size: %llu (*u_adjacencyList)[%d]: %llu\n", u_adjacencyList->size, i, u_adjacencyList->host_buffer[i]);
        printf("u_edgesOffset->size: %llu (*u_edgesOffset)[%d]: %llu\n", u_edgesOffset->size, i, u_edgesOffset->host_buffer[i]);
        printf("u_edgesSize->size: %llu (*u_edgesSize)[%d]: %llu G.edgesSize_r[%d]: %llu\n", u_edgesSize->size, i, u_edgesSize->host_buffer[i], i, G.edgesSize_r[i]);
    }
    printf("function: %s line: %d\n", __FILE__, __LINE__);
    //save results from sequential bfs
    std::vector<int> expectedDistance(distance);
    std::vector<int> expectedParent(parent);
    auto start = std::chrono::steady_clock::now();
    
    //run CUDA simple parallel bfs
    runCudaSimpleBfs(startVertex, G, distance, parent);
    u_distance->memcpyDtoH();
    checkOutput_rdma(distance, expectedDistance, G);

    // // //run CUDA queue parallel bfs
    // runCudaQueueBfs(startVertex, G, distance, parent);
    // checkOutput(distance, expectedDistance, G);

    // // //run CUDA scan parallel bfs
    // runCudaScanBfs(startVertex, G, distance, parent);
    // checkOutput(distance, expectedDistance, G);
    finalizeCuda();
    auto end = std::chrono::steady_clock::now();
    long duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    printf("Overall Elapsed time in milliseconds : %li ms.\n", duration);
    return 0;



    
    // // buf3->start((uint64_t) s_ctx->gpu_buffer, (uint64_t) s_ctx->server_mr.addr, N*sizeof(int));
    // // buf1->address = (uint64_t) s_ctx->gpu_buffer;
    // // buf1->size = 100;
    // printf("buf1->address: %p, buf1->size: %d, REQUEST_SIZE: %d buf1->host_address : %p N*sizeof(int): %llu\n", buf1->gpu_address, buf1->size, REQUEST_SIZE, buf1->host_address, N*sizeof(int));
    // printf("buf2->address: %p, buf2->size: %d, REQUEST_SIZE: %d buf2->host_address : %p\n", buf2->gpu_address, buf2->size, REQUEST_SIZE, buf2->host_address);
    // // cudaMemcpy(buf1, &buf, sizeof(rdma_buf<int>), cudaMemcpyHostToDevice);
    // // printf("buf[2]: %d a: %p\n", buf[2], a);
    // printf("Function name: %s, line number: %d mesg_size: %d num_iteration: %d sizeof(int): %d\n", __func__, __LINE__, mesg_size, num_msg, sizeof(int));
   
    // // allocate poll and post content
    // alloc_global_cont(&post_cont, &poll_cont, &post_cont2);  

    // int thr_per_blk = 2048*2; // s_ctx->n_bufs;
	// int blk_in_grid = 256;

    // int timer_size = 4;
    // clock_t *dtimer = NULL;

    // // Launch kernel
    // cudaError_t ret1 = cudaDeviceSynchronize();
    // printf("ret: %d\n", ret1);
    // if(cudaSuccess != ret1){    
    //     return -1;
    // }

    // cudaEvent_t event1, event2;
    // cudaEventCreate(&event1);
    // cudaEventCreate(&event2);
    // int data_size = mesg_size;

    // struct timespec start, finish, delta;
    // clock_gettime(CLOCK_REALTIME, &start);
    // cudaEventRecord(event1, (cudaStream_t)1); //where 0 is the default stream
    
    // test<<<2048, 512>>>(buf1, buf2, buf3, N);
    
    // cudaEventRecord(event2, (cudaStream_t) 1);
    // clock_gettime(CLOCK_REALTIME, &finish);
    // ret1 = cudaDeviceSynchronize();
    
    // //synchronize
    // cudaEventSynchronize(event1); //optional
    // cudaEventSynchronize(event2); //wait for the event to be executed!

    // float dt_ms;
    // cudaEventElapsedTime(&dt_ms, event1, event2);

    // printf("ret1: %d\n", ret1);
    // if(cudaSuccess != ret1){
    //     return -1;
    // }if (thid < n && k == level) {
        //     int u = thid;
        //     for (int i = (*d_edgesOffset)[u]; i < (*d_edgesOffset)[u] + (*d_edgesSize)[u]; i++) {
        //         int v = (*d_adjacencyList)[i];
        //         if (level + 1 < (*d_distance)[v]) {
        //             // (*d_distance)[v] = level + 1;
        //             // (*d_parent)[v] = i;
        //             d_distance->rvalue(v, level + 1);
        //             d_parent->rvalue(v, i);
        //             valueChange = 1;
        //         }
        //     }
        // }

        // if (valueChange) {
        //     *changed = valueChange;
        // }

    
    

    // clock_t cycles;
    // float g_usec_post;
    // cudaDeviceProp devProp;
    // cudaGetDeviceProperties(&devProp, 0);
    // printf("Cuda device clock rate = %d\n", devProp.clockRate);
    // float freq_post = (float)1/((float)devProp.clockRate*1000), max = 0;

    
    // g_usec_post = (float)((float)1/(devProp.clockRate*1000))*((cycles)) * 1000000;
    // printf("total timer: %f\n", g_usec_post);
    // printf("kernel time: %d.%.9ld dt_ms: %f\n", (int)delta.tv_sec, delta.tv_nsec, dt_ms);

	// // // Free CPU memory
	// // // free(A);
	// // // free(B);
	// // // free(C);

	// // // Free GPU memory
	// // cudaFree(d_A);
	// // cudaFree(d_B);
	// // cudaFree(d_C);

	// // printf("\n---------------------------\n");
	// // printf("__SUCCESS__\n");
	// // printf("---------------------------\n");
	// // printf("N                 = %d\n", N);
	// // printf("Threads Per Block = %d\n", thr_per_blk);
	// // printf("Blocks In Grid    = %d\n", blk_in_grid);
	// // printf("---------------------------\n\n");

    // // destroy(s_ctx);

	return 0;
}

__global__
void simpleBfs_rdma(int n, int level, rdma_buf<unsigned int> *d_adjacencyList, rdma_buf<unsigned int> *d_edgesOffset,
               rdma_buf<unsigned int> *d_edgesSize, rdma_buf<unsigned int> *d_distance, rdma_buf<unsigned int> *d_parent, uint *changed) {
    int thid = blockIdx.x * blockDim.x + threadIdx.x;
    int valueChange = 0;
    if(thid < d_distance->size/sizeof(uint)){
        unsigned int k = (*d_distance)[thid];
        // if(thid == 0 || thid == 1 || thid == 2 || thid == 3){
            // printf("d_distance->size: %llu (*d_distance)[0]: %d\n", d_distance->size, k);
        //     printf("d_edgesOffset->size: %d (*d_edgesOffset)[%d]: %llu (*d_distance)[%d]: %llu\n",\
        //             d_edgesOffset->size, thid, (*d_edgesOffset)[thid], thid, (*d_distance)[thid]);
        //     printf("d_edgesSize->size: %d (*d_edgesSize)[%d]: %llu\n", d_edgesSize->size, thid, (*d_edgesSize)[thid]);   
        // }
        
        // printf("(*d_distance)[thid]: %llu\n", (*d_distance)[thid]);
        if (thid < n && k == level) {
            int u = thid;
            for (int i = (*d_edgesOffset)[u]; i < (*d_edgesOffset)[u] + (*d_edgesSize)[u]; i++) {
                int v = (*d_adjacencyList)[i];
                if (level + 1 < (*d_distance)[v]) {
                    // (*d_distance)[v] = level + 1;
                    // (*d_parent)[v] = i;
                    d_distance->rvalue(v, level + 1);
                    // if(level + 1 == 2)
                        printf("d_distance[v]: %d\n", (*d_distance)[v]);
                    d_parent->rvalue(v, i);
                    valueChange = 1;
                }
            }
        }

        if (valueChange) {
            *changed = valueChange;
        }
    }
    // __syncthreads();
}

