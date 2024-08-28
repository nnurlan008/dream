#include <device_launch_parameters.h>
#include <cstdio>
#include "bfs.cuh"
// #include "../../include/runtime.h"

// #ifndef _BFSCUDA_H_
// #define _BFSCUDA_H_
// extern "C" {

// #include "../../include/runtime.h"

// __global__
// void simpleBfs_rdma(int N, int level, rdma_buf<uint> *d_adjacencyList, rdma_buf<uint> *d_edgesOffset,
//                rdma_buf<uint> *d_edgesSize, rdma_buf<uint> *d_distance, rdma_buf<uint> *d_parent, uint *changed);




__global__
void simpleBfs(int N, int level, uint *d_adjacencyList, uint *d_edgesOffset,
               uint *d_edgesSize, uint *d_distance, uint *d_parent, uint *changed) {
    int thid = blockIdx.x * blockDim.x + threadIdx.x;
    int valueChange = 0;

    if (thid < N && d_distance[thid] == level) {
        int u = thid;
        for (int i = d_edgesOffset[u]; i < d_edgesOffset[u] + d_edgesSize[u]; i++) {
            int v = d_adjacencyList[i];
            if (level + 1 < d_distance[v]) {
                d_distance[v] = level + 1;
                d_parent[v] = i;
                valueChange = 1;
            }
        }
    }

    if (valueChange) {
        *changed = valueChange;
    }
}

__global__
void queueBfs(int level, uint *d_adjacencyList, uint *d_edgesOffset, uint *d_edgesSize, uint *d_distance, uint *d_parent,
              int queueSize, uint *nextQueueSize, uint *d_currentQueue, uint *d_nextQueue) {
    int thid = blockIdx.x * blockDim.x + threadIdx.x;

    if (thid < queueSize) {
        int u = d_currentQueue[thid];
        for (int i = d_edgesOffset[u]; i < d_edgesOffset[u] + d_edgesSize[u]; i++) {
            int v = d_adjacencyList[i];
            if (d_distance[v] == INT_MAX && atomicMin(&d_distance[v], level + 1) == INT_MAX) {
                d_parent[v] = i;
                int position = atomicAdd(nextQueueSize, 1);
                d_nextQueue[position] = v;
            }
        }
    }
}

//Scan bfs
__global__
void nextLayer(int level, uint *d_adjacencyList, uint *d_edgesOffset, uint *d_edgesSize, uint *d_distance, uint *d_parent,
               int queueSize, uint *d_currentQueue) {
    int thid = blockIdx.x * blockDim.x + threadIdx.x;

    if (thid < queueSize) {
        int u = d_currentQueue[thid];
        for (int i = d_edgesOffset[u]; i < d_edgesOffset[u] + d_edgesSize[u]; i++) {
            int v = d_adjacencyList[i];
            if (level + 1 < d_distance[v]) {
                d_distance[v] = level + 1;
                d_parent[v] = i;
            }
        }
    }
}

__global__
void countDegrees(uint *d_adjacencyList, uint *d_edgesOffset, uint *d_edgesSize, uint *d_parent,
                  int queueSize, uint *d_currentQueue, uint *d_degrees) {
    int thid = blockIdx.x * blockDim.x + threadIdx.x;

    if (thid < queueSize) {
        int u = d_currentQueue[thid];
        int degree = 0;
        for (int i = d_edgesOffset[u]; i < d_edgesOffset[u] + d_edgesSize[u]; i++) {
            int v = d_adjacencyList[i];
            if (d_parent[v] == i && v != u) {
                ++degree;
            }
        }
        d_degrees[thid] = degree;
    }
}

__global__
void scanDegrees(int size, uint *d_degrees, uint *incrDegrees) {
    int thid = blockIdx.x * blockDim.x + threadIdx.x;

    if (thid < size) {
        //write initial values to shared memory
        __shared__ int prefixSum[1024];
        int modulo = threadIdx.x;
        prefixSum[modulo] = d_degrees[thid];
        __syncthreads();

        //calculate scan on this block
        //go up
        for (int nodeSize = 2; nodeSize <= 1024; nodeSize <<= 1) {
            if ((modulo & (nodeSize - 1)) == 0) {
                if (thid + (nodeSize >> 1) < size) {
                    int nextPosition = modulo + (nodeSize >> 1);
                    prefixSum[modulo] += prefixSum[nextPosition];
                }
            }
            __syncthreads();
        }

        //write information for increment prefix sums
        if (modulo == 0) {
            int block = thid >> 10;
            incrDegrees[block + 1] = prefixSum[modulo];
        }

        //go down
        for (int nodeSize = 1024; nodeSize > 1; nodeSize >>= 1) {
            if ((modulo & (nodeSize - 1)) == 0) {
                if (thid + (nodeSize >> 1) < size) {
                    int next_position = modulo + (nodeSize >> 1);
                    int tmp = prefixSum[modulo];
                    prefixSum[modulo] -= prefixSum[next_position];
                    prefixSum[next_position] = tmp;

                }
            }
            __syncthreads();
        }
        d_degrees[thid] = prefixSum[modulo];
    }

}

__global__
void assignVerticesNextQueue(uint *d_adjacencyList, uint *d_edgesOffset, uint *d_edgesSize, uint *d_parent, int queueSize,
                             uint *d_currentQueue, uint *d_nextQueue, uint *d_degrees, uint *incrDegrees,
                             int nextQueueSize) {
    int thid = blockIdx.x * blockDim.x + threadIdx.x;

    if (thid < queueSize) {
        __shared__ int sharedIncrement;
        if (!threadIdx.x) {
            sharedIncrement = incrDegrees[thid >> 10];
        }
        __syncthreads();

        int sum = 0;
        if (threadIdx.x) {
            sum = d_degrees[thid - 1];
        }

        int u = d_currentQueue[thid];
        int counter = 0;
        for (int i = d_edgesOffset[u]; i < d_edgesOffset[u] + d_edgesSize[u]; i++) {
            int v = d_adjacencyList[i];
            if (d_parent[v] == i && v != u) {
                int nextQueuePlace = sharedIncrement + sum + counter;
                d_nextQueue[nextQueuePlace] = v;
                counter++;
            }
        }
    }
}

// }
// #endif 