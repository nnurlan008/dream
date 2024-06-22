#ifndef _BFS_CUH_
#define _BFS_CUH_

__global__ void simpleBfs(int N, int level, uint *d_adjacencyList, uint *d_edgesOffset,
               uint *d_edgesSize, uint *d_distance, uint *d_parent, uint *changed);

__global__ void queueBfs(int level, uint *d_adjacencyList, uint *d_edgesOffset, uint *d_edgesSize, uint *d_distance, uint *d_parent,
              int queueSize, uint *nextQueueSize, uint *d_currentQueue, uint *d_nextQueue);

__global__ void nextLayer(int level, uint *d_adjacencyList, uint *d_edgesOffset, uint *d_edgesSize, uint *d_distance, uint *d_parent,
                              int queueSize, uint *d_currentQueue);

__global__ void countDegrees(uint *d_adjacencyList, uint *d_edgesOffset, uint *d_edgesSize, uint *d_parent,
                  int queueSize, uint *d_currentQueue, uint *d_degrees);

__global__ void scanDegrees(int size, uint *d_degrees, uint *incrDegrees);

__global__ void assignVerticesNextQueue(uint *d_adjacencyList, uint *d_edgesOffset, uint *d_edgesSize, uint *d_parent, int queueSize,
                             uint *d_currentQueue, uint *d_nextQueue, uint *d_degrees, uint *incrDegrees,
                             int nextQueueSize);

#endif