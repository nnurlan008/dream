#ifndef GRAPH_H
#define GRAPH_H

#include <vector>
#include <cstdio>
#include <cstdlib>

struct OutEdge{
    uint end;
};

typedef struct OutEdge E;

struct Graph_x {
    std::vector<uint> adjacencyList; // all edges
    std::vector<uint> edgesOffset; // offset to adjacencyList for every vertex
    std::vector<uint> edgesSize; //number of edges for every vertex
    uint64_t *adjacencyList_r; // all edges
    uint64_t *edgesOffset_r; // offset to adjacencyList for every vertex
    uint *edgesSize_r; //number of edges for every vertex
    uint64_t numVertices = 0;
    uint64_t numEdges = 0;
};

struct Graph_m {
    E *edgeList; // all edges
    uint *nodePointer; // offset to adjacencyList for every vertex
    std::vector<int> edgesSize; //number of edges for every vertex
    unsigned int numVertices = 0;
    unsigned int numEdges = 0;
};

void readfile(Graph_x &G, Graph_m &G_m, int argc, char **argv, uint *u_edgesOffset, \
              uint *u_edgesSize, uint *u_adjacencyList, uint *&u_weights);

void readfile(Graph_x &G, Graph_m &G_m, int argc, char **argv, uint *u_edgesOffset, \
              uint *u_edgesSize, uint *u_adjacencyList, float *&u_weights);

// void readGraph(Graph &G, int argc, char **argv);

#endif //BFS_CUDA_GRAPH_H
