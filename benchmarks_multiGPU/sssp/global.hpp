#ifndef GLOBAL_HPP
#define GLOBAL_HPP

#include <iostream>
#include <stdio.h>
#include <string>
#include <string.h>
#include <vector>


using namespace std;

typedef unsigned int uint;

const unsigned int MAX_DIST = 65535;


struct LoopInfo{
    uint numIteration;
    float time_cpu;
    float time_gpu;
    float time_cpu_message;
    float time_gpu_message;
    float time_dist_merge;
    float splitRatio;
};



void printDist(uint* dist, uint size);

void printProcessed(bool* processed, uint size);

void printPreNode(uint* preNode, uint size);

void compareResult(uint* dist1, uint* dist2, uint numNodes);

void printLoopInfo(vector<LoopInfo> info);

void printLoopInfoV1(vector<LoopInfo> info);

void printLoopInfoV2(vector<LoopInfo> info);

#endif