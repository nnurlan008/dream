#include "global.hpp"


void printDist(uint* dist, uint size) {
    for (int i = 0; i < size; i++) {
        std::cout << "dist[" << i << "]: " << dist[i] << std::endl;
    }
}

void printProcessed(bool* processed, uint size) {
    for (int i = 0; i < size; i++) {
        std::cout << "processed[" << i << "]: " << processed[i] << std::endl;
    }
}

void printPreNode(uint* preNode, uint size) {
    for (int i = 0; i < size; i++) {
        std::cout << "prevNode[" << i << "]: " << preNode[i] << std::endl;
    }
}

void compareResult(WeightT* dist1, WeightT* dist2, uint numNodes) {
    uint diffCount = 0;
    vector<int> nodesId;
    
    for (int i = 0; i < numNodes; i++) {
        if (dist1[i] != dist2[i]) {
            diffCount++;
            nodesId.push_back(i);
            
            // printf("index: %d dist1: %d, dist2: %d\n", i, dist1[i], dist2[i]);
        }
    }

    
    

    if (diffCount == 0) {
        std::cout << "Good! These two result are identical!" <<  std::endl;
    } else {
        std::cout << diffCount << " of " << numNodes << " does not match!" << std::endl;
        // std::cout << "\t\tNode: ";
        // for (int i = 0; i < nodesId.size(); i++) {
        //     std::cout << nodesId[i] << ", ";
        // }
        // std::cout << " does not match" << endl;
    }
}

void printLoopInfo(vector<LoopInfo> info) {
    for (int i = 0; i < info.size(); i++) {
        LoopInfo loopInfo = info[i];
        printf("No. itr: %d , updated CPU data size ratio: %f\n", loopInfo.numIteration, loopInfo.splitRatio);
        printf("CPU PART TIME: %f\n", loopInfo.time_cpu);
        printf("GPU PART TIME: %f\n", loopInfo.time_gpu);
        // printf("Copy dist from host to device : %f ms \n", timer_host_to_device.elapsedTime());
        // printf("Copy dist from device to host : %f ms \n", timer_device_to_host.elapsedTime()); 
    }
}

void printLoopInfoV1(vector<LoopInfo> info) {
    for (int i = 0; i < info.size(); i++) {
        LoopInfo loopInfo = info[i];
        printf("No. itr: %d , updated CPU data size ratio: %f\n", loopInfo.numIteration, loopInfo.splitRatio);
        printf("CPU PART TIME: %f\n", loopInfo.time_cpu);
        printf("GPU PART TIME: %f\n", loopInfo.time_gpu);
        printf("Dist Merge TIME: %f\n", loopInfo.time_dist_merge);
        // printf("Copy dist from host to device : %f ms \n", timer_host_to_device.elapsedTime());
        // printf("Copy dist from device to host : %f ms \n", timer_device_to_host.elapsedTime()); 
    }
}

void printLoopInfoV2(vector<LoopInfo> info) {
    for (int i = 0; i < info.size(); i++) {
        LoopInfo loopInfo = info[i];
        printf("No. itr: %d , updated CPU data size ratio: %f\n", loopInfo.numIteration, loopInfo.splitRatio);
        printf("CPU PART TIME: %f\n", loopInfo.time_cpu);
        printf("GPU PART TIME: %f\n", loopInfo.time_gpu);
        printf("CPU Process Msg TIME: %f\n", loopInfo.time_cpu_message);
        printf("GPU Process Msg TIME: %f\n", loopInfo.time_gpu_message);
    }
}