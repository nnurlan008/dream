#ifndef RUNTIME_PREFETCHING_H
#define RUNTIME_PREFETCHING_H

#ifndef __device__
#define __device__
#endif
#ifndef __host__
#define __host__
#endif
#ifndef __forceinline__
#define __forceinline__ inline
#endif

#ifdef __cplusplus
extern "C++" {
#endif

#include <simt/atomic>
#include <stdint.h>
#include <sys/types.h>

// #include <infiniband/verbs.h>


#include <iostream>
using namespace std;
// #include <simt/atomic>

#include "../src/rdma_utils.cuh"
#include "../src/rdma_utils.h"
#include "primitives.h"





#define htonl(x)  ((((uint32_t)(x) & 0xff000000) >> 24) |\
                   (((uint32_t)(x) & 0x00ff0000) >> 8) |\
                   (((uint32_t)(x) & 0x0000ff00) << 8) |\
                   (((uint32_t)(x) & 0x000000ff) << 24))

// remote address:
extern uint64_t remote_address_2nic[2];

// offset for buffers; in bytes
static size_t Address_Offset[2]; 
__device__ volatile uint64_t GPU_address_offset; 
__device__ uint64_t Global_GPU_address;
__device__ uint64_t d_remote_address[2];
__device__ unsigned long long int transfer_time;

// for eviction: 
// introduce two cursors; R: request, E: evict
__device__ size_t R_cursor, E_cursor, num_pages;

uint64_t GPU_address[2];
uint64_t GPU_addr_offset[2];
uint64_t allowed_size;

// __device__ uint64_t Global_Dev_address;
// device - info about QPs
__device__ struct post_content gpost_cont;
extern __device__ struct post_content rdma_utils_content;
__device__ struct batch gbatch;
__device__ struct post_content2 gpost_cont2;

__device__ struct server_content_2nic server_content;
__device__ struct poll_content gpoll_cont;
__device__ uint32_t gpu_mrs_lkey[2];
__device__ uint64_t gpu_qp_bufs[2];
__device__ uint64_t gpu_qp_num[2];
__device__ uint64_t gpu_cq_bufs[2];

extern __device__ uint32_t gpu_mrs_lkey_rdma_utils[2];
extern __device__ uint64_t gpu_qp_bufs_rdma_utils[2];
extern __device__ uint64_t gpu_qp_num_rdma_utils[2];
extern __device__ uint64_t gpu_cq_bufs_rdma_utils[2];

__device__ size_t g_qp_index;
__device__ size_t cq_wait[128];

__device__ int activeThreads[128]; // = 0;

// host - info about QPs
struct post_content hpost_cont;
struct post_content2 hpost_cont2;
struct poll_content hpoll_cont;
struct host_keys keys_for_host;

// global variable for holdinh main cq and qp
extern struct rdma_content main_content;

// server mode means data should be sent to server/pool
#define SERVER_MODE 0

#define KB(x) (long long int) x*1024
#define MB(x) (long long int) KB(x)*1024
#define GB(x) (long long int) MB(x)*1024

// wait for MAX_POST # of requests to be stored on qp buffer
#define MAX_POST 3 
// request size

#define REQUEST_SIZE 16*1024 // bytes
extern __device__ int GLOBAL_REQUEST_SIZE;

// define globale vaiable to save the number of post requests
// and compare them to max_post

// our data structure compared to cudaMallocManaged
void *rdmaAlloc(uint64_t global_start_address, size_t offset){
    return (void *)global_start_address + offset;
}



// typedef's:

// template <typename T>
struct rdma_tlb{
    size_t n_entries;
    uint8_t *tlb_buffer;

    // [] operator
    __device__ __host__
    uint8_t& operator[](size_t index) {
        // T *add = (T *) tlb_buffer;       
        return tlb_buffer[index];

    }
};

// Function to busy-wait for a specified number of nanoseconds
__device__ void sleep_nanoseconds(int nanoseconds) {
    clock_t start = clock();
    clock_t end = start + nanoseconds * CLOCKS_PER_SEC / 1000000000;

    while (clock() < end) {
        // Busy-wait until the specified time has elapsed
    }
}

__global__
void start_page_queue(const size_t size, const size_t page_size){
    R_cursor = 0;
    E_cursor = 0;
    // page_size = REQUEST_SIZE;
    num_pages = size/page_size;
    printf("num_pages: %llu\n", (unsigned long long) num_pages);
    printf("page_size: %llu\n", (unsigned long long) page_size);
}

void alloc_global_host_content(struct post_content post_cont, struct poll_content poll_cont, struct host_keys keys,
                                struct gpu_memory_info gpu_info){
    // copy poll and post content to global 
    printf("function: %s line: %d\n", __FILE__, __LINE__);
    hpost_cont = post_cont;
    printf("function: %s line: %d\n", __FILE__, __LINE__);
    hpoll_cont = poll_cont;
    printf("function: %s line: %d\n", __FILE__, __LINE__);
    keys_for_host = keys;
    printf("function: %s line: %d\n", __FILE__, __LINE__);
    GPU_address[0] = gpu_info.addrs[0];
    GPU_address[1] = gpu_info.addrs[1];
    GPU_addr_offset[0] = 0;
    GPU_addr_offset[1] = 0;
    Address_Offset[0] = 0;
    Address_Offset[1] = 0;
}

// __global__ void alloc_batch(){
//     for(int i = 0; i < 256; i++)
//     {
//       for(size_t k = 0; k < 64; k++)
//         gbatch.wait_queue[i*64+k] = 0;
//     }
// }

__global__ void 
alloc_global_content(struct post_content *post_cont, struct poll_content *poll_cont, struct server_content_2nic *post_cont2,
                                    struct gpu_memory_info gpu_infos){
    // copy poll and post content to global 
    gpost_cont = *post_cont;
    gpoll_cont = *poll_cont;
    // gpost_cont2 = *post_cont2;
    server_content = *post_cont2;
    d_remote_address[0] = gpu_infos.server_address[0];
    d_remote_address[1] = gpu_infos.server_address[1];
    // Global_GPU_address = post_cont->wr_sg_addr;
    // printf("alloc_global_content - Global_GPU_address: %p\n", Global_GPU_address);
    // Global_Dev_address = post_cont->wr_sg_addr;
    GPU_address_offset = 0;
    transfer_time = 0;
    gpu_mrs_lkey[0] = gpu_infos.wr_rdma_lkey[0];
    gpu_mrs_lkey[1] = gpu_infos.wr_rdma_lkey[1];

    gpu_qp_bufs[0] = gpu_infos.qp_buf_gpu[0];
    gpu_qp_bufs[1] = gpu_infos.qp_buf_gpu[1];

    gpu_cq_bufs[0] = gpu_infos.cq_buf_gpu[0];
    gpu_cq_bufs[1] = gpu_infos.cq_buf_gpu[1];

    gpu_qp_num[0] = gpu_infos.qp_num_gpu[0];
    gpu_qp_num[1] = gpu_infos.qp_num_gpu[1];

    gpu_mrs_lkey_rdma_utils[0] = gpu_infos.wr_rdma_lkey[0];
    gpu_mrs_lkey_rdma_utils[1] = gpu_infos.wr_rdma_lkey[1];

    gpu_qp_bufs_rdma_utils[0] = gpu_infos.qp_buf_gpu[0];
    gpu_qp_bufs_rdma_utils[1] = gpu_infos.qp_buf_gpu[1];

    gpu_cq_bufs_rdma_utils[0] = gpu_infos.cq_buf_gpu[0];
    gpu_cq_bufs_rdma_utils[1] = gpu_infos.cq_buf_gpu[1];

    gpu_qp_num_rdma_utils[0] = gpu_infos.qp_num_gpu[0];
    gpu_qp_num_rdma_utils[1] = gpu_infos.qp_num_gpu[1];

    printf("qp_num: %d\n", gpost_cont.qp_num);
    for(int i = 0; i < 256; i++)
    {
        gbatch.queue_lock[i] = 0;
        gbatch.global_post_number[i] = 0;
        for(size_t k = 0; k < 64; k++)
            gbatch.wait_queue[i*64+k] = 0;
    }
    g_qp_index = 0;
    for (size_t i = 0; i < 128; i++)
    {
        cq_wait[i] = 0;
        activeThreads[i] = 0;
        // QP_count.queue_count[i].store(0, simt::memory_order_relaxed);

    }
    GLOBAL_REQUEST_SIZE = REQUEST_SIZE;
    rdma_utils_content = gpost_cont;
     
}

struct tlb_entry {
    // uint64_t global_id;
    
    // data_page_t* page = nullptr;
    // TODO: implement LRU here per page
    // page size will be fixed for all pages
    // each page will have different gpu address - why?
    // for oversubscription and better eviction

    // 2: on-device, 0: on-host
    volatile uint8_t state; // state is also used as a lock
    uint lock; // 1: locked, 0: not locked
    volatile uint64_t device_address;
    uint64_t host_address;

    __forceinline__
    __host__ __device__
    tlb_entry(int state, uint64_t host_address) { init(state, host_address); }

    __forceinline__
    __host__
    void init(int i_state, uint64_t i_host_address) {
        state = i_state;
        lock = 0;
        host_address = i_host_address;
        device_address = NULL;
    }

    __forceinline__
    __device__
    uint lock_state(){
        volatile uint *entry = (volatile uint *) &lock;
        return *entry;
    }

    __forceinline__
    __device__
    uint inc_lock(){
        volatile uint *entry = (volatile uint *) &lock;
        uint locked = atomicAdd((unsigned int *)entry, 1);
        return locked;
    }

    __forceinline__
    __device__
    uint dec_lock(){
        volatile uint *entry = (volatile uint *) &lock;
        uint locked = atomicSub((unsigned int *)entry, 1);
        return locked;
    }

    __forceinline__
    __device__
    bool lock_entry(){
        volatile uint *entry = (volatile uint *) &lock;
        int locked = atomicCAS((unsigned int *)entry, 0, 1);
        return locked == 0;
    }

    __forceinline__
    __device__
    bool release_lock(){
        volatile uint *entry = (unsigned int *) &lock;
        int locked = atomicCAS((unsigned int *)entry, 1, 0);
        return locked == 1;
    }

    __forceinline__
    __device__
    bool update_dev_address(uint64_t new_address, uint64_t prev_address){
        unsigned long long int *address = (unsigned long long int *) &device_address;
        unsigned long long int check = atomicCAS(address, (unsigned long long int) prev_address, (unsigned long long int) new_address);
        return check == (unsigned long long int) prev_address;
    }

    // __forceinline__
    // __host__ __device__
    // bool update_host_address(uint64_t new_address, uint64_t prev_address){
    //     unsigned long long int *address = (unsigned long long int *) &host_address;
    //     unsigned long long int check = atomicCAS(address, (unsigned long long int) prev_address, (unsigned long long int) new_address);
    //     return check == (unsigned long long int) prev_address;
    // }
};

__global__ void memcpyDtoH_global(tlb_entry *d_TLB, size_t size, size_t tlb_size, int request_size, int sizeof_T){
    size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    size_t id = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;
    
        
    // for(int i = thid; i < tlb_size; i += stride)
    // {
        // if(i < tlb_size){
        
                uint64_t che;                
                // int buf_index = id/16384;
                struct ibv_wc wc;
                che = floor((double)index/request_size); //getTLBindex(index, request_size);
                // select which qp and cq to use:
                // volatile uint *entry = &tlb_buffer[che];
                // printf("sadasdrequest_size: %d che: %d checkTLB(0, tlb_buffer): %d\n", request_size, che, checkTLB(0, tlb_buffer));
                // TODO: for oversubscription, first check if gpu has enough free memory
                // __syncthreads();
                    
                    // lock entry:
                    // volatile uint8_t *state = &d_TLB[che].state;
                    // volatile uint *lock = &d_TLB[che].lock;
                if(index < size){    
                    if(d_TLB[che].state == 2 || d_TLB[che].state == 4){ // page completely on cpu or dirty on cpu
                        if(d_TLB[che].lock_entry()){
                            int qp_index = che & 127;
                            // unsigned long long int data_size = 256*sizeof(int);
                            unsigned long long int data_size;
                            if(che == tlb_size -1) data_size = size - che*request_size*sizeof(sizeof_T);
                            else data_size = request_size*sizeof(sizeof_T);
                            void *cq_dbrec = (void *) gpoll_cont.cq_dbrec[qp_index];
                            
                            bool isSet = false;
                            volatile uint *cq_lock = &gpost_cont.cq_lock[qp_index];
                            // printf("id: %d cq_lock: %d qp_index: %d\n", id, cq_lock, qp_index);

                            

                            while(atomicCAS((unsigned int *)cq_lock, 0, 1) != 0);
                                volatile size_t *p_index = &gpost_cont.n_post[qp_index];
                                int cur_post =  atomicAdd((unsigned long long int *)p_index, (unsigned long long int ) 1);
                                void *cqe = gpoll_cont.cq_buf + 2*4096*qp_index + (cur_post & 63) * 64;
                                struct mlx5_cqe64 *cqe64 = (struct mlx5_cqe64 *) cqe;
                                volatile uint8_t *op_flag = &cqe64->op_own;
                                // printf("cqe64->op_own: %d\n", cqe64->op_own);
                                // printf("*op_flag: %d\n", *op_flag);
                        //         // post_m(host_address + che*data_size, gpost_cont.wr_rdma_rkey, data_size, gpost_cont.wr_sg_lkey, gpu_address + che*data_size, 4, gpost_cont.qp_num + qp_index, 
                        //         //        cur_post, gpost_cont.qp_buf + 8192*qp_index, (void *) gpost_cont.bf_reg[qp_index], gpost_cont.qp_db[qp_index], gpost_cont.dev_qp_sq[qp_index], id);
                                // printf("GPU_address_offset: %p data size: %d\n", 0, 0);
                                unsigned long long int offset = atomicAdd((unsigned long long int *)&GPU_address_offset, (unsigned long long int) data_size);
                                // if(d_TLB[che].device_address == NULL)
                                // d_TLB[che].device_address = Global_GPU_address + offset; // update_gpu_offset( (unsigned long long int) data_size);
                                int rkey_index = (d_TLB[che].host_address - gpost_cont.wr_rdma_remote_addr)/(8*1024*1024*1024llu);
                                // if(gpost_cont2.addrs[rkey_index])
                                // printf("rkey: gpost_cont2.wr_rdma_rkey[rkey_index]: %d\n",\
                                //      gpost_cont2.wr_rdma_rkey[rkey_index]);
                                // printf("rkey_index: %d\n",\
                                //       rkey_index);
                                // printf("d_TLB[che].host_address: %p gpost_cont.wr_sg_lkey: %d, gpost_cont.qp_num: %d, qp_index: %d, cur_post: %d\n",\
                                //      d_TLB[che].host_address, gpost_cont.wr_sg_lkey, gpost_cont.qp_num, qp_index, cur_post);

                                // printf("gpost_cont.dev_qp_sq[qp_index]: %p, gpost_cont.qp_db[qp_index]: %p, gpost_cont.bf_reg[qp_index]: %p, gpost_cont.qp_buf: %p\n",\
                                //      gpost_cont.dev_qp_sq[qp_index], gpost_cont.qp_db[qp_index], gpost_cont.bf_reg[qp_index], gpost_cont.qp_buf);

                                // printf("gpoll_cont.cq_buf: %p, gpoll_cont.cq_dbrec[qp_index]: %p, che: %d data_size: %llu\n",\
                                //      gpoll_cont.cq_buf, gpoll_cont.cq_dbrec[qp_index], che, data_size);

                                

                                // post_m(d_TLB[che].host_address, gpost_cont2.wr_rdma_rkey[rkey_index], data_size, gpost_cont.wr_sg_lkey, d_TLB[che].device_address, 4, gpost_cont.qp_num + qp_index, 
                                //         cur_post, gpost_cont.qp_buf + 8192*qp_index, (void *) gpost_cont.bf_reg[qp_index], gpost_cont.qp_db[qp_index], gpost_cont.dev_qp_sq[qp_index], id);

                                post_write(d_TLB[che].host_address, gpost_cont2.wr_rdma_rkey[rkey_index], data_size, gpost_cont.wr_sg_lkey, d_TLB[che].device_address, 0, gpost_cont.qp_num + qp_index, 
                                        cur_post, gpost_cont.qp_buf + 8192*qp_index, (void *) gpost_cont.bf_reg[qp_index], gpost_cont.qp_db[qp_index], gpost_cont.dev_qp_sq[qp_index], id);
                                
                                
                                while(/*cqe64->op_own == 240*/*op_flag == 240){
                                    
                                    // printf("id: %d gpoll_cont.cq_buf: %p qp_index: %d d_TLB[che].device_address: %p\n", id, gpoll_cont.cq_buf, qp_index, d_TLB[che].device_address);
                                    // printf("*op_flag: %d cqe64->op_own: %d\n", *op_flag, cqe64->op_own);
                                    // if(*tlb_sync_host == 1) {
                                    //     tlb_sync_host = 0;
                                    // }
                                }
                                // printf("*op_flag: %d cqe64->op_own: %d\n", *op_flag, cqe64->op_own);
                                // printf("id: %d cqe64->op_own: %d qp_index: %d d_TLB[che].device_address: %p\n",\
                                //      id, cqe64->op_own, qp_index, d_TLB[che].device_address);
                                *op_flag = 240;
                                *(uint32_t *) cq_dbrec = (uint32_t) htonl((cur_post+1) & 0xffffff);
                                // if(che == 1024*1024*8){
                                //     T *tmp_array1 =  (T *)d_TLB[che].device_address;
                                //     printf("d_TLB[che].host_address: %p tmp_array1[0]: %d\n", d_TLB[che].host_address, tmp_array1[0]);
                                // }
                                // printf("index: %llu id: %d tmp_p[index%request_size]: %d index%request_size: %d d_TLB[che].device_address: %p Global_GPU_address + offset: %p\n",\
                                //         index, id, tmp_p[index%1024], index%1024, d_TLB[che].device_address, Global_GPU_address + offset);
                                d_TLB[che].state = 1;
                            
                            *cq_lock = 0;
                            d_TLB[che].release_lock();
                        }
                        
                        while(d_TLB[che].state != 1);
                    }
                }
        
}

// __global__ void memcpyDtoH_global(tlb_entry *tlb, size_t tlb_size){
//     int i = blockIdx.x * blockDim.x + threadIdx.x;
//     int stride = blockDim.x * gridDim.x;
    
        
//     // for(int i = thid; i < tlb_size; i += stride)
//     // {
//         if(i < tlb_size){
//             if(tlb[i].lock_entry()){
//                 if(tlb[i].device_address != NULL && (tlb[i].state == 2 || tlb[i].state == 4)){
//                     int qp_index = i & 255;
//                     unsigned long long int data_size = 1024; // request_size*sizeof(T);
//                     void *cq_dbrec = (void *) gpoll_cont.cq_dbrec[qp_index];
                    
//                     bool isSet = false;
//                     volatile uint *cq_lock = &gpost_cont.cq_lock[qp_index];
//                     // printf("id: %d cq_lock: %d qp_index: %d\n", id, cq_lock, qp_index);
//                     while(atomicCAS((unsigned int *)cq_lock, 0, 1) != 0);
//                         volatile size_t *p_index = &gpost_cont.n_post[qp_index];
//                         int cur_post =  atomicAdd((unsigned long long int *)p_index, (unsigned long long int ) 1);
//                         void *cqe = gpoll_cont.cq_buf + 2*4096*qp_index + (cur_post & 63) * 64;
//                         struct mlx5_cqe64 *cqe64 = (struct mlx5_cqe64 *) cqe;
//                 //         // post_m(host_address + che*data_size, gpost_cont.wr_rdma_rkey, data_size, gpost_cont.wr_sg_lkey, gpu_address + che*data_size, 4, gpost_cont.qp_num + qp_index, 
//                 //         //        cur_post, gpost_cont.qp_buf + 8192*qp_index, (void *) gpost_cont.bf_reg[qp_index], gpost_cont.qp_db[qp_index], gpost_cont.dev_qp_sq[qp_index], id);
//                         // printf("GPU_address_offset: %p data size: %d\n", 0, 0);
//                         // unsigned long long int offset = atomicAdd((unsigned long long int *)&GPU_address_offset, (unsigned long long int) data_size);
//                         // if(d_TLB[che].device_address == NULL)
//                         // tlb[thid].device_address = Global_GPU_address + offset; // update_gpu_offset( (unsigned long long int) data_size);


//                         int rkey_index = (tlb[i].host_address - gpost_cont.wr_rdma_remote_addr)/(8*1024*1024*1024llu);

//                         printf("rkey: gpost_cont2.wr_rdma_rkey[rkey_index]: %d\n",\
//                             gpost_cont2.wr_rdma_rkey[rkey_index]);
//                         printf("rkey_index: %d tlb[%d].device_address: %p\n",\
//                             rkey_index, i, tlb[i].device_address);
//                         printf("d_TLB[che].host_address: %p gpost_cont.wr_sg_lkey: %d, gpost_cont.qp_num: %d, qp_index: %d, cur_post: %d\n",\
//                             tlb[i].host_address, gpost_cont.wr_sg_lkey, gpost_cont.qp_num, qp_index, cur_post);

//                         printf("qp_index: %d gpost_cont.dev_qp_sq[qp_index]: %p, gpost_cont.qp_db[qp_index]: %p, gpost_cont.bf_reg[qp_index]: %p, gpost_cont.qp_buf: %p\n",\
//                             qp_index, gpost_cont.dev_qp_sq[qp_index], gpost_cont.qp_db[qp_index], gpost_cont.bf_reg[qp_index], gpost_cont.qp_buf);

//                         printf("gpoll_cont.cq_buf: %p, gpoll_cont.cq_dbrec[qp_index]: %p, che: %d data_size: %llu\n",\
//                             gpoll_cont.cq_buf, gpoll_cont.cq_dbrec[qp_index], i, data_size);

                        
//                         post_m(tlb[i].host_address, gpost_cont2.wr_rdma_rkey[rkey_index], data_size, gpost_cont.wr_sg_lkey, tlb[i].device_address, 4, gpost_cont.qp_num + qp_index, 
//                                 cur_post, gpost_cont.qp_buf + 8192*qp_index, (void *) gpost_cont.bf_reg[qp_index], gpost_cont.qp_db[qp_index], gpost_cont.dev_qp_sq[qp_index], 0);
//                         // int lkey_index = (tlb[i].host_address - gpost_cont.wr_rdma_remote_addr)/(8*1024*1024*1024llu);
//                         // post_m(tlb[i].device_address, gpost_cont.wr_rdma_rkey, data_size, gpost_cont2.wr_rdma_lkey[lkey_index], tlb[i].host_address, 4, gpost_cont.qp_num + qp_index, 
//                         //         cur_post, gpost_cont.qp_buf + 8192*qp_index, (void *) gpost_cont.bf_reg[qp_index], gpost_cont.qp_db[qp_index], gpost_cont.dev_qp_sq[qp_index], 0);

//                         volatile uint8_t *op_flag = &cqe64->op_own;
//                         while(/*cqe64->op_own == 240*/*op_flag == 240){
//                             // printf("id: %d cqe64->op_own: %d qp_index: %d d_TLB[che].device_address: %p\n", i, cqe64->op_own, qp_index, tlb[i].device_address);
//                             // if(*tlb_sync_host == 1) {
//                             //     tlb_sync_host = 0;
//                             // }
//                         }
//                         printf("id: %d cqe64->op_own: %d qp_index: %d d_TLB[che].device_address: %p\n", i, cqe64->op_own, qp_index, tlb[i].device_address);
//                         *op_flag = 240;
//                         *(uint32_t *) cq_dbrec = (uint32_t) htonl((cur_post +1) & 0xffffff);
//                         // if(che == 1024*1024*8){
//                         //     T *tmp_array1 =  (T *)d_TLB[che].device_address;
//                         //     printf("d_TLB[che].host_address: %p tmp_array1[0]: %d\n", d_TLB[che].host_address, tmp_array1[0]);
//                         // }
//                         // printf("index: %llu id: %d tmp_p[index%request_size]: %d index%request_size: %d d_TLB[che].device_address: %p Global_GPU_address + offset: %p\n",\
//                         //         index, id, tmp_p[index%1024], index%1024, d_TLB[che].device_address, Global_GPU_address + offset);
//                         tlb[i].state = 0; // completely on host

//                     atomicCAS((unsigned int *)cq_lock, 1, 0);
//                     // *cq_lock = 0;
//                 }
//                 tlb[i].release_lock();
//             }
//         }
//         __syncwarp();
//         __syncthreads();
//         // else{
//         //     printf("D->H transfer problem on index: %d!\n", thid);
//         // }
//     // }
// }




// data structure for device side
// TODO: change the name gpu_buf_d to sth meaningfull
template<typename T>
struct gpu_buf_d{
    
    T *gpu_address, *host_address;
    size_t size;
    int request_size;
    unsigned int *tlb_buffer;
    size_t tlb_size;
    // TODO:
    // gpu address should be dynamically assigned
    // based on the free mem on device
    // but for now it will be statically allocated

    __forceinline__
    __device__
    gpu_buf_d(uint64_t h_address, size_t user_size){
        // printf(" sREQUEST_SIZE: %d offset: %d\n", REQUEST_SIZE, Address_Offset);
        // TODO: change static allocation of gpu address to dynamical version when there is a need 
        // for free dev memory -> evict some memory
        // gpu_address = (T *) (Global_Dev_address + Address_Offset);
        host_address = (T *) h_address;
        size = user_size;
        request_size = 65536; // bytes
    }

};

int memcpyServerToHost_global(){
            int i = 0;
            while(i < 5)
            {
               
                i++;
                printf("i: %d\n", i);
            }
    return 0;
}

const int Q_SIZE = 128;
// Kernel function to initialize the queue_count array
__global__ void initQueueCount(simt::atomic<size_t, simt::thread_scope_device> *queue_count, 
                               simt::atomic<size_t, simt::thread_scope_device> *qp_filled) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < Q_SIZE) {
        // Initialize each atomic variable to 0
        queue_count[idx].store(0, simt::memory_order_relaxed);
        qp_filled[idx].store(0, simt::memory_order_relaxed);
    }
}

template<typename T>
struct rdma_buf {
    // private:
        // uint64_t wr_rdma_remote_addr, uint32_t wr_rdma_rkey,            
        //               uint32_t wr_sg_length, uint32_t wr_sg_lkey, uint64_t wr_sg_addr,
        //               int wr_opcode, uint32_t qp_num, int cur_post, void *qp_buf, void *bf_reg, unsigned int *qp_db, void *dev_qp_sq, int id

    // public:
        int gpu;
        uint64_t gpu_address;
        uint64_t host_address;
        size_t size;
        
        uint32_t request_size;
        uint64_t dev_addr;
        T *dev_buffer;
        T *host_buffer;  // in server mode this will be server address
        T *local_buffer; // in server mode this will be host buffer
        size_t tlb_size;
        tlb_entry *host_TLB, *d_TLB;
        unsigned int *h_tlb, *d_tlb;

        // server mode only:
        struct ibv_mr *buffer_mr;
        // int *page_number;
        // int *h_page_number;
        // uint *page_lock;
        // int *page_map;
        // int *h_page_map;

        size_t numPages;


        simt::atomic<size_t, simt::thread_scope_device> *queue_count;
        simt::atomic<size_t, simt::thread_scope_device> *qp_filled; // 0: empty

    // Constructor
        
        // constructor
        // __device__ __host__
        // rdma_buf(size_t user_size){
        //     #ifdef  __CUDA_ARCH__
        //         printf("Devcie declaration needs assignment!\n");
        //     #else
        //         size = user_size;
        //         request_size = REQUEST_SIZE/sizeof(T);
        //     // printf(" sREQUEST_SIZE: %d offset: %d\n", REQUEST_SIZE, Address_Offset);
        //     // uint64_t offset = (uint64_t) Address_Offset;
        //     // // gpu_address = user_address + Address_Offset;
        //     // host_address = h_address + Address_Offset;
        //     // size = user_size;
        //     // request_size = REQUEST_SIZE/sizeof(T);

        //     // // allocate memory on gpu and host
        //     // alloc_memory((T *) host_address, (T *) gpu_address);
        //     // printf("gpu_address: %p\n", gpu_address);

        //     // // allocate TLB on device for now
        //     // if(alloc_tlb()){
        //     //     printf("Error on TLB Buffer allocation!\n");
        //     //     exit(-1);
        //     // }
        //     #endif

        // }

        __device__
        rdma_buf& operator=(const rdma_buf& obj) {
            this->d_TLB = obj.d_TLB;
            this->host_address = obj.host_address;
            this->request_size = obj.request_size;
            this->dev_buffer = obj.dev_buffer;
            this->dev_addr = obj.dev_addr;
            this->size = obj.size;
            this->tlb_size = obj.size;
            this->d_tlb = obj.d_tlb;
        
            return *this;
        }

        void reset(){
            if(update_device_tlb() == -1){
                printf("error on reset, file: %s, line: %d\n", __FILE__, __LINE__);
                exit(-1);
            }
        }

        // constructor for pointer declaration:
        void start(size_t user_size, int gpu_number, void *shared_host_address){
            gpu = gpu_number;
            uint64_t offset = (uint64_t) Address_Offset[gpu];
            // gpu_address = user_address + Address_Offset;
            if(shared_host_address == NULL){
                host_address = remote_address_2nic[gpu] + Address_Offset[gpu];
                Address_Offset[gpu] = Address_Offset[gpu] + size;
            }
            else{
                host_address = (uint64_t) shared_host_address;
            }

            size = user_size;
            request_size = REQUEST_SIZE/sizeof(T);
            dev_addr = GPU_address[gpu] + GPU_addr_offset[gpu];
            dev_buffer = (T *) dev_addr;
             
            printf("request_size: %d sizeof(T): %d\n", request_size, sizeof(T));
             
            // allocate memory on gpu and host
            alloc_memory((T *) host_address, (T *) gpu_address);
            printf("remote_address_2nic[0]: %p host_address: %p Address_Offset: %p\n", remote_address_2nic[0], host_address, Address_Offset);
            printf("remote_address_2nic[1]: %p host_address: %p Address_Offset: %p\n", remote_address_2nic[1], host_address, Address_Offset);
            printf("dev_address: %p gpu: %d\n", dev_addr, gpu);

            // allocate TLB on device for now
            if(alloc_tlb()){
                printf("Error on TLB Buffer allocation!\n");
                exit(-1);
            }
            printf("RDMA buf started.\n");
        }

        // allocate tlb data structure on device so that device can access
        
        int alloc_memory(T *remote_address, T* gpu_address){
            // printf("gpu_address: %p\n", gpu_address);
            host_buffer = (T *) host_address;
            
            // device_buffer = (T *) gpu_address;
            
            if(SERVER_MODE){
                local_buffer = (T *) malloc(size);
                if (local_buffer == NULL){
                    printf("Error on Local Buffer allocation!\n");
                    exit(-1);
                }

                buffer_mr = ibv_reg_mr( main_content.pd, local_buffer, size,
                                        IBV_ACCESS_LOCAL_WRITE);
                if(!buffer_mr){
                    printf("Error on Memory region allocation!\n");
                    exit(-1);
                }

            }
            else{
                local_buffer = host_buffer;
            }
            // check if the offset does not exceed allowed memory
            return 0; // for success
        }
        
        
        int alloc_tlb(){
            int req_size = REQUEST_SIZE;
            tlb_size = ceil((double)size/req_size);
            
            // if(cudaSuccess != cudaMalloc(&tlb_buffer, tlb_size*sizeof(unsigned int)))
            //     return -1;
            cudaSetDevice(gpu);
            if(cudaSuccess != cudaMalloc(&d_TLB, tlb_size*sizeof(tlb_entry)))
                return -1;
            cudaSetDevice(gpu);
            if(cudaSuccess != cudaMalloc(&d_tlb, tlb_size*sizeof(unsigned int)))
                return -1;

            printf("tlb_size: %llu sizeof(tlb_entry): %d\n", tlb_size, sizeof(uint8_t));
            
            host_TLB = (tlb_entry *) malloc(tlb_size*sizeof(tlb_entry));
            h_tlb = (unsigned int *) malloc(tlb_size*sizeof(unsigned int));
            // h_page_number = (int *) malloc(tlb_size*sizeof(int));
            printf("line: %d\n", __LINE__);
            for(size_t i = 0; i < tlb_size; i++){
                // printf("host_address: %p\n", host_address);
                // printf("REQUEST_SIZE: %d\n", REQUEST_SIZE);
                // printf("host_address + %llu*REQUEST_SIZE: %p\n", i, host_address + i*REQUEST_SIZE);
                host_TLB[i].init(0, host_address + i*REQUEST_SIZE);
                h_tlb[i] = 0;
                host_TLB[i].device_address = GPU_address[gpu] + GPU_addr_offset[gpu];
                GPU_addr_offset[gpu] += REQUEST_SIZE;
                // h_page_number[i] = -1;
                // printf("host_TLB[%d].device_address: %p\n", i, host_TLB[i].device_address);
                if(i == tlb_size-1) printf("host_address + %llu*REQUEST_SIZE: %p\n", i, host_address + i*REQUEST_SIZE);
            }
            printf("line: %d\n", __LINE__);
            // h_page_map = (int *) malloc(numPages*sizeof(int));
            // for (size_t i = 0; i < numPages; i++)
            // {
            //     h_page_map[i] = -1;
            // }
            
            printf("line: %d\n", __LINE__);
            
            if(update_device_tlb() == -1) return -1;
            // free(h_page_number);
            // free(h_page_map);
            // printf("tlb_buffer: %p\n", tlb_buffer);
            return 0;
        }

        __forceinline__
        __host__
        int update_device_tlb(){
            if(cudaSuccess != cudaDeviceSynchronize()) 
                return -1;
            printf("line: %d\n", __LINE__);
            cudaSetDevice(gpu);
            if(cudaSuccess != cudaMemcpy(d_TLB, host_TLB, tlb_size*sizeof(tlb_entry), cudaMemcpyHostToDevice))
                return -1;
            printf("line: %d\n", __LINE__);
            cudaSetDevice(gpu);
            if(cudaSuccess != cudaMemcpy(d_tlb, h_tlb, tlb_size*sizeof(unsigned int), cudaMemcpyHostToDevice))
                return -1;
            printf("line: %d\n", __LINE__);
            // if(cudaSuccess != cudaMemcpy(page_number, h_page_number, tlb_size*sizeof(int), cudaMemcpyHostToDevice))
            //     return -1;
            // printf("line: %d\n", __LINE__);
            // if(cudaSuccess != cudaMemcpy(page_map, h_page_map, numPages*sizeof(int), cudaMemcpyHostToDevice))
            //     return -1;
            // printf("line: %d\n", __LINE__);
            cudaSetDevice(gpu);
            if(cudaSuccess != cudaDeviceSynchronize()) return -1;
                return 0;
            printf("line: %d\n", __LINE__);
        }

        // use the below function online in local mode 
        __forceinline__
        __host__
        int memcpyDtoH(void){
            size_t threads = 1024;
            size_t n_blks = size/threads + 1; ; // tlb_size/threads + 1;
            if(cudaSuccess != cudaDeviceSynchronize()) return -1;
            memcpyDtoH_global<<< n_blks, threads>>>(d_TLB, size, tlb_size, request_size, sizeof(T));
            if(cudaSuccess != cudaDeviceSynchronize()) return -1;
            return 0;
        }

        // // use the below 2 functions online in local mode
        // __forceinline__
        // __host__
        void memcpyHostToServer(void){
            // use rdma to send data from host to server
            // I will use main qp and cq

            struct ibv_wc wc;
            struct ibv_send_wr wr, *bad_wr = NULL;
            struct ibv_sge sge;
            size_t transfer_size = 1*1024*1024*1024llu;
            size_t total_size = size;
            size_t already_sent = 0;
            double n_region = (double) ((size)/(transfer_size));
            int n_regions = (int) n_region;
            n_regions++;
            // printf("debug - line: %d function: %s transfer_size: %llu n_regions: %d local_buffer[0]: %f\n", 
            //         __LINE__, __func__, transfer_size, n_regions, local_buffer[0]);
            // for (int i = 0; i < n_regions; i++)
            int i = 0;
            while(i < n_regions)
            {
                if (i < n_regions == 0) break;
                if(total_size < transfer_size) transfer_size = total_size;

                int rkey_index = (host_address - keys_for_host.addrs[0])/(8*1024*1024*1024llu);
                if(host_address + already_sent + transfer_size <= keys_for_host.addrs[rkey_index + 1]){
                    memset(&wr, 0, sizeof(wr));
                    // wr.wr_id = (uintptr_t)conn;
                    wr.opcode = IBV_WR_RDMA_WRITE;
                    wr.sg_list = &sge;
                    wr.num_sge = 1;
                    wr.send_flags = IBV_SEND_SIGNALED;
                    wr.wr.rdma.remote_addr = host_address + already_sent; // + i*Region_Size; // + 8*1024*1024*1024llu; 
                    wr.wr.rdma.rkey = keys_for_host.rkeys[rkey_index]; // s_ctx->server_mr.rkey;
                    sge.addr = (uintptr_t) local_buffer + already_sent; // + i*Region_Size; // (uintptr_t)srv_buffer;
                    sge.length = transfer_size;
                    sge.lkey = buffer_mr->lkey;
                    int ret = ibv_post_send(main_content.qp, &wr, &bad_wr);
                    do{
                        ret = ibv_poll_cq(main_content.cq, 1, &wc);
                    }while(ret == 0);
                    already_sent += transfer_size;
                }
                else{
                    memset(&wr, 0, sizeof(wr));
                    // wr.wr_id = (uintptr_t)conn;
                    wr.opcode = IBV_WR_RDMA_READ;
                    wr.sg_list = &sge;
                    wr.num_sge = 1;
                    wr.send_flags = IBV_SEND_SIGNALED;
                    wr.wr.rdma.remote_addr = host_address + already_sent; // + i*Region_Size; // + 8*1024*1024*1024llu; 
                    wr.wr.rdma.rkey = keys_for_host.rkeys[rkey_index]; // s_ctx->server_mr.rkey;
                    sge.addr = (uintptr_t) local_buffer + already_sent; // + i*Region_Size; // (uintptr_t)srv_buffer;
                    sge.length = keys_for_host.addrs[rkey_index+1] - (host_address + already_sent);
                    sge.lkey = buffer_mr->lkey;
                    int ret = ibv_post_send(main_content.qp, &wr, &bad_wr);
                    do{
                        ret = ibv_poll_cq(main_content.cq, 1, &wc);
                    }while(ret == 0);

                    already_sent += keys_for_host.addrs[rkey_index+1] - (host_address - + already_sent);

                    memset(&wr, 0, sizeof(wr));
                    // wr.wr_id = (uintptr_t)conn;
                    wr.opcode = IBV_WR_RDMA_READ;
                    wr.sg_list = &sge;
                    wr.num_sge = 1;
                    wr.send_flags = IBV_SEND_SIGNALED;
                    wr.wr.rdma.remote_addr = host_address + already_sent; // + i*Region_Size; // + 8*1024*1024*1024llu; 
                    wr.wr.rdma.rkey = keys_for_host.rkeys[rkey_index+1]; // s_ctx->server_mr.rkey;
                    sge.addr = (uintptr_t) local_buffer + already_sent; // + i*Region_Size; // (uintptr_t)srv_buffer;
                    sge.length = transfer_size - (keys_for_host.rkeys[rkey_index+1] - host_address);
                    sge.lkey = buffer_mr->lkey;
                    ret = ibv_post_send(main_content.qp, &wr, &bad_wr);
                    do{
                        ret = ibv_poll_cq(main_content.cq, 1, &wc);
                    }while(ret == 0);
                    already_sent += transfer_size - (keys_for_host.rkeys[rkey_index+1] - host_address);
                }

                total_size = total_size - transfer_size;
                i++;
                // break;
                // printf("debug - host_address: %p keys_for_host.rkeys[i]: %u sge.length: %u total_size: %llu i<n_regions: %d\n",
                // host_address, keys_for_host.rkeys[rkey_index], sge.length, total_size, i<n_regions);
                // printf("debug - line: %d function: %s i: %zu n_regions: %zu local_buffer[0]: %f\n", __LINE__, __func__, i, n_regions, local_buffer[0]);
                
            }
            
        }

        // __forceinline__
        // __host__
        void memcpyServerToHost(void){
            // memcpyServerToHost_global(size, local_buffer, host_address, buffer_mr);
            // // use rdma to send data from server to host
            // // I will use main qp and cq

            struct ibv_wc wc;
            struct ibv_send_wr wr, *bad_wr = NULL;
            struct ibv_sge sge;
            size_t transfer_size = 1*1024*1024*1024llu;
            size_t total_size = size;
            size_t already_sent = 0;
            double n_region = (double) ((size)/(transfer_size));
            int n_regions = (int) n_region;
            n_regions++;
            // printf("debug - line: %d function: %s transfer_size: %llu n_regions: %d local_buffer[0]: %f\n", 
            //         __LINE__, __func__, transfer_size, n_regions, local_buffer[0]);
            // for (int i = 0; i < n_regions; i++)
            int i = 0;
            while(i < n_regions)
            {
                if (i < n_regions == 0) break;
                if(total_size < transfer_size) transfer_size = total_size;

                int rkey_index = (host_address - keys_for_host.addrs[0])/(8*1024*1024*1024llu);
                if(host_address + already_sent + transfer_size <= keys_for_host.addrs[rkey_index + 1]){
                    memset(&wr, 0, sizeof(wr));
                    // wr.wr_id = (uintptr_t)conn;
                    wr.opcode = IBV_WR_RDMA_READ;
                    wr.sg_list = &sge;
                    wr.num_sge = 1;
                    wr.send_flags = IBV_SEND_SIGNALED;
                    wr.wr.rdma.remote_addr = host_address + already_sent; // + i*Region_Size; // + 8*1024*1024*1024llu; 
                    wr.wr.rdma.rkey = keys_for_host.rkeys[rkey_index]; // s_ctx->server_mr.rkey;
                    sge.addr = (uintptr_t) local_buffer + already_sent; // + i*Region_Size; // (uintptr_t)srv_buffer;
                    sge.length = transfer_size;
                    sge.lkey = buffer_mr->lkey;
                    int ret = ibv_post_send(main_content.qp, &wr, &bad_wr);
                    do{
                        ret = ibv_poll_cq(main_content.cq, 1, &wc);
                    }while(ret == 0);
                    already_sent += transfer_size;
                }
                else{
                    memset(&wr, 0, sizeof(wr));
                    // wr.wr_id = (uintptr_t)conn;
                    wr.opcode = IBV_WR_RDMA_READ;
                    wr.sg_list = &sge;
                    wr.num_sge = 1;
                    wr.send_flags = IBV_SEND_SIGNALED;
                    wr.wr.rdma.remote_addr = host_address + already_sent; // + i*Region_Size; // + 8*1024*1024*1024llu; 
                    wr.wr.rdma.rkey = keys_for_host.rkeys[rkey_index]; // s_ctx->server_mr.rkey;
                    sge.addr = (uintptr_t) local_buffer + already_sent; // + i*Region_Size; // (uintptr_t)srv_buffer;
                    sge.length = keys_for_host.addrs[rkey_index+1] - (host_address + already_sent);
                    sge.lkey = buffer_mr->lkey;
                    int ret = ibv_post_send(main_content.qp, &wr, &bad_wr);
                    do{
                        ret = ibv_poll_cq(main_content.cq, 1, &wc);
                    }while(ret == 0);

                    already_sent += keys_for_host.addrs[rkey_index+1] - (host_address - + already_sent);

                    memset(&wr, 0, sizeof(wr));
                    // wr.wr_id = (uintptr_t)conn;
                    wr.opcode = IBV_WR_RDMA_READ;
                    wr.sg_list = &sge;
                    wr.num_sge = 1;
                    wr.send_flags = IBV_SEND_SIGNALED;
                    wr.wr.rdma.remote_addr = host_address + already_sent; // + i*Region_Size; // + 8*1024*1024*1024llu; 
                    wr.wr.rdma.rkey = keys_for_host.rkeys[rkey_index+1]; // s_ctx->server_mr.rkey;
                    sge.addr = (uintptr_t) local_buffer + already_sent; // + i*Region_Size; // (uintptr_t)srv_buffer;
                    sge.length = transfer_size - (keys_for_host.rkeys[rkey_index+1] - host_address);
                    sge.lkey = buffer_mr->lkey;
                    ret = ibv_post_send(main_content.qp, &wr, &bad_wr);
                    do{
                        ret = ibv_poll_cq(main_content.cq, 1, &wc);
                    }while(ret == 0);
                    already_sent += transfer_size - (keys_for_host.rkeys[rkey_index+1] - host_address);
                }

                total_size = total_size - transfer_size;
                i++;
                // break;
                // printf("debug - host_address: %p keys_for_host.rkeys[i]: %u sge.length: %u total_size: %llu i<n_regions: %d\n",
                // host_address, keys_for_host.rkeys[rkey_index], sge.length, total_size, i<n_regions);
                // printf("debug - line: %d function: %s i: %zu n_regions: %zu local_buffer[0]: %f\n", __LINE__, __func__, i, n_regions, local_buffer[0]);
                
            }

        }

        __forceinline__
        __device__
        void request_queue(uint64_t che){

        }

    __forceinline__
    __device__
    void read_batch(uint64_t che){

        int qp_index = get_smid()%24; // che%8; // (); // warp_id() % 256; // ;
        unsigned long long int data_size = request_size*sizeof(T);
        void *cq_dbrec = (void *) gpoll_cont.cq_dbrec[qp_index];
        
        volatile int *activeThreads_ptr = &activeThreads[qp_index];
        volatile size_t *global_qp_number = &gpost_cont.queue_count[qp_index];

        volatile uint *queue_count1 = &gpost_cont.queue_lock[qp_index];
        volatile uint *cq_lock = &gpost_cont.cq_lock[qp_index];
        
        int currentCount = atomicAdd((int *) activeThreads_ptr, 1);
        int retries = 1, retries2 = 1;
        // printf("qp_index: %d retries: %d currentCount: %d  global_qp_number: %d before in > 63 \n", qp_index, retries, currentCount, (int *) global_qp_number);
        while(currentCount - 8*(*global_qp_number) > 7){
        

            if(retries2 > 1000000) {
               
                retries2 = -1;
            }
            retries2++;
        }

        volatile size_t *p_index = &gpost_cont.n_post[qp_index];
        // unsigned int cur_post = currentCount;
        retries = 1;
        uint ticket = atomicAdd((unsigned int *)queue_count1, (unsigned int ) 1);
        __threadfence_system();
        void *cqe = gpoll_cont.cq_buf + 2*4096*qp_index + (ticket & 63) * 64;
        struct mlx5_cqe64 *cqe64 = (struct mlx5_cqe64 *) cqe;
        
        uint64_t rem_addr = host_address + che*request_size*sizeof(T);
        int rkey_index = (/*d_TLB[che].host_address*/ rem_addr - d_remote_address[gpu]/*gpost_cont.wr_rdma_remote_addr*/)/(8*1024*1024*1024llu);

        d_TLB[che].device_address = dev_addr + che*request_size*sizeof(T);
        
        uint64_t value_ctrl;
        post_m(/*d_TLB[che].host_address*/ rem_addr, gpost_cont2.wr_rdma_rkey[rkey_index], data_size, gpost_cont.wr_sg_lkey, d_TLB[che].device_address, 4, gpost_cont.qp_num + qp_index, 
                /*(*global_qp_number)*64+*/ticket, &value_ctrl, gpost_cont.qp_buf + 8192*qp_index, (void *) gpost_cont.bf_reg[qp_index], gpost_cont.qp_db[qp_index], gpost_cont.dev_qp_sq[qp_index], 0);

        __threadfence_system();
        
        atomicAdd((unsigned long long int *)p_index, 1);
        __threadfence_system();
        if(atomicCAS((unsigned int *)cq_lock, 0, 1) == 0){
 
            __nanosleep(10000);
            __threadfence_system();
            int n = 1;
            while((*p_index) != (*queue_count1)/*qp_filled[qp_index].load(simt::memory_order_acquire)*/){
                // __nanosleep(1000*n);
                __threadfence_system();
                n++;
            }
            // __threadfence_system();
            uint max = (*p_index)-1; // qp_filled[qp_index].load(simt::memory_order_acquire) - 1;
            update_db_spec((void *) gpost_cont.bf_reg[qp_index], gpost_cont.qp_buf + 8192*qp_index, (*p_index)-1);
            retries = 1;
            volatile uint8_t *op_flag = &cqe64->op_own;
            while(*op_flag == 240){
                    
                    if(retries > 1000000) {
                        printf("updating polling: qp_index: %d retries: %d cur_post: %d max: %d (*p_index): %d ticket: %d\n",
                         qp_index, retries, (int) currentCount, (int) max, (int) (*p_index), (int) ticket);
                        retries = -1;
                        // break;
                    }
                    retries++;
            }
            *op_flag = 240;
            __threadfence_system();
            *(uint32_t *) cq_dbrec = (uint32_t) htonl((/*(*global_qp_number)*64 + */(*p_index)-1 + 1) & 0xffffff);
            __threadfence_system();
            
            atomicCAS((unsigned int *)cq_lock, 1, 0);
        }
        else{
            retries = 1;
            __threadfence_system();
            uint max = (*p_index) - 1;
            // qp_filled[qp_index].load(simt::memory_order_acquire) - 1;
            volatile uint8_t *op_flag = &cqe64->op_own;
            while(*op_flag == 240){
                    if(retries > 9500){
                    //    read(che);
                        post_m(/*d_TLB[che].host_address*/ rem_addr, gpost_cont2.wr_rdma_rkey[rkey_index], data_size, gpost_cont.wr_sg_lkey, /*dev_addr + che*request_size*sizeof(T)*/d_TLB[che].device_address, 4, gpost_cont.qp_num + qp_index, 
                        currentCount, &value_ctrl, gpost_cont.qp_buf + 8192*qp_index, (void *) gpost_cont.bf_reg[qp_index], gpost_cont.qp_db[qp_index], gpost_cont.dev_qp_sq[qp_index], 0);
                        update_db(&value_ctrl, (void *) gpost_cont.bf_reg[qp_index]);
                        retries = 1;
                    }
                    if(retries > 1000000) {
                        printf("waiting polling: qp_index: %d retries: %d cur_post: %d p_index: %d activeThreads: %d, ticket: %d\n", 
                                qp_index, retries, (int) currentCount, (int)  max, activeThreads[qp_index], (int) ticket);
                        retries = -1;
                        // break;
                    }
                    retries++;
            }
            *op_flag = 240;
            __threadfence_system();
            retries = 1;
            while((*cq_lock) == 1){
                if(retries > 1000000) {
                        printf("stuck in cq_lock==1: qp_index: %d retries: %d cur_post: %d max: %d (*p_index): %d\n",
                         qp_index, retries, (int) currentCount, (int) max, (int) (*p_index));
                        retries = -1;
                    }
                    retries++;
            }
        }
        if((currentCount+1) % 8 == 0){
            retries = 1;
            atomicAdd((unsigned long long int *)global_qp_number, 1);
        }
    }

        __forceinline__
        __device__
        void read(uint64_t che){
            int warpId = warp_id()%2;
            int qp_index = get_smid()%84 + 84*warpId; // che%256;   // che%8; // (); // warp_id() % 256; // ;
            // warpId = qp_index/84; // 128;
            // atomicAdd((unsigned long long int *)&g_qp_index, 1)
            // unsigned int mask = __match_any_sync(__activemask(), qp_index);
            // unsigned int leader = __ffs(mask) - 1; // mask ? 31 - __clz(mask) : 0;    // select a leader


            unsigned long long int data_size = request_size*sizeof(T);
            void *cq_dbrec = (void *) gpoll_cont.cq_dbrec[qp_index];
            
            // bool isSet = false;
            volatile size_t *p_index = &gpost_cont.n_post[qp_index];
            // unsigned int cur_post = atomicAdd((unsigned long long int *)p_index, (unsigned long long int ) 1);

            volatile uint *cq_lock = &gpost_cont.cq_lock[qp_index/**64+cur_post%64*/];
            int retries = 1;
            
            uint64_t rem_addr = host_address + che*request_size*sizeof(T);
            int rkey_index = (/*d_TLB[che].host_address*/ rem_addr - d_remote_address[warpId]/*gpost_cont.wr_rdma_remote_addr*/)/(8*1024*1024*1024llu);
            
            void *qp_buf = (void *) gpu_qp_bufs[warpId];
            uint32_t lkey = gpu_mrs_lkey[warpId];

            while(atomicCAS((unsigned int *)cq_lock, 0, 1) != 0){
                
                // if(retries >= 100000){
                //     // qp_index = atomicAdd((unsigned long long int *)&g_qp_index, 1)%128;
                //     // cq_lock = &gpost_cont.cq_lock[qp_index];
                //     printf("qp_index: %d retries: %d stuck in lock cq\n", qp_index, retries);
                //     retries = -1;
                // }
                // retries++;
            }
            // printf("qp_index: %d retries: %d\n", qp_index, retries);
            p_index = &gpost_cont.queue_count[qp_index];
            // cur_post = atomicAdd((unsigned long long int *)p_index, (unsigned long long int ) 1);
            unsigned int cur_post = atomicAdd((unsigned long long int *)p_index, (unsigned long long int ) 1);

            void *cqe = /*gpoll_cont.cq_buf*/ (void *) gpu_cq_bufs[warpId] + 2*4096*(qp_index-warpId*84) + (cur_post & 63) * 64;
            struct mlx5_cqe64 *cqe64 = (struct mlx5_cqe64 *) cqe;
            
            uint64_t value_ctrl;
            atomicAdd((unsigned long long int *)&g_qp_index, (unsigned long long int) 1);
            
            // if(che <= 100){
            //     printf("rkey_index: %d nic: %d d_remote_address[nic]: %p rem_addr: %p qp_index: %d\n", rkey_index, warpId, d_remote_address[warpId], rem_addr, qp_index);
            //     printf("server_content.wr_rdma_rkey[rkey_index+nic*5]: %d lkey: %d\n", server_content.wr_rdma_rkey[rkey_index+warpId*5], lkey);
            //     printf("nic: %d gpu_cq_bufs[nic]: %p\n", warpId, gpu_cq_bufs[warpId]);
            // }
            // printf("gpost_cont.bf_reg[%d]: %p\n", (qp_index), gpost_cont.bf_reg[qp_index]);
            // post_m(/*d_TLB[che].host_address*/ rem_addr, server_content.wr_rdma_rkey[rkey_index+warpId*5], data_size, lkey /*gpost_cont.wr_sg_lkey*/, dev_addr + che*request_size*sizeof(T)/*d_TLB[che].device_address*/, 4, gpu_qp_num[warpId] + (qp_index-84*warpId), 
            //         cur_post, &value_ctrl, /*gpost_cont.qp_buf*/ qp_buf + 8192*(qp_index-84*warpId), (void *) gpost_cont.bf_reg[qp_index], gpost_cont.qp_db[qp_index], gpost_cont.dev_qp_sq[qp_index], 0);
            
            post_opt_2nic(rem_addr, server_content.wr_rdma_rkey[rkey_index+warpId*5],            
                        dev_addr + che*request_size*sizeof(T),
                        cur_post, qp_index-84*warpId, warpId);

            // printf("posted db qp_index: %d curpost&63: %d cur_post&63 == 1: %d\n", qp_index, cur_post&63, (cur_post&63) == 1);
            // if(((cur_post+1)&63)%16 == 0){
            //     // printf("ringing db qp_index: %d curpost: %d\n", qp_index, (int) cur_post);
            //     update_db_spec(gpost_cont.bf_reg[qp_index], qp_buf + 8192*(qp_index-84*warpId), cur_post);

                
            // }
            
            
            
            volatile uint8_t *op_flag = &cqe64->op_own;
            retries = 0;
            // bool stop = true;
            while(*op_flag == 240){
                    
                    // retries++;
                    // if(retries > 5){
                    //     post_m(/*d_TLB[che].host_address*/ rem_addr, server_content.wr_rdma_rkey[rkey_index+warpId*5], data_size, lkey /*gpost_cont.wr_sg_lkey*/, dev_addr + che*request_size*sizeof(T)/*d_TLB[che].device_address*/, 4, gpu_qp_num[warpId] + (qp_index-84*warpId), 
                    //         cur_post, &value_ctrl, /*gpost_cont.qp_buf*/ qp_buf + 8192*(qp_index-84*warpId), (void *) gpost_cont.bf_reg[qp_index], gpost_cont.qp_db[qp_index], gpost_cont.dev_qp_sq[qp_index], 0);
                    //     retries = -1;
                    // }
                    // if(retries>100000) {
                    //     printf("stuck in completion of single gpu and 2 nic\n");
                    //     // printf("rkey_index: %d nic: %d d_remote_address[nic]: %p rem_addr: %p qp_index: %d\n", rkey_index, warpId, d_remote_address[warpId], rem_addr, qp_index);
                    //     // printf("server_content.wr_rdma_rkey[rkey_index+nic*5]: %d lkey: %d qp_num: %d\n", server_content.wr_rdma_rkey[rkey_index+warpId*5], lkey, gpu_qp_num[warpId]);
                    //     // printf("nic: %d gpu_cq_bufs[nic]: %p\n", warpId, gpu_cq_bufs[warpId]);
                    //     retries=-1;
                    //     // stop = false;
                    // }
                    
            }
            

            // atomicAdd((unsigned long long int *)&cq_wait[qp_index], (unsigned long long int) retries);
            // __syncwarp(mask);
            // __threadfence_system();
            *op_flag = 240;
            // if(lane_id() == leader){  
            // __threadfence_system();
            *(uint32_t *) cq_dbrec = (uint32_t) htonl((cur_post + 1) & 0xffffff);
            *cq_lock = 0;  
        }

        // [] operator for lvalue the default one
        __forceinline__
        __device__ 
        T operator[](const size_t index) {
                uint64_t che;
                uint64_t id = blockDim.x * blockIdx.x + threadIdx.x; 
                che = index/request_size;
                // if(id == 0)
                //     printf("id: %llu che: %llu dev_buffer: %p index: %d\n", id, che, dev_buffer, (int) index);

                
                    if(/*d_TLB[che].state == 2*/d_tlb[che] == 2){ // page completely on gpu or dirty on gpu
                        // T *tmp_array =  (T *) d_TLB[che].device_address;
                        // // LRU is incremented
                        // // printf(" tlb: 2 ");
                        // return tmp_array[index%request_size];
                        
                        return dev_buffer[index];
                    }
                    if(/*d_TLB[che].state == 0 || d_TLB[che].state == 5*/d_tlb[che] == 0){ // page completely on cpu or dirty on cpu
                    
                        // unsigned int mask1 = __activemask(); // __match_any_sync(__activemask(), (unsigned long long)qp_index);
                        // unsigned int leader1 = __ffs(mask) - 1; // mask ? 31 - __clz(mask) : 0;    // select a leader

                        if(d_TLB[che].lock_entry()){
                            read(che);
                            atomicCAS(&d_tlb[che], 0, 2);
                            // *page = 2;
                            // if(lane_id() == leader)
                                
                            d_TLB[che].release_lock();
                        }
                        // else{
                        //     // volatile unsigned int *tlb_value = (volatile unsigned int *) &d_tlb[che];
                        //     int prefetched = 0;
                        //     int prefetch_index = che+1;
                        //     int tries = 0;
                        //     while (prefetch_index < tlb_size){
                        //         if(d_tlb[prefetch_index] == 0) {
                        //             if(d_TLB[prefetch_index].lock_entry()){
                        //                 prefetched = 1;
                        //                 break;
                        //             }
                        //         }
                        //         if(tries>5) break;
                        //         tries++;
                        //         prefetch_index++;   
                        //     }
                            
                        //     if(prefetched){
                        //         read(prefetch_index);
                        //         // printf("prefetching page: %llu\n", prefetch_index);
                                
                        //         atomicCAS(&d_tlb[prefetch_index], 0, 2);
                        //         d_TLB[prefetch_index].release_lock();
                        //     }

                        // }

                        volatile unsigned int *tlb_value = (volatile unsigned int *) &d_tlb[che];
                        while(true){
                            if(*tlb_value == 2) break;
                            // __nanosleep(100);
                        };

                       
                        
                    }
                   
                    return dev_buffer[index];
        }

        // __forceinline__
        // __device__
        // void rvalue(size_t index, T new_value){
            
        //     // #ifdef  __CUDA_ARCH__
        //         T *tmp_array;
        //         int che = floor((double)index/request_size);
        //         if(d_TLB[che].state == 2){
        //             tmp_array = (T *)d_TLB[che].device_address;
        //             int ind = index&(request_size-1);
        //             tmp_array[ind] = new_value;
        //         }
        //         else if(d_TLB[che].state == 0){
                    
        //             if(d_TLB[che].lock_entry()){
        //                 unsigned long long int data_size = request_size*sizeof(T);
        //                 unsigned long long int offset = atomicAdd((unsigned long long int *)&GPU_address_offset, (unsigned long long int) data_size);
        //                 // d_TLB[che].device_address = Global_GPU_address + offset;
                        
        //                 d_TLB[che].state = 2;
        //                 d_TLB[che].release_lock();
        //             }
        //             else{
        //                 while(d_TLB[che].state != 2);
        //             }
        //             tmp_array = (T *)d_TLB[che].device_address;
        //             int ind = index&(request_size-1);
        //             tmp_array[ind] = new_value;
                    
        //         }
        //     // return tmp_array[ind];
              
            
        //     // else
                
        //     // #endif
        // }
};






// typedef struct rdma_buf rdma_buf;



// typedef struct rdma_tlb rdma_tlb;






#ifdef __cplusplus
}
#endif

#endif