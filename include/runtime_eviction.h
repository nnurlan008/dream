#ifndef RUNTIME_EVICTION_H
#define RUNTIME_EVICTION_H

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
extern uint64_t remote_address;

// offset for buffers; in bytes
static size_t Address_Offset = 0; 
__device__ volatile uint64_t GPU_address_offset; 
__device__ uint64_t Global_GPU_address;
__device__ uint64_t d_remote_address;
__device__ unsigned long long int transfer_time;

// for eviction: 
// introduce two cursors; R: request, E: evict
__device__ size_t R_cursor, E_cursor, num_pages;
// __device__ long long int *page_map;


uint64_t GPU_address;
uint64_t GPU_addr_offset;



// __device__ uint64_t Global_Dev_address;
// device - info about QPs
__device__ struct post_content gpost_cont;
__device__ struct post_content2 gpost_cont2;
__device__ struct poll_content gpoll_cont;

// host - info about QPs
struct post_content hpost_cont;
struct post_content2 hpost_cont2;
struct poll_content hpoll_cont;
struct host_keys keys_for_host;

// global variable for holdinh main cq and qp
extern struct rdma_content main_content;

// server mode means data should be sent to server/pool
#define SERVER_MODE 1

#define KB(x) (long long int) x*1024
#define MB(x) (long long int) KB(x)*1024
#define GB(x) (long long int) MB(x)*1024

// wait for MAX_POST # of requests to be stored on qp buffer
#define MAX_POST 3 
// request size

#define REQUEST_SIZE 64*1024 // bytes

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
void start_page_queue(const size_t size,const size_t page_size){
    R_cursor = 0;
    E_cursor = 0;
    num_pages = size/page_size;
    printf("num_pages: %llu\n", (unsigned long long) num_pages);
    printf("page_size: %llu\n", (unsigned long long) page_size);

}

void alloc_global_host_content(struct post_content post_cont, struct poll_content poll_cont, struct host_keys keys){
    // copy poll and post content to global 
    printf("function: %s line: %d\n", __FILE__, __LINE__);
    hpost_cont = post_cont;
    printf("function: %s line: %d\n", __FILE__, __LINE__);
    hpoll_cont = poll_cont;
    printf("function: %s line: %d\n", __FILE__, __LINE__);
    keys_for_host = keys;
    printf("function: %s line: %d\n", __FILE__, __LINE__);
    GPU_address = post_cont.wr_sg_addr;
    GPU_addr_offset = 0;

    // printf("page_map_size: %llu\n", (unsigned long long) page_map_size);
    // if(cudaSuccess != cudaMalloc((void **) &page_map, page_map_size*sizeof(long long int)))
    // {
    //     printf("Page map could not be allocated on device!\n");
    //     exit(-1);
    // }
    // long long int *temp = (long long int *) malloc(page_map_size*sizeof(long long int));
    // for (size_t i = 0; i < page_map_size; i++)
    // {
    //     temp[i] = -1;
    // }
    
    // int ret = cudaMemcpy(page_map, temp, page_map_size*sizeof(long long int), cudaMemcpyHostToDevice);
    // if(cudaSuccess != ret)
    // {
    //     printf("Page map could not be set; ret: %d!\n", ret);
    //     exit(-1);
    // }
    // free(temp);

}

__global__ void alloc_global_content(struct post_content *post_cont, struct poll_content *poll_cont, struct post_content2 *post_cont2){
    // copy poll and post content to global 
    gpost_cont = *post_cont;
    gpoll_cont = *poll_cont;
    gpost_cont2 = *post_cont2;
    d_remote_address = gpost_cont.wr_rdma_remote_addr;
    Global_GPU_address = post_cont->wr_sg_addr;
    printf("alloc_global_content - Global_GPU_address: %p\n", Global_GPU_address);
    // Global_Dev_address = post_cont->wr_sg_addr;
    GPU_address_offset = 0;
    transfer_time = 0;
    printf("qp_num: %d\n", gpost_cont.qp_num);
}

struct page_map_entry{
    long long int state;
    uint lock; // 1: locked, 0: not locked

    __forceinline__
    __host__ __device__
    page_map_entry() { init(); }

    __forceinline__
    __host__
    void init() {
        state = -1;
        lock = 0;
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

};

struct page_entry {
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
    long long int page_number; // virtual number for 

    __forceinline__
    __host__ __device__
    page_entry(int state, uint64_t host_address) { init(state, host_address); }

    __forceinline__
    __host__
    void init(int i_state, uint64_t i_host_address) {
        state = i_state;
        lock = 0;
        host_address = i_host_address;
        device_address = NULL;
        page_number = -1;
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

__global__ void memcpyDtoH_global(page_entry *d_TLB, size_t size, size_t tlb_size, int request_size, int sizeof_T){
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

template<typename T>
struct rdma_buf {
    // private:
        // uint64_t wr_rdma_remote_addr, uint32_t wr_rdma_rkey,            
        //               uint32_t wr_sg_length, uint32_t wr_sg_lkey, uint64_t wr_sg_addr,
        //               int wr_opcode, uint32_t qp_num, int cur_post, void *qp_buf, void *bf_reg, unsigned int *qp_db, void *dev_qp_sq, int id

    // public:
        uint64_t gpu_address;
        uint64_t host_address;
        size_t size;
        
        uint32_t request_size;
        uint64_t dev_addr;
        T *dev_buffer;
        T *host_buffer;  // in server mode this will be server address
        T *local_buffer; // in server mode this will be host buffer
        size_t tlb_size;
        page_entry *host_TLB, *d_TLB;
        unsigned int *h_tlb, *d_tlb;

        // server mode only:
        struct ibv_mr *buffer_mr;
        page_map_entry *h_page_map;
        __device__ /*long long int*/ page_map_entry *page_map;
        
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

        // constructor for pointer declaration:
        void start(size_t user_size){
            uint64_t offset = (uint64_t) Address_Offset;
            // gpu_address = user_address + Address_Offset;
            host_address = remote_address + Address_Offset;
            size = user_size;
            request_size = REQUEST_SIZE/sizeof(T);
            // dev_addr = GPU_address + GPU_addr_offset;
            // dev_buffer = (T *) dev_addr;
             
            printf("request_size: %d sizeof(T): %d\n", request_size, sizeof(T));
             
            // allocate memory on gpu and host
            alloc_memory((T *) host_address, (T *) gpu_address);
            printf("remote_address: %p host_address: %p Address_Offset: %p\n", remote_address, host_address, Address_Offset);

            // allocate TLB on device for now
            if(alloc_tlb()){
                printf("Error on TLB Buffer allocation!\n");
                exit(-1);
            }
        }

        // allocate tlb data structure on device so that device can access
        
        int alloc_memory(T *remote_address, T* gpu_address){
            // printf("gpu_address: %p\n", gpu_address);
            host_buffer = (T *) host_address;
            
            // device_buffer = (T *) gpu_address;
            Address_Offset = Address_Offset + size;
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
            alloc_page_map();
            return 0; // for success
        }
        
        
        int alloc_tlb(){
            int req_size = REQUEST_SIZE;
            tlb_size = ceil((double)size/req_size);
            
            // if(cudaSuccess != cudaMalloc(&tlb_buffer, tlb_size*sizeof(unsigned int)))
            //     return -1;
            if(cudaSuccess != cudaMalloc(&d_TLB, tlb_size*sizeof(page_entry)))
                return -1;

            if(cudaSuccess != cudaMalloc(&d_tlb, tlb_size*sizeof(unsigned int)))
                return -1;

            // if(cudaSuccess != cudaMallocManaged(&tlb_sync_host, 1*sizeof(tlb_entry)))
            //     return -1;

            printf("tlb_size: %llu sizeof(tlb_entry): %d\n", tlb_size, sizeof(uint8_t));
            
            host_TLB = (page_entry *) malloc(tlb_size*sizeof(page_entry));
            h_tlb = (unsigned int *) malloc(tlb_size*sizeof(unsigned int));

            for(size_t i = 0; i < tlb_size; i++){
                // printf("host_address: %p\n", host_address);
                // printf("REQUEST_SIZE: %d\n", REQUEST_SIZE);
                // printf("host_address + %llu*REQUEST_SIZE: %p\n", i, host_address + i*REQUEST_SIZE);
                host_TLB[i].init(0, host_address + i*REQUEST_SIZE);
                h_tlb[i] = 0;
                // host_TLB[i].device_address = GPU_address + GPU_addr_offset;
                // GPU_addr_offset += REQUEST_SIZE;
                // printf("host_TLB[%d].device_address: %p\n", i, host_TLB[i].device_address);
                if(i == tlb_size-1) printf("host_address + %llu*REQUEST_SIZE: %p\n", i, host_address + i*REQUEST_SIZE);
            }
            // printf("Global_GPU_address: %p\n", Global_GPU_address);
            if(update_device_tlb() == -1) return -1;

            // printf("tlb_buffer: %p\n", tlb_buffer);
            return 0;
        }

        int alloc_page_map(){
            
            size_t restricted_gpu_mem = sizeof(unsigned int)*1963263822llu;
            restricted_gpu_mem = restricted_gpu_mem/3;
            const size_t page_size = REQUEST_SIZE;

            size_t page_map_size = restricted_gpu_mem/page_size;

            // allocating page map with page_map_entry
            printf("restricted_gpu_mem: %llu\n", (long long int) restricted_gpu_mem);
            h_page_map = (page_map_entry *) malloc(page_map_size*sizeof(page_map_entry)); 

            for (size_t i = 0; i < page_map_size; i++)
            {
                h_page_map[i].init();
            }
            
            if(cudaSuccess != cudaMalloc((void **) &page_map, page_map_size*sizeof(page_map_entry)))
            {
                printf("Page map could not be allocated on device!\n");
                exit(-1);
            }

            int ret = cudaMemcpy(page_map, h_page_map, page_map_size*sizeof(page_map_entry), cudaMemcpyHostToDevice);
            if(cudaSuccess != ret)
            {
                printf("Page map could not be set; ret: %d!\n", ret);
                exit(-1);
            }

            // printf("page_map_size: %llu\n", (unsigned long long) page_map_size);
            // if(cudaSuccess != cudaMalloc((void **) &page_map, page_map_size*sizeof(long long int)))
            // {
            //     printf("Page map could not be allocated on device!\n");
            //     exit(-1);
            // }
            // long long int *temp = (long long int *) malloc(page_map_size*sizeof(long long int));
            // for (size_t i = 0; i < page_map_size; i++)
            // {
            //     temp[i] = -1;
            // }
            
            // int ret = cudaMemcpy(page_map, temp, page_map_size*sizeof(long long int), cudaMemcpyHostToDevice);
            // if(cudaSuccess != ret)
            // {
            //     printf("Page map could not be set; ret: %d!\n", ret);
            //     exit(-1);
            // }
            // free(temp);
            return 0;
        }

        __forceinline__
        __host__
        int update_device_tlb(){
            if(cudaSuccess != cudaDeviceSynchronize()) return -1;
            if(cudaSuccess != cudaMemcpy(d_TLB, host_TLB, tlb_size*sizeof(page_entry), cudaMemcpyHostToDevice))
                return -1;

            if(cudaSuccess != cudaMemcpy(d_tlb, h_tlb, tlb_size*sizeof(unsigned int), cudaMemcpyHostToDevice))
                return -1;
            if(cudaSuccess != cudaDeviceSynchronize()) return -1;
            return 0;
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
        size_t get_page(){
            size_t ret = atomicAdd( (unsigned long long int *) &R_cursor, 1)%num_pages;
            return ret;
        }

        __forceinline__
        __device__
        size_t evict_page(){
            size_t ret = atomicAdd( (unsigned long long int *) &E_cursor, 1)%num_pages;
            return ret;
        }

        __forceinline__
        __device__
        float available_pages(){
            if(E_cursor == R_cursor) return (float) 100;
            return (float) ((num_pages - R_cursor + E_cursor)%num_pages)/num_pages*100;
        }

        __forceinline__
        __device__
        void read(uint64_t che){
            int qp_index = get_smid()%128; // get_smid(); // warp_id() % 256; // ;

            // unsigned int mask = __match_any_sync(__activemask(), qp_index);
            // unsigned int leader = __ffs(mask) - 1; // mask ? 31 - __clz(mask) : 0;    // select a leader


            unsigned long long int data_size = request_size*sizeof(T);
            void *cq_dbrec = (void *) gpoll_cont.cq_dbrec[qp_index];
            
            bool isSet = false;
            volatile uint *cq_lock = &gpost_cont.cq_lock[qp_index];
            
            // __syncwarp(mask);
            int retries = 0;
            // if(lane_id() == leader){
                while(atomicCAS((unsigned int *)cq_lock, 0, 1) != 0){
                    // retries++;
                }
            
            volatile size_t *p_index = &gpost_cont.n_post[qp_index];
            unsigned int cur_post = atomicAdd((unsigned long long int *)p_index, (unsigned long long int ) 1);
            void *cqe = gpoll_cont.cq_buf + 2*4096*qp_index + (cur_post & 63) * 64;
            struct mlx5_cqe64 *cqe64 = (struct mlx5_cqe64 *) cqe;
            
            // get_page is circular 
            if(d_TLB[che].page_number == -1){
                size_t page_number = get_page();
                size_t page_to_evict = page_map[page_number].state;
                if(page_to_evict != -1){
                    // printf("1. che: %llu evicting: %llu\n", (unsigned long long) che, (unsigned long long) page_number);
                    
                    while(!d_TLB[page_to_evict].lock_entry());
                    // printf("che: %llu evicting: %llu d_tlb[page_to_evict]: %d\n", (unsigned long long) che, (unsigned long long) page_number, d_tlb[page_to_evict]);
                    atomicCAS(&d_tlb[page_to_evict], 2, 0);
                    d_TLB[page_to_evict].device_address = NULL;
                    d_TLB[page_to_evict].page_number = -1;
                    
                    d_TLB[page_to_evict].release_lock();
                }
                page_map[page_number].state = che;
                d_TLB[che].page_number = page_number;
            }
            d_TLB[che].device_address = d_TLB[che].page_number*data_size + Global_GPU_address;

            // if(d_TLB[che].device_address == NULL) 
            //     printf("inside read - d_TLB[%llu].device_address: %p\n", che, d_TLB[che].device_address);

            uint64_t rem_addr = host_address + che*request_size*sizeof(T);
            int rkey_index = (/*d_TLB[che].host_address*/ rem_addr - d_remote_address/*gpost_cont.wr_rdma_remote_addr*/)/(8*1024*1024*1024llu);
            uint64_t value_ctrl;
            int finished;
            

            post_m(/*d_TLB[che].host_address*/ rem_addr, gpost_cont2.wr_rdma_rkey[rkey_index], data_size, gpost_cont.wr_sg_lkey, d_TLB[che].device_address /*dev_addr + che*request_size*sizeof(T)*/, 4, gpost_cont.qp_num + qp_index, 
                    cur_post, &value_ctrl, gpost_cont.qp_buf + 8192*qp_index, (void *) gpost_cont.bf_reg[qp_index], gpost_cont.qp_db[qp_index], gpost_cont.dev_qp_sq[qp_index], 0);
            // __syncwarp(mask);
            // printf("mask: %d\n", mask);
            // if(cur_post == (*p_index-1)){
                
                // update_db(&value_ctrl, (void *) gpost_cont.bf_reg[qp_index]);
            // }
            // __syncwarp(mask);
            volatile uint8_t *op_flag = &cqe64->op_own;
            retries = 0;
            
            while(*op_flag == 240){
                    if(retries > 15){
                    //     unsigned int mask2 = __match_any_sync(__activemask(), (unsigned long long)qp_index);
                        
                    //     cur_post = atomicAdd((unsigned long long int *)p_index, (unsigned long long int ) 1);
                    //     cqe = gpoll_cont.cq_buf + 2*4096*qp_index + (cur_post & 63) * 64;
                    //     op_flag = &cqe64->op_own;
                    //     // unsigned int leader2 = __ffs(mask) - 1; // mask ? 31 - __clz(mask) : 0;    // select a leader
                        post_m(rem_addr, gpost_cont2.wr_rdma_rkey[rkey_index], data_size, gpost_cont.wr_sg_lkey, d_TLB[che].device_address /*dev_addr + che*request_size*sizeof(T)*/, 4, gpost_cont.qp_num + qp_index, 
                        cur_post, &value_ctrl, gpost_cont.qp_buf + 8192*qp_index, (void *) gpost_cont.bf_reg[qp_index], gpost_cont.qp_db[qp_index], gpost_cont.dev_qp_sq[qp_index], 0);
                    //     __syncwarp(mask2);
                    //     if(cur_post == (*p_index-1)){
                            // update_db(&value_ctrl, (void *) gpost_cont.bf_reg[qp_index]);
                    //     }
                    //     __syncwarp(mask2);
                        retries = -1;
                    }
                    // atomicAdd((unsigned long long int *) &transfer_time, 1);
                    retries++;
                    
            }
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
            // T *add = (T *) address;
            // #ifdef  __CUDA_ARCH__
                uint64_t che;
                uint64_t id = blockDim.x * blockIdx.x + threadIdx.x; 

                che = floor((double)index/request_size); // floor((double)index/request_size); //getTLBindex(index, request_size);
                // select which qp and cq to use:
                // TODO: for oversubscription, first check if gpu has enough free memory
                // volatile uint8_t *page_entry = &d_tlb[che];



                unsigned int mask1 = __match_any_sync(__activemask(), (unsigned long long)che);
                unsigned int leader1 = __ffs(mask1) - 1; // mask ? 31 - __clz(mask) : 0;    // select a leader
                T return_val;
                if(leader1 == lane_id()){
                    while(!d_TLB[che].lock_entry());
                    if(d_tlb[che] == 0){
                        read(che);
                        atomicCAS(&d_tlb[che], 0, 2);
                    }
                }
                

                __syncwarp(mask1);
                if(d_TLB[che].device_address == NULL){
                    // printf("mask1: %d, d_TLB[%llu].device_address: %p\n", mask1, che, d_TLB[che].device_address);
                    return_val = (T) 0;
                }
                else{
                    T *tmp_array = (T *) d_TLB[che].device_address; 
                    return_val = tmp_array[index%request_size];
                }
                __syncwarp(mask1);
                // __syncthreads();
                if(leader1 == lane_id()){
                    d_TLB[che].release_lock();
                }
                // __syncwarp(mask1);

                return return_val;

        }

        // destructor ~

        // [] operator for rvalue
        // a lot left to do here in rvalue function
        

        __forceinline__
        __device__
        void rvalue(size_t index, T new_value){
            
            // #ifdef  __CUDA_ARCH__
                T *tmp_array;
                int che = floor((double)index/request_size);
                if(d_TLB[che].state == 2){
                    tmp_array = (T *)d_TLB[che].device_address;
                    int ind = index&(request_size-1);
                    tmp_array[ind] = new_value;
                }
                else if(d_TLB[che].state == 0){
                    
                    if(d_TLB[che].lock_entry()){
                        unsigned long long int data_size = request_size*sizeof(T);
                        unsigned long long int offset = atomicAdd((unsigned long long int *)&GPU_address_offset, (unsigned long long int) data_size);
                        d_TLB[che].device_address = Global_GPU_address + offset;
                        
                        d_TLB[che].state = 2;
                        d_TLB[che].release_lock();
                    }
                    else{
                        while(d_TLB[che].state != 2);
                    }
                    tmp_array = (T *)d_TLB[che].device_address;
                    int ind = index&(request_size-1);
                    tmp_array[ind] = new_value;
                    
                }
            // return tmp_array[ind];
              
            
            // else
                
            // #endif
        }
    // tlb entry update
};






// typedef struct rdma_buf rdma_buf;



// typedef struct rdma_tlb rdma_tlb;






#ifdef __cplusplus
}
#endif

#endif