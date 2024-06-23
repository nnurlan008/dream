#ifndef RUNTIME_H
#define RUNTIME_H

#include <stdint.h>
#include <sys/types.h>
#include <infiniband/verbs.h>


#include <iostream>
// #include <simt/atomic>

#include "../src/rdma_utils.cuh"
#include "../src/rdma_utils.h"

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

#define KB(x) (long long int) x*1024
#define MB(x) (long long int) KB(x)*1024
#define GB(x) (long long int) MB(x)*1024

// wait for MAX_POST # of requests to be stored on qp buffer
#define MAX_POST 3 
// request size

#define REQUEST_SIZE 1*1024 // bytes

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

void alloc_global_host_content(struct post_content post_cont, struct poll_content poll_cont, struct host_keys keys){
    // copy poll and post content to global 
    printf("function: %s line: %d\n", __FILE__, __LINE__);
    hpost_cont = post_cont;
    printf("function: %s line: %d\n", __FILE__, __LINE__);
    hpoll_cont = poll_cont;
    printf("function: %s line: %d\n", __FILE__, __LINE__);
    keys_for_host = keys;
    printf("function: %s line: %d\n", __FILE__, __LINE__);
}

__global__ void alloc_global_content(struct post_content *post_cont, struct poll_content *poll_cont, struct post_content2 *post_cont2){
    // copy poll and post content to global 
    gpost_cont = *post_cont;
    gpoll_cont = *poll_cont;
    gpost_cont2 = *post_cont2;
    Global_GPU_address = post_cont->wr_sg_addr;
    printf("alloc_global_content - Global_GPU_address: %p\n", Global_GPU_address);
    // Global_Dev_address = post_cont->wr_sg_addr;
    GPU_address_offset = 0;
    printf("qp_num: %d\n", gpost_cont.qp_num);
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

__global__ void memcpyDtoH_global(tlb_entry *d_TLB, size_t size, size_t tlb_size){
    size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    size_t id = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;
    
        
    // for(int i = thid; i < tlb_size; i += stride)
    // {
        // if(i < tlb_size){
        
                uint64_t che;                
                // int buf_index = id/16384;
                struct ibv_wc wc;
                che = floor((double)index/256); //getTLBindex(index, request_size);
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
                            int qp_index = che & 255;
                            // unsigned long long int data_size = 256*sizeof(int);
                            unsigned long long int data_size;
                            if(che == tlb_size -1) data_size = size - che*1024;
                            else data_size = 1024;
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
        // rdma_tlb *tlb;
        uint32_t request_size;
        T *host_buffer;
        // __device__ T *device_buffer;
        // ibv_mr *host_mr, *dev_mr;
        // size_t offset=0;
        // private:
        // unsigned int * tlb_buffer;
        size_t tlb_size;
        tlb_entry *host_TLB, *d_TLB;
        // uint8_t *tlb_sync_host;
        //  rdma_buf_d<T> *dev_buf;

        // constructor
        
        rdma_buf(uint64_t user_address, uint64_t h_address, size_t user_size){
            printf(" sREQUEST_SIZE: %d offset: %d\n", REQUEST_SIZE, Address_Offset);
            uint64_t offset = (uint64_t) Address_Offset;
            // gpu_address = user_address + Address_Offset;
            host_address = h_address + Address_Offset;
            size = user_size;
            request_size = REQUEST_SIZE/sizeof(T);

            // allocate memory on gpu and host
            alloc_memory((T *) host_address, (T *) gpu_address);
            printf("gpu_address: %p\n", gpu_address);

            // allocate TLB on device for now
            if(alloc_tlb()){
                printf("Error on TLB Buffer allocation!\n");
                exit(-1);
            }

            // dev_buf

        }

        // constructor for pointer declaration:
        void start(size_t user_size){
            uint64_t offset = (uint64_t) Address_Offset;
            // gpu_address = user_address + Address_Offset;
            host_address = remote_address + Address_Offset;
            size = user_size;
            request_size = REQUEST_SIZE/sizeof(T);
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
            // check if the offset does not exceed allowed memory
            return 0; // for success
        }
        
        
        int alloc_tlb(){
            int req_size = REQUEST_SIZE;
            tlb_size = ceil((double)size/req_size);
            
            // if(cudaSuccess != cudaMalloc(&tlb_buffer, tlb_size*sizeof(unsigned int)))
            //     return -1;
            if(cudaSuccess != cudaMalloc(&d_TLB, tlb_size*sizeof(tlb_entry)))
                return -1;

            // if(cudaSuccess != cudaMallocManaged(&tlb_sync_host, 1*sizeof(tlb_entry)))
            //     return -1;

            printf("tlb_size: %llu sizeof(tlb_entry): %d\n", tlb_size, sizeof(uint8_t));
            
            host_TLB = (tlb_entry *) malloc(tlb_size*sizeof(tlb_entry));

            for(size_t i = 0; i < tlb_size; i++){
                // printf("host_address: %p\n", host_address);
                // printf("REQUEST_SIZE: %d\n", REQUEST_SIZE);
                // printf("host_address + %llu*REQUEST_SIZE: %p\n", i, host_address + i*REQUEST_SIZE);
                host_TLB[i].init(0, host_address + i*REQUEST_SIZE);
                if(i == tlb_size-1) printf("host_address + %llu*REQUEST_SIZE: %p\n", i, host_address + i*REQUEST_SIZE);
            }
            // printf("Global_GPU_address: %p\n", Global_GPU_address);
            if(update_device_tlb() == -1) return -1;

            // printf("tlb_buffer: %p\n", tlb_buffer);
            return 0;
        }

        __forceinline__
        __host__
        int update_device_tlb(){
            if(cudaSuccess != cudaDeviceSynchronize()) return -1;
            if(cudaSuccess != cudaMemcpy(d_TLB, host_TLB, tlb_size*sizeof(tlb_entry), cudaMemcpyHostToDevice))
                return -1;
            if(cudaSuccess != cudaDeviceSynchronize()) return -1;
            return 0;
        }

        __forceinline__
        __host__
        int memcpyDtoH(void){
            size_t threads = 1024;
            size_t n_blks = size/threads + 1; ; // tlb_size/threads + 1;
            if(cudaSuccess != cudaDeviceSynchronize()) return -1;
            memcpyDtoH_global<<< n_blks, threads>>>(d_TLB, size, tlb_size);
            if(cudaSuccess != cudaDeviceSynchronize()) return -1;
            return 0;
        }

        // // check TLB
        // __forceinline__
        // __device__ __host__
        // uint8_t checkTLB(size_t index, unsigned int *tlb_buffer){
        //     // printf("index: %d\n", index);
        //     return tlb_buffer[index];
        // }

        // __forceinline__
        // __device__ __host__
        // size_t getTLBindex(size_t index, int request_size){
        //     // printf("index: %d sadasdrequest_size: %d\n", index, request_size);
        //     return floor((double)index/request_size);
        // }

        // // lock TLB entry
        // __forceinline__
        // __device__
        // int lock_tlb_entry(size_t index){
        //     // if(atomicCAS((unsigned int *)&tlb_buffer[index], 0, 1) == 0){
        //     //     return 1; // on success
        //     // }
        //     return 0; // on failure
        // }

        // __forceinline__
        // __device__
        // int lock_cq_entry(size_t index){
        //     bool isSet = false; 
        //     do 
        //     {
        //         if (isSet = atomicCAS((unsigned int *)&gpost_cont.cq_lock[index], (unsigned int) 0, (unsigned int) 1) == 0) 
        //         {
        //             return isSet;
        //         }
        //         // if (isSet) 
        //         // {
        //         //     mutex = 0;
        //         // }
        //     } 
        //     while (!isSet);
            
        //     // while(atomicCAS((unsigned int *)&gpost_cont.cq_lock[index], (unsigned int) 0, (unsigned int) 1)) != 0){
        //     //     // printf("index: %d gpost_cont.cq_lock[index]: %d isSet: %d\n", 
        //     //     //         index, gpost_cont.cq_lock[index], isSet);
        //     //     // sleep_d(20);
        //     // };
            
        //     // return 1;
        //     // if(atomicCAS((unsigned int *)&gpost_cont.cq_lock[index], 0, 1) == 0){
        //     //     return 1; // on success
        //     // }
        //     // return 0; // on failure
        // }
        
        // // sleep for device
        // __forceinline__
        // __device__
        // void sleep_d(clock_t delay){
        //     clock_t start_clock = clock();
        //     // clock_t clock_offset = 0;
        //     while (clock() - start_clock < delay)
        //     {
                
        //     }
        // }

        // __forceinline__
        // __device__ 
        // unsigned long long int update_gpu_offset(unsigned long long int data_size){
        //     unsigned long long int offset = atomicAdd((unsigned long long int *)&GPU_address_offset, (unsigned long long int) data_size);
        //     return offset;
        // }

        // [] operator for lvalue
        __forceinline__
        __device__ 
        T& operator[](size_t index) {
            // T *add = (T *) address;
            // #ifdef  __CUDA_ARCH__
            // printf("GPU access\n");
                uint64_t che;
                uint64_t id = blockDim.x * blockIdx.x + threadIdx.x; 
                
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
                    // int reg_371200 = 0;
                    // if(index == 371200){
                    //     printf("index: %d d_TLB[che].state: %d d_TLB[che].host_address: %p reg_371200++: %d\n", 371200, d_TLB[che].state, d_TLB[che].host_address, reg_371200++);
                    // }
                    if(index < 0 || index >= size/sizeof(T) || che < 0 || che >= tlb_size) {
                        printf("index: %llu Invalid index\n", index);
                        return;
                    }
                    if(d_TLB[che].state == 2 || d_TLB[che].state == 4){ // page completely on gpu or dirty on gpu
                        T *tmp_array =  (T *) d_TLB[che].device_address;
                        // printf("d_TLB[che].state: %d index: %llu tmp_array[0]: %d\n", d_TLB[che].state, index, tmp_array[0]);
                        // LRU is incremented
                        // if(index == 371200){
                        //     printf("index: %d  data is in device reg_371200++: %d\n", 371200, reg_371200++);
                        // }
                        // if(index == 0) return tmp_array[0];
                        // else return tmp_array[index%request_size];
                        return tmp_array[index&255];
                    }
                    // if(index == 371200){
                    //     printf("index: %d trying to get tlb lock reg_371200++: %d\n", 371200, reg_371200++);
                    // }
                    if(d_TLB[che].state == 0 || d_TLB[che].state == 3){ // page completely on cpu or dirty on cpu
                        if(d_TLB[che].lock_entry()){
                            int qp_index = che & 255;
                            unsigned long long int data_size;
                            if(che == tlb_size -1) data_size = size - che*request_size*sizeof(T);
                            else data_size = request_size*sizeof(T);
                            void *cq_dbrec = (void *) gpoll_cont.cq_dbrec[qp_index];
                            
                            bool isSet = false;
                            volatile uint *cq_lock = &gpost_cont.cq_lock[qp_index];
                            // printf("id: %d cq_lock: %d qp_index: %d\n", id, cq_lock, qp_index);
                            // if(index == 371200){
                            //     printf("index: %d trying to get cq_lock reg_371200++: %d\n", 371200, reg_371200++);
                            // }
                            while(atomicCAS((unsigned int *)cq_lock, 0, 1) != 0);
                            // if(index == 371200){
                            //     printf("index: %d got cq_lock reg_371200++: %d\n", 371200, reg_371200++);
                            // }
                                volatile size_t *p_index = &gpost_cont.n_post[qp_index];
                                int cur_post = atomicAdd((unsigned long long int *)p_index, (unsigned long long int ) 1);
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
                                d_TLB[che].device_address = Global_GPU_address + offset; // update_gpu_offset( (unsigned long long int) data_size);
                                int rkey_index = (d_TLB[che].host_address - gpost_cont.wr_rdma_remote_addr)/(8*1024*1024*1024llu);
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

                                

                                post_m(d_TLB[che].host_address, gpost_cont2.wr_rdma_rkey[rkey_index], data_size, gpost_cont.wr_sg_lkey, d_TLB[che].device_address, 4, gpost_cont.qp_num + qp_index, 
                                        cur_post, gpost_cont.qp_buf + 8192*qp_index, (void *) gpost_cont.bf_reg[qp_index], gpost_cont.qp_db[qp_index], gpost_cont.dev_qp_sq[qp_index], id);
                                
                                // if(index == 371200){
                                //     printf("index: %d posted - be waiting reg_371200++: %d\n", 371200, reg_371200++);
                                // }
                                // printf("entering - id: %d d_TLB[che].device_address: %p d_TLB[che].host_address: %p\n",\
                                //     id, d_TLB[che].device_address, d_TLB[che].host_address);
                                while(/*cqe64->op_own == 240*/*op_flag == 240){
                                    
                                    // printf("id: %d gpoll_cont.cq_buf: %p qp_index: %d d_TLB[che].device_address: %p\n", id, gpoll_cont.cq_buf, qp_index, d_TLB[che].device_address);
                                    // printf("*op_flag: %d cqe64->op_own: %d\n", *op_flag, cqe64->op_own);
                                    // if(*tlb_sync_host == 1) {
                                    //     tlb_sync_host = 0;
                                    // }
                                }
                                // printf("cq done - id: %d d_TLB[che].device_address: %p d_TLB[che].host_address: %p\n",\
                                //     id, d_TLB[che].device_address, d_TLB[che].host_address);
                                // if(index == 371200){
                                //     printf("index: %d cq completed reg_371200++: %d\n", 371200, reg_371200++);
                                // }
                                // printf("*op_flag: %d cqe64->op_own: %d\n", *op_flag, cqe64->op_own);
                                // printf("id: %d cqe64->op_own: %d qp_index: %d d_TLB[che].device_address: %p\n",\
                                //      id, cqe64->op_own, qp_index, d_TLB[che].device_address);
                                *op_flag = 240;
                                // __threadfence_system();
                                *(uint32_t *) cq_dbrec = (uint32_t) htonl((cur_post+1) & 0xffffff);
                                // __threadfence();
                                // if(che == 1024*1024*8){
                                //     T *tmp_array1 =  (T *)d_TLB[che].device_address;
                                //     printf("d_TLB[che].host_address: %p tmp_array1[0]: %d\n", d_TLB[che].host_address, tmp_array1[0]);
                                // }
                                // printf("index: %llu id: %d tmp_p[index%request_size]: %d index%request_size: %d d_TLB[che].device_address: %p Global_GPU_address + offset: %p\n",\
                                //         index, id, tmp_p[index%1024], index%1024, d_TLB[che].device_address, Global_GPU_address + offset);
                                d_TLB[che].state = 2;
                            
                            *cq_lock = 0;
                            d_TLB[che].release_lock();
                        }
                        // if(index == 371200){
                        //     printf("index: %d  other thread posting reg_371200++: %d\n", 371200, reg_371200++);
                        // }
                        while(d_TLB[che].state != 2);
                        // if(index == 371200){
                        //     printf("index: %d  other thread posted and updated tlb reg_371200++: %d\n", 371200, reg_371200++);
                        // }
                    }

                    T *tmp_array = (T *) d_TLB[che].device_address;

                    // printf("tmp_array[0]: %d index: %d\n", tmp_array[0], index);
                    // printf("tmp_array[1]: %d request_size: %d\n", tmp_array[1], request_size);
                    // printf("tmp_array[2]: %d index%request_size: %d\n", tmp_array[2], index%request_size);

                    
                    // if(index == 0) return tmp_array[0];
                    // else return tmp_array[index%request_size];
                    // size_t cur_index = index&255;
                    // printf("tmp_array[0]: %d index: %d cur_index: %d\n", tmp_array[0], index, cur_index);
                    // printf("tmp_array[1]: %d request_size: %d\n", tmp_array[1], request_size);
                    // printf("tmp_array[index&255]: %llu tmp_array[0]: %llu\n", tmp_array[cur_index], tmp_array[0]);
                    // printf("index%request_size: %llu index&255: %llu index: %llu\n", \
                    //         index%request_size, index&255, index);
                    // for(int i = 0; i < 256; i++){
                    //     printf(" tmp_array[%d]: %d ", i&255, tmp_array[i&255]);
                    //     printf(" tmp_array[index&%d]: %d ", i&255, tmp_array[i&255]);
                    // }
                // T a = 2;
                int i = index&255;
                // printf("\ntest (*d_distance)[%d]: %d\n", i, tmp_array[i]);
                // T a = tmp_array[i];
                // printf("\nresult to return (*d_distance)[%d]: %d\n", index, a);
                return tmp_array[i]; // tmp_array[(int)index&255]; // tmp_array[index%request_size];
            // #else
            //     printf("CPU access\n");
            //     //  CPU access
            //     if(*tlb_sync_host == 0){ // it means tlb is not sync with device tlb
            //         if(cudaSuccess != cudaDeviceSynchronize()) return -1;
            //         if(cudaSuccess != cudaMemcpy(host_TLB, d_TLB, tlb_size*sizeof(tlb_entry), cudaMemcpyDeviceToHost))
            //             return -1;
            //         if(cudaSuccess != cudaDeviceSynchronize()) return -1;
            //         *tlb_sync_host = 1;
            //     }

            //     uint64_t che;
            //     struct ibv_wc wc;
            //     che = floor((double)index/request_size);

            //     if(host_TLB[che].state == 2 || host_TLB[che].state == 4){
            //         // post and poll on CPU!
            //         unsigned long long int data_size = request_size*sizeof(T);
            //         // post(uint64_t wr_rdma_remote_addr, uint32_t wr_rdma_rkey,            
            //         //   uint32_t wr_sg_length, uint32_t wr_sg_lkey, uint64_t wr_sg_addr,
            //         //   int wr_opcode, uint32_t qp_num, int cur_post, void *qp_buf, void *bf_reg, unsigned int *qp_db, void *dev_qp_sq)
            //         int lkey_index = (host_TLB[che].host_address - hpost_cont.wr_rdma_remote_addr)/(8*1024*1024*1024llu);
            //         post(host_TLB[che].device_address, hpost_cont.wr_rdma_rkey, data_size, keys_for_host.lkeys[lkey_index], host_TLB[che].host_address,
            //             4, hpost_cont.qp_num, hpost_cont.n_post[0], hpost_cont.qp_buf, hpost_cont.bf_reg[0], hpost_cont.qp_db[0], hpost_cont.dev_qp_sq[0]);
                    
            //         void *cq_dbrec = (void *) hpoll_cont.cq_dbrec[0];
            //         void *cqe = hpoll_cont.cq_buf + (hpost_cont.n_post[0] & 63) * 64;
            //         struct mlx5_cqe64 *cqe64 = (struct mlx5_cqe64 *) cqe;

            //         volatile uint8_t *op_flag = &cqe64->op_own;
            //         while(/*cqe64->op_own == 240*/*op_flag == 240){
            //             // printf("id: %d isSet: %d cqe64->op_own: %d qp_index: %d d_TLB[che].device_address: %p\n", id, isSet, cqe64->op_own, qp_index, d_TLB[che].device_address);
                        
            //         }
            //         *op_flag = 240;
            //         *(uint32_t *) cq_dbrec = (uint32_t) htonl((hpost_cont.n_post[0]+1) & 0xffffff);
            //         hpost_cont.n_post[0]++;
                    
            //     }


            //     return host_buffer[index];
            // #endif
            

        }

        // destructor ~

        // [] operator for rvalue
        // a lot left to do here in rvalue function
        __forceinline__
        __device__
        void rvalue(size_t index, T new_value){
            
            // #ifdef  __CUDA_ARCH__
                // device_buffer[index] = i;
                // printf("writing to c[%d]: %d\n", index, i);
                int che = floor((double)index/request_size);
                

                if(d_TLB[che].state == 2){
                    T *tmp_array = (T *)d_TLB[che].device_address;
                    // int ind = index&255;
                    // while(!d_TLB[che].lock_entry());
                    // T a = new_value;
                    // printf("i: %d tmp_array[%d]: %d \n", new_value, ind, tmp_array[ind]);
                    // tmp_array[ind] = new_value;
                    // if(new_value == 1)
                        // tmp_array[index&255] = (new_value == 1)*1 + (new_value == 2)*2;
                    // else if(new_value == 2)
                    //     tmp_array[index&255] = 2;
                    tmp_array[index&255] = new_value;
                    // atomicExch((unsigned int *)&tmp_array[ind], (unsigned int)new_value);
                    // printf("i: %d tmp_array[%d]: %d \n", new_value, ind, tmp_array[ind]);
                    // d_TLB[che].state = 4;
                    // d_TLB[che].release_lock();
                    // atomicCAS((unsigned int *)&tmp_array[ind], (unsigned int) tmp_array[ind], (unsigned int) i);
                }

                // return new_value;
                // while(!d_TLB[che].lock_entry());
                // if(d_TLB[che].device_address == NULL){
                //     unsigned long long int data_size = request_size*sizeof(T);
                //     unsigned long long int offset = atomicAdd((unsigned long long int *)&GPU_address_offset, (unsigned long long int) data_size);
                //     d_TLB[che].device_address = Global_GPU_address + offset;
                //     // d_TLB[che].release_lock();
                // }

                // if(d_TLB[che].device_address == NULL){
                    
                //     if(d_TLB[che].lock_entry()){
                //         unsigned long long int data_size = request_size*sizeof(T);
                //         unsigned long long int offset = atomicAdd((unsigned long long int *)&GPU_address_offset, (unsigned long long int) data_size);
                //         d_TLB[che].device_address = Global_GPU_address + offset;
                //         d_TLB[che].release_lock();
                //     }
                //     while(d_TLB[che].device_address == NULL); 
                // }


                // if(d_TLB[che].device_address == NULL){
                //     printf("rvalue - d_TLB[che].device_address: %p \n", NULL);
                // }
                // else{
                //     printf("rvalue - d_TLB[che].device_address: %p \n", d_TLB[che].device_address);
                //     T *tmp_array = (T *)d_TLB[che].device_address;
                //     int ind = index&255;
                //     printf("ind: %d index: %d \n", ind, index);
                //     printf("i: %d tmp_array[%d]: %d \n", i, ind, tmp_array[ind]);
                // }

                // __syncwarp();
                // else if(d_TLB[che].state == 0 || d_TLB[che].state == 3){

                //     uint64_t che;
                //     uint64_t id = blockDim.x * blockIdx.x + threadIdx.x; 
                    
                //     // int buf_index = id/16384;
                //     struct ibv_wc wc;
                //     che = floor((double)index/request_size); //getTLBindex(index, request_size);
                
                    
                        
                //         if(d_TLB[che].state == 0 || d_TLB[che].state == 3){ // page completely on cpu or dirty on cpu
                //             if(d_TLB[che].lock_entry()){
                //                 int qp_index = che & 255;
                //                 unsigned long long int data_size = request_size*sizeof(T);
                //                 void *cq_dbrec = (void *) gpoll_cont.cq_dbrec[qp_index];
                                
                //                 bool isSet = false;
                //                 volatile uint *cq_lock = &gpost_cont.cq_lock[qp_index];
                                
                //                 while(atomicCAS((unsigned int *)cq_lock, 0, 1) != 0);
                                
                //                     volatile size_t *p_index = &gpost_cont.n_post[qp_index];
                //                     int cur_post = atomicAdd((unsigned long long int *)p_index, (unsigned long long int ) 1);
                //                     void *cqe = gpoll_cont.cq_buf + 2*4096*qp_index + (cur_post & 63) * 64;
                //                     struct mlx5_cqe64 *cqe64 = (struct mlx5_cqe64 *) cqe;
                //                     volatile uint8_t *op_flag = &cqe64->op_own;
                //                     unsigned long long int offset = atomicAdd((unsigned long long int *)&GPU_address_offset, (unsigned long long int) data_size);
                //                     d_TLB[che].device_address = Global_GPU_address + offset; // update_gpu_offset( (unsigned long long int) data_size);
                //                     int rkey_index = (d_TLB[che].host_address - gpost_cont.wr_rdma_remote_addr)/(8*1024*1024*1024llu);
                //                     post_m(d_TLB[che].host_address, gpost_cont2.wr_rdma_rkey[rkey_index], data_size, gpost_cont.wr_sg_lkey, d_TLB[che].device_address, 4, gpost_cont.qp_num + qp_index, 
                //                             cur_post, gpost_cont.qp_buf + 8192*qp_index, (void *) gpost_cont.bf_reg[qp_index], gpost_cont.qp_db[qp_index], gpost_cont.dev_qp_sq[qp_index], id);
                                   
                                    
                //                     while(/*cqe64->op_own == 240*/*op_flag == 240){
                                        
                //                     }
                                    
                                    
                //                     *op_flag = 240;
                //                     // __threadfence_system();
                //                     *(uint32_t *) cq_dbrec = (uint32_t) htonl((cur_post+1) & 0xffffff);
                //                     // __threadfence();
                //                     d_TLB[che].state = 2;
                                
                //                 *cq_lock = 0;
                //                 d_TLB[che].release_lock();
                //             }
                            
                //             while(d_TLB[che].state != 2);
                            
                //         }

                //         T *tmp_array = (T *) d_TLB[che].device_address;
                        
                        
                //     int offsett = index&255;
                
                //     tmp_array[offsett] = i;

                // }
                // printf("rvalue - index: %d che: %d d_TLB[che].device_address: %p d_TLB[che].host_address: %p\n", index, che, d_TLB[che].device_address, d_TLB[che].host_address);
                // printf("d_TLB[che].device_address: %p \n", d_TLB[che].device_address);
                // T *tmp_array = (T *)d_TLB[che].device_address;
                // int ind = index&255;
                // tmp_array[ind] = i;

                

                // if(d_TLB[che].state != 4){ // data is already on device
                //     d_TLB[che].state = 4; // 4: dirty data on gpu
                // }
                // d_TLB[che].release_lock();
                // while(!d_TLB[che].lock_entry());
                // d_TLB[che]
                // d_TLB[che]. = 2;
            
            // else
                
            // #endif
        }
    // tlb entry update
};






// typedef struct rdma_buf rdma_buf;



// typedef struct rdma_tlb rdma_tlb;








#endif
