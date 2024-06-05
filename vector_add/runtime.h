#ifndef RUNTIME_H
#define RUNTIME_H

#include <stdint.h>
#include <sys/types.h>
#include <infiniband/verbs.h>


#include <iostream>
// #include <simt/atomic>

#include "rdma_utils.cuh"

#define htonl(x)  ((((uint32_t)(x) & 0xff000000) >> 24) |\
                   (((uint32_t)(x) & 0x00ff0000) >> 8) |\
                   (((uint32_t)(x) & 0x0000ff00) << 8) |\
                   (((uint32_t)(x) & 0x000000ff) << 24))

// offset for buffers; in bytes
size_t Address_Offset = 0; 
__device__ uint64_t Global_Dev_address;

__device__ struct post_content gpost_cont;
__device__ struct poll_content gpoll_cont;

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

__global__ void alloc_global_content(struct post_content *post_cont, struct poll_content *poll_cont){
    // copy poll and post content to global 
    gpost_cont = *post_cont;
    gpoll_cont = *poll_cont;
    Global_Dev_address = post_cont->wr_sg_addr;
    printf("qp_num: %d\n", gpost_cont.qp_num);
}

//tlb entry taken from bam source code
// template<simt::thread_scope _scope = simt::thread_scope_device>
// struct tlb_entry {
//     // uint64_t global_id;
//     simt::atomic<uint32_t, _scope> state;
//     // data_page_t* page = nullptr;
//     // TODO: implement LRU here per page
//     // page size will be fixed for all pages
//     // each page will have different gpu address - why?
//     // for oversubscription

//     __forceinline__
//     __host__ __device__
//     tlb_entry() { init(); }

//     __forceinline__
//     __host__ __device__
//     void init() {
//         // global_id = 0;
//         state.store(0, simt::memory_order_relaxed);
//         // page = nullptr;
//     }

//     __forceinline__
//     __device__
//     int lock_entry(size_t index){
//          do {
//             // 1 means is being posted currently
//             uint32_t st = state.fetch_or(1, simt::memory_order_acquire);
//             if ((st & 1) == 0)
//                 break;
//             sleep_nanoseconds(100);
//         } while (true);
//     }

//     __forceinline__
//     __device__
//     void release(const uint32_t count) {
// //		    if (global_id == 515920192)
// //			printf("--(2)st: %llx\tcount: %llu\n", (unsigned long long) state.load(simt::memory_order_relaxed), (unsigned long long) count);

//         state.fetch_sub(count, simt::memory_order_release); }

//     __forceinline__
//     __device__
//     void release() { if (page != nullptr)  {
// //		    if (global_id == 515920192)
// //			printf("--(1)st: %llx\tcount: %llu\n", (unsigned long long) state.load(simt::memory_order_relaxed), (unsigned long long) 1);

            // page->state.fetch_sub(1, simt::memory_order_release); }}

// };    


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
        gpu_address = (T *) (Global_Dev_address + Address_Offset);
        host_address = (T *) h_address;
        size = user_size;
        request_size = 65536; // bytes
    }

    //  // [] operator for lvalue
    //     __forceinline__
    //     __device__ 
    //     T& operator[](size_t index) {
    //             int che;
    //             int id = blockDim.x * blockIdx.x + threadIdx.x; 
    //             struct ibv_wc wc;
    //             // che is the tlb index
    //             che = floor((double)index/request_size); //getTLBindex(index, request_size);
    //             // printf("sadasdrequest_size: %d che: %d checkTLB(0, tlb_buffer): %d\n", request_size, che, checkTLB(0, tlb_buffer));
    //             // TODO: for oversubscription, first check if gpu has enough free memory
    //             // __syncthreads();
    //             if(/*checkTLB(che, tlb_buffer)*/tlb_buffer[che] == 0){
    //                 if(/*lock_tlb_entry(che)*/atomicCAS((unsigned int *)&tlb_buffer[che], 0, 1) == 0){
    //                     // select which qp and cq to use:
    //                     int qp_index = che % 15; // we have 15 QP resources
    //                     int data_size = 524288;
    //                     void *cq_dbrec = (void *) gpoll_cont.cq_dbrec[qp_index];
    //                         int cur_post = atomicAdd((unsigned long long int *)&gpost_cont.n_post[qp_index], (unsigned long long int ) 1);
    //                         post_m(host_address + che*data_size, gpost_cont.wr_rdma_rkey, data_size, gpost_cont.wr_sg_lkey, gpu_address + che*data_size, 4, gpost_cont.qp_num + qp_index, 
    //                             cur_post, gpost_cont.qp_buf + 8192*qp_index, (void *) gpost_cont.bf_reg[qp_index], gpost_cont.qp_db[qp_index], gpost_cont.dev_qp_sq[qp_index], id);
                            
    //                         void *cqe = gpoll_cont.cq_buf + 4096*qp_index + (cur_post & 15) * 64;
    //                         struct mlx5_cqe64 *cqe64 = (struct mlx5_cqe64 *) cqe;
                            
    //                         while(cqe64->op_own == 240);
    //                         cqe64->op_own = 240;
    //                         *(uint32_t *) cq_dbrec = (uint32_t) htonl((cur_post +1) & 0xffffff);
    //                         __threadfence();
    //                         tlb_buffer[che] = 2; // 2 means data is in device
    //                         // atomicCAS((unsigned int *)&tlb_buffer[che], 1, 2);
    //                 }
    //             }
    //             if(checkTLB(che, tlb_buffer) == 1){
    //                 while(tlb_buffer[che] != 2);
    //                 // __syncthreads();
    //             }
    //             return device_buffer[index];
    //     }

    //      // [] operator for lvalue
    //     __forceinline__
    //     __device__ 
    //     T operator[](size_t index) const {
    //             int che;
    //             int id = blockDim.x * blockIdx.x + threadIdx.x; 
    //             struct ibv_wc wc;
    //             che = floor((double)index/request_size); //getTLBindex(index, request_size);
    //             if(/*checkTLB(che, tlb_buffer)*/tlb_buffer[che] == 0){
    //                 if(/*lock_tlb_entry(che)*/atomicCAS((unsigned int *)&tlb_buffer[che], 0, 1) == 0){
    //                     int qp_index = che % 15;
    //                     printf("rvlue\n");
    //                     int data_size = 524288;
    //                     void *cq_dbrec = (void *) gpoll_cont.cq_dbrec[qp_index];
    //                         int cur_post = atomicAdd((unsigned long long int *)&gpost_cont.n_post[qp_index], (unsigned long long int ) 1);
    //                         post_m(host_address + che*data_size, gpost_cont.wr_rdma_rkey, data_size, gpost_cont.wr_sg_lkey, gpu_address + che*data_size, 4, gpost_cont.qp_num + qp_index, 
    //                             cur_post, gpost_cont.qp_buf + 8192*qp_index, (void *) gpost_cont.bf_reg[qp_index], gpost_cont.qp_db[qp_index], gpost_cont.dev_qp_sq[qp_index], id);
    //                         void *cqe = gpoll_cont.cq_buf + 4096*qp_index + (cur_post & 15) * 64;
    //                         struct mlx5_cqe64 *cqe64 = (struct mlx5_cqe64 *) cqe;
    //                         while(cqe64->op_own == 240);
    //                         cqe64->op_own = 240;
    //                         *(uint32_t *) cq_dbrec = (uint32_t) htonl((cur_post +1) & 0xffffff);
    //                         __threadfence();
    //                         tlb_buffer[che] = 2;
    //                 }
    //             }
    //             if(checkTLB(che, tlb_buffer) == 1){
    //                 while(tlb_buffer[che] != 2);
    //                 // __syncthreads();
    //             }
                
    //             return device_buffer[index];
    //     }

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
        __device__ T *device_buffer;
        // ibv_mr *host_mr, *dev_mr;
        // size_t offset=0;
        // private:
        unsigned int * tlb_buffer;
        size_t tlb_size;
        //  rdma_buf_d<T> *dev_buf;

        // constructor
        
        rdma_buf(uint64_t user_address, uint64_t h_address, size_t user_size){
            printf(" sREQUEST_SIZE: %d offset: %d\n", REQUEST_SIZE, Address_Offset);
            uint64_t offset = (uint64_t) Address_Offset;
            gpu_address = user_address + Address_Offset;
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
        void start(uint64_t user_address, uint64_t h_address, size_t user_size){
            uint64_t offset = (uint64_t) Address_Offset;
            gpu_address = user_address + Address_Offset;
            host_address = h_address + Address_Offset;
            size = user_size;
            request_size = REQUEST_SIZE/sizeof(T);
            
            // allocate memory on gpu and host
            alloc_memory((T *) host_address, (T *) gpu_address);
            printf("gpu_address: %p\n", user_address + offset);

            // allocate TLB on device for now
            if(alloc_tlb()){
                printf("Error on TLB Buffer allocation!\n");
                exit(-1);
            }


        }

        // allocate tlb data structure on device so that device can access
        
        int alloc_memory(T *remote_address, T* gpu_address){
            // printf("gpu_address: %p\n", gpu_address);
            host_buffer = (T *) remote_address;
            
            device_buffer = (T *) gpu_address;
            Address_Offset = Address_Offset + size;
            // check if the offset does not exceed allowed memory
            return 0; // for success
        }
        
        
        int alloc_tlb(){
            int req_size = REQUEST_SIZE;
            tlb_size = ceil((double)size/req_size);
            
            if(cudaSuccess != cudaMalloc(&tlb_buffer, tlb_size*sizeof(unsigned int)))
                return -1;
            printf("tlb_buffer: %p\n", tlb_buffer);
            return 0;
        }

        // check TLB
        __forceinline__
        __device__ __host__
        uint8_t checkTLB(size_t index, unsigned int *tlb_buffer){
            // printf("index: %d\n", index);
            return tlb_buffer[index];
        }

        __forceinline__
        __device__ __host__
        size_t getTLBindex(size_t index, int request_size){
            // printf("index: %d sadasdrequest_size: %d\n", index, request_size);
            return floor((double)index/request_size);
        }

        //lock TLB entry
        __forceinline__
        __device__
        int lock_tlb_entry(size_t index){
            if(atomicCAS((unsigned int *)&tlb_buffer[index], 0, 1) == 0){
                return 1; // on success
            }
            return 0; // on failure
        }

        __forceinline__
        __device__
        int lock_cq_entry(size_t index){
            bool isSet = false; 
            do 
            {
                if (isSet = atomicCAS((unsigned int *)&gpost_cont.cq_lock[index], (unsigned int) 0, (unsigned int) 1) == 0) 
                {
                    return isSet;
                }
                // if (isSet) 
                // {
                //     mutex = 0;
                // }
            } 
            while (!isSet);
            
            // while(atomicCAS((unsigned int *)&gpost_cont.cq_lock[index], (unsigned int) 0, (unsigned int) 1)) != 0){
            //     // printf("index: %d gpost_cont.cq_lock[index]: %d isSet: %d\n", 
            //     //         index, gpost_cont.cq_lock[index], isSet);
            //     // sleep_d(20);
            // };
            
            // return 1;
            // if(atomicCAS((unsigned int *)&gpost_cont.cq_lock[index], 0, 1) == 0){
            //     return 1; // on success
            // }
            // return 0; // on failure
        }
        
        // sleep for device
        __forceinline__
        __device__
        void sleep_d(clock_t delay){
            clock_t start_clock = clock();
            // clock_t clock_offset = 0;
            while (clock() - start_clock < delay)
            {
                
            }
        }

        // destructor ~

        // [] operator for lvalue
        __forceinline__
        __device__ 
        T& operator[](int index) {
            // T *add = (T *) address;
            // #ifdef  __CUDA_ARCH__
                int che;
                int id = blockDim.x * blockIdx.x + threadIdx.x; 
                
                // int buf_index = id/16384;
                struct ibv_wc wc;
                che = floor((double)index/request_size); //getTLBindex(index, request_size);
                // select which qp and cq to use:
                volatile uint *entry = &tlb_buffer[che];
                // printf("sadasdrequest_size: %d che: %d checkTLB(0, tlb_buffer): %d\n", request_size, che, checkTLB(0, tlb_buffer));
                // TODO: for oversubscription, first check if gpu has enough free memory
                // __syncthreads();

                // if(/*checkTLB(che, tlb_buffer)*/tlb_buffer[che] == 0){
                    // if(id % 16384 == 0){
                    // opcode: 4 for read, 0 for write
                    
                    // lock entry:
                    if(/*lock_tlb_entry(che)*/atomicCAS((unsigned int *)entry, 0, 1) == 0){
                        int qp_index = che % 128;
                        // printf("lvlue\n");
                        int data_size = request_size*sizeof(T);
                        // printf("data_size: %d\n", data_size);
                        /********************************************************************************************/
                        // uint64_t remote_addr = host_address + che*65536; // + data_size*4*index;
                        // uint64_t local_addr = gpu_address + che*65536; // + data_size*4*index;
                        // void *qp_buf = gpost_cont.qp_buf + 8192*qp_index;
                        // uint32_t qp_num = gpost_cont.qp_num + qp_index;
                        // void *dev_qp_sq = gpost_cont.dev_qp_sq[qp_index];
                        // void *bf_reg = (void *) gpost_cont.bf_reg[qp_index];
                        // unsigned int *qp_db = gpost_cont.qp_db[qp_index];
                        // void *cq_buf = gpoll_cont.cq_buf + 4096*qp_index;
                        // uint32_t *cons_index = (uint32_t *) gpoll_cont.cons_index[qp_index];
                        void *cq_dbrec = (void *) gpoll_cont.cq_dbrec[qp_index];
                        // uint32_t length = 4*data_size;
                        // int wr_opcode = gpost_cont.wr_opcode;
                        // uint64_t wr_rdma_remote_addr = remote_addr;
                        // uint32_t wr_rdma_rkey = gpost_cont.wr_rdma_rkey;
                        // int wr_sg_length = 4*data_size;
                        // uint32_t wr_sg_lkey = gpost_cont.wr_sg_lkey;
                        // uint64_t wr_sg_addr = local_addr;
                        /********************************************************************************************/
                        
                            // printf("qp_index: %d gpost_cont.cq_lock[index]: %d\n", qp_index, gpost_cont.cq_lock[qp_index]);
                            // int cur_post = gpost_cont->n_post[qp_index];gpost_cont->n_post[qp_index]++;
                            
                            // printf("cur_post: %d\n", cur_post);
                            // printf("before while: id: %d qp_index: %d\n", id, qp_index);
                            // printf("before while: id: %d cur_post: %d\n", id, cur_post);
                            // printf("before while: id: %d gpost_cont.n_post[qp_index]: %d\n", id, gpost_cont.n_post[qp_index]);
                            // printf("before while: id: %d h_address: %p\n", id, host_address);
                            // while(cur_post != atomicCAS((unsigned int *)&gpost_cont.n_post[qp_index], cur_post, cur_post + 1)) {
                            // // while(cur_post != atomicAdd((unsigned long long int *)&gpost_cont.n_post[qp_index], (unsigned long long int ) 1)) {
                            //     cur_post = gpost_cont.n_post[qp_index];
                            //     // printf("in while:     id: %d cur_post: %d\n",id, cur_post);
                            //     // printf("in while:     id: %d gpost_cont.n_post[qp_index]: %d\n",id, cur_post, gpost_cont.n_post[qp_index]);
                                
                            // }
                            
                            // printf("after while:  gpost_cont.n_post[qp_index]: %d id: %d\n", gpost_cont.n_post[qp_index], id);
                            // printf("after while:  id: %d cur_post: %d\n", id, cur_post);
                            // sleep_d(che*2);
                            // int cur_post = atomicAdd((unsigned long long int *)&gpost_cont.n_post[qp_index], (unsigned long long int ) 1);
                            // printf("locked by id: %d tlb_buffer[che]: %d che: %d qp_num: %d\n", id, tlb_buffer[che], che, qp_index);
                            // printf("__line: %d\n", __LINE__);


                                    // volatile size_t *p_index = &gpost_cont.n_post[qp_index];
                                    // int cur_post = atomicAdd((unsigned long long int *)p_index, (unsigned long long int ) 1);
                                    // // (cons_index_dev & ibv_cqe) * cqe_sz
                                    // void *cqe = gpoll_cont.cq_buf + 2*4096*qp_index + (cur_post & 63) * 64;
                                    // struct mlx5_cqe64 *cqe64 = (struct mlx5_cqe64 *) cqe;
                                    
                                    // post_m(host_address + che*data_size, gpost_cont.wr_rdma_rkey, data_size, gpost_cont.wr_sg_lkey, gpu_address + che*data_size, 4, gpost_cont.qp_num + qp_index, 
                                    //     cur_post, gpost_cont.qp_buf + 8192*qp_index, (void *) gpost_cont.bf_reg[qp_index], gpost_cont.qp_db[qp_index], gpost_cont.dev_qp_sq[qp_index], id);
                            
                              
                            // printf("__line: %d\n", __LINE__);
                            // cqe64->op_own = 240;
                            // void *cq_dbrec;// = (void *) gpoll_cont.cq_dbrec[qp_index];
                            
                            // atomicAdd(gpost_cont.n_post[che], (unsigned long long int) 1);
                            
                            // poll(gpoll_cont.cq_buf + 4096*2*qp_index, &wc, (uint32_t *) gpoll_cont.cons_index[qp_index], gpoll_cont.ibv_cqe, gpoll_cont.cqe_sz,1,cq_dbrec);
                            // tlb_buffer[che] = 2;
                        // if(lock_cq_entry(qp_index)){

                            // /* working part */
                            // // while(atomicCAS((unsigned int *)&cqe64->op_own, 0, 240) == 240){
                            // //     printf("id: %d cqe64->op_own: %d\n",id, cqe64->op_own);
                            // // }
                            // // printf("__line: %d\n", __LINE__);
                            // volatile uint8_t *op_flag = &cqe64->op_own;
                            // // printf("__line: %d\n", __LINE__);
                            // while(/*atomicCAS((unsigned int *)op_flag, 0, 240) == 240*/*op_flag == 240){
                            //     // *op_flag = cqe64->op_own;
                            //     printf("id: %d cqe64->op_own: %d\n",id, cqe64->op_own);
                            //     // sleep_d(500);
                            // }
                            
                            // *op_flag = 240;
                            // *(uint32_t *) cq_dbrec = (uint32_t) htonl((cur_post +1) & 0xffffff);
                            // // __threadfence();
                            // *entry = 2;


                            // gpost_cont.cq_lock[qp_index] = 0;
                        // }
                        
                        // cqe = gpoll_cont.cq_buf + 4096*qp_index + (cur_post & 15) * 64;
                        // cqe64 = (struct mlx5_cqe64 *) cqe;
                        
                        // printf("cur_post & 15: %d cqe64->op_own: %d gpost_cont.cq_lock[%d]: %d\n", cur_post & 15, cqe64->op_own, cur_post & 15, gpost_cont.cq_lock[cur_post & 15]);
                        
                        bool isSet = false;
                        volatile uint *cq_lock = &gpost_cont.cq_lock[qp_index];
                        while(atomicCAS((unsigned int *)cq_lock, 0, 1) != 0);
                        // do 
                        // {
                        //     printf("id: %d qp_index: %d\n", id, qp_index);
                        //     if ((atomicCAS((unsigned int *)cq_lock, 0, 1) == 0)) 
                        //     {
                        //         isSet = true;

                            // printf("id: %d, che: %d cur_post: %d qp_index: %d\n", id, che, cur_post, qp_index);
                            // printf("locked by id: %d tlb_buffer[che]: %d che: %d qp_num: %d\n", id, tlb_buffer[che], che, qp_index);
                            // printf("qp_index: %d cur_post: %d cqe64->op_own: %d gpost_cont.cq_lock[%d]: %d\n",qp_index, cur_post, cqe64->op_own, cur_post & 15, gpost_cont.cq_lock[cur_post & 15]);
                            // cqe = gpoll_cont.cq_buf+4096*qp_index + (cur_post & 15) * 64;
                            // cqe64 = (struct mlx5_cqe64 *) cqe;
                            // cqe64->op_own = 240;
                            // cq_dbrec = (void *) gpoll_cont.cq_dbrec[qp_index];
                                
                                   
                           
                                volatile size_t *p_index = &gpost_cont.n_post[qp_index];
                                int cur_post =  atomicAdd((unsigned long long int *)p_index, (unsigned long long int ) 1);
                                // gpost_cont.n_post[qp_index]++;
                                void *cqe = gpoll_cont.cq_buf + 2*4096*qp_index + (cur_post & 63) * 64;
                                struct mlx5_cqe64 *cqe64 = (struct mlx5_cqe64 *) cqe;
                                
                                // post_m(host_address + che*data_size, gpost_cont.wr_rdma_rkey, data_size, gpost_cont.wr_sg_lkey, gpu_address + che*data_size, 4, gpost_cont.qp_num + qp_index, 
                                // cur_post, gpost_cont.qp_buf + 8192*qp_index, (void *) gpost_cont.bf_reg[qp_index], gpost_cont.qp_db[qp_index], gpost_cont.dev_qp_sq[qp_index], id);
                                post_m(host_address + che*data_size, gpost_cont.wr_rdma_rkey, data_size, gpost_cont.wr_sg_lkey, gpu_address + che*data_size, 4, gpost_cont.qp_num + qp_index, 
                                       cur_post, gpost_cont.qp_buf + 8192*qp_index, (void *) gpost_cont.bf_reg[qp_index], gpost_cont.qp_db[qp_index], gpost_cont.dev_qp_sq[qp_index], id);
                                volatile uint8_t *op_flag = &cqe64->op_own;
                                while(/*cqe64->op_own == 240*/*op_flag == 240){
                                    // printf("id: %d isSet: %d cqe64->op_own: %d qp_index: %d\n", id, isSet, cqe64->op_own, qp_index);
                                }
                                // printf("chunk: %d - %d\n", che*data_size/4, che*data_size/4+data_size/4);
                                // cqe64->op_own = 240;
                                *op_flag = 240;
                                *(uint32_t *) cq_dbrec = (uint32_t) htonl((cur_post +1) & 0xffffff);
                                __threadfence();
                                // atomicCAS((unsigned int *)&cq_lock, 1, 0);
                                
                                *entry = 2;
                                // break;
                        //     }
                        // } 
                        // while (!isSet);
                        *cq_lock = 0;

                        // gpost_cont.cq_lock[qp_index] = 0;
                            // __threadfence();
                            
                            // atomicCAS((unsigned int *)&cqe64->op_own, 0, 240);
                            
                            
                            // atomicCAS((unsigned int *)&tlb_buffer[che], 1, 2);
                            
                            // printf("qp_index: %d gpost_cont.cq_lock[index]: %d\n", qp_index, gpost_cont.cq_lock[qp_index]);
                            // atomicCAS((unsigned int *)&gpost_cont.cq_lock[qp_index], 1, 0);
                            // __threadfence();
                            // sleep_d(1000);
                        
                    }
                    // wait_on_tlb;
                    // while(tlb_buffer[che] != 2);
                    
                // }
                // if(checkTLB(che, tlb_buffer) == 1){
                    // printf("id: %d tlb_buffer[%d]: %d\n", id, che, tlb_buffer[che]);
                    
                    while(*entry != 2);
                    // printf("id: %d tlb_buffer[%d]: %d\n", id, che, tlb_buffer[che]);
                    // __syncthreads();
                // }
                // while(tlb_buffer[che] != 2);
                // __syncthreads();
                // if(15728640)
                return device_buffer[index];
            // #else

            //     // }

            // #endif
            

        }

         // [] operator for rvalue
        __forceinline__
        __device__
        void rvalue(size_t index, T i){
            device_buffer[index] = i;
            int che = floor((double)index/request_size);
            // atomicCAS((unsigned int *)&tlb_buffer[che], tlb_buffer[che], 2);
            tlb_buffer[che] = 2;
        }


    // tlb entry update
};






// typedef struct rdma_buf rdma_buf;



// typedef struct rdma_tlb rdma_tlb;








#endif
