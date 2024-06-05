#include <stdio.h>

// extern "C"{
//   #include "rdma_utils.h"
// }

#include "rdma_utils.cuh"
#include <time.h>
#include "runtime.h"


// Size of array
#define N 1024*1024*150llu

// Kernel
__global__ void add_vectors_uvm(int *a, int *b, int *c, int size)
{
	int id = blockDim.x * blockIdx.x + threadIdx.x;
	// if(id < size) {
		c[id] = a[id] + b[id];
		// printf("c[%d]: %d\n", id, c[id]);
	// }
}


struct __attribute__((__packed__)) mlx5_tm_cqe {
	__be32		success;
	__be16		hw_phase_cnt;
	uint8_t		rsvd0[12];
};

struct __attribute__((__packed__)) ibv_tmh {
	uint8_t		opcode;      /* from enum ibv_tmh_op */
	uint8_t		reserved[3]; /* must be zero */
	__be32		app_ctx;     /* opaque user data */
	__be64		tag;
};

struct __attribute__((__packed__)) mlx5_cqe64 {
	union {
		struct {
			uint8_t		rsvd0[2];
			__be16		wqe_id;
			uint8_t		rsvd4[13];
			uint8_t		ml_path;
			uint8_t		rsvd20[4];
			__be16		slid;
			__be32		flags_rqpn;
			uint8_t		hds_ip_ext;
			uint8_t		l4_hdr_type_etc;
			__be16		vlan_info;
		};
		struct mlx5_tm_cqe tm_cqe;
		/* TMH is scattered to CQE upon match */
		struct ibv_tmh tmh;
	};
	__be32		srqn_uidx;
	__be32		imm_inval_pkey;
	uint8_t		app;
	uint8_t		app_op;
	__be16		app_info;
	__be32		byte_cnt;
	__be64		timestamp;
	__be32		sop_drop_qpn;
	__be16		wqe_counter;
	uint8_t		signature;
	uint8_t		op_own;
};

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


// __global__ void add_vectors_rdma(int *a, int *b, int *c, int size, \
//                                 uint8_t *tlb_A, uint8_t *tlb_B, uint8_t *tlb_C, struct post_content *post_cont1, struct poll_content *poll_cont1)
// {   

    

// 	int id = blockDim.x * blockIdx.x + threadIdx.x;
    
//     // if(id < blockDim.x) {
//         // if(id == 1023)
//         // printf("hello\n");

//         // post A[id] if tlb_A[blockIdx.x] == 0
//         // if(tlb_A[blockIdx.x] == 0)
//         //      post(post_cont(A[id]))

//         // post B[id] if tlb_B[blockIdx.x] == 0

//         // one thread out of the block should poll and
//         // update the tlb entry

//         // all other threads poll on the tlb array entry
//         // when the entry is 2 - on-device, computation continues
        
//         // printf("a[%d]: %d\n", id, a[id]);
//         int single_thread = threadIdx.x;
//         // if (tlb_A[blockIdx.x] == 0){
            
//             // __threadfence_system();
//             // printf("threadIdx.x: %d\n", threadIdx.x);
//             // __syncthreads();
//             // if (threadIdx.x == 0){
//                 // printf("a[%d]: %d\n", id, a[id]);
//                 struct post_content post_cont = *post_cont1; 
//                 struct poll_content poll_cont = *poll_cont1;
//                 struct ibv_wc wc;
//                 // configure work request here
//                 // default works fpr this request
//                 // post and poll using CQ
//                 struct post_wr wr;
//                 wr.qp_num = post_cont.qp_num;
//                 wr.wr_opcode = IBV_WR_RDMA_READ; // post_cont.wr_opcode;
//                 wr.wr_rdma_remote_addr = post_cont.wr_rdma_remote_addr;
//                 wr.wr_rdma_rkey = post_cont.wr_rdma_rkey;
//                 wr.wr_sg_addr = post_cont.wr_sg_addr;
//                 wr.wr_sg_length = 128*4; // post_cont.wr_sg_length;
//                 wr.wr_sg_lkey = post_cont.wr_sg_lkey;
//                 int cur_post = 0;
//                 // void *buf = post_cont.qp_buf + 8192*id;
//                 // void *reg = (void *) post_cont.bf_reg[id];
//                 // if(id == 0 || id == 516 || id == 1028){
//                 // int condition = (threadIdx.x == 0 && blockIdx.x == 0) || (threadIdx.x == 16 && blockIdx.x > 0 && blockIdx.x < 3);
                
//                 // if(id == 0 || id == 1){
//                 // // if(condition){
//                 //     cur_post = 0;
//                 //     printf("blockIdx.x1: %d a[%d]: %d\n",blockIdx.x, id, a[id]);
//                 //     printf("threadIdx.x1: %d a[%d]: %d\n",threadIdx.x, id, a[id]);
//                 //     printf("threadIdx.y1: %d a[%d]: %d\n",threadIdx.y, id, a[id]);
//                 //     printf("id: %d, a[%d]: %d cur_post: %d\n", id, id, a[id], cur_post);
//                 //     printf("qp_num: %d\n",post_cont.qp_num + id);
//                 //     printf("blockIdx.x: %d, post_cont.qp_buf + 8192*blockIdx.x: %p\n", blockIdx.x, post_cont.qp_buf + 8192*id);
//                 //     printf("blockIdx.x: %d, post_cont.wr_sg_addr + 896*blockIdx.x*4: %p\n", blockIdx.x, post_cont.wr_sg_addr + 512*id*4);
//                 //     printf("blockIdx.x: %d, post_cont.bf_reg[blockIdx.x]: %p\n", blockIdx.x, post_cont.bf_reg[id]);
//                 //     printf("blockIdx.x: %d, post_cont.wr_rdma_remote_addr + 896*blockIdx.x*4: %p\n", blockIdx.x, post_cont.wr_rdma_remote_addr + 896*id*4);

//                 //     post(post_cont.wr_rdma_remote_addr + 1*id*4 , post_cont.wr_rdma_rkey, 
//                 //         /*post_cont.wr_sg_length*/1*4, post_cont.wr_sg_lkey, post_cont.wr_sg_addr + 1*id*4, post_cont.wr_opcode, 
//                 //         post_cont.qp_num + id, cur_post, post_cont.qp_buf + 8192*id, (void *) post_cont.bf_reg[id]);
//                 //     // post_s(wr, cur_post, buf, reg);
//                 //     cur_post++;
                
//                 //     while(poll(poll_cont.cq_buf + 4096*id, &wc, (uint32_t *) poll_cont.cons_index[id], poll_cont.ibv_cqe, poll_cont.cqe_sz, 1, \ 
//                 //     (void *) poll_cont.cq_dbrec[id])==0);
//                 //     // __syncwarp();
//                 //     // __syncthreads();
//                 // }

                
                
//                 // if(threadIdx.x == 31){
//                 //     cur_post = 0;
//                 //     // printf("blockIdx.x1: %d a[%d]: %d\n",blockIdx.x, id, a[id]);
//                 //     // printf("threadIdx.x1: %d a[%d]: %d\n",threadIdx.x, id, a[id]);
//                 //     // printf("threadIdx.y1: %d a[%d]: %d\n",threadIdx.y, id, a[id]);
//                 //     // printf("id1 %d a[%d]: %d cur_post: %d\n", id, id, a[id], cur_post);
                    
//                 //     // printf("blockIdx.x: %d, post_cont.qp_buf + 8192*blockIdx.x: %p\n", blockIdx.x, post_cont.qp_buf + 8192*blockIdx.x);
//                 //     // printf("blockIdx.x: %d, post_cont.wr_sg_addr + 896*blockIdx.x*4: %p\n", blockIdx.x, post_cont.wr_sg_addr + 512*blockIdx.x*4);
//                 //     // printf("blockIdx.x: %d, post_cont.bf_reg[blockIdx.x]: %p\n", blockIdx.x, post_cont.bf_reg[blockIdx.x]);
//                 //     // printf("pblockIdx.x: %d, post_cont.wr_rdma_remote_addr + 896*blockIdx.x*4: %p\n", blockIdx.x, post_cont.wr_rdma_remote_addr + 896*blockIdx.x*4);
                    
//                 //     void *bf_reg = (void *) post_cont.bf_reg[0];
//                 //     void *cq_buf = poll_cont.cq_buf + 4096*0;
//                 //     uint32_t *cons_index = (uint32_t *) poll_cont.cons_index[0];
//                 //     void *cq_dbrec = (void *) poll_cont.cq_dbrec[0];
//                 //     printf("id: %d, blockIdx.x: %d, qp_num: %d\n",id, blockIdx.x, post_cont.qp_num);
//                 //     post_m(post_cont.wr_rdma_remote_addr, post_cont.wr_rdma_rkey, 
//                 //         1024, post_cont.wr_sg_lkey, post_cont.wr_sg_addr , post_cont.wr_opcode, 
//                 //         post_cont.qp_num, cur_post, post_cont.qp_buf, bf_reg, id);
//                 //     cur_post++;
                
//                 //     // while(poll(cq_buf, &wc, cons_index, poll_cont.ibv_cqe, poll_cont.cqe_sz, 1, cq_dbrec)==0);
//                 //     // tlb_A[0] = 3;
//                 //     // tlb_A[1] = 3;
//                 // }
                
//                 // if(threadIdx.x == 0){
//                     cur_post = 0;
//                     // printf("blockIdx.x1: %d a[%d]: %d\n",blockIdx.x, id, a[id]);
//                     // printf("threadIdx.x1: %d a[%d]: %d\n",threadIdx.x, id, a[id]);
//                     // printf("threadIdx.y1: %d a[%d]: %d\n",threadIdx.y, id, a[id]);
//                     // printf("id1 %d a[%d]: %d cur_post: %d\n", id, id, a[id], cur_post);
                    
//                     // printf("blockIdx.x: %d, post_cont.qp_buf + 8192*blockIdx.x: %p\n", blockIdx.x, post_cont.qp_buf + 8192*blockIdx.x);
//                     // printf("blockIdx.x: %d, post_cont.wr_sg_addr + 896*blockIdx.x*4: %p\n", blockIdx.x, post_cont.wr_sg_addr + 512*blockIdx.x*4);
//                     // printf("blockIdx.x: %d, post_cont.bf_reg[blockIdx.x]: %p\n", blockIdx.x, post_cont.bf_reg[blockIdx.x]);
//                     // printf("pblockIdx.x: %d, post_cont.wr_rdma_remote_addr + 896*blockIdx.x*4: %p\n", blockIdx.x, post_cont.wr_rdma_remote_addr + 896*blockIdx.x*4);
                    
//                     // post_cont.wr_rdma_remote_addr = post_cont.wr_rdma_remote_addr + blockDim.x*2*4;
//                     // post_cont.wr_sg_addr = post_cont.wr_sg_addr + blockDim.x*2*4;
//                     // post_cont.qp_buf = post_cont.qp_buf + 8192*1;
//                     // post_cont.qp_num = post_cont.qp_num + 1;
//                     int index = blockIdx.x;
//                     uint64_t remote_addr = post_cont.wr_rdma_remote_addr + /*blockDim.x*/1*4*index;
//                     uint64_t local_addr = post_cont.wr_sg_addr + blockDim.x*1*4*index;
//                     void *qp_buf = post_cont.qp_buf + 8192*index;
//                     uint32_t qp_num = post_cont.qp_num + index;
//                     void *bf_reg = (void *) post_cont.bf_reg[index];
//                     // void *cq_buf = poll_cont.cq_buf + 4096*index;
//                     // uint32_t *cons_index = (uint32_t *) poll_cont.cons_index[index];
//                     // void *cq_dbrec = (void *) poll_cont.cq_dbrec[index];
//                     uint32_t length = 4*blockDim.x;
//                     printf("id: %d, blockIdx.x: %d, qp_num: %d\n",id, index, qp_num);
//                     __syncthreads();
//                     // if(blockIdx.x == 0){
//                         // __syncthreads();
//                         // if (/*(threadIdx.x | threadIdx.y | threadIdx.z)*/ threadIdx.x == 0) {
//                             post_m(remote_addr, post_cont.wr_rdma_rkey, 
//                                 4, post_cont.wr_sg_lkey, local_addr, post_cont.wr_opcode, 
//                                 qp_num, cur_post, qp_buf, bf_reg, id);
//                     // }
//                     // else if(blockIdx.x == 1){
//                     //     // __syncthreads();
//                     //     // if (/*(threadIdx.x | threadIdx.y | threadIdx.z)*/ threadIdx.x == 0) {
//                     //         post_m(remote_addr, post_cont.wr_rdma_rkey, 
//                     //             8, post_cont.wr_sg_lkey, local_addr, post_cont.wr_opcode, 
//                     //             qp_num, cur_post, qp_buf, bf_reg, id);
//                     // }
//                     __syncthreads();
//                         // __syncthreads();
//                             //    tlb_A[blockIdx.x] = 3;
//                     // }
//                     // else while(tlb_A[blockIdx.x] != 3); 
//                     // __threadfence_system();
//                     // while(tlb_A[blockIdx.x] != 3);
//                     // __threadfence_block();
//                     // else {
//                     //     while(tlb_A[blockIdx.x] != 3);
//                     // }
//                     // cur_post++;
                
//                     // while(poll(cq_buf, &wc, cons_index, poll_cont.ibv_cqe, poll_cont.cqe_sz, 1, cq_dbrec)==0);
//                     // tlb_A[2] = 3;
//                     // tlb_A[3] = 3;
//                 // }
//                 // __syncthreads();
//                 // if(id == 514){
//                 //     cur_post = 0;
//                 //     // printf("blockIdx.x1: %d a[%d]: %d\n",blockIdx.x, id, a[id]);
//                 //     // printf("threadIdx.x1: %d a[%d]: %d\n",threadIdx.x, id, a[id]);
//                 //     // printf("threadIdx.y1: %d a[%d]: %d\n",threadIdx.y, id, a[id]);
//                 //     // printf("id1 %d a[%d]: %d cur_post: %d\n", id, id, a[id], cur_post);
                    
//                 //     // printf("blockIdx.x: %d, post_cont.qp_buf + 8192*blockIdx.x: %p\n", blockIdx.x, post_cont.qp_buf + 8192*blockIdx.x);
//                 //     // printf("blockIdx.x: %d, post_cont.wr_sg_addr + 896*blockIdx.x*4: %p\n", blockIdx.x, post_cont.wr_sg_addr + 512*blockIdx.x*4);
//                 //     // printf("blockIdx.x: %d, post_cont.bf_reg[blockIdx.x]: %p\n", blockIdx.x, post_cont.bf_reg[blockIdx.x]);
//                 //     // printf("pblockIdx.x: %d, post_cont.wr_rdma_remote_addr + 896*blockIdx.x*4: %p\n", blockIdx.x, post_cont.wr_rdma_remote_addr + 896*blockIdx.x*4);

//                 //     // post_cont.wr_rdma_remote_addr = post_cont.wr_rdma_remote_addr + blockDim.x*4*4;
//                 //     // post_cont.wr_sg_addr = post_cont.wr_sg_addr + blockDim.x*4*4;
//                 //     // post_cont.qp_buf = post_cont.qp_buf + 8192*10;
//                 //     // post_cont.qp_num = post_cont.qp_num + 10;
//                 //     uint64_t remote_addr = post_cont.wr_rdma_remote_addr + blockDim.x*4*4;
//                 //     uint64_t local_addr = post_cont.wr_sg_addr + blockDim.x*4*4;
//                 //     void *qp_buf = post_cont.qp_buf + 8192*2;
//                 //     uint32_t qp_num = post_cont.qp_num + 2;
//                 //     void *bf_reg = (void *) post_cont.bf_reg[2];
//                 //     void *cq_buf = poll_cont.cq_buf + 4096*2;
//                 //     uint32_t *cons_index = (uint32_t *) poll_cont.cons_index[2];
//                 //     void *cq_dbrec = (void *) poll_cont.cq_dbrec[2];
//                 //     printf("id: %d, blockIdx.x: %d, qp_num: %d\n",id, blockIdx.x, qp_num);
//                 //     post_m(remote_addr, post_cont.wr_rdma_rkey, 
//                 //         1024, post_cont.wr_sg_lkey, local_addr, post_cont.wr_opcode, 
//                 //         qp_num, cur_post, qp_buf, bf_reg, id);
//                 //     // cur_post++;
                
//                 //     // while(poll(cq_buf, &wc, cons_index, poll_cont.ibv_cqe, poll_cont.cqe_sz, 1, cq_dbrec)==0);
//                 //     // tlb_A[4] = 3;
//                 //     // tlb_A[5] = 3;
//                 // }
//                 // // __syncthreads();
//                 // if(id == 780){
//                 //     cur_post = 0;
//                 //     // printf("blockIdx.x1: %d a[%d]: %d\n",blockIdx.x, id, a[id]);
//                 //     // printf("threadIdx.x1: %d a[%d]: %d\n",threadIdx.x, id, a[id]);
//                 //     // printf("threadIdx.y1: %d a[%d]: %d\n",threadIdx.y, id, a[id]);
//                 //     // printf("id1 %d a[%d]: %d cur_post: %d\n", id, id, a[id], cur_post);
                    
//                 //     // printf("blockIdx.x: %d, post_cont.qp_buf + 8192*blockIdx.x: %p\n", blockIdx.x, post_cont.qp_buf + 8192*blockIdx.x);
//                 //     // printf("blockIdx.x: %d, post_cont.wr_sg_addr + 896*blockIdx.x*4: %p\n", blockIdx.x, post_cont.wr_sg_addr + 512*blockIdx.x*4);
//                 //     // printf("blockIdx.x: %d, post_cont.bf_reg[blockIdx.x]: %p\n", blockIdx.x, post_cont.bf_reg[blockIdx.x]);
//                 //     // printf("pblockIdx.x: %d, post_cont.wr_rdma_remote_addr + 896*blockIdx.x*4: %p\n", blockIdx.x, post_cont.wr_rdma_remote_addr + 896*blockIdx.x*4);

//                 //     // post_cont.wr_rdma_remote_addr = post_cont.wr_rdma_remote_addr + blockDim.x*6*4;
//                 //     // post_cont.wr_sg_addr = post_cont.wr_sg_addr + blockDim.x*6*4;
//                 //     // post_cont.qp_buf = post_cont.qp_buf + 8192*3;
//                 //     // post_cont.qp_num = post_cont.qp_num + 3;
//                 //     uint64_t remote_addr = post_cont.wr_rdma_remote_addr + blockDim.x*6*4;
//                 //     uint64_t local_addr = post_cont.wr_sg_addr + blockDim.x*6*4;
//                 //     void *qp_buf = post_cont.qp_buf + 8192*3;
//                 //     uint32_t qp_num = post_cont.qp_num + 3;
//                 //     void *bf_reg = (void *) post_cont.bf_reg[3];
//                 //     void *cq_buf = poll_cont.cq_buf + 4096*3;
//                 //     uint32_t * cons_index = (uint32_t *) poll_cont.cons_index[3];
//                 //     void *cq_dbrec = (void *) poll_cont.cq_dbrec[3];
//                 //     printf("id: %d, blockIdx.x: %d, qp_num: %d\n",id, blockIdx.x, qp_num);
//                 //     post_m(remote_addr, post_cont.wr_rdma_rkey, 
//                 //         1024, post_cont.wr_sg_lkey, local_addr, post_cont.wr_opcode, 
//                 //         qp_num, cur_post, qp_buf, bf_reg, id);
//                 //     // cur_post++;
                
//                 //     // while(poll(cq_buf, &wc, cons_index, poll_cont.ibv_cqe, poll_cont.cqe_sz, 1, cq_dbrec)==0);
//                 //     // tlb_A[6] = 3;
//                 //     // tlb_A[7] = 3;
//                 // }

//                 // if(id == 1027){
//                 //     cur_post = 0;
//                 //     // printf("blockIdx.x1: %d a[%d]: %d\n",blockIdx.x, id, a[id]);
//                 //     // printf("threadIdx.x1: %d a[%d]: %d\n",threadIdx.x, id, a[id]);
//                 //     // printf("threadIdx.y1: %d a[%d]: %d\n",threadIdx.y, id, a[id]);
//                 //     // printf("id1 %d a[%d]: %d cur_post: %d\n", id, id, a[id], cur_post);
                    
//                 //     // printf("blockIdx.x: %d, post_cont.qp_buf + 8192*blockIdx.x: %p\n", blockIdx.x, post_cont.qp_buf + 8192*blockIdx.x);
//                 //     // printf("blockIdx.x: %d, post_cont.wr_sg_addr + 896*blockIdx.x*4: %p\n", blockIdx.x, post_cont.wr_sg_addr + 512*blockIdx.x*4);
//                 //     // printf("blockIdx.x: %d, post_cont.bf_reg[blockIdx.x]: %p\n", blockIdx.x, post_cont.bf_reg[blockIdx.x]);
//                 //     // printf("pblockIdx.x: %d, post_cont.wr_rdma_remote_addr + 896*blockIdx.x*4: %p\n", blockIdx.x, post_cont.wr_rdma_remote_addr + 896*blockIdx.x*4);

//                 //     uint64_t remote_addr = post_cont.wr_rdma_remote_addr + blockDim.x*8*4;
//                 //     uint64_t local_addr = post_cont.wr_sg_addr + blockDim.x*8*4;
//                 //     void *qp_buf = post_cont.qp_buf + 8192*4;
//                 //     uint32_t qp_num = post_cont.qp_num + 4;
//                 //     // post_cont.wr_rdma_remote_addr = post_cont.wr_rdma_remote_addr + blockDim.x*8*4;
//                 //     // post_cont.wr_sg_addr = post_cont.wr_sg_addr + blockDim.x*8*4;
//                 //     // post_cont.qp_buf = post_cont.qp_buf + 8192*4;
//                 //     // post_cont.qp_num = post_cont.qp_num + 4;
//                 //     void *bf_reg = (void *) post_cont.bf_reg[4];
//                 //     void *cq_buf = poll_cont.cq_buf + 4096*4;
//                 //     uint32_t * cons_index = (uint32_t *) poll_cont.cons_index[4];
//                 //     void *cq_dbrec = (void *) poll_cont.cq_dbrec[4];
//                 //     printf("id: %d, blockIdx.x: %d, qp_num: %d\n",id, blockIdx.x, qp_num);
//                 //     post_m(remote_addr, post_cont.wr_rdma_rkey, 
//                 //         1024, post_cont.wr_sg_lkey, local_addr, post_cont.wr_opcode, 
//                 //         qp_num, cur_post, qp_buf, bf_reg, id);
//                 //     // cur_post++;
                
//                 //     // while(poll(cq_buf, &wc, cons_index, poll_cont.ibv_cqe, poll_cont.cqe_sz, 1, cq_dbrec)==0);
//                 //     // tlb_A[8] = 3;
//                 //     // tlb_A[9] = 3;
//                 // }

//                 // __syncwarp();
                
//                 // printf("blockIdx.x: %d a[%d]: %d\n", blockIdx.x, id, a[id]);
//                 // printf("blockIdx.x: %d a[%d]: %d\n", blockIdx.x, id, a[id]);
//                 // printf("blockIdx.x: %d a[%d]: %d\n", blockIdx.x, id, a[id]);
//                 // // printf("a[%d]: %d\n", id, a[id]);
//                 // // printf("a[%d]: %d\n", id, a[id]);
//                 // printf("blockIdx.x: %d a[%d]: %d\n", blockIdx.x, id, a[id]);
//                 // __threadfence_system();
//                 // cur_post++;
//                 // __syncwarp();
//                 // while(tlb_A[blockIdx.x] == 0);
//                 __syncthreads();
//                 for(int del = 0; del < 20000000; del++);
//                 __syncthreads();    
//             // }
            
//         // for(int del = 0; del < 20000000; del++);
            
            
//         // }
        
//         // __threadfence_system();
        
//         printf("a[%d]: %d\n", id, a[id]);
//         // __syncthreads();
//         // if(threadIdx.x != 0)
//             // for(int del = 0; del < 10000000; del++);
//         // __threadfence_system();
//         // __syncthreads();
//         // if(a[id] == 2)
//         //     printf("a[%d]: %d\n", id, a[id]);

//         // if(id = 0){
//         //         // printf("blockIdx.x: %d a[%d]: %d\n",blockIdx.x, id, a[id]);
//         //         // printf("a[%d]: %d\n", id, a[id]);
//         //         // printf("a[%d]: %d\n", id, a[id]);
//         //         // printf("a[%d]: %d\n", id, a[id]);
//         //         struct poll_content poll_cont = *poll_cont1;
//         //         struct ibv_wc wc;
//         //         int n = 256;
//         //         do{
//         //             while(poll(poll_cont.cq_buf, &wc, (uint32_t *) poll_cont.cons_index[blockIdx.x], poll_cont.ibv_cqe, poll_cont.cqe_sz, 1, \ 
//         //              (void *) poll_cont.cq_dbrec[blockIdx.x])==0);
//         //             n--;
//         //         }while(n);

//         //         // while(poll(poll_cont.cq_buf + 4096*blockIdx.x, &wc, (uint32_t *) poll_cont.cons_index[blockIdx.x], poll_cont.ibv_cqe, poll_cont.cqe_sz, 1, \ 
//         //         //      (void *) poll_cont.cq_dbrec[blockIdx.x])==0);

//         //         // void *cqe;
//         //         // struct mlx5_cqe64 *cqe64;
//         //         // uint32_t cons_index_dev = *(uint32_t *) poll_cont.cons_index[blockIdx.x];
//         //         // cqe = poll_cont.cq_buf /*+ 4096*blockIdx.x*/ + (cons_index_dev & poll_cont.ibv_cqe) * poll_cont.cqe_sz;
//         //         // printf("blockIdx.x: %d poll_cont.cq_buf: 0x%llx\n", blockIdx.x, poll_cont.cq_buf + 4096*blockIdx.x);
                
//         //         // cqe64 = (struct mlx5_cqe64 *)((poll_cont.cqe_sz == 64) ? cqe : cqe + 64);
//         //         // // ((cqe64->op_own != 240) && !((cqe64->op_own & 1) ^ !!(1 & (poll_cont.ibv_cqe + 1))))==0
//         //         // while(cqe64->op_own == 240);
                
//         //         // uint32_t *gpu_dbrec = (uint32_t *) poll_cont.cq_dbrec[blockIdx.x];
//         //         // (*(uint32_t *) poll_cont.cons_index[blockIdx.x])++;
//         //         // gpu_dbrec[0] = htonl((cons_index_dev) & 0xffffff);

//         //         // __threadfence_system();
//         //         // __syncthreads();
//         // //         // update tlb_A[blockIdx.x] = 2
//         //         tlb_A[blockIdx.x] = 3;
//         //         // __threadfence_system();
//         // //         cur_post++;
//         //         // printf("a[%d]: %d\n", id, a[id]);
            
            
            
            
//         //     // __threadfence_system();
            
//         // }
//         // while(tlb_A[blockIdx.x] == 0);
//         // __syncthreads();
//         // printf("a[%d]: %d\n", id, a[id]);
//         // if(a[id] == 2)
//         // printf("a[%d]: %d\n", id, a[id]);
//         // if (tlb_B[blockIdx.x] == 0){
//         //     if (threadIdx.x == blockDim.x - 1){
//         //         struct ibv_wc wc;
//         //         struct post_content post_cont = *post_cont1; 
//         //         struct poll_content poll_cont = *poll_cont1;
//         //         // configure work request here
//         //         // default works fpr this request
//         //         // post and poll using CQ
//         //         post(post_cont.wr_rdma_remote_addr + blockDim.x*blockIdx.x*4 + size, post_cont.wr_rdma_rkey, \
//         //              post_cont.wr_sg_length, post_cont.wr_sg_lkey, post_cont.wr_sg_addr + blockDim.x*blockIdx.x*4 + size, post_cont.wr_opcode, \
//         //              post_cont.qp_num + blockIdx.x, cur_post, post_cont.qp_buf + 8192*blockIdx.x, (void *) post_cont.bf_reg[blockIdx.x]);
//         //         poll(poll_cont.cq_buf + 4096*blockIdx.x, &wc,  (uint32_t *) poll_cont.cons_index[blockIdx.x], poll_cont.ibv_cqe, poll_cont.cqe_sz, 1, \ 
//         //              (void *) poll_cont.cq_dbrec[blockIdx.x]);
//         //         // update tlb_A[blockIdx.x] = 2
//         //         tlb_B[blockIdx.x] = 2;
//         //         cur_post++;
//         //         printf("a[%d]: %d\n", id, b[id]);
//         //     }
//         //     else while(!tlb_B[blockIdx.x]);
//         // }
        
//         // if(tlb_A[blockIdx.x] == 0)
//         // __syncthreads();
// 		// c[id] = a[id] + b[id];
// 		// printf("c[%d]: %d\n", id, c[id]);
// 	// }
    
// }

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

__global__ void read_nonstop(int *a, int size){
    int i = 0;
    int stop = 0;
    while(1){
        for(i = 0; i < size; i++)
            if(a[i] == 3){
                printf("a[%d]: %d\n", i, a[i]);
                stop = 1;
                break;
            }
        if(stop) break;
    }
}

__global__ void read(int *a, int index){
    printf("a[%d]: %d\n", index, a[index]);
}

__global__ void write(int *a, int index, int number){
    a[index] = number;
    printf("a[%d]: %d\n", index, a[index]);
}

int alloc_global_cont(struct post_content *post_cont, struct poll_content *poll_cont){
    struct post_content *d_post;
    struct poll_content *d_poll;

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

    alloc_content<<<1,1>>>(d_post, d_poll);
    alloc_global_content<<<1,1>>>(d_post, d_poll);
    ret0 = cudaDeviceSynchronize();
    if(ret0 != cudaSuccess){
        printf("Error on alloc_content!\n");
        return -1;
    }
    return 0;
}

__global__ void test(rdma_buf<int> *a, rdma_buf<int> *b, rdma_buf<int> *c){
    // rdma_buf<int> A = *a;
    // rdma_buf<int> B = *b;
    int id = blockDim.x * blockIdx.x + threadIdx.x;
    // if(id == 0) printf("a->size: %d\n", a->size);
    // if(id==0) ( *a)[id] = 0;
    // if(id == 15728640)
    //     printf("b[15728640]: %d a[15728640]: %d\n", (*a)[id], (*b)[id]);

    for(int i = id; i < 16777216; i += 524288){
        c->rvalue(i, (*a)[i] + (*b)[i]);
    }

    // if(id < a->size){
    //     c->rvalue(id, (*a)[id] + (*b)[id]);

    // }  
    
    // c[id] = (*a)[id] + (*b)[id];
    // printf("buf1[2]: %d buf1->address: %p, buf1->size: %d, REQUEST_SIZE12: %d\n", (*buf1)[2], buf1->gpu_address, buf1->size, 1);
}

__global__ void test2(rdma_buf<int> *a, rdma_buf<int> *b, rdma_buf<int> *c){
    int id = blockDim.x * blockIdx.x + threadIdx.x;

    int k = (*a)[id]; // + (*b)[id];

    // c->rvalue(id, (*a)[id] + (*b)[id]); 
}

// Main program
int main(int argc, char **argv)
{   
    if (argc != 7)
        usage(argv[0]);
    // else
    //     usage(argv[0]);
    init_gpu(0);
    printf("Function: %s line number: %d 1024MB: %d bytes REQUEST_SIZE: %d\n",__func__, __LINE__, MB(1024), REQUEST_SIZE);
    int num_msg = (unsigned long) atoi(argv[4]);
    int mesg_size = (unsigned long) atoi(argv[5]);
    int num_bufs = (unsigned long) atoi(argv[6]);

    struct context *s_ctx = (struct context *)malloc(sizeof(struct context));
    struct post_content post_cont, *d_post;
    struct poll_content poll_cont, *d_poll;

    int num_iteration = num_msg;
    s_ctx->n_bufs = num_bufs;
    s_ctx->gpu_buf_size = N*sizeof(int)*3;

    int ret = connect(argv[2], s_ctx);
    ret = prepare_post_poll_content(s_ctx, &post_cont, &poll_cont);


    int a[100], *b;
    a[2] = 5;
    // rdma_buf<int> buf((uint64_t) s_ctx->gpu_buffer, (uint64_t) s_ctx->server_mr.addr, 100);
    rdma_buf<int> *buf1, *buf2, *buf3;
    cudaMallocManaged(&buf1, sizeof(rdma_buf<int>));
    cudaMallocManaged(&buf2, sizeof(rdma_buf<int>));
    cudaMallocManaged(&buf3, sizeof(rdma_buf<int>));
    // cudaMallocManaged(&b, sizeof(int)*100);
    // printf("s_ctx->gpu_buffer: %p, buf1->size: %d, Address_Offset: %d\n", s_ctx->gpu_buffer, buf1->size, Address_Offset);
    buf1->start((uint64_t) s_ctx->gpu_buffer, (uint64_t) s_ctx->server_mr.addr, N*sizeof(int));
    // printf("s_ctx->gpu_buffer: %p, buf1->size: %d, Address_Offset: %d\n", s_ctx->gpu_buffer, buf1->size, Address_Offset);
    buf2->start((uint64_t) s_ctx->gpu_buffer, (uint64_t) s_ctx->server_mr.addr, N*sizeof(int));
    buf3->start((uint64_t) s_ctx->gpu_buffer, (uint64_t) s_ctx->server_mr.addr, N*sizeof(int));
    // buf1->address = (uint64_t) s_ctx->gpu_buffer;
    // buf1->size = 100;
    printf("buf1->address: %p, buf1->size: %d, REQUEST_SIZE: %d buf1->host_address : %p\n", buf1->gpu_address, buf1->size, REQUEST_SIZE, buf1->host_address);
    printf("buf2->address: %p, buf2->size: %d, REQUEST_SIZE: %d buf2->host_address : %p\n", buf2->gpu_address, buf2->size, REQUEST_SIZE, buf2->host_address);
    // cudaMemcpy(buf1, &buf, sizeof(rdma_buf<int>), cudaMemcpyHostToDevice);
    // printf("buf[2]: %d a: %p\n", buf[2], a);
    printf("Function name: %s, line number: %d mesg_size: %d num_iteration: %d sizeof(int): %d\n", __func__, __LINE__, mesg_size, num_msg, sizeof(int));
    // exit(0);
    uint8_t access_size = sizeof(int);
    size_t bytes = N*sizeof(int);
    void *A = (void *) s_ctx->gpu_buffer;
    void *B = (void *) s_ctx->gpu_buffer + bytes;
    void *C = (void *) s_ctx->gpu_buffer + 2*bytes;
    // int *C_dev;
    int *h_array = (int *) malloc(bytes);
    for(int i = 0; i < bytes/sizeof(int); i++)
        h_array[i] = 0;

    // allocate poll and post content
    alloc_global_cont(&post_cont, &poll_cont);
    

    cudaError_t rtr1 = cudaMemcpy(A, h_array, bytes, cudaMemcpyHostToDevice);
    cudaError_t rtr2 = cudaMemcpy(B, h_array, bytes, cudaMemcpyHostToDevice);
    cudaError_t rtr3 = cudaMemcpy(C, h_array, bytes, cudaMemcpyHostToDevice);
    if(rtr1 != cudaSuccess || rtr2 != cudaSuccess || rtr3 != cudaSuccess){
        printf("Error on array copy! line: %d\n", __LINE__);
        return -1;
    }

    int thr_per_blk = 2048*2; // s_ctx->n_bufs;
	int blk_in_grid = 256;

    // int thr_per_blk = s_ctx->n_bufs;
	// int blk_in_grid = 512;
    

    // Allocate TLB for array A
    uint8_t *tlb_A, *tlb_B, *tlb_C, *h_tlb;
    int tlb_size = bytes/(64*1024); // divided by access size //16*1024*1024/(64*1024); // thr_per_blk;
    cudaError_t ret1 = cudaMalloc((void **)&tlb_A, tlb_size*sizeof(uint8_t));
    cudaError_t ret2 = cudaMalloc((void **)&tlb_B, tlb_size*sizeof(uint8_t));
    cudaError_t ret3 = cudaMalloc((void **)&tlb_C, tlb_size*sizeof(uint8_t));
    if(ret1 != cudaSuccess || ret2 != cudaSuccess || ret3 != cudaSuccess){
        printf("Error on allocation TLB!\n");
        return -1;
    }
    h_tlb = (uint8_t *) malloc(tlb_size*sizeof(uint8_t));
    for (int i = 0; i < tlb_size; i++) h_tlb[i] = 0;
    ret1 = cudaMemcpy(tlb_A, h_tlb, tlb_size*sizeof(uint8_t), cudaMemcpyHostToDevice);
    ret2 = cudaMemcpy(tlb_B, h_tlb, tlb_size*sizeof(uint8_t), cudaMemcpyHostToDevice);
    ret3 = cudaMemcpy(tlb_C, h_tlb, tlb_size*sizeof(uint8_t), cudaMemcpyHostToDevice);
    if(ret1 != cudaSuccess || ret2 != cudaSuccess || ret3 != cudaSuccess){
        printf("Error on allocation TLB!\n");
        return -1;
    }

    
    


            // // Allocate memory for arrays d_A, d_B, and d_C on device
            // int *d_A, *d_B, *d_C;
            // cudaError_t state;
            // state = cudaMallocManaged(&d_A, bytes);
            // if(cudaSuccess != state){
            // 	printf("error on cudaMallocManaged(&d_A, bytes): %d\n", state);
            // }
            // state = cudaMallocManaged(&d_B, bytes);
            // if(cudaSuccess != state){
            // 	printf("error on cudaMallocManaged(&d_B, bytes): %d\n", state);
            // }
            // state = cudaMallocManaged(&d_C, bytes);
            // if(cudaSuccess != state){
            // 	printf("error on cudaMallocManaged(&d_C, bytes): %d\n", state);
            // }
            // printf("line number %d\n", __LINE__);
            // // Fill host arrays A and B
            // for(int i=0; i<N; i++)
            // {
            // 	d_A[i] = 1.0;
            // 	d_B[i] = 2.0;
            // }
   
        

    

    int *dev_a, *dev_b, *dev_c, *host_a;                      // 107374182
    // ret1 = cudaMalloc((void **)&dev_a, thr_per_blk*blk_in_grid*sizeof(int));
    // ret2 = cudaMalloc((void **)&dev_b, thr_per_blk*blk_in_grid*sizeof(int));
    // ret3 = cudaMalloc((void **)&dev_c, thr_per_blk*blk_in_grid*sizeof(int));
    // host_a = (int *) malloc(thr_per_blk*blk_in_grid*sizeof(int));
    // if(ret1 != cudaSuccess || ret2 != cudaSuccess || ret3 != cudaSuccess){
    //     printf("cuda error: %s, %d\n", __func__, __LINE__);
    // }

    // for(int i = 0; i < thr_per_blk*blk_in_grid; i++) host_a[i] = 2;
    // ret2 = cudaMemcpy(dev_b, host_a, thr_per_blk*blk_in_grid*sizeof(int), cudaMemcpyHostToDevice);
    // ret3 = cudaMemcpy(dev_b, host_a, thr_per_blk*blk_in_grid*sizeof(int), cudaMemcpyHostToDevice);
    // if(ret2 != cudaSuccess || ret3 != cudaSuccess){
    //     printf("cuda error: %s, %d\n", __func__, __LINE__);
    // }

    
    printf("thr_per_blk: %d, blk_in_grid: %d tlb_size: %d a: %p\n", thr_per_blk, blk_in_grid, tlb_size, A);

    int timer_size = 4;
    clock_t *dtimer = NULL;
	clock_t timer[thr_per_blk*timer_size];

    if (cudaSuccess != cudaMalloc((void **)&dtimer, sizeof(clock_t) * timer_size * thr_per_blk)) 
	{
        printf("Error on timer allocation!\n");
        return -1;
    }

    // Launch kernel
    ret1 = cudaDeviceSynchronize();
    printf("ret: %d\n", ret1);
    if(cudaSuccess != ret1){    
        return -1;
    }

    cudaEvent_t event1, event2;
    cudaEventCreate(&event1);
    cudaEventCreate(&event2);
    int data_size = mesg_size;


    struct timespec start, finish, delta;
    clock_gettime(CLOCK_REALTIME, &start);
    cudaEventRecord(event1, (cudaStream_t)1); //where 0 is the default stream
    
    // add_vectors_uvm<<< thr_per_blk,sblk_in_grid >>>(dev_a, dev_a, dev_c, thr_per_blk*blk_in_grid);
    // add_vectors_rdma_64MB_512KB<<< thr_per_blk, blk_in_grid>>>((int *) A, (int *) B, (int *) C, bytes/sizeof(int), tlb_A, tlb_B, tlb_C, dtimer, data_size, num_iteration);
    // add_vectors_rdma<<< thr_per_blk, blk_in_grid>>>((int *) A, (int *) B, (int *) C, bytes/sizeof(int), tlb_A, tlb_B, tlb_C, dtimer, /*d_post, d_poll,*/ data_size, num_iteration);
    // test<<<2048*16, 512>>>(buf1, buf2, (int *) C);
    // test<<<2048, 256>>>(buf1, buf2, buf3);
    test2<<< 4096, 1024>>>(buf1, buf2, buf3);
    cudaEventRecord(event2, (cudaStream_t) 1);
    clock_gettime(CLOCK_REALTIME, &finish);
    ret1 = cudaDeviceSynchronize();
    
    
    //synchronize
    cudaEventSynchronize(event1); //optional
    cudaEventSynchronize(event2); //wait for the event to be executed!

    //calculate time
    float dt_ms;
    cudaEventElapsedTime(&dt_ms, event1, event2);
    sub_timespec(start, finish, &delta);
    
        // read<<<1,1>>>(dev_array, 0);
        // write<<<1,1>>>(dev_array, 71808, 3);
        // read<<<1,1>>>(dev_array2, 71808-256);
        // read_nonstop<<<1,1>>>(dev_array, 256);

        // add_vectors_uvm<<< blk_in_grid, thr_per_blk >>>(d_A, d_B, d_C, bytes);
	
    printf("ret1: %d\n", ret1);
    if(cudaSuccess != ret1){
        return -1;
    }
    rtr3 = cudaMemcpy(timer, dtimer, sizeof(clock_t)*timer_size*thr_per_blk, cudaMemcpyDeviceToHost);
    if(rtr3 != cudaSuccess){
        printf("Error on array copy!\n");
        return -1;
    }

    clock_t cycles;
    float g_usec_post;
    cudaDeviceProp devProp;
    cudaGetDeviceProperties(&devProp, 0);
    printf("Cuda device clock rate = %d\n", devProp.clockRate);
    float freq_post = (float)1/((float)devProp.clockRate*1000), max = 0;
    printf("timer: \n");
    float div, sum_div = 0, sum_time = 0;
    // for(int i = 0; i < thr_per_blk; i++){
    //     cycles = timer[timer_size*i+1] - timer[i*timer_size];
	//     g_usec_post = (float)((float)1/(devProp.clockRate*1000))*((cycles)) * 1000000;
    //     // if (max < g_usec_post) max = g_usec_post;
    //     printf("Posting - blockIdx.x: %d: %f \n", i, g_usec_post);
    //     // div = dt_ms/(g_usec_post/1000);
    //     // sum_div += div;
    //     // sum_time += g_usec_post;
    //     cycles = timer[timer_size*i+3] - timer[i*timer_size+2];
	//     g_usec_post = (float)((float)1/(devProp.clockRate*1000))*((cycles)) * 1000000;
    //     // if (max < g_usec_post) max = g_usec_post;
    //     printf("polling - blockIdx.x: %d: %f div: %f\n", i, g_usec_post, div);
    //     // cycles = timer[timer_size*i+3] - timer[i*timer_size+2];
	//     // g_usec_post = (float)((float)1/(devProp.clockRate*1000))*((cycles)) * 1000000;
    //     if (max < g_usec_post) max = g_usec_post;
    //     // printf("Array B - blockIdx.x: %d: %f \n", i, g_usec_post);
    // }

    sum_div = sum_div/thr_per_blk;
    sum_time = sum_time/thr_per_blk/1000;
    printf("\nmax: %f\n", max);
    printf("\nsum_div: %f sum_time: %f total time: %f\n", sum_div, sum_time, sum_time*sum_div);
    printf("BW: %f GBps for data size: %d\n", (float)(thr_per_blk)*data_size*4*num_iteration/(dt_ms*0.001*1024*1024*1024), data_size*4);
    clock_t max1 = timer[1];
    clock_t min = timer[0];
    for(int i = 0; i < thr_per_blk; i++){
        if(max1 < timer[timer_size*i+1]) max1 = timer[timer_size*i + 1];
        if(min > timer[timer_size*i]) min = timer[timer_size*i];
        // cycles = timer[2*i+1] - timer[i*2];
	    // g_usec_post = (float)((float)1/(devProp.clockRate*1000))*((cycles)) * 1000000;
        // if (max < g_usec_post) max = g_usec_post;
        // printf("blockIdx.x: %d: %f \n", i, g_usec_post);
    }
    cycles = max1 - min;
    g_usec_post = (float)((float)1/(devProp.clockRate*1000))*((cycles)) * 1000000;
    printf("total timer: %f\n", g_usec_post);
    printf("kernel time: %d.%.9ld dt_ms: %f\n", (int)delta.tv_sec, delta.tv_nsec, dt_ms);



    // for(int i = 0; i < 512*100; i++)
    //     printf(" %d ", h_array[i]);
    // printf("\n");
    // sleep(5);
    rtr3 = cudaMemcpy(h_array, A, bytes, cudaMemcpyDeviceToHost);
    if(rtr3 != cudaSuccess){
        printf("Error on array copy of A to host!\n");
        return -1;
    }
    printf("H_array: \n");
    for(int i = 0; i < bytes/4; i++){
        if(h_array[i] != 2){ 
            if(i>0 && h_array[i-1] == 2){
                printf("start: A[%d]: %d qp: %d\n", i, h_array[i], (i/(REQUEST_SIZE/4))%15);
            }
            else if(i == 0){
                printf("start: A[%d]: %d qp: %d\n", i, h_array[i], (i/(REQUEST_SIZE/4))%15);
            }
            else if(h_array[i+1] == 2){
                printf("end: A[%d]: %d qp: %d\n", i, h_array[i], (i/(REQUEST_SIZE/4))%15);
            }
           
        }
    }
    printf("----------------------\n");
    rtr3 = cudaMemcpy(h_array, B, bytes, cudaMemcpyDeviceToHost);
    if(rtr3 != cudaSuccess){
        printf("Error on array copy of B to host!\n");
        return -1;
    }
    for(int i = 0; i < bytes/4; i++){
        if(h_array[i] != 2){ 
            if((i>0 && h_array[i-1] == 2) || (i == 0)){
                    printf("start: B[%d]: %d qp: %d\n", i, h_array[i], (i/16384)%15);
                }
                else if(h_array[i+1] == 2){
                    printf("end: B[%d]: %d qp: %d\n", i, h_array[i], (i/16384)%15);
                }
        }
    }
    rtr3 = cudaMemcpy((void *) h_array, C, bytes, cudaMemcpyDeviceToHost);
    if(rtr3 != cudaSuccess){
        printf("Error on array copy of C to host!\n");
        return -1;
    }
    for(int i = 0; i < bytes/4; i++){
        if(h_array[i] != 4){ 
            printf("error in C: C[%d]: %d\n", i, h_array[i]);
            break;
        }
    }
    // printf("C[0]: %d\n", h_array[0]);
    // printf("C[1]: %d\n", h_array[1]);
    // printf("C[49151]: %d\n", h_array[49151]);
    // printf("C[524287]: %d\n", h_array[524287]);
    // printf("C[524288]: %d\n", h_array[524288]);
    // printf("C[524289]: %d\n", h_array[524289]);
    // printf("C[600000]: %d\n", h_array[600000]);
    // printf("C: %p\n", C);
    printf("\n");

    // delay(40);
	// // Number of bytes to allocate for N doubles
	// size_t bytes = N*sizeof(int);
	// printf("size: %d GB\n", sizeof(int)/4);
	// // Allocate memory for arrays A, B, and C on host
	// // int *A = (int*)malloc(bytes);
	// // int *B = (int*)malloc(bytes);
	// // int *C = (int*)malloc(bytes);

	// // Allocate memory for arrays d_A, d_B, and d_C on device
	// int *d_A, *d_B, *d_C;
	// cudaError_t state;
	// state = cudaMallocManaged(&d_A, bytes);
	// if(cudaSuccess != state){
	// 	printf("error on cudaMallocManaged(&d_A, bytes): %d\n", state);
	// }
	// state = cudaMallocManaged(&d_B, bytes);
	// if(cudaSuccess != state){
	// 	printf("error on cudaMallocManaged(&d_B, bytes): %d\n", state);
	// }
	// state = cudaMallocManaged(&d_C, bytes);
	// if(cudaSuccess != state){
	// 	printf("error on cudaMallocManaged(&d_C, bytes): %d\n", state);
	// }
	// printf("line number %d\n", __LINE__);
	// // Fill host arrays A and B
	// for(int i=0; i<N; i++)
	// {
	// 	d_A[i] = 1.0;
	// 	d_B[i] = 2.0;
	// }

	// // Copy data from host arrays A and B to device arrays d_A and d_B
	// // cudaMemcpy(d_A, A, bytes, cudaMemcpyHostToDevice);
	// // cudaMemcpy(d_B, B, bytes, cudaMemcpyHostToDevice);

	// // Set execution configuration parameters
	// //		thr_per_blk: number of CUDA threads per grid block
	// //		blk_in_grid: number of blocks in grid
	// int thr_per_blk = 256;
	// int blk_in_grid = ceil( float(N) / thr_per_blk );

	
	// // Copy data from device array d_C to host array C
	// // cudaMemcpy(C, d_C, bytes, cudaMemcpyDeviceToHost);

	// // Verify results
    // double tolerance = 1.0e-14;
	// for(int i=0; i<N; i++)
	// {
	// 	if( fabs(d_C[i] - 3.0) > tolerance)
	// 	{ 
	// 		printf("\nError: value of d_C[%d] = %d instead of 3.0\n\n", i, d_C[i]);
	// 		exit(1);
	// 	}
	// }	

	// // Free CPU memory
	// // free(A);
	// // free(B);
	// // free(C);

	// // Free GPU memory
	// cudaFree(d_A);
	// cudaFree(d_B);
	// cudaFree(d_C);

	// printf("\n---------------------------\n");
	// printf("__SUCCESS__\n");
	// printf("---------------------------\n");
	// printf("N                 = %d\n", N);
	// printf("Threads Per Block = %d\n", thr_per_blk);
	// printf("Blocks In Grid    = %d\n", blk_in_grid);
	// printf("---------------------------\n\n");

    // destroy(s_ctx);

	return 0;
}