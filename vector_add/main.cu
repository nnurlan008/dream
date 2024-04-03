#include <stdio.h>

// extern "C"{
//   #include "rdma_utils.h"
// }

#include "rdma_utils.cuh"

// Size of array
#define N 256*1024

// Kernel
__global__ void add_vectors_uvm(int *a, int *b, int *c, int size)
{
	int id = blockDim.x * blockIdx.x + threadIdx.x;
	if(id < size) {
		c[id] = a[id] + b[id];
		printf("c[%d]: %d\n", id, c[id]);
	}
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

__global__ void add_vectors_rdma(int *a, int *b, int *c, int size, \
                                uint8_t *tlb_A, uint8_t *tlb_B, uint8_t *tlb_C, struct post_content *post_cont1, struct poll_content *poll_cont1)
{
	int id = blockDim.x * blockIdx.x + threadIdx.x;
    if(id < size) {
        // if(id == 1023)
        // printf("hello\n");

        // post A[id] if tlb_A[blockIdx.x] == 0
        // if(tlb_A[blockIdx.x] == 0)
        //      post(post_cont(A[id]))

        // post B[id] if tlb_B[blockIdx.x] == 0

        // one thread out of the block should poll and
        // update the tlb entry

        // all other threads poll on the tlb array entry
        // when the entry is 2 - on-device, computation continues
        int cur_post = 0;
        // printf("a[%d]: %d\n", id, a[id]);
        
        // if (tlb_A[blockIdx.x] == 0 && blockIdx.x <= 7 /*&& blockIdx.x <= 9)*/){
            
            // __threadfence_system();
            // printf("threadIdx.x: %d\n", threadIdx.x);
            // __syncthreads();
            if (threadIdx.x == 0){
                // printf("a[%d]: %d\n", id, a[id]);
                struct post_content post_cont = *post_cont1; 
                struct poll_content poll_cont = *poll_cont1;
                struct ibv_wc wc;
                // configure work request here
                // default works fpr this request
                // post and poll using CQ
                struct post_wr wr;
                wr.qp_num = post_cont.qp_num;
                wr.wr_opcode = IBV_WR_RDMA_READ; // post_cont.wr_opcode;
                wr.wr_rdma_remote_addr = post_cont.wr_rdma_remote_addr;
                wr.wr_rdma_rkey = post_cont.wr_rdma_rkey;
                wr.wr_sg_addr = post_cont.wr_sg_addr;
                wr.wr_sg_length = 4096; // post_cont.wr_sg_length;
                wr.wr_sg_lkey = post_cont.wr_sg_lkey;
                // printf("blockIdx.x: %d post_cont.bf_reg[blockIdx.x]: 0x%llx\n", blockIdx.x, post_cont.bf_reg[blockIdx.x]);
                post(post_cont.wr_rdma_remote_addr + 1024*blockIdx.x*4 , post_cont.wr_rdma_rkey, 
                     post_cont.wr_sg_length, post_cont.wr_sg_lkey, post_cont.wr_sg_addr + 1024*blockIdx.x*4, post_cont.wr_opcode, 
                     wr.qp_num + blockIdx.x, cur_post, post_cont.qp_buf + 8192*blockIdx.x, (void *) post_cont.bf_reg[blockIdx.x]);
                     
                printf("blockIdx.x: %d a[%d]: %d\n",blockIdx.x, id, a[id]);
                // printf("a[%d]: %d\n", id, a[id]);
                // printf("a[%d]: %d\n", id, a[id]);
                // printf("a[%d]: %d\n", id, a[id]);

                // post_s(wr, cur_post, post_cont.qp_buf, (void *) post_cont.bf_reg[0]);
                cur_post++;
                
                while(poll(poll_cont.cq_buf + 4096*blockIdx.x, &wc, (uint32_t *) poll_cont.cons_index[blockIdx.x], poll_cont.ibv_cqe, poll_cont.cqe_sz, 1, \ 
                (void *) poll_cont.cq_dbrec[blockIdx.x])==0);
                tlb_A[blockIdx.x] = 3;
                // printf("blockIdx.x: %d a[%d]: %d\n", blockIdx.x, id, a[id]);
                // printf("blockIdx.x: %d a[%d]: %d\n", blockIdx.x, id, a[id]);
                // printf("blockIdx.x: %d a[%d]: %d\n", blockIdx.x, id, a[id]);
                // // printf("a[%d]: %d\n", id, a[id]);
                // // printf("a[%d]: %d\n", id, a[id]);
                // printf("blockIdx.x: %d a[%d]: %d\n", blockIdx.x, id, a[id]);
                // __threadfence_system();
                // cur_post++;
                // __syncwarp();
            }
            while(tlb_A[blockIdx.x] == 0);
            // __syncthreads();
            // for(int del = 0; del < 10000000; del++);
            __syncthreads();
            printf("a[%d]: %d\n", id, a[id]);
        // }
        // if(threadIdx.x != 0)
            // for(int del = 0; del < 10000000; del++);
        // __threadfence_system();
        // __syncthreads();
        // if(a[id] == 2)
        //     printf("a[%d]: %d\n", id, a[id]);

        // if(id = 0){
        //         // printf("blockIdx.x: %d a[%d]: %d\n",blockIdx.x, id, a[id]);
        //         // printf("a[%d]: %d\n", id, a[id]);
        //         // printf("a[%d]: %d\n", id, a[id]);
        //         // printf("a[%d]: %d\n", id, a[id]);
        //         struct poll_content poll_cont = *poll_cont1;
        //         struct ibv_wc wc;
        //         int n = 256;
        //         do{
        //             while(poll(poll_cont.cq_buf, &wc, (uint32_t *) poll_cont.cons_index[blockIdx.x], poll_cont.ibv_cqe, poll_cont.cqe_sz, 1, \ 
        //              (void *) poll_cont.cq_dbrec[blockIdx.x])==0);
        //             n--;
        //         }while(n);

        //         // while(poll(poll_cont.cq_buf + 4096*blockIdx.x, &wc, (uint32_t *) poll_cont.cons_index[blockIdx.x], poll_cont.ibv_cqe, poll_cont.cqe_sz, 1, \ 
        //         //      (void *) poll_cont.cq_dbrec[blockIdx.x])==0);

        //         // void *cqe;
        //         // struct mlx5_cqe64 *cqe64;
        //         // uint32_t cons_index_dev = *(uint32_t *) poll_cont.cons_index[blockIdx.x];
        //         // cqe = poll_cont.cq_buf /*+ 4096*blockIdx.x*/ + (cons_index_dev & poll_cont.ibv_cqe) * poll_cont.cqe_sz;
        //         // printf("blockIdx.x: %d poll_cont.cq_buf: 0x%llx\n", blockIdx.x, poll_cont.cq_buf + 4096*blockIdx.x);
                
        //         // cqe64 = (struct mlx5_cqe64 *)((poll_cont.cqe_sz == 64) ? cqe : cqe + 64);
        //         // // ((cqe64->op_own != 240) && !((cqe64->op_own & 1) ^ !!(1 & (poll_cont.ibv_cqe + 1))))==0
        //         // while(cqe64->op_own == 240);
                
        //         // uint32_t *gpu_dbrec = (uint32_t *) poll_cont.cq_dbrec[blockIdx.x];
        //         // (*(uint32_t *) poll_cont.cons_index[blockIdx.x])++;
        //         // gpu_dbrec[0] = htonl((cons_index_dev) & 0xffffff);

        //         // __threadfence_system();
        //         // __syncthreads();
        // //         // update tlb_A[blockIdx.x] = 2
        //         tlb_A[blockIdx.x] = 3;
        //         // __threadfence_system();
        // //         cur_post++;
        //         // printf("a[%d]: %d\n", id, a[id]);
            
            
            
            
        //     // __threadfence_system();
            
        // }
        // while(tlb_A[blockIdx.x] == 0);
        // __syncthreads();
        // printf("a[%d]: %d\n", id, a[id]);
        // if(a[id] == 2)
        // printf("a[%d]: %d\n", id, a[id]);
        // if (tlb_B[blockIdx.x] == 0){
        //     if (threadIdx.x == blockDim.x - 1){
        //         struct ibv_wc wc;
        //         struct post_content post_cont = *post_cont1; 
        //         struct poll_content poll_cont = *poll_cont1;
        //         // configure work request here
        //         // default works fpr this request
        //         // post and poll using CQ
        //         post(post_cont.wr_rdma_remote_addr + blockDim.x*blockIdx.x*4 + size, post_cont.wr_rdma_rkey, \
        //              post_cont.wr_sg_length, post_cont.wr_sg_lkey, post_cont.wr_sg_addr + blockDim.x*blockIdx.x*4 + size, post_cont.wr_opcode, \
        //              post_cont.qp_num + blockIdx.x, cur_post, post_cont.qp_buf + 8192*blockIdx.x, (void *) post_cont.bf_reg[blockIdx.x]);
        //         poll(poll_cont.cq_buf + 4096*blockIdx.x, &wc,  (uint32_t *) poll_cont.cons_index[blockIdx.x], poll_cont.ibv_cqe, poll_cont.cqe_sz, 1, \ 
        //              (void *) poll_cont.cq_dbrec[blockIdx.x]);
        //         // update tlb_A[blockIdx.x] = 2
        //         tlb_B[blockIdx.x] = 2;
        //         cur_post++;
        //         printf("a[%d]: %d\n", id, b[id]);
        //     }
        //     else while(!tlb_B[blockIdx.x]);
        // }
        
        // if(tlb_A[blockIdx.x] == 0)
        // __syncthreads();
		// c[id] = a[id] + b[id];
		// printf("c[%d]: %d\n", id, c[id]);
	}
    
}

void usage(const char *argv0)
{
  fprintf(stderr, "usage: %s <mode> <server-address> <server-port>\n  mode = \"read\", \"write\"\n", argv0);
  exit(1);
}

// Main program
int main(int argc, char **argv)
{   
    if (argc != 6)
        usage(argv[0]);
    // else
    //     usage(argv[0]);
    init_gpu(0);
    printf("Function: %s line number: %d\n",__func__, __LINE__);
    int num_msg = (unsigned long) atoi(argv[4]);
    int mesg_size = (unsigned long) atoi(argv[5]);

    struct context *s_ctx = (struct context *)malloc(sizeof(struct context));
    struct post_content post_cont, *d_post;
    struct poll_content poll_cont, *d_poll;

    s_ctx->n_bufs = 20;
    s_ctx->gpu_buf_size = 3*1024*1024;

    int ret = connect(argv[2], s_ctx);
    ret = prepare_post_poll_content(s_ctx, &post_cont, &poll_cont);

    printf("Function name: %s, line number: %d\n", __func__, __LINE__);

    uint8_t access_size = sizeof(int);
    size_t bytes = N*sizeof(int);
    void *A = (void *) s_ctx->gpu_buffer;
    void *B = (void *) s_ctx->gpu_buffer + bytes;
    void *C = (void *) s_ctx->gpu_buffer + 2*bytes;
    int *h_array = (int *) malloc(bytes);
    for(int i = 0; i < bytes/sizeof(int); i++)
        h_array[i] = 0;

    // allocate poll and post content
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
    ret0 = cudaMemcpy(d_post, &post_cont, sizeof(struct post_content), cudaMemcpyHostToDevice);
    if(ret0 != cudaSuccess){
        printf("Error on post copy!\n");
        return -1;
    }
    ret0 = cudaMemcpy(d_poll, &poll_cont, sizeof(struct poll_content), cudaMemcpyHostToDevice);
    if(ret0 != cudaSuccess){
        printf("Error on poll copy!\n");
        return -1;
    }

    cudaError_t rtr1 = cudaMemcpy(A, h_array, bytes, cudaMemcpyHostToDevice);
    cudaError_t rtr2 = cudaMemcpy(B, h_array, bytes, cudaMemcpyHostToDevice);
    cudaError_t rtr3 = cudaMemcpy(C, h_array, bytes, cudaMemcpyHostToDevice);
    if(rtr1 != cudaSuccess && rtr2 != cudaSuccess && rtr3 != cudaSuccess){
        printf("Error on array copy!\n");
        return -1;
    }

    // Allocate TLB for array A
    uint8_t *tlb_A, *tlb_B, *tlb_C, *h_tlb;
    int tlb_size = bytes/(access_size*s_ctx->n_bufs);
    cudaError_t ret1 = cudaMalloc((void **)&tlb_A, tlb_size*sizeof(uint8_t));
    cudaError_t ret2 = cudaMalloc((void **)&tlb_B, tlb_size*sizeof(uint8_t));
    cudaError_t ret3 = cudaMalloc((void **)&tlb_C, tlb_size*sizeof(uint8_t));
    if(ret1 != cudaSuccess && ret2 != cudaSuccess && ret3 != cudaSuccess){
        printf("Error on allocation TLB!\n");
        return -1;
    }
    h_tlb = (uint8_t *) malloc(tlb_size*sizeof(uint8_t));
    for (int i = 0; i < tlb_size; i++) h_tlb[i] = 0;
    ret1 = cudaMemcpy(tlb_A, h_tlb, tlb_size*sizeof(uint8_t), cudaMemcpyHostToDevice);
    ret2 = cudaMemcpy(tlb_B, h_tlb, tlb_size*sizeof(uint8_t), cudaMemcpyHostToDevice);
    ret3 = cudaMemcpy(tlb_C, h_tlb, tlb_size*sizeof(uint8_t), cudaMemcpyHostToDevice);
    if(ret1 != cudaSuccess && ret2 != cudaSuccess && ret3 != cudaSuccess){
        printf("Error on allocation TLB!\n");
        return -1;
    }

    int thr_per_blk = 1024;
	int blk_in_grid = ceil( float(N) / thr_per_blk );

    printf("thr_per_blk: %d, blk_in_grid: %d tlb_size: %d\n", thr_per_blk, blk_in_grid, tlb_size);

    


    // Allocate memory for arrays d_A, d_B, and d_C on device
	int *d_A, *d_B, *d_C;
	cudaError_t state;
	state = cudaMallocManaged(&d_A, bytes);
	if(cudaSuccess != state){
		printf("error on cudaMallocManaged(&d_A, bytes): %d\n", state);
	}
	state = cudaMallocManaged(&d_B, bytes);
	if(cudaSuccess != state){
		printf("error on cudaMallocManaged(&d_B, bytes): %d\n", state);
	}
	state = cudaMallocManaged(&d_C, bytes);
	if(cudaSuccess != state){
		printf("error on cudaMallocManaged(&d_C, bytes): %d\n", state);
	}
	printf("line number %d\n", __LINE__);
	// Fill host arrays A and B
	for(int i=0; i<N; i++)
	{
		d_A[i] = 1.0;
		d_B[i] = 2.0;
	}

    // Launch kernel
    ret1 = cudaDeviceSynchronize();
    printf("ret: %d\n", ret1);
    if(cudaSuccess != ret1){    
        return -1;
    }
	add_vectors_rdma<<< 2, 896 >>>((int *) A, (int *) B, (int *) C, bytes/sizeof(int), tlb_A, tlb_B, tlb_C, d_post, d_poll);
    // add_vectors_uvm<<< blk_in_grid, thr_per_blk >>>(d_A, d_B, d_C, bytes);
	ret1 = cudaDeviceSynchronize();
    printf("ret1: %d\n", ret1);
    if(cudaSuccess != ret1){
        return -1;
    }

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

	return 0;
}