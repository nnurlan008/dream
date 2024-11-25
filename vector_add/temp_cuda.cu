#include <stdio.h>

// extern "C"{
//   #include "rdma_utils.h"
// }

#include "rdma_utils.cuh"
#include <time.h>  
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


void usage(const char *argv0)
{
  fprintf(stderr, "usage: %s <mode> <server-address> <server-port>\n  mode = \"read\", \"write\"\n", argv0);
  exit(1);
}

__global__ void read(int *a, int index){
    printf("a[%d]: %d\n", index, a[index]);
}

__global__ void write_nonstop(int *a, int size, int index, int number){
    int i = 0;
    // while(1){
        for(i = 0; i < index; i++){
            a[i] = number;
        }
    // }
    // a[index] = number;
    printf("a[%d]: %d\n", 0, a[0]);
}

__global__ void write(int *a, int index, int number){
    a[index] = number;
    printf("a[%d]: %d\n", index, a[index]);
}

// Main program
int main(int argc, char **argv)
{   
   
    // init_gpu(0);
    
    int *dev_array, *dev_array2;                      // 107374182
    cudaError_t ret1 = cudaMalloc((void **)&dev_array, 1024*1024*20);
    if(ret1 != cudaSuccess){
        printf("cuda error: %s\n", cudaGetErrorString(ret1));
    }
    ret1 = cudaMalloc((void **)&dev_array2, 1024);
    if(ret1 != cudaSuccess){
        printf("cuda error: %s\n", cudaGetErrorString(ret1));
    }
    printf("dev_array: 0x%llx, dev_array2: 0x%llx \n", dev_array, dev_array2);
    // Launch kernel
    ret1 = cudaDeviceSynchronize();
    printf("ret: %d\n", ret1);
    if(cudaSuccess != ret1){    
        return -1;
    }
    // write_nonstop<<<1,1>>>(dev_array, 0, 1024*1024*20, 3);
    // read<<<1,1>>>(dev_array2, 71808-256);
	// add_vectors_rdma<<< 2, 896 >>>((int *) A, (int *) B, (int *) C, bytes/sizeof(int), tlb_A, tlb_B, tlb_C, d_post, d_poll);

    // add_vectors_uvm<<< blk_in_grid, thr_per_blk >>>(d_A, d_B, d_C, bytes);
	ret1 = cudaDeviceSynchronize();
    printf("ret1: %d\n", ret1);
    if(cudaSuccess != ret1){
        return -1;
    }
    delay(40);
	
	return 0;
}