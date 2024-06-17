#include <stdio.h>

// extern "C"{
//   #include "rdma_utils.h"
// }

#include "rdma_utils.cuh"
#include <time.h>
#include "runtime.h"


// Size of array
#define N 3*1024*1024*1024llu

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

int alloc_global_cont(struct post_content *post_cont, struct poll_content *poll_cont, struct post_content2 *post_cont2){
    struct post_content *d_post;
    struct poll_content *d_poll;
    struct post_content2 *d_post2;

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

    ret0 = cudaMalloc((void **)&d_post2, sizeof(struct post_content2));
    if(ret0 != cudaSuccess){
        printf("Error on allocation post content!\n");
        return -1;
    }
    ret0 = cudaMemcpy(d_post2, post_cont2, sizeof(struct post_content2), cudaMemcpyHostToDevice);
    if(ret0 != cudaSuccess){
        printf("Error on poll copy!\n");
        return -1;
    }

    alloc_content<<<1,1>>>(d_post, d_poll);
    alloc_global_content<<<1,1>>>(d_post, d_poll, d_post2);
    ret0 = cudaDeviceSynchronize();
    if(ret0 != cudaSuccess){
        printf("Error on alloc_content!\n");
        return -1;
    }
    return 0;
}

__global__ void test(rdma_buf<int> *a, rdma_buf<int> *b, rdma_buf<int> *c, size_t size){
    // rdma_buf<int> A = *a;
    // rdma_buf<int> B = *b;
    size_t id = blockDim.x * blockIdx.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;
    // if(id == 0) printf("a->size: %d\n", a->size);
    // if(id==0) ( *a)[id] = 0;
    // if(id == 15728640)
    //     printf("b[15728640]: %d a[15728640]: %d\n", (*a)[id], (*b)[id]);

    for(size_t i = id; i < size; i += stride){
        int k = (*a)[i];// + (*b)[i];
        // int j = (*b)[i];
        // c->rvalue(i, (*a)[i] + (*b)[i]);
    }

    // if(id < a->size){
    //     c->rvalue(id, (*a)[id] + (*b)[id]);

    // }  
    
    // c[id] = (*a)[id] + (*b)[id];
    // printf("buf1[2]: %d buf1->address: %p, buf1->size: %d, REQUEST_SIZE12: %d\n", (*buf1)[2], buf1->gpu_address, buf1->size, 1);
}

__global__ void test2(rdma_buf<int> *a, rdma_buf<int> *b, rdma_buf<int> *c){
    size_t id = blockDim.x * blockIdx.x + threadIdx.x;

    int k = (*a)[id] + (*b)[id];

    // c->rvalue(id, (*a)[id] + (*b)[id]); 
    // if(id == 0) printf("(*b)[%d]: %d\n", id, (*b)[id]);
}

__global__ void check(rdma_buf<int> *a, rdma_buf<int> *b, rdma_buf<int> *c, size_t size){
    size_t id = blockDim.x * blockIdx.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;
//    for(int i = 0; i < (*a).size; i++){
    if(id == 0) printf("a tlb size: %llu id: %d\n", a->tlb_size, id);
    if(id == 0) printf("b tlb size: %llu id: %d\n", b->tlb_size, id);
    if(id == 0) printf("c tlb size: %llu id: %d\n", c->tlb_size, id);
        // array a:
    int k = a->tlb_size;
    // if(id < k){
    for (size_t o = id; o < k; o += stride){
        if(a->d_TLB[o].device_address == NULL){
            // printf("  %p  ", a->d_TLB[id].device_address);
            if(o > 0 && a->d_TLB[o-1].device_address != NULL){
                printf("middle start: tlb index: %d qp: %d\n", o, o%256);
            }
            else if(o == 0){
                printf("first start: tlb index: %d qp: %d\n", o, o%256);
            }
            else if(o < a->tlb_size-1 && a->d_TLB[o+1].device_address != NULL){
                printf("middle end: tlb index: %d qp: %d\n", o, o%256);
            }
            else if(o == a->tlb_size-1){
                printf("last end: tlb index: %d qp: %d\n", o, o%256);
            }
        }
        int flag1 = 0;
        int *tmp_p2 = (int *)a->d_TLB[o].device_address;
        for(uint64_t p = 0; p < 256; p++ )
            {
                if(tmp_p2[p] != 2){
                    flag1++;
                    // printf(" case for unmatched data a.tlb index: %d tmp_p2[p]: %d\n", id, tmp_p2[p]);
                }
            }
        if(flag1 > 0) printf("matched data a tlb index: %d flag: %d d_TLB[che].device_address: %p\n", o, flag1, a->d_TLB[o].device_address);
        if(o == 1024*1024*8){
            int *tmp_array1 = (int *) a->d_TLB[o].device_address;
            printf("d_TLB[che].host_address: %p tmp_array1[0]: %d\n", a->d_TLB[o].host_address, tmp_array1[0]);
        }
        

        // if(b->d_TLB[id].device_address == NULL){
        //     // printf("  %p  ", a->d_TLB[id].device_address);
        //     if(id > 0 && b->d_TLB[id-1].device_address != NULL){
        //         printf("middle start: b tlb index: %d qp: %d\n", id, id%256);
        //     }
        //     else if(id == 0){
        //         printf("first start: b tlb index: %d qp: %d\n", id, id%256);
        //     }
        //     else if(id < b->tlb_size-1 && b->d_TLB[id+1].device_address != NULL){
        //         printf("middle end: b tlb index: %d qp: %d\n", id, id%256);
        //     }
        //     else if(id == b->tlb_size-1){
        //         printf("last end: b tlb index: %d qp: %d\n", id, id%256);
        //     }
        // }
        // int flag2 = 0;
        // for(uint64_t p = 0; p < 256; p++)
        // {
        //     int *tmp_p = (int *)b->d_TLB[id].device_address;
        //     if(tmp_p[p] == 2){
        //         flag2++;
        //     }
        // }
        // int *tmp_p1 = (int *)b->d_TLB[id].device_address;
        // if(flag2 < 256) printf("unmatched data b tlb index: %d flag: %d b->d_TLB[id].device_address[0]: %d b->d_TLB[id].state: %d\n",\
        //                        id, flag2, tmp_p1[0], b->d_TLB[id].state);

        // if(c->d_TLB[id].device_address == NULL){
        //     // printf("  %p  ", a->d_TLB[id].device_address);
        //     if(id > 0 && c->d_TLB[id-1].device_address != NULL){
        //         printf("middle start: c tlb index: %d qp: %d\n", id, id%256);
        //     }
        //     else if(id == 0){
        //         printf("first start: c tlb index: %d qp: %d\n", id, id%256);
        //     }
        //     else if(id < c->tlb_size-1 && c->d_TLB[id+1].device_address != NULL){
        //         printf("middle end: c tlb index: %d qp: %d\n", id, id%256);
        //     }
        //     else if(id == c->tlb_size-1){
        //         printf("last end: c tlb index: %d qp: %d\n", id, id%256);
        //     }
        // }
        // flag = 0;
        // for(uint64_t p = 0; p < 256; p++ )
        //     {
        //         int *tmp_p = (int *)c->d_TLB[id].device_address;
        //         if(tmp_p[p] == 4){
        //             flag++;
        //         }
        //     }
        // if(flag < 256) printf("matched data c tlb index: %d flag: %d\n", id, flag);
    }
}

// Main program
int main(int argc, char **argv)
{   
    if (argc != 7)
        usage(argv[0]);
    
    init_gpu(0);
    printf("Function: %s line number: %d 1024MB: %d bytes REQUEST_SIZE: %d\n",__func__, __LINE__, MB(1024), REQUEST_SIZE);
    int num_msg = (unsigned long) atoi(argv[4]);
    int mesg_size = (unsigned long) atoi(argv[5]);
    int num_bufs = (unsigned long) atoi(argv[6]);

    struct context *s_ctx = (struct context *)malloc(sizeof(struct context));
    struct post_content post_cont, *d_post;
    struct poll_content poll_cont, *d_poll;
    struct post_content2 post_cont2, *d_post2;

    int num_iteration = num_msg;
    s_ctx->n_bufs = num_bufs;
    s_ctx->gpu_buf_size = 30*1024*1024*1024llu; // N*sizeof(int)*3llu;

    int ret = connect(argv[2], s_ctx);
    ret = prepare_post_poll_content(s_ctx, &post_cont, &poll_cont, &post_cont2);


    int a[100], *b;
    a[2] = 5;
    // rdma_buf<int> buf((uint64_t) s_ctx->gpu_buffer, (uint64_t) s_ctx->server_mr.addr, 100);
    rdma_buf<int> *buf1, *buf2, *buf3;
    cudaMallocManaged(&buf1, sizeof(rdma_buf<int>));
    cudaMallocManaged(&buf2, sizeof(rdma_buf<int>));
    cudaMallocManaged(&buf3, sizeof(rdma_buf<int>));
    // cudaMallocManaged(&b, sizeof(int)*100);
    // printf("s_ctx->gpu_buffer: %p, buf1->size: %d, Address_Offset: %d\n", s_ctx->gpu_buffer, buf1->size, Address_Offset);
    
    buf1->start((uint64_t) s_ctx->server_memory.addresses[0], N*sizeof(int));
    printf("s_ctx->gpu_buffer: %p, buf1->size: %d, Address_Offset: %d\n", s_ctx->gpu_buffer, buf1->size, Address_Offset);
    buf2->start((uint64_t) s_ctx->server_memory.addresses[0], N*sizeof(int));
    
    printf("s_ctx->gpu_buffer: %p, buf1->size: %d, Address_Offset: %d\n", s_ctx->gpu_buffer, buf1->size, Address_Offset);
    // buf3->start((uint64_t) s_ctx->gpu_buffer, (uint64_t) s_ctx->server_mr.addr, N*sizeof(int));
    // buf1->address = (uint64_t) s_ctx->gpu_buffer;
    // buf1->size = 100;
    printf("buf1->address: %p, buf1->size: %d, REQUEST_SIZE: %d buf1->host_address : %p N*sizeof(int): %llu\n", buf1->gpu_address, buf1->size, REQUEST_SIZE, buf1->host_address, N*sizeof(int));
    printf("buf2->address: %p, buf2->size: %d, REQUEST_SIZE: %d buf2->host_address : %p\n", buf2->gpu_address, buf2->size, REQUEST_SIZE, buf2->host_address);
    // cudaMemcpy(buf1, &buf, sizeof(rdma_buf<int>), cudaMemcpyHostToDevice);
    // printf("buf[2]: %d a: %p\n", buf[2], a);
    printf("Function name: %s, line number: %d mesg_size: %d num_iteration: %d sizeof(int): %d\n", __func__, __LINE__, mesg_size, num_msg, sizeof(int));
   
    // allocate poll and post content
    alloc_global_cont(&post_cont, &poll_cont, &post_cont2);  

    int thr_per_blk = 2048*2; // s_ctx->n_bufs;
	int blk_in_grid = 256;

    int timer_size = 4;
    clock_t *dtimer = NULL;

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
    
    test<<<2048, 512>>>(buf1, buf2, buf3, N);
    
    cudaEventRecord(event2, (cudaStream_t) 1);
    clock_gettime(CLOCK_REALTIME, &finish);
    ret1 = cudaDeviceSynchronize();
    
    //synchronize
    cudaEventSynchronize(event1); //optional
    cudaEventSynchronize(event2); //wait for the event to be executed!

    printf("ret1: %d\n", ret1);
    if(cudaSuccess != ret1){
        return -1;
    }

    
    

    clock_t cycles;
    float g_usec_post;
    cudaDeviceProp devProp;
    cudaGetDeviceProperties(&devProp, 0);
    printf("Cuda device clock rate = %d\n", devProp.clockRate);
    float freq_post = (float)1/((float)devProp.clockRate*1000), max = 0;

    
    g_usec_post = (float)((float)1/(devProp.clockRate*1000))*((cycles)) * 1000000;
    printf("total timer: %f\n", g_usec_post);
    printf("kernel time: %d.%.9ld dt_ms: %f\n", (int)delta.tv_sec, delta.tv_nsec, dt_ms);

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


// TODO:
/*
1. Correct the allocation mechanism.
2. Create page cache; each request or tlb entry is not just an array entry. 
    It is a different data structure where we keep the info about the location of the data,
    host address and possible device location if data has been fetched. We do this to make sure 
    that we are not dependent on continuous assignment and each request can be allocated to device
    memory during post time. - somehow done
3. Add device allocation - when the data is brought to device memory for the first time, 
    the device address should be given to that data request, which will be stored in the corresponding 
    tlb entry. - somehow done
4. Add eviction mechanism - when the memory of the GPU is completely filled, polling threads are 
    allowed to evict pages based on LRU policy. - not done
5. Think about possible prefetching but not too important - not done
6. Think about thread-level request handling; each thread will put the data request corresponding to one entry;
    then the coalescer unit will coalesce these smal requests and ring/update the db register - 
    this way no unnecessary data will be brought to the memory. similar to BaM. 
    For this, I think we also need to extend the QPs in depth and number as well. This step is an important point.
*/