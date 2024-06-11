// #include <stdio.h>

// // extern "C"{
// //   #include "rdma_utils.h"
// // }

// #include "rdma_utils.cuh"
// #include <time.h>
// #include "runtime.h"


// // Size of array
// #define N 1024*1024*150llu

// // Kernel
// __global__ void add_vectors_uvm(int *a, int *b, int *c, int size)
// {
// 	int id = blockDim.x * blockIdx.x + threadIdx.x;
// 	// if(id < size) {
// 		c[id] = a[id] + b[id];
// 		// printf("c[%d]: %d\n", id, c[id]);
// 	// }
// }


// struct __attribute__((__packed__)) mlx5_tm_cqe {
// 	__be32		success;
// 	__be16		hw_phase_cnt;
// 	uint8_t		rsvd0[12];
// };

// struct __attribute__((__packed__)) ibv_tmh {
// 	uint8_t		opcode;      /* from enum ibv_tmh_op */
// 	uint8_t		reserved[3]; /* must be zero */
// 	__be32		app_ctx;     /* opaque user data */
// 	__be64		tag;
// };

// struct __attribute__((__packed__)) mlx5_cqe64 {
// 	union {
// 		struct {
// 			uint8_t		rsvd0[2];
// 			__be16		wqe_id;
// 			uint8_t		rsvd4[13];
// 			uint8_t		ml_path;
// 			uint8_t		rsvd20[4];
// 			__be16		slid;
// 			__be32		flags_rqpn;
// 			uint8_t		hds_ip_ext;
// 			uint8_t		l4_hdr_type_etc;
// 			__be16		vlan_info;
// 		};
// 		struct mlx5_tm_cqe tm_cqe;
// 		/* TMH is scattered to CQE upon match */
// 		struct ibv_tmh tmh;
// 	};
// 	__be32		srqn_uidx;
// 	__be32		imm_inval_pkey;
// 	uint8_t		app;
// 	uint8_t		app_op;
// 	__be16		app_info;
// 	__be32		byte_cnt;
// 	__be64		timestamp;
// 	__be32		sop_drop_qpn;
// 	__be16		wqe_counter;
// 	uint8_t		signature;
// 	uint8_t		op_own;
// };

// #define htonl(x)  ((((uint32_t)(x) & 0xff000000) >> 24) |\
//                    (((uint32_t)(x) & 0x00ff0000) >>  8) |\
//                    (((uint32_t)(x) & 0x0000ff00) <<  8) |\
//                    (((uint32_t)(x) & 0x000000ff) << 24))


// void delay(int number_of_seconds)
// {
//     // Converting time into milli_seconds
//     int milli_seconds = 1000000 * number_of_seconds;
 
//     // Storing start time
//     clock_t start_time = clock();
 
//     // looping till required time is not achieved
//     while (clock() < start_time + milli_seconds)
//         ;
// }

// enum { NS_PER_SECOND = 1000000000 };

// void sub_timespec(struct timespec t1, struct timespec t2, struct timespec *td)
// {
//     td->tv_nsec = t2.tv_nsec - t1.tv_nsec;
//     td->tv_sec  = t2.tv_sec - t1.tv_sec;
//     if (td->tv_sec > 0 && td->tv_nsec < 0)
//     {
//         td->tv_nsec += NS_PER_SECOND;
//         td->tv_sec--;
//     }
//     else if (td->tv_sec < 0 && td->tv_nsec > 0)
//     {
//         td->tv_nsec -= NS_PER_SECOND;
//         td->tv_sec++;
//     }
// }

// void usage(const char *argv0)
// {
//   fprintf(stderr, "usage: %s <mode> <server-address> <server-port>\n  mode = \"read\", \"write\"\n", argv0);
//   exit(1);
// }

// __global__ void read_nonstop(int *a, int size){
//     int i = 0;
//     int stop = 0;
//     while(1){
//         for(i = 0; i < size; i++)
//             if(a[i] == 3){
//                 printf("a[%d]: %d\n", i, a[i]);
//                 stop = 1;
//                 break;
//             }
//         if(stop) break;
//     }
// }

// __global__ void read(int *a, int index){
//     printf("a[%d]: %d\n", index, a[index]);
// }

// __global__ void write(int *a, int index, int number){
//     a[index] = number;
//     printf("a[%d]: %d\n", index, a[index]);
// }

// int alloc_global_cont(struct post_content *post_cont, struct poll_content *poll_cont){
//     struct post_content *d_post;
//     struct poll_content *d_poll;

//     cudaError_t ret0 = cudaMalloc((void **)&d_post, sizeof(struct post_content));
//     if(ret0 != cudaSuccess){
//         printf("Error on allocation post content!\n");
//         return -1;
//     }
//     ret0 = cudaMalloc((void **)&d_poll, sizeof(struct poll_content));
//     if(ret0 != cudaSuccess){
//         printf("Error on allocation poll content!\n");
//         return -1;
//     }
//     printf("sizeof(struct post_content): %d, sizeof(struct poll_content): %d\n", sizeof(struct post_content), sizeof(struct poll_content));
//     ret0 = cudaMemcpy(d_post, post_cont, sizeof(struct post_content), cudaMemcpyHostToDevice);
//     if(ret0 != cudaSuccess){
//         printf("Error on post copy!\n");
//         return -1;
//     }
//     ret0 = cudaMemcpy(d_poll, poll_cont, sizeof(struct poll_content), cudaMemcpyHostToDevice);
//     if(ret0 != cudaSuccess){
//         printf("Error on poll copy!\n");
//         return -1;
//     }

//     alloc_content<<<1,1>>>(d_post, d_poll);
//     alloc_global_content<<<1,1>>>(d_post, d_poll);
//     ret0 = cudaDeviceSynchronize();
//     if(ret0 != cudaSuccess){
//         printf("Error on alloc_content!\n");
//         return -1;
//     }
//     return 0;
// }

// __global__ void test(rdma_buf<int> *a, rdma_buf<int> *b, rdma_buf<int> *c){
//     // rdma_buf<int> A = *a;
//     // rdma_buf<int> B = *b;
//     int id = blockDim.x * blockIdx.x + threadIdx.x;
//     // if(id == 0) printf("a->size: %d\n", a->size);
//     // if(id==0) ( *a)[id] = 0;
//     // if(id == 15728640)
//     //     printf("b[15728640]: %d a[15728640]: %d\n", (*a)[id], (*b)[id]);

//     for(int i = id; i < 16777216; i += 524288){
//         c->rvalue(i, (*a)[i] + (*b)[i]);
//     }

//     // if(id < a->size){
//     //     c->rvalue(id, (*a)[id] + (*b)[id]);

//     // }  
    
//     // c[id] = (*a)[id] + (*b)[id];
//     // printf("buf1[2]: %d buf1->address: %p, buf1->size: %d, REQUEST_SIZE12: %d\n", (*buf1)[2], buf1->gpu_address, buf1->size, 1);
// }

// __global__ void test2(rdma_buf<int> *a, rdma_buf<int> *b, rdma_buf<int> *c){
//     int id = blockDim.x * blockIdx.x + threadIdx.x;

//     // int k = (*a)[id] + (*b)[id];

//     c->rvalue(id, (*a)[id] + (*b)[id]); 
// }

// // Main program
// int main(int argc, char **argv)
// {   
//     if (argc != 7)
//         usage(argv[0]);
//     // else
//     //     usage(argv[0]);
//     init_gpu(0);
//     printf("Function: %s line number: %d 1024MB: %d bytes REQUEST_SIZE: %d\n",__func__, __LINE__, MB(1024), REQUEST_SIZE);
//     int num_msg = (unsigned long) atoi(argv[4]);
//     int mesg_size = (unsigned long) atoi(argv[5]);
//     int num_bufs = (unsigned long) atoi(argv[6]);

//     struct context *s_ctx = (struct context *)malloc(sizeof(struct context));
//     struct post_content post_cont, *d_post;
//     struct poll_content poll_cont, *d_poll;

//     int num_iteration = num_msg;
//     s_ctx->n_bufs = num_bufs;
//     s_ctx->gpu_buf_size = N*sizeof(int)*3llu;

//     int ret = connect(argv[2], s_ctx);
//     ret = prepare_post_poll_content(s_ctx, &post_cont, &poll_cont);


//     int a[100], *b;
//     a[2] = 5;
//     // rdma_buf<int> buf((uint64_t) s_ctx->gpu_buffer, (uint64_t) s_ctx->server_mr.addr, 100);
//     rdma_buf<int> *buf1, *buf2, *buf3;
//     cudaMallocManaged(&buf1, sizeof(rdma_buf<int>));
//     cudaMallocManaged(&buf2, sizeof(rdma_buf<int>));
//     cudaMallocManaged(&buf3, sizeof(rdma_buf<int>));
//     // cudaMallocManaged(&b, sizeof(int)*100);
//     // printf("s_ctx->gpu_buffer: %p, buf1->size: %d, Address_Offset: %d\n", s_ctx->gpu_buffer, buf1->size, Address_Offset);
//     buf1->start((uint64_t) s_ctx->gpu_buffer, (uint64_t) s_ctx->server_mr.addr, N*sizeof(int));
//     // printf("s_ctx->gpu_buffer: %p, buf1->size: %d, Address_Offset: %d\n", s_ctx->gpu_buffer, buf1->size, Address_Offset);
//     buf2->start((uint64_t) s_ctx->gpu_buffer, (uint64_t) s_ctx->server_mr.addr, N*sizeof(int));
//     buf3->start((uint64_t) s_ctx->gpu_buffer, (uint64_t) s_ctx->server_mr.addr, N*sizeof(int));
//     // buf1->address = (uint64_t) s_ctx->gpu_buffer;
//     // buf1->size = 100;
//     printf("buf1->address: %p, buf1->size: %d, REQUEST_SIZE: %d buf1->host_address : %p\n", buf1->gpu_address, buf1->size, REQUEST_SIZE, buf1->host_address);
//     printf("buf2->address: %p, buf2->size: %d, REQUEST_SIZE: %d buf2->host_address : %p\n", buf2->gpu_address, buf2->size, REQUEST_SIZE, buf2->host_address);
//     // cudaMemcpy(buf1, &buf, sizeof(rdma_buf<int>), cudaMemcpyHostToDevice);
//     // printf("buf[2]: %d a: %p\n", buf[2], a);
//     printf("Function name: %s, line number: %d mesg_size: %d num_iteration: %d sizeof(int): %d\n", __func__, __LINE__, mesg_size, num_msg, sizeof(int));
//     // exit(0);
//     uint8_t access_size = sizeof(int);
//     size_t bytes = N*sizeof(int);
//     void *A = (void *) s_ctx->gpu_buffer;
//     void *B = (void *) s_ctx->gpu_buffer + bytes;
//     void *C = (void *) s_ctx->gpu_buffer + 2*bytes;
//     // int *C_dev;
//     int *h_array = (int *) malloc(bytes);
//     for(int i = 0; i < bytes/sizeof(int); i++)
//         h_array[i] = 0;

//     // allocate poll and post content
//     alloc_global_cont(&post_cont, &poll_cont);
    

//     cudaError_t rtr1 = cudaMemcpy(A, h_array, bytes, cudaMemcpyHostToDevice);
//     cudaError_t rtr2 = cudaMemcpy(B, h_array, bytes, cudaMemcpyHostToDevice);
//     cudaError_t rtr3 = cudaMemcpy(C, h_array, bytes, cudaMemcpyHostToDevice);
//     if(rtr1 != cudaSuccess || rtr2 != cudaSuccess || rtr3 != cudaSuccess){
//         printf("Error on array copy! line: %d\n", __LINE__);
//         return -1;
//     }

//     int thr_per_blk = 2048*2; // s_ctx->n_bufs;
// 	int blk_in_grid = 256;

//     // int thr_per_blk = s_ctx->n_bufs;
// 	// int blk_in_grid = 512;
    

//     // Allocate TLB for array A
//     uint8_t *tlb_A, *tlb_B, *tlb_C, *h_tlb;
//     int tlb_size = bytes/(64*1024); // divided by access size //16*1024*1024/(64*1024); // thr_per_blk;
//     cudaError_t ret1 = cudaMalloc((void **)&tlb_A, tlb_size*sizeof(uint8_t));
//     cudaError_t ret2 = cudaMalloc((void **)&tlb_B, tlb_size*sizeof(uint8_t));
//     cudaError_t ret3 = cudaMalloc((void **)&tlb_C, tlb_size*sizeof(uint8_t));
//     if(ret1 != cudaSuccess || ret2 != cudaSuccess || ret3 != cudaSuccess){
//         printf("Error on allocation TLB!\n");
//         return -1;
//     }
//     h_tlb = (uint8_t *) malloc(tlb_size*sizeof(uint8_t));
//     for (int i = 0; i < tlb_size; i++) h_tlb[i] = 0;
//     ret1 = cudaMemcpy(tlb_A, h_tlb, tlb_size*sizeof(uint8_t), cudaMemcpyHostToDevice);
//     ret2 = cudaMemcpy(tlb_B, h_tlb, tlb_size*sizeof(uint8_t), cudaMemcpyHostToDevice);
//     ret3 = cudaMemcpy(tlb_C, h_tlb, tlb_size*sizeof(uint8_t), cudaMemcpyHostToDevice);
//     if(ret1 != cudaSuccess || ret2 != cudaSuccess || ret3 != cudaSuccess){
//         printf("Error on allocation TLB!\n");
//         return -1;
//     }

    
    


//             // // Allocate memory for arrays d_A, d_B, and d_C on device
//             // int *d_A, *d_B, *d_C;
//             // cudaError_t state;
//             // state = cudaMallocManaged(&d_A, bytes);
//             // if(cudaSuccess != state){
//             // 	printf("error on cudaMallocManaged(&d_A, bytes): %d\n", state);
//             // }
//             // state = cudaMallocManaged(&d_B, bytes);
//             // if(cudaSuccess != state){
//             // 	printf("error on cudaMallocManaged(&d_B, bytes): %d\n", state);
//             // }
//             // state = cudaMallocManaged(&d_C, bytes);
//             // if(cudaSuccess != state){
//             // 	printf("error on cudaMallocManaged(&d_C, bytes): %d\n", state);
//             // }
//             // printf("line number %d\n", __LINE__);
//             // // Fill host arrays A and B
//             // for(int i=0; i<N; i++)
//             // {
//             // 	d_A[i] = 1.0;
//             // 	d_B[i] = 2.0;
//             // }
   
        

    

//     int *dev_a, *dev_b, *dev_c, *host_a;                      // 107374182
//     // ret1 = cudaMalloc((void **)&dev_a, thr_per_blk*blk_in_grid*sizeof(int));
//     // ret2 = cudaMalloc((void **)&dev_b, thr_per_blk*blk_in_grid*sizeof(int));
//     // ret3 = cudaMalloc((void **)&dev_c, thr_per_blk*blk_in_grid*sizeof(int));
//     // host_a = (int *) malloc(thr_per_blk*blk_in_grid*sizeof(int));
//     // if(ret1 != cudaSuccess || ret2 != cudaSuccess || ret3 != cudaSuccess){
//     //     printf("cuda error: %s, %d\n", __func__, __LINE__);
//     // }

//     // for(int i = 0; i < thr_per_blk*blk_in_grid; i++) host_a[i] = 2;
//     // ret2 = cudaMemcpy(dev_b, host_a, thr_per_blk*blk_in_grid*sizeof(int), cudaMemcpyHostToDevice);
//     // ret3 = cudaMemcpy(dev_b, host_a, thr_per_blk*blk_in_grid*sizeof(int), cudaMemcpyHostToDevice);
//     // if(ret2 != cudaSuccess || ret3 != cudaSuccess){
//     //     printf("cuda error: %s, %d\n", __func__, __LINE__);
//     // }

    
//     printf("thr_per_blk: %d, blk_in_grid: %d tlb_size: %d a: %p\n", thr_per_blk, blk_in_grid, tlb_size, A);

//     int timer_size = 4;
//     clock_t *dtimer = NULL;
// 	clock_t timer[thr_per_blk*timer_size];

//     if (cudaSuccess != cudaMalloc((void **)&dtimer, sizeof(clock_t) * timer_size * thr_per_blk)) 
// 	{
//         printf("Error on timer allocation!\n");
//         return -1;
//     }

//     // Launch kernel
//     ret1 = cudaDeviceSynchronize();
//     printf("ret: %d\n", ret1);
//     if(cudaSuccess != ret1){    
//         return -1;
//     }

//     cudaEvent_t event1, event2;
//     cudaEventCreate(&event1);
//     cudaEventCreate(&event2);
//     int data_size = mesg_size;


//     struct timespec start, finish, delta;
//     clock_gettime(CLOCK_REALTIME, &start);
//     cudaEventRecord(event1, (cudaStream_t)1); //where 0 is the default stream
    
//     // add_vectors_uvm<<< thr_per_blk,sblk_in_grid >>>(dev_a, dev_a, dev_c, thr_per_blk*blk_in_grid);
//     // add_vectors_rdma_64MB_512KB<<< thr_per_blk, blk_in_grid>>>((int *) A, (int *) B, (int *) C, bytes/sizeof(int), tlb_A, tlb_B, tlb_C, dtimer, data_size, num_iteration);
//     // add_vectors_rdma<<< thr_per_blk, blk_in_grid>>>((int *) A, (int *) B, (int *) C, bytes/sizeof(int), tlb_A, tlb_B, tlb_C, dtimer, /*d_post, d_poll,*/ data_size, num_iteration);
//     // test<<<2048*16, 512>>>(buf1, buf2, (int *) C);
//     // test<<<2048, 256>>>(buf1, buf2, buf3);
//     test2<<< 4096*2*4*2, 1024>>>(buf1, buf2, buf3);
//     cudaEventRecord(event2, (cudaStream_t) 1);
//     clock_gettime(CLOCK_REALTIME, &finish);
//     ret1 = cudaDeviceSynchronize();
    
    
//     //synchronize
//     cudaEventSynchronize(event1); //optional
//     cudaEventSynchronize(event2); //wait for the event to be executed!

//     //calculate time
//     float dt_ms;
//     cudaEventElapsedTime(&dt_ms, event1, event2);
//     sub_timespec(start, finish, &delta);
    
//         // read<<<1,1>>>(dev_array, 0);
//         // write<<<1,1>>>(dev_array, 71808, 3);
//         // read<<<1,1>>>(dev_array2, 71808-256);
//         // read_nonstop<<<1,1>>>(dev_array, 256);

//         // add_vectors_uvm<<< blk_in_grid, thr_per_blk >>>(d_A, d_B, d_C, bytes);
	
//     printf("ret1: %d\n", ret1);
//     if(cudaSuccess != ret1){
//         return -1;
//     }
//     rtr3 = cudaMemcpy(timer, dtimer, sizeof(clock_t)*timer_size*thr_per_blk, cudaMemcpyDeviceToHost);
//     if(rtr3 != cudaSuccess){
//         printf("Error on array copy!\n");
//         return -1;
//     }

//     clock_t cycles;
//     float g_usec_post;
//     cudaDeviceProp devProp;
//     cudaGetDeviceProperties(&devProp, 0);
//     printf("Cuda device clock rate = %d\n", devProp.clockRate);
//     float freq_post = (float)1/((float)devProp.clockRate*1000), max = 0;
//     printf("timer: \n");
//     float div, sum_div = 0, sum_time = 0;
//     // for(int i = 0; i < thr_per_blk; i++){
//     //     cycles = timer[timer_size*i+1] - timer[i*timer_size];
// 	//     g_usec_post = (float)((float)1/(devProp.clockRate*1000))*((cycles)) * 1000000;
//     //     // if (max < g_usec_post) max = g_usec_post;
//     //     printf("Posting - blockIdx.x: %d: %f \n", i, g_usec_post);
//     //     // div = dt_ms/(g_usec_post/1000);
//     //     // sum_div += div;
//     //     // sum_time += g_usec_post;
//     //     cycles = timer[timer_size*i+3] - timer[i*timer_size+2];
// 	//     g_usec_post = (float)((float)1/(devProp.clockRate*1000))*((cycles)) * 1000000;
//     //     // if (max < g_usec_post) max = g_usec_post;
//     //     printf("polling - blockIdx.x: %d: %f div: %f\n", i, g_usec_post, div);
//     //     // cycles = timer[timer_size*i+3] - timer[i*timer_size+2];
// 	//     // g_usec_post = (float)((float)1/(devProp.clockRate*1000))*((cycles)) * 1000000;
//     //     if (max < g_usec_post) max = g_usec_post;
//     //     // printf("Array B - blockIdx.x: %d: %f \n", i, g_usec_post);
//     // }

//     sum_div = sum_div/thr_per_blk;
//     sum_time = sum_time/thr_per_blk/1000;
//     printf("\nmax: %f\n", max);
//     printf("\nsum_div: %f sum_time: %f total time: %f\n", sum_div, sum_time, sum_time*sum_div);
//     printf("BW: %f GBps for data size: %d\n", (float)(thr_per_blk)*data_size*4*num_iteration/(dt_ms*0.001*1024*1024*1024), data_size*4);
//     clock_t max1 = timer[1];
//     clock_t min = timer[0];
//     for(int i = 0; i < thr_per_blk; i++){
//         if(max1 < timer[timer_size*i+1]) max1 = timer[timer_size*i + 1];
//         if(min > timer[timer_size*i]) min = timer[timer_size*i];
//         // cycles = timer[2*i+1] - timer[i*2];
// 	    // g_usec_post = (float)((float)1/(devProp.clockRate*1000))*((cycles)) * 1000000;
//         // if (max < g_usec_post) max = g_usec_post;
//         // printf("blockIdx.x: %d: %f \n", i, g_usec_post);
//     }
//     cycles = max1 - min;
//     g_usec_post = (float)((float)1/(devProp.clockRate*1000))*((cycles)) * 1000000;
//     printf("total timer: %f\n", g_usec_post);
//     printf("kernel time: %d.%.9ld dt_ms: %f\n", (int)delta.tv_sec, delta.tv_nsec, dt_ms);



//     // for(int i = 0; i < 512*100; i++)
//     //     printf(" %d ", h_array[i]);
//     // printf("\n");
//     // sleep(5);
//     rtr3 = cudaMemcpy(h_array, A, bytes, cudaMemcpyDeviceToHost);
//     if(rtr3 != cudaSuccess){
//         printf("Error on array copy of A to host!\n");
//         return -1;
//     }
//     printf("H_array: \n");
//     for(int i = 0; i < bytes/4; i++){
//         if(h_array[i] != 2){ 
//             if(i>0 && h_array[i-1] == 2){
//                 printf("start: A[%d]: %d qp: %d\n", i, h_array[i], (i/(REQUEST_SIZE/4))%15);
//             }
//             else if(i == 0){
//                 printf("start: A[%d]: %d qp: %d\n", i, h_array[i], (i/(REQUEST_SIZE/4))%15);
//             }
//             else if(h_array[i+1] == 2){
//                 printf("end: A[%d]: %d qp: %d\n", i, h_array[i], (i/(REQUEST_SIZE/4))%15);
//             }
           
//         }
//     }
//     printf("----------------------\n");
//     rtr3 = cudaMemcpy(h_array, B, bytes, cudaMemcpyDeviceToHost);
//     if(rtr3 != cudaSuccess){
//         printf("Error on array copy of B to host!\n");
//         return -1;
//     }
//     for(int i = 0; i < bytes/4; i++){
//         if(h_array[i] != 2){ 
//             if((i>0 && h_array[i-1] == 2) || (i == 0)){
//                     printf("start: B[%d]: %d qp: %d\n", i, h_array[i], (i/16384)%15);
//                 }
//                 else if(h_array[i+1] == 2){
//                     printf("end: B[%d]: %d qp: %d\n", i, h_array[i], (i/16384)%15);
//                 }
//         }
//     }
//     rtr3 = cudaMemcpy((void *) h_array, C, bytes, cudaMemcpyDeviceToHost);
//     if(rtr3 != cudaSuccess){
//         printf("Error on array copy of C to host!\n");
//         return -1;
//     }
//     for(int i = 0; i < bytes/4; i++){
//         if(h_array[i] != 2){ 
//             printf("error in C: C[%d]: %d\n", i, h_array[i]);
//             break;
//         }
//     }
//     // printf("C[0]: %d\n", h_array[0]);
//     // printf("C[1]: %d\n", h_array[1]);
//     // printf("C[49151]: %d\n", h_array[49151]);
//     // printf("C[524287]: %d\n", h_array[524287]);
//     // printf("C[524288]: %d\n", h_array[524288]);
//     // printf("C[524289]: %d\n", h_array[524289]);
//     // printf("C[600000]: %d\n", h_array[600000]);
//     // printf("C: %p\n", C);
//     printf("\n");

//     // delay(40);
// 	// // Number of bytes to allocate for N doubles
// 	// size_t bytes = N*sizeof(int);
// 	// printf("size: %d GB\n", sizeof(int)/4);
// 	// // Allocate memory for arrays A, B, and C on host
// 	// // int *A = (int*)malloc(bytes);
// 	// // int *B = (int*)malloc(bytes);
// 	// // int *C = (int*)malloc(bytes);

// 	// // Allocate memory for arrays d_A, d_B, and d_C on device
// 	// int *d_A, *d_B, *d_C;
// 	// cudaError_t state;
// 	// state = cudaMallocManaged(&d_A, bytes);
// 	// if(cudaSuccess != state){
// 	// 	printf("error on cudaMallocManaged(&d_A, bytes): %d\n", state);
// 	// }
// 	// state = cudaMallocManaged(&d_B, bytes);
// 	// if(cudaSuccess != state){
// 	// 	printf("error on cudaMallocManaged(&d_B, bytes): %d\n", state);
// 	// }
// 	// state = cudaMallocManaged(&d_C, bytes);
// 	// if(cudaSuccess != state){
// 	// 	printf("error on cudaMallocManaged(&d_C, bytes): %d\n", state);
// 	// }
// 	// printf("line number %d\n", __LINE__);
// 	// // Fill host arrays A and B
// 	// for(int i=0; i<N; i++)
// 	// {
// 	// 	d_A[i] = 1.0;
// 	// 	d_B[i] = 2.0;
// 	// }

// 	// // Copy data from host arrays A and B to device arrays d_A and d_B
// 	// // cudaMemcpy(d_A, A, bytes, cudaMemcpyHostToDevice);
// 	// // cudaMemcpy(d_B, B, bytes, cudaMemcpyHostToDevice);

// 	// // Set execution configuration parameters
// 	// //		thr_per_blk: number of CUDA threads per grid block
// 	// //		blk_in_grid: number of blocks in grid
// 	// int thr_per_blk = 256;
// 	// int blk_in_grid = ceil( float(N) / thr_per_blk );

	
// 	// // Copy data from device array d_C to host array C
// 	// // cudaMemcpy(C, d_C, bytes, cudaMemcpyDeviceToHost);

// 	// // Verify results
//     // double tolerance = 1.0e-14;
// 	// for(int i=0; i<N; i++)
// 	// {
// 	// 	if( fabs(d_C[i] - 3.0) > tolerance)
// 	// 	{ 
// 	// 		printf("\nError: value of d_C[%d] = %d instead of 3.0\n\n", i, d_C[i]);
// 	// 		exit(1);
// 	// 	}
// 	// }	

// 	// // Free CPU memory
// 	// // free(A);
// 	// // free(B);
// 	// // free(C);

// 	// // Free GPU memory
// 	// cudaFree(d_A);
// 	// cudaFree(d_B);
// 	// cudaFree(d_C);

// 	// printf("\n---------------------------\n");
// 	// printf("__SUCCESS__\n");
// 	// printf("---------------------------\n");
// 	// printf("N                 = %d\n", N);
// 	// printf("Threads Per Block = %d\n", thr_per_blk);
// 	// printf("Blocks In Grid    = %d\n", blk_in_grid);
// 	// printf("---------------------------\n\n");

//     // destroy(s_ctx);

// 	return 0;
// }