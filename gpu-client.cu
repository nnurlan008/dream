
#include <infiniband/verbs.h>
#include <rdma/rdma_cma.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <stdio.h>
#include <stdlib.h>
#include <netdb.h>
#include <unistd.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <infiniband/mlx5dv.h>
#include <linux/kernel.h>
#include <valgrind/memcheck.h>
#include <rdma/mlx5-abi.h>
#include <rdma/ib_user_verbs.h>
#include <linux/types.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stddef.h>
#include <endian.h>
#include <sys/time.h>

#include <sys/mman.h>

extern "C"{
  #include "gpu-utils.h"
}

#include <time.h>
 
void delay(int number_of_seconds)
{
    // Converting time into milli_seconds
    int milli_seconds = 1000 * number_of_seconds;
 
    // Storing start time
    clock_t start_time = clock();
 
    // looping till required time is not achieved
    while (clock() < start_time + milli_seconds)
        ;
}


#define mr_buffer_gpu 1
#define control_on_cpu 1 // 1: qp and cq on cpu, otherwise on gpu
#define wq_buffer_gpu !control_on_cpu
#define cq_buffer_gpu !control_on_cpu

const int TIMEOUT_IN_MS = 500000; /* ms */

static int on_addr_resolved(struct rdma_cm_id *id);
static int on_connection(struct rdma_cm_id *id, unsigned long offset);
static int on_disconnect(struct rdma_cm_id *id);
static int on_event(struct rdma_cm_event *event);
static int on_route_resolved(struct rdma_cm_id *id);
static void usage(const char *argv0);
void post_receives(struct connection *conn);
void build_context(struct ibv_context *verbs);
void build_qp_attr(struct ibv_qp_init_attr *qp_attr);
void register_memory(struct connection *conn);

__device__ int gpu_poll_cq(void *cq_buf, void *twc, uint32_t *cons_index,
                           int ibv_cqe, uint32_t cqe_sz, int n, void *cq_dbrec,
                           void *mctx_t, void *dev_rsc, int refcnt,
                           void *qp_context, int dump_fill_mkey_be,
                           void *dev_rq, void *dev_wrid, uint64_t wrid_1,
                           void *cqe_dev, int cond, void *dev_scat_address,
						               clock_t *timer, uint64_t wrid_0, unsigned int wqe_head_0,
                           void *dev_wq
                           /*, void **table table, int refcnt*/);

__device__ int poll( void *cq_buf, struct ibv_wc *twc, uint32_t *cons_index,
                            int ibv_cqe, uint32_t cqe_sz, int n, void *cq_dbrec,
                            void *dev_rsc, void *dev_wrid, 
							              uint64_t wrid_0, unsigned int wqe_head_0,
                            void *dev_wq) ;

__device__ int post(unsigned int qpbf_bufsize,
          uint64_t wr_rdma_remote_addr, uint32_t wr_rdma_rkey,
          uint32_t wr_sg_length, uint32_t wr_sg_lkey, uint64_t wr_sg_addr, int wr_opcode, 
          uint32_t qp_num, uint64_t wr_id, struct ibv_send_wr *wr,
          void *qp_buf,  void *dev_qpsq_wqe_head, 
          void *dev_qp_sq, void *dev_qp_db, void *dev_wrid,
          void *bf_reg);

// __device__ int device_gpu_post_send(
//           unsigned int qpbf_bufsize, uint64_t wr_rdma_remote_addr, uint32_t wr_rdma_rkey,
//           int wr_opcode, unsigned int bf_offset, uint32_t qp_num, uint64_t wr_id,
//           void *qp_buf, void *dev_qpsq_wr_data, void *dev_qpsq_wqe_head, 
//           void *dev_qp_sq, void *bf_reg, void *dev_qp_db, void *dev_wr_sg, void *dev_wr_data);

// __global__ void global_gpu_post_send(
//           unsigned int qpbf_bufsize, uint64_t wr_rdma_remote_addr, uint32_t wr_rdma_rkey,
//           int wr_opcode, unsigned int bf_offset, uint32_t qp_num, uint64_t wr_id,
//           void *qp_buf, void *dev_qpsq_wr_data, void *dev_qpsq_wqe_head, 
//           void *dev_qp_sq, void *bf_reg, void *dev_qp_db, void *dev_wr_sg, void *dev_wr_data, int *ret);

static struct context *s_ctx = NULL;
static enum mode s_mode = M_WRITE;

void die(const char *reason)
{
  fprintf(stderr, "%s\n", reason);
  exit(-1);
}

#define TEST_NZ(x) do { if ( (x)) die("error: " #x " failed (returned non-zero)." ); } while (0)
#define TEST_Z(x)  do { if (!(x)) die("error: " #x " failed (returned zero/null)."); } while (0)

#define PORT 9700

int host_gpu_post_send(struct ibv_qp *ibqp, struct ibv_send_wr *wr, 
                        struct ibv_send_wr **bad_wr,unsigned long offset);

void set_mode(enum mode m)
{
  s_mode = m;
}

void * get_local_message_region(void *context)
{
  if (s_mode == M_WRITE)
    return ((struct connection *)context)->rdma_local_region;
  else
    return ((struct connection *)context)->rdma_remote_region;
}

void on_connect(void *context)
{
  ((struct connection *)context)->connected = 1;
}

void destroy_connection(void *context)
{
  struct connection *conn = (struct connection *)context;

  rdma_destroy_qp(conn->id);

  ibv_dereg_mr(conn->send_mr);
  ibv_dereg_mr(conn->recv_mr);
  ibv_dereg_mr(conn->rdma_local_mr);
  ibv_dereg_mr(conn->rdma_remote_mr);

  free(conn->send_msg);
  free(conn->recv_msg);
  free(conn->rdma_local_region);
  free(conn->rdma_remote_region);

  rdma_destroy_id(conn->id);

  free(conn);
}

void build_params(struct rdma_conn_param *params)
{
  memset(params, 0, sizeof(*params));

  params->initiator_depth = params->responder_resources = 1;
  params->rnr_retry_count = 7; /* infinite retry */
}

void send_message(struct connection *conn, unsigned long offset)
{
  struct ibv_send_wr wr, *bad_wr = NULL;
  struct ibv_sge sge;

  memset(&wr, 0, sizeof(wr));

  wr.wr_id = (uintptr_t)conn;
  wr.opcode = IBV_WR_SEND;
  wr.sg_list = &sge;
  wr.num_sge = 1;
  wr.send_flags = IBV_SEND_SIGNALED;

  sge.addr = (uintptr_t)conn->send_msg;
  sge.length = sizeof(struct message);
  sge.lkey = conn->send_mr->lkey;
  printf("Function: %s line number: %d sge.addr: 0x%llx\n", __func__, __LINE__, sge.addr);
  printf("Function: %s line number: %d sge.length: %d\n", __func__, __LINE__, sge.length);
  while (!conn->connected);

  if(control_on_cpu)
    TEST_NZ(ibv_post_send(conn->qp, &wr, &bad_wr));
  else 
    TEST_NZ(host_gpu_post_send(conn->qp, &wr, &bad_wr, offset));
}

void send_mr(void *context, unsigned long offset)
{
  struct connection *conn = (struct connection *)context;

  // conn->send_msg->type = MSG_MR;
  memcpy(&conn->send_msg->data.mr, conn->rdma_remote_mr, sizeof(struct ibv_mr));

  send_message(conn, offset);
}

int process_cm_event(struct rdma_event_channel *cm_channel,
                     enum rdma_cm_event_type expected_event,
                     struct rdma_cm_event **cm_event,
                     struct rdma_cm_event *copy_event);

int process_work_completion_events (struct ibv_comp_channel *comp_channel, 
		struct ibv_wc *wc, int max_wc);

int gpu_process_work_completion_events (struct ibv_comp_channel *comp_channel, 
		struct ibv_wc *wc, int max_wc);

int host_gpu_poll_cq (struct ibv_cq *cq_ptr, int num_entries, struct ibv_wc *wc);

int host_gpu_poll_cq2 (struct ibv_cq *cq_ptr, int num_entries, struct ibv_wc *wc);

int host_gpu_post_recv(struct ibv_qp *ibqp, struct ibv_recv_wr *wr,
		   struct ibv_recv_wr **bad_wr);

int host_gpu_post_write(struct ibv_qp *ibqp, struct ibv_send_wr *wr, struct ibv_send_wr **bad_wr);

int cpu_poll_cq(struct ibv_cq *ibcq, int n, struct ibv_wc *wc) ;

void process_gpu_mr(int *addr, int size);

int* /*volatile*/ read_after_write_buffer = NULL;
struct ibv_mr *read_after_write_mr;
// void *cqbuf, *wqbuf;
void* /*volatile*/ cqbuf = NULL;
void* /*volatile*/ wqbuf = NULL;
int wqbuf_size = 8192, cqbuf_size = 4096; // 32768;

__global__ void gpu_benchmark(uint32_t mesg_size, uint64_t wr_id, uint32_t peer_rkey, uintptr_t peer_addr, uintptr_t local_address, uint32_t lkey,
              /* post*/       uint32_t qpn,
              /* post*/       void *dev_qpsq_wqe_head, void *qp_buf,void *dev_qp_sq, void *dev_qp_db,
              /* post*/       void *qp_sq_wrid, void *bf_reg,
              /*poll*/        int ibv_cqe, uint32_t cqe_sz, int n, uint64_t wrid_0, unsigned int wqe_head_0,
                              void *dev_wq, 
              /*poll*/        void *cq_buf, uint32_t *cons_index, void *cq_dbrec, void *dev_rsc, 
                              void *dev_wrid, clock_t *dtimer, int result);

__device__ int poll_read_write( void *cq_buf, void *twc, uint32_t *cons_index,
                            int ibv_cqe, uint32_t cqe_sz, int n, void *cq_dbrec,
                            void *dev_rsc, 
                            // void *dev_wrid,
							              uint64_t wrid_0, unsigned int wqe_head_0,
                            void *dev_wq);

__global__ void gpu_whole(
          unsigned int qpbf_bufsize, struct ibv_send_wr wr1, 
          uint32_t wr_sg_length, uint32_t wr_sg_lkey, uint64_t wr_sg_addr,
          int wr_opcode, uint32_t qp_num, uint64_t wr_id,
          void *qp_buf, /*void *dev_qpsq_wr_data,*/ void *dev_qpsq_wqe_head, 
          void *dev_qp_sq, void *dev_qp_db,/*void *dev_wr_sg,*/ void *dev_wrid1,
          void *bf_reg,

          void *cq_buf, void *cons_index,
          int ibv_cqe, uint32_t cqe_sz, int max_wc, 
          void *dev_cq_dbrec,
          void* dev_rsc,
          void *dev_wrid,
          clock_t *timer, uint64_t wrid_0, 
          unsigned int wqe_head_0, void *dev_wq    
);

__device__ int read_write_post(
          uint64_t wr_rdma_remote_addr, uint32_t wr_rdma_rkey,
          uint32_t wr_sg_length, uint32_t wr_sg_lkey, uint64_t wr_sg_addr, int wr_opcode, 
          uint32_t qp_num, uint64_t wr_id, 
          void *qp_buf, void *dev_qpsq_wqe_head, 
          void *dev_qp_sq, void *dev_qp_db, void *dev_wrid,
          void *bf_reg);

int process_work_completion_events2 (struct ibv_comp_channel *comp_channel, 
		struct ibv_wc *wc, int max_wc);

int host_gpu_post_send2(struct ibv_qp *ibqp, struct ibv_send_wr *wr,
                       struct ibv_send_wr **bad_wr, unsigned long offset);

long int calculate_matrix(int *array, int length);

int start_polling_on_cpu(struct ibv_cq *ibcq, int n, struct ibv_wc *wc);

int cpu_poll_cq2(void *cq_buf, void *twc, uint32_t *cons_index,
                 int ibv_cqe, uint32_t cqe_sz, int n, void *cq_dbrec,
                 void *mctx_t, void *dev_rsc, int refcnt,
                 void *qp_context, int dump_fill_mkey_be,
                 void *dev_rq, void *dev_wrid, uint64_t wrid_1,
                 void *cqe_dev, int cond, void *dev_scat_address,
                 clock_t *timer, uint64_t wrid_0, unsigned int wqe_head_0,
                 void *dev_wq);

__global__ void multiple_packets(int num_of_packets,
          unsigned int qpbf_bufsize, struct ibv_send_wr wr1,  int mesg_size,
          uint32_t wr_sg_length, uint32_t wr_sg_lkey, uint64_t wr_sg_addr,
          int wr_opcode, uint32_t qp_num, uint64_t wr_id,
          void *qp_buf, void *dev_qpsq_wqe_head, 
          void *dev_qp_sq, void *dev_qp_db, void *dev_wrid1,
          void *bf_reg,

          void *cq_buf, void *cons_index,
          int ibv_cqe, uint32_t cqe_sz, int max_wc, 
          void *dev_cq_dbrec,
          void* dev_rsc,
          void *dev_wrid,
          clock_t *timer, uint64_t wrid_0, 
          unsigned int wqe_head_0, void *dev_wq    
);

__global__ void invalidate_cq(void *buf){
  int i;
  int nent = 16;
  int cqe_sz = 64;
  struct mlx5_cqe64 *cqe;
  for (i = 0; i < nent; ++i) {
    cqe = (struct mlx5_cqe64 *) (buf + i * cqe_sz);
    cqe += cqe_sz == 128 ? 1 : 0;
    cqe->op_own = MLX5_CQE_INVALID << 4;
  }
}

int cpu_benchmark(struct ibv_cq *cq_ptr, int num_entries, struct ibv_wc *wc,
                   struct ibv_qp *ibqp, struct ibv_send_wr *wr,
                   struct ibv_send_wr **bad_wr);

int cpu_benchmark_whole(struct ibv_cq *cq_ptr, int num_entries, struct ibv_wc *wc,
                   struct ibv_qp *ibqp, struct ibv_send_wr *wr,
                   struct ibv_send_wr **bad_wr, int num_packets, int mesg_size, float *bandwidth);

int main(int argc, char **argv)
{
  struct addrinfo *addr;
  struct rdma_cm_event *event = NULL;
  struct rdma_cm_event event_copy;
  struct rdma_cm_id *conn_id= NULL;
  struct rdma_event_channel *ec = NULL;
 
  if (argc != 6)
    usage(argv[0]);
  if (strcmp(argv[1], "write") == 0)
    set_mode(M_WRITE);
  else if (strcmp(argv[1], "read") == 0)
    set_mode(M_READ);
  else
    usage(argv[0]);

  int num_msg = (unsigned long) atoi(argv[4]);
  int mesg_size = (unsigned long) atoi(argv[5]);

  TEST_NZ(getaddrinfo(argv[2], "9700", NULL, &addr));

  TEST_Z(ec = rdma_create_event_channel());
  TEST_NZ(rdma_create_id(ec, &conn_id, NULL, RDMA_PS_IB));
  TEST_NZ(rdma_resolve_addr(conn_id, NULL, addr->ai_addr, TIMEOUT_IN_MS));

  freeaddrinfo(addr);
  int ret;

  if (wq_buffer_gpu == 1){

  ret = cudaMalloc((void **)&wqbuf, wqbuf_size);
    if (cudaSuccess != ret) {
      printf("error on cudaMalloc for wqbuf: %d\n", ret);
      exit(0);
    }
    ret = cudaMemset(wqbuf,0,wqbuf_size);
    if (cudaSuccess != ret) {
      printf("error on cudaMemset for wqbuf: %d\n", ret);
      exit(0);
    }
  }
  else if(wq_buffer_gpu == 2){
    wqbuf = malloc(wqbuf_size);
    memset(wqbuf,0,wqbuf_size);
    if(posix_memalign((void **) &wqbuf, 4096, wqbuf_size)) exit(-1);
  }
  else{
    wqbuf = NULL;
    wqbuf_size = 0;
  }
  printf("Function: %s line number: %d\n",__func__, __LINE__);
  /*Allocate control buffers - CQ and WQ*/
  if(cq_buffer_gpu == 1){
    ret = cudaMalloc((void **) &cqbuf, cqbuf_size);
    if (cudaSuccess != ret) {
      printf("error on cudaMalloc for cqbuf: %d\n", ret);
      exit(0);
    }
    ret = cudaMemset(cqbuf,0,cqbuf_size);
    if (cudaSuccess != ret) {
      printf("error on cudaMemset for cqbuf: %d\n", ret);
      exit(0);
    }
    invalidate_cq<<<1,1>>>(cqbuf);
  }
  else if(cq_buffer_gpu == 2){
    cqbuf = malloc(cqbuf_size);
    memset(cqbuf,0,cqbuf_size);
    if(posix_memalign((void **) &cqbuf, 4096, cqbuf_size)) exit(-1);
  }
  else {
    cqbuf = NULL;
    cqbuf_size = 0;
  }

  printf("Function: %s line number: %d\n",__func__, __LINE__);
  printf("Function: %s line number: %d cqbuf: 0x%llx\n",__func__, __LINE__, cqbuf);
  if ((unsigned long)cqbuf & 0xF00000000 != 0xb00000000)
    printf("true\n");
  else printf("false\n");
  // exit(0);
    /*Connection Set up*/
  if (process_cm_event(ec, RDMA_CM_EVENT_ADDR_RESOLVED, &event, &event_copy))
    return -1;
 
  if (on_addr_resolved(event_copy.id)){
    printf("error on on_addr_resolved\n");
    return -1;
  }
  printf("Function: %s line number: %d\n",__func__, __LINE__);

  if (process_cm_event(ec, RDMA_CM_EVENT_ROUTE_RESOLVED, &event, &event_copy))
    return -1;
  printf("Function: %s line number: %d\n",__func__, __LINE__);
  if (on_route_resolved(event_copy.id)){
    printf("error on on_route_resolved!\n");
    return -1;
  }
  printf("Function: %s line number: %d\n",__func__, __LINE__);
  if (process_cm_event(ec, RDMA_CM_EVENT_ESTABLISHED, &event, &event_copy))
    return -1;
  printf("Function: %s line number: %d\n",__func__, __LINE__);
  if(on_connection(event_copy.id, 56)){
    printf("error on on_connection!\n");
    return -1;
  }
  printf("Function: %s line number: %d\n",__func__, __LINE__);
  struct ibv_wc wc;
  // struct mlx5_qp *qp = to_mqp(event->id->qp);
  // printf("Function: %s line number: %d qp->buf.length: %d\n",__func__, __LINE__, qp->buf.length);

  process_work_completion_events (s_ctx->comp_channel, &wc, 1);
  printf("Function: %s line number: %d wc.opcode: %d\n",__func__, __LINE__, wc.opcode);
  printf("Function: %s line number: %d\n",__func__, __LINE__);
  // exit(0);
  process_work_completion_events (s_ctx->comp_channel, &wc, 1);
  struct connection *conn = (struct connection *)(uintptr_t)wc.wr_id;
  if (wc.opcode & IBV_WC_RECV){
    printf("receive completed\n");
    printf("Function: %s line number: %d \n",__func__, __LINE__);
    memcpy(&conn->peer_mr, &conn->recv_msg->data.mr, sizeof(conn->peer_mr));
    printf("Function: %s line number: %d conn->recv_msg->data.mr.addr: 0x%llx\n",__func__, __LINE__, conn->recv_msg->data.mr.addr);
    printf("Function: %s line number: %d conn->recv_msg->data.mr.rkey: %u\n",__func__, __LINE__, conn->recv_msg->data.mr.rkey);
  //   conn->peer_mr.addr;
  // wr.wr.rdma.rkey = conn->peer_mr.rkey
    post_receives(conn);
    // exit(0);
  }
  else {
    printf("not received: %d\n", wc.opcode);
    exit(-1);
  }
  printf("Function: %s line number: %d\n",__func__, __LINE__);
  /*Post some requests*/
  // read request:
  conn = (struct connection *)(uintptr_t)wc.wr_id;
  struct ibv_send_wr wr, *bad_wr = NULL;
  struct ibv_sge sge;

  // if (s_mode == M_WRITE)
  //   printf("received MSG_MR. writing message to remote memory...\n");
  // else // M_READ
  //   printf("received MSG_MR. reading message from remote memory...\n");

  // memset(&wr, 0, sizeof(wr));
  // wr.wr_id = (uintptr_t)conn;
  // wr.opcode = IBV_WR_RDMA_READ;
  // wr.sg_list = &sge;
  // wr.num_sge = 1;
  // wr.send_flags = IBV_SEND_SIGNALED;
  // wr.wr.rdma.remote_addr = (uintptr_t)conn->peer_mr.addr;
  // wr.wr.rdma.rkey = conn->peer_mr.rkey;
  // sge.addr = (uintptr_t)conn->rdma_local_region;
  // sge.length = RDMA_BUFFER_SIZE;
  // sge.lkey = conn->rdma_local_mr->lkey;
  // TEST_NZ(ibv_post_send(conn->qp, &wr, &bad_wr));
  // process_work_completion_events (s_ctx->comp_channel, &wc, 1);
  // printf("remote buffer: %s\n", conn->rdma_local_region);

  // conn = (struct connection *)(uintptr_t)wc.wr_id;
  // bad_wr = NULL;
  // memset(&wr, 0, sizeof(wr));
  // wr.wr_id = (uintptr_t)conn;
  // wr.opcode = IBV_WR_RDMA_READ;
  // wr.sg_list = &sge;
  // wr.num_sge = 1;
  // wr.send_flags = IBV_SEND_SIGNALED;
  // wr.wr.rdma.remote_addr = (uintptr_t)conn->peer_mr.addr;
  // wr.wr.rdma.rkey = conn->peer_mr.rkey;
  // sge.addr = (uintptr_t)conn->rdma_local_region;
  // sge.length = RDMA_BUFFER_SIZE;
  // sge.lkey = conn->rdma_local_mr->lkey;
  // TEST_NZ(ibv_post_send(conn->qp, &wr, &bad_wr));
  // process_work_completion_events (s_ctx->comp_channel, &wc, 1);
  // printf("remote buffer: %s\n", conn->rdma_local_region);

  // printf("sizeof(struct mlx5_wqe_data_seg): %d \n\n\n", sizeof(struct mlx5_wqe_data_seg));
  // exit(0);

  
  // struct ibv_wc wc1;
  // conn = (struct connection *)(uintptr_t)wc.wr_id;
  // bad_wr = NULL;
  // memset(&wr, 0, sizeof(wr));
  // wr.wr_id = (uintptr_t)conn;
  // wr.opcode = IBV_WR_RDMA_READ;
  // wr.sg_list = &sge;
  // wr.num_sge = 1;
  // wr.send_flags = IBV_SEND_SIGNALED;
  // wr.wr.rdma.remote_addr = (uintptr_t)conn->peer_mr.addr;
  // wr.wr.rdma.rkey = conn->peer_mr.rkey;
  // sge.addr = (uintptr_t)read_after_write_buffer; // conn->rdma_local_region;
  // sge.length = RDMA_BUFFER_SIZE;
  // sge.lkey = read_after_write_mr->lkey; // conn->rdma_local_mr->lkey;
  // TEST_NZ(host_gpu_post_send(conn->qp, &wr, &bad_wr, offset));
  // process_work_completion_events (s_ctx->comp_channel, &wc1, 1);
  // printf("\n\nread from remote buffer: %s\n\n", conn->rdma_local_region);
  // printf("Function: %s line number: %d\n",__func__, __LINE__);
  // process_gpu_mr(read_after_write_buffer);

  /*post*/
  uintptr_t wr_id = (uintptr_t) conn;
  uint32_t peer_rkey  = conn->peer_mr.rkey;
  uintptr_t peer_addr = (uintptr_t)conn->peer_mr.addr;
  uintptr_t local_address = (uintptr_t)read_after_write_buffer;
  uint32_t lkey = read_after_write_mr->lkey;
  uint32_t qpn = conn->qp->qp_num;

  void *dev_qpsq_wqe_head;
  void *qp_buf = wqbuf; 
  void *dev_qp_sq; 
  void *dev_qp_db;
  void *qp_sq_wrid; 
  void *bf_reg;

  struct mlx5_qp *qp = to_mqp(conn->qp);

  cudaError_t cudaState = cudaHostRegister(qp->sq.wqe_head, sizeof(qp->sq.wqe_head), cudaHostRegisterMapped);
  if(cudaState != cudaErrorHostMemoryAlreadyRegistered && cudaState !=  cudaSuccess)
        exit(0);
  // get GPU pointer for qp->sq.wqe_head
  if(cudaHostGetDevicePointer(&dev_qpsq_wqe_head, qp->sq.wqe_head, 0) != cudaSuccess)
      exit(0);

  cudaState = cudaHostRegister(&qp->sq, sizeof(qp->sq), cudaHostRegisterMapped);
  if(cudaState != cudaErrorHostMemoryAlreadyRegistered && cudaState !=  cudaSuccess)
        exit(0);
  // get GPU pointer for qp->sq
  if(cudaHostGetDevicePointer(&dev_qp_sq, &qp->sq, 0) != cudaSuccess)
      exit(0);

  cudaState = cudaHostRegister(qp->db, sizeof(qp->db), cudaHostRegisterMapped);
  if(cudaState != cudaSuccess && cudaState != cudaErrorHostMemoryAlreadyRegistered)
        exit(0);
  // printf("Function name: %s, line number: %d\n", __func__, __LINE__);
  // get GPU pointer for qp->db
  if(cudaHostGetDevicePointer(&dev_qp_db, qp->db, 0) != cudaSuccess)
      exit(0);

  cudaState = cudaHostRegister(qp->sq.wrid, sizeof(qp->sq.wrid), cudaHostRegisterMapped);
  // printf("Function name: %s, line number: %d cudaStatus: %d\n", __func__, __LINE__, cudaState);
  if(cudaState != cudaErrorHostMemoryAlreadyRegistered && cudaState != cudaSuccess)
        exit(0);
  // get GPU pointer for qp->sq
  if(cudaHostGetDevicePointer(&qp_sq_wrid, qp->sq.wrid, 0) != cudaSuccess)
      exit(0);

  cudaError_t cudaStatus1;

  cudaStatus1 = cudaHostRegister(qp->bf->reg,  8, cudaHostRegisterIoMemory);
    if (cudaStatus1 != cudaSuccess and cudaStatus1 != cudaErrorHostMemoryAlreadyRegistered) {
      exit(0);
    }
    cudaStatus1 = cudaHostGetDevicePointer(&bf_reg, qp->bf->reg, 0);
    if (cudaStatus1 != cudaSuccess) {
      printf("cudaHostGetDevicePointer successful with no error: %s\n", cudaGetErrorString(cudaStatus1));
      exit(0);
    }
    

  /*poll*/
  int ibv_cqe;
  uint32_t cqe_sz; 
  int n = 1;
  uint64_t wrid_0; 
  unsigned int wqe_head_0;

  void *dev_wq; 
  void *cq_buf = cqbuf; 
  uint32_t *cons_index; 
  void *cq_dbrec; 
  void *dev_rsc; 
  void *dev_wrid;
  struct mlx5_cq *cq = to_mcq(s_ctx->cq);
  struct mlx5_context *mctx = container_of(cq->verbs_cq.cq.context, struct mlx5_context, ibv_ctx.context);
  struct mlx5_resource *rsc = mctx->uidx_table[0].table[0];

  // struct mlx5_wq * wq_sq = &qp->sq;
  cudaState = cudaHostRegister(&qp->sq, sizeof(qp->sq), cudaHostRegisterMapped);
  if(cudaState !=  cudaSuccess && cudaState != cudaErrorHostMemoryAlreadyRegistered) 
      exit(0);
  if(cudaHostGetDevicePointer(&dev_wq, &qp->sq, 0) != cudaSuccess)
      exit(0);

  cudaState = cudaHostRegister(&cq->cons_index, sizeof(cq->cons_index), cudaHostRegisterMapped);
  if(cudaState != cudaSuccess && cudaState != cudaErrorHostMemoryAlreadyRegistered)
      exit(0);
  // get GPU pointer for cons index
  if(cudaHostGetDevicePointer(&cons_index, &cq->cons_index, 0) != cudaSuccess)
      exit(0);

  cudaState = cudaHostRegister(cq->dbrec, sizeof(cq->dbrec), cudaHostRegisterMapped);
  if(cudaState != cudaErrorHostMemoryAlreadyRegistered && cudaState !=  cudaSuccess) 
      exit(0);
  if(cudaHostGetDevicePointer(&cq_dbrec, cq->dbrec, 0) != cudaSuccess)
      exit(0);

  cudaState = cudaHostRegister(rsc, sizeof(rsc), cudaHostRegisterMapped);
  if(cudaState != cudaErrorHostMemoryAlreadyRegistered && cudaState !=  cudaSuccess) 
      exit(0);
  if(cudaHostGetDevicePointer(&dev_rsc, rsc, 0) != cudaSuccess)
      exit(0);

  cudaState = cudaHostRegister(qp->rq.wrid, sizeof(qp->rq.wrid), cudaHostRegisterMapped);
  if(cudaState != cudaSuccess && cudaState != cudaErrorHostMemoryAlreadyRegistered) 
      exit(0);
  if(cudaHostGetDevicePointer(&dev_wrid, qp->rq.wrid, 0) != cudaSuccess)
      exit(0);

  // int a = cudaDeviceSynchronize();
  // if (a != 0){
  //     printf("cudasynchronize: %d\n", a); 
  //     exit(0);
  // }
  cudaDeviceProp devProp;
  cudaGetDeviceProperties(&devProp, 0);
  printf("Cuda device clock rate = %d\n", devProp.clockRate);
  clock_t *dtimer = NULL;
  const size_t time_size = 4;
	clock_t timer[time_size];
  // uint32_t mesg_size = RDMA_BUFFER_SIZE;
	if (cudaSuccess != cudaMalloc((void **)&dtimer, sizeof(clock_t) * time_size)) 
		exit(0);
  if (cudaSuccess != cudaMemset(dtimer, 0, sizeof(clock_t) * time_size)) 
		exit(0);

  int *array = (int *) malloc(mesg_size*sizeof(int));
  for(int i = 0; i < mesg_size; i++)
    array[i] = 2;
  
  int result = calculate_matrix(array, mesg_size);
  

  

  int *cpu_buffer = (int *) malloc(RDMA_BUFFER_SIZE*sizeof(int));
  struct ibv_mr *cpu_mr;

  for(int i = 0; i < mesg_size; i++)
    cpu_buffer[i] = 0;

  // TEST_Z(cpu_mr = ibv_reg_mr(
  //   s_ctx->pd, cpu_buffer, RDMA_BUFFER_SIZE*sizeof(int),
  //   IBV_ACCESS_LOCAL_WRITE
  // ));
  // printf("conn->peer_mr.addr: 0x%llx conn->peer_mr.rkey: %d\n", conn->peer_mr.addr, conn->peer_mr.rkey);
  // struct ibv_wc wc1;
  // conn = (struct connection *)(uintptr_t)wc.wr_id;
  // bad_wr = NULL;
  // memset(&wr, 0, sizeof(wr));
  // wr.wr_id = (uintptr_t)conn;
  // wr.opcode = IBV_WR_RDMA_READ;
  // wr.sg_list = &sge;
  // wr.num_sge = 1;
  // wr.send_flags = IBV_SEND_SIGNALED;
  // wr.wr.rdma.remote_addr = (uintptr_t)conn->peer_mr.addr;
  // wr.wr.rdma.rkey = conn->peer_mr.rkey;
  // sge.addr = (uintptr_t)read_after_write_buffer; // conn->rdma_local_region;
  // sge.length = RDMA_BUFFER_SIZE*sizeof(int)-4;
  // sge.lkey = read_after_write_mr->lkey; // conn->rdma_local_mr->lkey;
  // // TEST_NZ(host_gpu_post_send(conn->qp, &wr, &bad_wr, offset));
  // // clock_t start, end;
  // // double cpu_time_used;
  
  // struct timeval start, end;

  // gettimeofday(&start, NULL);
  // TEST_NZ(host_gpu_post_send(conn->qp, &wr, &bad_wr, offset)); 
  // // ret = (ibv_post_send(conn->qp, &wr, &bad_wr));
  // gettimeofday(&end, NULL);
  // printf("errno:%d \n", errno);
  // // if (ret != 0) exit(-1);

  // double time_taken;
 
  // time_taken = (end.tv_sec - start.tv_sec) * 1e6;
  // time_taken = (time_taken + (end.tv_usec - 
  //                           start.tv_usec)) * 1e-6;

  // printf("ret from post; %d\n", ret);
  // printf("POSTING - INTERNAL MEASUREMENT: %f seconds to execute \n", time_taken);
  // gettimeofday(&start, NULL);
  // // process_work_completion_events (s_ctx->comp_channel, &wc, 1);
  // while (ibv_poll_cq(s_ctx->cq /* the CQ, we got notification for */, 
  //     1 /* number of remaining WC elements*/,
  //     &wc/* where to store */) < 0);
  // gettimeofday(&end, NULL);
  // time_taken = (end.tv_sec - start.tv_sec) * 1e6;
  // time_taken = (time_taken + (end.tv_usec - 
  //                           start.tv_usec)) * 1e-6;

  // printf("ret from post; %d\n", ret);
  // printf("POLLING - INTERNAL MEASUREMENT: %f seconds to execute \n", time_taken);
  // // start_polling_on_cpu(s_ctx->cq, 1, &wc);
  // // printf("\n\nread from remote buffer: %s\n\n", conn->rdma_local_region);
  // printf("Function: %s line number: %d\n",__func__, __LINE__);
  // // process_gpu_mr(read_after_write_buffer);
  // int sum = calculate_matrix(cpu_buffer, mesg_size);
  // result = calculate_matrix(array, mesg_size);

  // // for(int i = 0; i < mesg_size; i++)
  // //   printf("%d ", cpu_buffer[i]);
  // process_gpu_mr(read_after_write_buffer);
  // if(sum == result - 2) {
  //   printf("equal - sum: %d\n", sum);
  //   printf("equal - result: %d\n", result);
  // }
  // else {
  //   printf("unequal - sum: %d\n", sum);
  //   printf("unequal - result: %d\n", result);
  // }

  // // working part:
  int num_tests = 1;

  float bandwidth;
  // for (int i = 0; i < num_tests; i++){
    // for (int )
    conn = (struct connection *)(uintptr_t)wc.wr_id;
    bad_wr = NULL;
    memset(&wr, 0, sizeof(wr));
    wr.wr_id = (uintptr_t)conn;
    wr.opcode = IBV_WR_RDMA_READ;
    wr.sg_list = &sge;
    wr.num_sge = 1;
    wr.send_flags = IBV_SEND_SIGNALED;
    wr.wr.rdma.remote_addr = (uintptr_t)conn->peer_mr.addr;
    wr.wr.rdma.rkey = conn->peer_mr.rkey;
    sge.addr = (uintptr_t)read_after_write_mr->addr; // conn->rdma_local_region;
    sge.length = mesg_size*sizeof(int);
    sge.lkey = read_after_write_mr->lkey; // conn->rdma_local_mr->lkey;
    printf("read_after_write_mr->addr: 0x%llx\n\n\n", read_after_write_mr->addr);
    printf("read_after_write_buffer: 0x%llx\n", read_after_write_buffer);
    struct timeval start1, end1, start2, end2;
    double time_taken1, time_taken2;
    if(control_on_cpu){
      gettimeofday(&start1, NULL);
      ibv_post_send(conn->qp, &wr, &bad_wr);
      gettimeofday(&end1, NULL);
      

      gettimeofday(&start2, NULL);
      while(cpu_poll_cq(s_ctx->cq, 1, &wc) != 0);
      gettimeofday(&end2, NULL);
      time_taken1 = (end1.tv_usec - start1.tv_usec);
      // time_taken1 = (time_taken1 + (end1.tv_usec - 
      //                       start1.tv_usec)) * 1e-6;
      printf("POSTING - INTERNAL MEASUREMENT: %f useconds to execute \n", time_taken1);
      time_taken2 = (end2.tv_usec - start2.tv_usec);
      // time_taken2 = (time_taken2 + (end2.tv_usec - 
      //                       start2.tv_usec)) * 1e-6;
      printf("POLLING - INTERNAL MEASUREMENT: %f useconds to execute \n", time_taken2);
    }
    else{
    cpu_benchmark_whole(s_ctx->cq, 1, &wc, conn->qp, &wr, &bad_wr, num_msg, mesg_size, &bandwidth);
    }
    process_gpu_mr(read_after_write_buffer, mesg_size);
    printf("Function: %s line number: %d bandwidth: %f\n",__func__, __LINE__, bandwidth);
    // delay(10);
  //   for (int c = 1; c <= 32767; c++)
  //      for (int d = 1; d <= 32767; d++)
  //      {}
  // }

  // TEST_NZ(host_gpu_post_send(conn->qp, &wr, &bad_wr, offset));
  // // TEST_NZ(ibv_post_send(conn->qp, &wr, &bad_wr));
  // process_work_completion_events (s_ctx->comp_channel, &wc, 1);
  // start_polling_on_cpu(s_ctx->cq, 1, &wc);
  // printf("\n\nread from remote buffer: %s\n\n", conn->rdma_local_region);
  printf("Function: %s line number: %d bandwidth: %f\n",__func__, __LINE__, bandwidth);


  process_gpu_mr(read_after_write_buffer, mesg_size);
  // int sum = calculate_matrix(cpu_buffer, mesg_size);
  // // result = calculate_matrix(array, mesg_size);

  // // for(int i = 0; i < mesg_size; i++)
  // //   printf("%d ", cpu_buffer[i]);

  // if(sum == result - 2) {
  //   printf("equal - sum: %d\n", sum);
  //   printf("equal - result: %d\n", result);
  // }
  // else {
  //   printf("unequal - sum: %d\n", sum);
  //   printf("unequal - result: %d\n", result);
  // }



  // conn = (struct connection *)(uintptr_t)wc.wr_id;
  // bad_wr = NULL;
  // memset(&wr, 0, sizeof(wr));
  // wr.wr_id = (uintptr_t)conn;
  // wr.opcode = IBV_WR_RDMA_READ;
  // wr.sg_list = &sge;
  // wr.num_sge = 1;
  // wr.send_flags = IBV_SEND_SIGNALED;
  // wr.wr.rdma.remote_addr = (uintptr_t)conn->peer_mr.addr;
  // wr.wr.rdma.rkey = conn->peer_mr.rkey;
  // sge.addr = (uintptr_t)read_after_write_mr->addr; // conn->rdma_local_region;
  // sge.length = RDMA_BUFFER_SIZE*sizeof(int)-4;
  // sge.lkey = read_after_write_mr->lkey; // conn->rdma_local_mr->lkey;
  // printf("read_after_write_mr->addr: 0x%llx\n\n\n", read_after_write_mr->addr);
  // cpu_benchmark_whole(s_ctx->cq, 1, &wc, conn->qp, &wr, &bad_wr);
  // // TEST_NZ(host_gpu_post_send(conn->qp, &wr, &bad_wr, offset));
  // // // TEST_NZ(ibv_post_send(conn->qp, &wr, &bad_wr));
  // // process_work_completion_events (s_ctx->comp_channel, &wc, 1);
  // // start_polling_on_cpu(s_ctx->cq, 1, &wc);
  // // printf("\n\nread from remote buffer: %s\n\n", conn->rdma_local_region);
  // printf("Function: %s line number: %d\n",__func__, __LINE__);


  // process_gpu_mr(read_after_write_buffer);
  // sum = calculate_matrix(cpu_buffer, mesg_size);
  // // result = calculate_matrix(array, mesg_size);

  // // for(int i = 0; i < mesg_size; i++)
  // //   printf("%d ", cpu_buffer[i]);

  // if(sum == result - 2) {
  //   printf("equal - sum: %d\n", sum);
  //   printf("equal - result: %d\n", result);
  // }
  // else {
  //   printf("unequal - sum: %d\n", sum);
  //   printf("unequal - result: %d\n", result);
  // }

  // for(int i = 0; i < mesg_size; i++)
  //   cpu_buffer[i] = 4;

  // conn = (struct connection *)(uintptr_t)wc.wr_id;
  // bad_wr = NULL;
  // memset(&wr, 0, sizeof(wr));
  // wr.wr_id = (uintptr_t)conn;
  // wr.opcode = IBV_WR_RDMA_WRITE;
  // wr.sg_list = &sge;
  // wr.num_sge = 1;
  // wr.send_flags = IBV_SEND_SIGNALED;
  // wr.wr.rdma.remote_addr = (uintptr_t)conn->peer_mr.addr;
  // wr.wr.rdma.rkey = conn->peer_mr.rkey;
  // sge.addr = (uintptr_t)cpu_buffer; // conn->rdma_local_region;
  // sge.length = RDMA_BUFFER_SIZE*sizeof(int)-4;
  // sge.lkey = cpu_mr->lkey; // conn->rdma_local_mr->lkey;
  // // printf("cpu_benchmark: \n\n\n");
  // // cpu_benchmark(s_ctx->cq, 1, &wc, conn->qp, &wr, &bad_wr);
  // TEST_NZ(host_gpu_post_send(conn->qp, &wr, &bad_wr, offset));
  // // // TEST_NZ(ibv_post_send(conn->qp, &wr, &bad_wr));
  // process_work_completion_events (s_ctx->comp_channel, &wc, 1);

  // conn = (struct connection *)(uintptr_t)wc.wr_id;
  // bad_wr = NULL;
  // memset(&wr, 0, sizeof(wr));
  // wr.wr_id = (uintptr_t)conn;
  // wr.opcode = IBV_WR_RDMA_READ;
  // wr.sg_list = &sge;
  // wr.num_sge = 1;
  // wr.send_flags = IBV_SEND_SIGNALED;
  // wr.wr.rdma.remote_addr = (uintptr_t)conn->peer_mr.addr;
  // wr.wr.rdma.rkey = conn->peer_mr.rkey;
  // sge.addr = (uintptr_t)cpu_buffer; // conn->rdma_local_region;
  // sge.length = RDMA_BUFFER_SIZE*sizeof(int)-4;
  // sge.lkey = cpu_mr->lkey; // conn->rdma_local_mr->lkey;
  // // printf("cpu_benchmark: \n\n\n");
  // // cpu_benchmark(s_ctx->cq, 1, &wc, conn->qp, &wr, &bad_wr);
  // TEST_NZ(host_gpu_post_send(conn->qp, &wr, &bad_wr, offset));
  // // // TEST_NZ(ibv_post_send(conn->qp, &wr, &bad_wr));
  // process_work_completion_events (s_ctx->comp_channel, &wc, 1);
  // // start_polling_on_cpu(s_ctx->cq, 1, &wc);
  // // printf("\n\nread from remote buffer: %s\n\n", conn->rdma_local_region);
  // printf("Function: %s line number: %d\n",__func__, __LINE__);
  // // process_gpu_mr(read_after_write_buffer);

  // for(int i = 0; i < mesg_size; i++)
  //   printf("%d ", cpu_buffer[i]);

  // cudaState = cudaDeviceSynchronize();
  // if (cudaState != 0){
  //     printf("cudasynchronize: %d\n", cudaState); 
  //     exit(0);
  // }
  // gpu_benchmark<<<1,1>>>(mesg_size, wr_id, peer_rkey, peer_addr, local_address, lkey,
  //         /* post*/       qpn,
  //         /* post*/       dev_qpsq_wqe_head, qp_buf, dev_qp_sq, dev_qp_db,
  //         /* post*/       qp_sq_wrid, bf_reg,
  //         /*poll*/        ibv_cqe, cqe_sz, n, wrid_0, wqe_head_0,
  //                         dev_wq, 
  //         /*poll*/        cq_buf, cons_index, cq_dbrec, dev_rsc, 
  //                         dev_wrid, dtimer, result);
  // cudaState = cudaDeviceSynchronize();
  // if (cudaState != 0){
  //     printf("cudasynchronize: %d\n", cudaState); 
  //     exit(0);
  // }

  // cudaMemcpy(timer, dtimer, sizeof(clock_t)*(time_size), cudaMemcpyDeviceToHost);
  // cudaFree(dtimer);
  
	// float freq = (float)1/(devProp.clockRate*1000);
	// float posting_usec = (float)((float)1/(devProp.clockRate*1000))*((timer[1]-timer[0])) * 1000000;
  // float polling_usec = (float)((float)1/(devProp.clockRate*1000))*((timer[3]-timer[2])) * 1000000;
  // float total_usec = (float)((float)1/(devProp.clockRate*1000))*((timer[3]-timer[0])-(timer[2] - timer[1])) * 1000000;
	// printf("POSTING - INTERNAL MEASUREMENT: %f useconds to execute \n", posting_usec);
  // printf("POLLING - INTERNAL MEASUREMENT: %f useconds to execute \n", polling_usec);
  // printf("Total - INTERNAL MEASUREMENT: %f useconds to execute \n", total_usec);


  // if (a != 0){
  //     printf("cudasynchronize: %d\n", a); 
  //     exit(0);
  // }

  // process_gpu_mr(read_after_write_buffer);

  printf("Final: Function name: %s, line number: %d\n", __func__, __LINE__);

  // conn = (struct connection *)(uintptr_t)wc.wr_id;
  // bad_wr = NULL;
  // memset(&wr, 0, sizeof(wr));
  // wr.wr_id = (uintptr_t)conn;
  // wr.opcode = IBV_WR_RDMA_WRITE;
  // wr.sg_list = &sge;
  // wr.num_sge = 1;
  // wr.send_flags = IBV_SEND_SIGNALED;
  // wr.wr.rdma.remote_addr = (uintptr_t)conn->peer_mr.addr;
  // wr.wr.rdma.rkey = conn->peer_mr.rkey;
  // sge.addr = (uintptr_t)conn->rdma_remote_region;
  // sge.length = RDMA_BUFFER_SIZE;
  // sge.lkey = conn->rdma_remote_mr->lkey;
  // TEST_NZ(host_gpu_post_send(conn->qp, &wr, &bad_wr));
  // process_work_completion_events (s_ctx->comp_channel, &wc, 1);
  // printf("\n\nwrite to remote buffer: %s\n\n", conn->rdma_remote_region);
  // printf("Function name: %s, line number: %d\n", __func__, __LINE__);

  

  // while (rdma_get_cm_event(ec, &event) == 0) {
  //   struct rdma_cm_event event_copy;

  //   memcpy(&event_copy, event, sizeof(*event));
  //   rdma_ack_cm_event(event);

  //   if (on_event(&event_copy))
  //     break;
  // }


  


  rdma_destroy_event_channel(ec);

  return 0;
}

long int calculate_matrix(int *array, int length){
  long int sum = 0; 
  for (int i = 0; i < length; i++)
    sum += array[i];

  return sum;
}

int process_cm_event(struct rdma_event_channel *cm_channel,
                     enum rdma_cm_event_type expected_event,
                     struct rdma_cm_event **cm_event,
                     struct rdma_cm_event *copy_event){

  int ret = rdma_get_cm_event(cm_channel, cm_event);
  
  if (ret != 0) {
    printf("Failed to retrieve a cm event\n");
    return -1;
  }
  if((*cm_event)->status != 0) {
    printf("Exceptected event: %d, status: %d\n", expected_event, (*cm_event)->status);
    rdma_ack_cm_event(*cm_event);
    return -1;
  }
  if((*cm_event)->event != expected_event){
    printf("Exceptected event: %d, Event on cm channel: %d\n", expected_event, (*cm_event)->event);
    rdma_ack_cm_event(*cm_event);
    return -1;
  }

  memcpy(copy_event, *cm_event, sizeof(**cm_event));
  rdma_ack_cm_event(*cm_event);
  return 0;
}
int mlx5_cq_event(struct ibv_comp_channel *channel, struct ibv_cq **cq);

int process_work_completion_events (struct ibv_comp_channel *comp_channel, 
		struct ibv_wc *wc, int max_wc){
  
  // struct ibv_cq *cq;
  void *context = NULL;

  // TEST_NZ(ibv_get_cq_event(comp_channel, &cq, &context));
  // mlx5_cq_event(comp_channel, &cq);
  // ibv_ack_cq_events(cq, 1);
  // TEST_NZ(ibv_req_notify_cq(cq, 0));
  // while (ibv_poll_cq(cq, 1, &wc))
  //     on_completion(&wc);

  // // int total_wc = 0;
  // struct mlx5_cq *cq = to_mcq(s_ctx->cq);
  // void *cqe;
  // struct mlx5_cqe64 *cqe64;
	// cqe = cq->active_buf->buf + (cq->cons_index & cq->verbs_cq.cq.cqe) * cq->cqe_sz;
  
  printf("Function: %s line number: %d\n",__func__, __LINE__);
  //   // cqe = cqe_dev;// cq_buf + ((*cons_index) & ibv_cqe) * cqe_sz;
	// cqe64 = (struct mlx5_cqe64 *)((cq->cqe_sz == 64) ? cqe : cqe + 64);
  // // printf("Function name: %s, line number: %d cqe64->op_own >> 4: %d\n", __func__, __LINE__, cqe64->op_own >> 4);
  host_gpu_poll_cq(s_ctx->cq /* the CQ, we got notification for */, 
      1  /* number of remaining WC elements*/,
      wc/* where to store */);

  // printf("Function: %s line number: %d\n",__func__, __LINE__);

  // clock_t start, end;
  // double cpu_time_used;
  
  // start = clock();
  
  // int ret;
  // do {
  //   ret = /*ibv_poll_cq host_gpu*/ibv_poll_cq(s_ctx->cq /* the CQ, we got notification for */, 
  //     1 /* number of remaining WC elements*/,
  //     wc/* where to store */);
  //     printf("polling: %d\n", ret);
  //   // printf("Failed to poll cq for wc due to %d \n", ret);
  //   if (ret < 0) {
  //     printf("Failed to poll cq for wc due to %d \n", ret);
  //     continue;
  //   /* ret is errno here */
  //     // return ret;
  //   } 
  // } while (ret < 0);

  // end = clock();
  // cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
  // printf("POLLING - INTERNAL MEASUREMENT: %f useconds to execute \n", cpu_time_used);

  // struct connection *conn = (struct connection *)(uintptr_t)wc->wr_id;
  printf("wc->status: %d\n", wc->status);
  if (wc->status != IBV_WC_SUCCESS){
    printf("wc->status: %d\n", wc->status);
    die("on_completion: status is not IBV_WC_SUCCESS.");
  }

  // if (wc->opcode & IBV_WC_RECV){
  //   printf("receive completed\n");
  // }
  // if(wc->opcode == IBV_WC_SEND){
  //   printf("send completed\n");
  // }
  // else printf("wc.opcode: %d\n", wc->opcode);
  return 0;
}

int process_work_completion_events2 (struct ibv_comp_channel *comp_channel, 
		struct ibv_wc *wc, int max_wc){
  
  struct ibv_cq *cq;
  void *context = NULL;
  int total_wc = 0;

  host_gpu_poll_cq2(s_ctx->cq /* the CQ, we got notification for */, 
      1 - total_wc /* number of remaining WC elements*/,
      wc + total_wc/* where to store */);

  
  return 0;
}

int gpu_process_work_completion_events (struct ibv_comp_channel *comp_channel, 
		struct ibv_wc *wc, int max_wc){
  
  struct ibv_cq *cq;
  void *context = NULL;

  TEST_NZ(ibv_get_cq_event(comp_channel, &cq, &context));
  ibv_ack_cq_events(cq, 1);
  TEST_NZ(ibv_req_notify_cq(cq, 0));
  // printf("Function name: %s, line number: %d\n", __func__, __LINE__);
  // while (ibv_poll_cq(cq, 1, &wc))
  //     on_completion(&wc);

  int total_wc = 0;
  do {
    int ret = ibv_poll_cq(cq /* the CQ, we got notification for */, 
      1 - total_wc /* number of remaining WC elements*/,
      wc + total_wc/* where to store */);

    if (ret < 0) {
      printf("Failed to poll cq for wc due to %d \n", ret);
      continue;
    /* ret is errno here */
      // return ret;
    }
    total_wc += ret;
  } while (total_wc < 1);

  struct connection *conn = (struct connection *)(uintptr_t)wc->wr_id;
  // printf("wc->status: %d\n", wc->status);
  if (wc->status != IBV_WC_SUCCESS){
    printf("wc->status: %d\n", wc->status);
    die("on_completion: status is not IBV_WC_SUCCESS.");
  }
  // if (wc->opcode & IBV_WC_RECV){
  //   printf("receive completed\n");
  // }
  // if(wc->opcode == IBV_WC_SEND){
  //   printf("send completed\n");
  // }
  // else printf("wc.opcode: %d\n", wc->opcode);
  return 0;
}

void build_connection(struct rdma_cm_id *id)
{
  struct connection *conn;
  struct ibv_qp_init_attr qp_attr;
  build_context(id->verbs);
  build_qp_attr(&qp_attr);
  if(wq_buffer_gpu == 1 || wq_buffer_gpu == 2)
    TEST_NZ(rdmax_create_qp(id, s_ctx->pd, &qp_attr, wqbuf, wqbuf_size));
  else
    TEST_NZ(rdma_create_qp(id, s_ctx->pd, &qp_attr));
     
  // printf("Function: %s line number: %d\n",__func__, __LINE__);

  // printf("Function: %s line number: %d\n",__func__, __LINE__);

  // printf("Function Name: %s, line number: %d\n", __func__, __LINE__);
  struct mlx5_context *ctx = to_mctx(s_ctx->pd->context);
  // printf("Function Name: %s, line number: %d,  ctx->bfs: 0x%llx\n", __func__, __LINE__, ctx->bfs);
  
  id->context = conn = (struct connection *)malloc(sizeof(struct connection));

  conn->id = id;
  conn->qp = id->qp;

//   conn->send_state = SS_INIT;
//   conn->recv_state = RS_INIT;

  conn->connected = 0;
  // printf("Function: %s line number: %d\n",__func__, __LINE__);
  register_memory(conn);
  post_receives(conn);
  // printf("Function: %s line number: %d\n",__func__, __LINE__);
}

void post_receives(struct connection *conn)
{
  // printf("Function: %s line number: %d\n",__func__, __LINE__);
  struct ibv_recv_wr wr, *bad_wr = NULL;
  struct ibv_sge sge;

  wr.wr_id = (uintptr_t)conn;
  wr.next = NULL;
  wr.sg_list = &sge;
  wr.num_sge = 1;

  sge.addr = (uintptr_t)conn->recv_msg;
  sge.length = sizeof(struct message);
  sge.lkey = conn->recv_mr->lkey;
  // printf("Function: %s line number: %d sge.length: %d\n",__func__, __LINE__, sge.length);
  // TEST_NZ(ibv_post_recv(conn->qp, &wr, &bad_wr));
  TEST_NZ(host_gpu_post_recv(conn->qp, &wr, &bad_wr));
  // printf("Function: %s line number: %d\n",__func__, __LINE__);
}

void register_memory(struct connection *conn)
{
  conn->send_msg = (struct message *) malloc(sizeof(struct message));
  conn->recv_msg = (struct message *) malloc(sizeof(struct message));

  conn->rdma_local_region = (char *) malloc(RDMA_BUFFER_SIZE);
  conn->rdma_remote_region = (char *) malloc(RDMA_BUFFER_SIZE);

  if (mr_buffer_gpu)
    cudaMalloc((void **) &read_after_write_buffer, RDMA_BUFFER_SIZE*sizeof(int));
  else
    read_after_write_buffer = (int *) malloc(RDMA_BUFFER_SIZE*sizeof(int));

  TEST_Z(read_after_write_mr = ibv_reg_mr(
    s_ctx->pd, read_after_write_buffer, RDMA_BUFFER_SIZE*sizeof(int),
    IBV_ACCESS_LOCAL_WRITE
  ));

  TEST_Z(conn->send_mr = ibv_reg_mr(
    s_ctx->pd, 
    conn->send_msg, 
    sizeof(struct message), 
    0));

  TEST_Z(conn->recv_mr = ibv_reg_mr(
    s_ctx->pd, 
    conn->recv_msg, 
    sizeof(struct message), 
    IBV_ACCESS_LOCAL_WRITE));

  TEST_Z(conn->rdma_local_mr = ibv_reg_mr(
    s_ctx->pd, 
    conn->rdma_local_region, 
    RDMA_BUFFER_SIZE, 
    ((s_mode == M_WRITE) ? 0 :  IBV_ACCESS_LOCAL_WRITE)));

  TEST_Z(conn->rdma_remote_mr = ibv_reg_mr(
    s_ctx->pd, 
    conn->rdma_remote_region, 
    RDMA_BUFFER_SIZE, 
    IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ));
    // ((s_mode == M_WRITE) ? (IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE) : IBV_ACCESS_REMOTE_READ)));
}

void build_qp_attr(struct ibv_qp_init_attr *qp_attr)
{
  memset(qp_attr, 0, sizeof(*qp_attr));

  qp_attr->send_cq = s_ctx->cq;
  qp_attr->recv_cq = s_ctx->cq;
  qp_attr->qp_type = IBV_QPT_RC;

  qp_attr->cap.max_send_wr = 10;
  qp_attr->cap.max_recv_wr = 10;
  qp_attr->cap.max_send_sge = 1;
  qp_attr->cap.max_recv_sge = 1;
}

// #define list_empty(h) list_empty_(h, LIST_LOC)
// #define list_debug(h, loc) ((void)loc, h)
// #define list_debug_node(n, loc) ((void)loc, n)
// #define LIST_LOC __FILE__  ":" stringify(__LINE__)
// #define list_off_(type, member)					\
// 	(container_off(type, member) +				\
// 	 check_type(((type *)0)->member, struct list_node))

// #define list_pop(h, type, member)					\
// 	((type *)list_pop_((h), list_off_(type, member)))

// #define list_del(n) list_del_(n, LIST_LOC)
// static inline void list_del_(struct list_node *n, const char* abortstr)
// {
// 	(void)list_debug_node(n, abortstr);
// 	n->next->prev = n->prev;
// 	n->prev->next = n->next;
// }

// static inline bool list_empty_(const struct list_head *h, const char* abortstr)
// {
// 	(void)list_debug(h, abortstr);
// 	return h->n.next == &h->n;
// }



// static inline const void *list_pop_(const struct list_head *h, size_t off)
// {
// 	struct list_node *n;

// 	if (list_empty(h))
// 		return NULL;
// 	n = h->n.next;
// 	list_del(n);
// 	return (const char *)n - off;
// }

void build_context(struct ibv_context *verbs)
{
  if (s_ctx) {
    if (s_ctx->ctx != verbs)
      die("cannot handle events in more than one context.");

    return;
  }

  s_ctx = (struct context *)malloc(sizeof(struct context));

  s_ctx->ctx = verbs;

  TEST_Z(s_ctx->pd = ibv_alloc_pd(s_ctx->ctx));
  TEST_Z(s_ctx->comp_channel = ibv_create_comp_channel(s_ctx->ctx));
  if (cq_buffer_gpu == 1 || cq_buffer_gpu == 2){
    TEST_Z(s_ctx->cq = ibvx_create_cq(s_ctx->ctx, 10, NULL, s_ctx->comp_channel, 0, cqbuf, cqbuf_size)); /* cqe=10 is arbitrary */
  }
  else
    TEST_Z(s_ctx->cq = ibv_create_cq(s_ctx->ctx, 10, NULL, s_ctx->comp_channel, 0)); /* cqe=10 is arbitrary */
  TEST_NZ(ibv_req_notify_cq(s_ctx->cq, 0));
  

 

  // TEST_NZ(pthread_create(&s_ctx->cq_poller_thread, NULL, poll_cq, NULL));
}

int on_addr_resolved(struct rdma_cm_id *id)
{
  printf("address resolved.\n");

  build_connection(id);
  // printf("Function: %s line number: %d\n",__func__, __LINE__);
  sprintf((char *) get_local_message_region(id->context), "Hello my friend.  with pid %d", getpid());
  TEST_NZ(rdma_resolve_route(id, TIMEOUT_IN_MS));

  return 0;
}

int on_connection(struct rdma_cm_id *id, unsigned long offset)
{
  on_connect(id->context);
  send_mr(id->context, offset);

  return 0;
}

int on_disconnect(struct rdma_cm_id *id)
{
  printf("disconnected.\n");

  destroy_connection(id->context);
  return 1; /* exit event loop */
}

// int on_event(struct rdma_cm_event *event)
// {
//   int r = 0;

//   if (event->event == RDMA_CM_EVENT_ADDR_RESOLVED)
//     r = on_addr_resolved(event->id);
//   else if (event->event == RDMA_CM_EVENT_ROUTE_RESOLVED)
//     r = on_route_resolved(event->id);
//   else if (event->event == RDMA_CM_EVENT_ESTABLISHED)
//     r = on_connection(event->id);
//   else if (event->event == RDMA_CM_EVENT_DISCONNECTED)
//     r = on_disconnect(event->id);
//   else {
//     fprintf(stderr, "on_event: %d\n", event->event);
//     die("on_event: unknown event.");
//   }

//   return r;
// }

int on_route_resolved(struct rdma_cm_id *id)
{
  struct rdma_conn_param cm_params;

  printf("route resolved.\n");
  build_params(&cm_params);
  // printf("Function: %s line number: %d\n",__func__, __LINE__);
  TEST_NZ(rdma_connect(id, &cm_params));
  // printf("Function: %s line number: %d\n",__func__, __LINE__);
  return 0;
}

void usage(const char *argv0)
{
  fprintf(stderr, "usage: %s <mode> <server-address> <server-port>\n  mode = \"read\", \"write\"\n", argv0);
  exit(1);
}

__global__ void global_gpu_poll_cq(void *cq_buf, void *twc, void *cons_index,
                            int ibv_cqe, uint32_t cqe_sz, int max_wc, 
                            int *total_wc, int *dev_ret, void *dev_cq_dbrec,
                            void *mctx, void* dev_rsc, int refcnt, void *qp_context,
                            int dump_fill_mkey_be, void *dev_rq,
                            void *dev_wrid, uint64_t wrid, void *cqe_dev, int cond,
                            void *dev_scat_address, clock_t *timer, uint64_t wrid_0, 
                            unsigned int wqe_head_0, void *dev_wq) 
{  
	// printf("Function: %s line number: %d\n",__func__, __LINE__);
    int ret;
    int total = 0;
    // printf("\n\n\ngpu polling\n\n\n");
    void *cqe = cq_buf + ((* (int *) cons_index) & ibv_cqe) * cqe_sz;
    struct mlx5_cqe64 *cqe64 = (struct mlx5_cqe64 *)((cqe_sz == 64) ? cqe_dev : cqe_dev + 64);
    
    printf("In: %s, cqe64->op_own: %d\n",__func__, cqe64->op_own);
    /*
    poll_read_write( void *cq_buf, void *twc, uint32_t *cons_index,
                            int ibv_cqe, uint32_t cqe_sz, int n, void *cq_dbrec,
                            void *dev_rsc, 
                            // void *dev_wrid,
							              uint64_t wrid_0, unsigned int wqe_head_0,
                            void *dev_wq)
    */
    
    timer[0] = clock();
    do {
        ret = gpu_poll_cq(cq_buf /* the CQ, we got notification for */, 
            twc + (total)/* where to store */,
            (uint32_t *) cons_index,
            ibv_cqe,
            cqe_sz,
            max_wc /* number of remaining WC elements*/,
            (uint32_t *) dev_cq_dbrec,
            (struct mlx5_context *) mctx,
            dev_rsc, refcnt, qp_context, dump_fill_mkey_be, dev_rq,
            dev_wrid, wrid, cqe_dev, cond, dev_scat_address, timer, wrid_0, wqe_head_0,
            dev_wq);
        // printf("ret: %d total_wc: %d\n", ret, *total_wc);
        // if (ret < 0) {
        //     // printf("Failed to poll cq for wc due to %d \n", ret);
        //     continue;
           
        // }
        // printf("gpu polling\n");
        // total += ret;
        // (*dev_ret) = ret;
    } while (ret < 0); 
    timer[1] = clock();
    // printf("%d WC are completed \n", *total_wc);
    
    // __syncthreads();

   
}

__global__ void global_gpu_poll_cq2(void *cq_buf, void *twc, void *cons_index,
                            int ibv_cqe, uint32_t cqe_sz, int max_wc, 
                            int *total_wc, int *dev_ret, void *dev_cq_dbrec,
                            void *mctx, void* dev_rsc, int refcnt, void *qp_context,
                            int dump_fill_mkey_be, void *dev_rq,
                            void *dev_wrid, uint64_t wrid, void *cqe_dev, int cond,
                            void *dev_scat_address, clock_t *timer, uint64_t wrid_0, 
                            unsigned int wqe_head_0, void *dev_wq) 
{  
	// printf("Function: %s line number: %d\n",__func__, __LINE__);
    int ret;
    int total = 0;
    // printf("\n\n\ngpu polling\n\n\n");

    /*
    poll_read_write( void *cq_buf, void *twc, uint32_t *cons_index,
                            int ibv_cqe, uint32_t cqe_sz, int n, void *cq_dbrec,
                            void *dev_rsc, 
                            // void *dev_wrid,
							              uint64_t wrid_0, unsigned int wqe_head_0,
                            void *dev_wq)
    */
    
    timer[0] = clock();
    do {
        ret = poll_read_write( cq_buf, twc, (uint32_t *) cons_index,
                            ibv_cqe, cqe_sz, 1, dev_cq_dbrec,
                            dev_rsc, 
                            // void *dev_wrid,
							              wrid_0, wqe_head_0,
                            dev_wq);
        // printf("ret: %d total_wc: %d\n", ret, *total_wc);
        if (ret < 0) {
            // printf("Failed to poll cq for wc due to %d \n", ret);
            continue;
           
        }
        // printf("gpu polling\n");
        // total += ret;
        // (*dev_ret) = ret;
    } while (total < max_wc); 
    timer[1] = clock();
    // printf("%d WC are completed \n", *total_wc);
    
    // __syncthreads();

   
}

__device__ int gpu_poll_cq( void *cq_buf, void *twc, uint32_t *cons_index,
                            int ibv_cqe, uint32_t cqe_sz, int n, void *cq_dbrec,
                            void *mctx_t, void *dev_rsc, int refcnt,
                            void *qp_context, int dump_fill_mkey_be,
                            void *dev_rq, void *dev_wrid, uint64_t wrid_1,
                            void *cqe_dev, int cond, void *dev_scat_address,
							              clock_t *timer, uint64_t wrid_0, unsigned int wqe_head_0,
                            void *dev_wq
                            /*, void **table table, int refcnt*/) 
{
   
	// timer[0] = clock();
  uint32_t *gpu_dbrec = (uint32_t *) cq_dbrec;
  // timer[1] = clock();
  struct ibv_wc *wc = (struct ibv_wc *)twc;
  int npolled=0;
	int err = CQ_OK;
	struct mlx5_resource *rsc = NULL;
	struct mlx5_srq *srq = NULL;
	void *cqe;
	struct mlx5_cqe64 *cqe64;
	int cqe_ver = 1;
	struct mlx5_wq *wq;
	uint16_t wqe_ctr;
	uint32_t qpn;
	uint32_t srqn_uidx;
	int idx;
	uint8_t opcode;
	struct mlx5_err_cqe *ecqe;
	struct mlx5_sigerr_cqe *sigerr_cqe;
	struct mlx5_mkey *mkey;
	struct mlx5_qp *mqp;
	struct mlx5_context *mctx;
	uint8_t is_srq;
	uint32_t cons_index_dev = *cons_index;

  // for (npolled = 0 ; npolled < 1; ++npolled) {

    // timer[0] = clock();
    
    if(cq_buffer_gpu == 1)
		  cqe = cq_buf + (cons_index_dev & ibv_cqe) * cqe_sz;
    else 
      cqe = cqe_dev;
    // printf("Function: %s line number: %d cq_buf: 0x%llx\n",__func__, __LINE__, cq_buf);
    // cqe = cqe_dev;// cq_buf + ((*cons_index) & ibv_cqe) * cqe_sz;
		cqe64 = (struct mlx5_cqe64 *)((cqe_sz == 64) ? cqe : cqe + 64);
    // timer[1] = clock();
		
        // (cqe64->op_own >> 4 != MLX5_CQE_INVALID) &&
		// 	!((cqe64->op_own & MLX5_CQE_OWNER_MASK) ^ !!(n & (ibv_cqe + 1)))
		int cond1 = (cqe64->op_own / 16 != MLX5_CQE_INVALID) &&
			!((cqe64->op_own & MLX5_CQE_OWNER_MASK) ^ !!(1 & (ibv_cqe + 1)));
      // printf("cqe64->op_own: %d\n", cqe64->op_own);
		if (cond1) {
			// printf("Function: %s line number: %d\n",__func__, __LINE__);
            (*cons_index)++;
            // printf("cq not empty: %d\n", cond1);

		} else {
      // printf("cq empty: %d\n", cond1 );
			err = CQ_EMPTY;
			// break;
			goto out1;
		}
		// printf("Function: %s line number: %d\n",__func__, __LINE__);
		is_srq = 0;
		err = 0; 
    
		// printf("Function: %s line number: %d\n",__func__, __LINE__);
        // mctx = (struct mlx5_context *) mctx_t;
        
		qpn = htonl(cqe64->sop_drop_qpn) & 0xffffff;
    // printf("Function: %s line number: %d\n",__func__, __LINE__);  
        wc->wc_flags = 0;
        wc->qp_num = qpn;
		opcode = cqe64->op_own >> 4;
		// printf("Function: %s line number: %d opcode: %d\n",__func__, __LINE__, opcode);
		if(opcode == MLX5_CQE_REQ)
		{
			// uint32_t rsn = cqe_ver ? (htonl(cqe64->srqn_uidx) & 0xffffff) : qpn;
      // printf("Function: %s line number: %d\n",__func__, __LINE__);
			// if (!rsc || (rsn != rsc->rsn)){
                // printf("Function: %s line number: %d\n",__func__, __LINE__);
				// if(cqe_ver) {
                    // printf("Function: %s line number: %d\n",__func__, __LINE__);
					// int tind = rsn >> MLX5_UIDX_TABLE_SHIFT;
                    // printf("Function: %s line number: %d rsn & MLX5_UIDX_TABLE_MASK: %d tind: %d\n",__func__, __LINE__, rsn & MLX5_UIDX_TABLE_MASK, tind);
                    // printf("mctx->uidx_table[tind].refcnt: %d\n", /*mctx->uidx_table[tind].*/refcnt);
					// if (/*(mctx->uidx_table[tind].*/refcnt){
          //               // printf("Function: %s line number: %d\n",__func__, __LINE__);
          //               rsc = (struct mlx5_resource *) dev_rsc;// mctx->uidx_table[tind].table[rsn & MLX5_UIDX_TABLE_MASK];
          //               // printf("Function: %s line number: %d\n",__func__, __LINE__);
          //           }
          //           else rsc = NULL;
			    // }
            // }
            // printf("Function: %s line number: %d\n",__func__, __LINE__);
			mqp = (struct mlx5_qp *) dev_rsc;
			if ((!mqp)){
                // printf("Function: %s line number: %d\n",__func__, __LINE__);
				err = CQ_POLL_ERR;
				// break;
			  goto out1;
			}
			wq = (struct mlx5_wq *) dev_wq; // &mqp->sq;
			wqe_ctr = htons (cqe64->wqe_counter);
			idx = wqe_ctr & (wq->wqe_cnt - 1);
     
      if(htonl(cqe64->sop_drop_qpn) >> 24 == MLX5_OPCODE_RDMA_WRITE){
				wc->opcode    = IBV_WC_RDMA_WRITE;
      }
      else {
        wc->opcode    = IBV_WC_RDMA_READ;
        wc->byte_len  = htonl(cqe64->byte_cnt);
      }
      // printf("wq->tail: %d\n", wq->tail);
      wc->wr_id = wrid_0; // wq->wrid[idx];
      wc->status = (ibv_wc_status) err;
			wq->tail = wqe_head_0 + 1; // wq->wqe_head[idx] + 1;
      // printf("wqe_head_0: %d\n", wqe_head_0);
      // printf("wq->tail: %d\n", wq->tail);
      // printf("wc->status: %d\n", wc->status);
    }
	else // MLX5_CQE_RESP_SEND:
      {
        // printf("Function: %s line number: %d\n",__func__, __LINE__);
        uint16_t	wqe_ctr;
        struct mlx5_wq *wq;
        struct mlx5_qp *qp = (struct mlx5_qp *)(dev_rsc);

        wc->byte_len = htonl(cqe64->byte_cnt);
        wq = (struct mlx5_wq *) dev_rq; // &qp->rq;
                
        wqe_ctr = wq->tail & (wq->wqe_cnt - 1);
        wc->wr_id = wrid_1;// wq->wrid[wqe_ctr];
        ++wq->tail;
            
        int size = wc->byte_len;
        
        int copy;

        err = IBV_WC_SUCCESS;
        wc->opcode   = IBV_WC_RECV;
        wc->status = IBV_WC_SUCCESS;
      }
		
        	
        // if (err != CQ_OK){
		// 	break;
		// }
        // printf("Function: %s line number: %d\n",__func__, __LINE__);
//  }
	/* Update cons index */
	// cq->dbrec[0] = htonl(*cons_index & 0xffffff);
    // printf("Function: %s line number: %d\n",__func__, __LINE__);
out1:
  gpu_dbrec[0] = htonl((*cons_index) & 0xffffff);
    
	// timer[1] = clock();
  return err; // == CQ_POLL_ERR ? err : 1;
}

int host_gpu_poll_cq (struct ibv_cq *cq_ptr, int num_entries, struct ibv_wc *wc)
{
  struct mlx5_cq *cq = to_mcq(cq_ptr);
  void *cqe;
  // printf("Function: %s line number: %d cq->buf.length: %d\n",__func__, __LINE__, cq->buf_a.length);
  struct mlx5_cqe64 *cqe64;
  // int cond = 0;
  uint32_t cons_index = cq->cons_index;
  int cq_cqe = cq->verbs_cq.cq.cqe;
  int cq_cqe_sz = cq->cqe_sz;
  void *cq_buf_a = cq->buf_a.buf; 
  cqe = cq_buf_a ;//+ (cons_index & cq_cqe) * cq_cqe_sz;

  printf("cons_index: %d\n", cons_index);
  printf("cq_cqe: %d\n", cq_cqe);
  printf("cq_cqe_sz: %d\n", cq_cqe_sz);
  
    // printf("Function: %s line number: %d cq_buf: 0x%llx\n",__func__, __LINE__, cq_buf);
    // cqe = cqe_dev;// cq_buf + ((*cons_index) & ibv_cqe) * cqe_sz;

	// cqe64 = (struct mlx5_cqe64 *)((cq->cqe_sz == 64) ? cqe : cqe);
  // printf("cq buffer length: %d", cq->buf_a.length);
  // printf("Function name: %s, line number: %d cqe64->op_own >> 4: %d\n", __func__, __LINE__, cqe64->op_own >> 4);

  // for (int i = 0; i < 4097; i++){
  //   cqe64 = (struct mlx5_cqe64 *)( cqe + i);
  //   if (cqe64->op_own)
  //   printf("i: %d cqe64->op_own: %d\n", i, cqe64->op_own);
  // } 

  // void *cqe2 = cq->active_buf->buf + (cq->cons_index & cq->verbs_cq.cq.cqe) * cq->cqe_sz;
  // cqe64 = (struct mlx5_cqe64 *)((cq->cqe_sz == 64) ? cqe2 : cqe2 + 64);
  // cond = (cqe64->op_own >> 4 != MLX5_CQE_INVALID) &&
  //   !((cqe64->op_own & MLX5_CQE_OWNER_MASK) ^ !!(num_entries & (cq->verbs_cq.cq.cqe + 1)));
  // if (cond) {
  //   // ++cq->cons_index;
  //   printf("Function: %s line number: %d\n",__func__, __LINE__);
  //   printf("opcode: %d\n", cqe64->op_own >> 4);
  // }

  void *dev_cq_ptr; // cq pointer for GPU
  void *dev_wc; // wc pointer for GPU memory
  void *dev_cons_index; // 
  void *dev_ret; // 
  void *dev_total_wc; // 
  void *dev_cq_dbrec;
  struct mlx5_context *mctx = container_of(cq->verbs_cq.cq.context, struct mlx5_context, ibv_ctx.context);
  // mctx->uidx_table[0].table
  void *dev_mctx;
  void **dev_table;
  void *dev_rsc;
  void *dev_qp_context;
  void *dev_qp_buf;
  int total_wc = 0, ret = -1;
  struct mlx5_resource *rsc = mctx->uidx_table[0].table[0];
  if(!rsc) printf("rsc is null\n\n\n");
  struct mlx5_qp *qp = (struct mlx5_qp *)(rsc);
  // printf("qp buf address: 0x%llx\n\n\n", qp->buf.buf);
  struct mlx5_context *qp_ctx = to_mctx(qp->ibv_qp->pd->context); 
  // container_of(qp->ibv_qp->pd->context, struct mlx5_context, ibv_ctx.context);
  // printf("qp_ctx address: 0x%llx\n\n\n", qp_ctx);
  // printf("Function: %s line number: %d ctx->dump_fill_mkey_be:%d\n",__func__, __LINE__, qp_ctx->dump_fill_mkey);
  // printf("Function: %s line number: %d\n",__func__, __LINE__);
  //struct ibv_wc *twc = wc;
  cudaError_t crc = cudaSuccess;
  cudaError_t cudaStatus;
  //register cq in host memory
  // printf("Function name: %s, line number: %d cqe64->op_own >> 4: %d\n", __func__, __LINE__, cqe64->op_own >> 4);
  if (cq_buffer_gpu != 1){
    cudaStatus = cudaHostRegister(cq->active_buf->buf /*cqbuf*/, cq->active_buf->length /*cqbuf_size*/, cudaHostRegisterMapped);
    if(cudaStatus != cudaErrorHostMemoryAlreadyRegistered && cudaStatus != cudaSuccess)
        exit(0);
    // get GPU pointer for cq
    if(cudaHostGetDevicePointer(&dev_cq_ptr, /*cqbuf*/ cq->active_buf->buf, 0) !=  cudaSuccess)
        exit(0);
  }
  else{
    dev_cq_ptr = cq->active_buf->buf;
  }
  // printf("Function name: %s, line number: %d cqe64->op_own >> 4: %d\n", __func__, __LINE__, cqe64->op_own >> 4);
  // printf("Function: %s line number: %d\n",__func__, __LINE__);
  // register wc in host memory 
  cudaStatus = cudaHostRegister(wc, sizeof(wc), cudaHostRegisterMapped);
  if(cudaStatus != cudaErrorHostMemoryAlreadyRegistered && cudaStatus !=  cudaSuccess)
      exit(0);
  // get GPU pointer for wc
  if(cudaHostGetDevicePointer(&dev_wc, wc, 0) != cudaSuccess)
      exit(0);
      // printf("Function name: %s, line number: %d cqe64->op_own >> 4: %d\n", __func__, __LINE__, cqe64->op_own >> 4);
  // printf("Function: %s line number: %d cq->cons_index: %d\n ",__func__, __LINE__, cq->cons_index);
  // register cons index in host memory 
  cudaStatus = cudaHostRegister(&cq->cons_index, sizeof(cq->cons_index), cudaHostRegisterMapped);
  if(cudaStatus != cudaSuccess && cudaStatus != cudaErrorHostMemoryAlreadyRegistered)
      exit(0);
  // get GPU pointer for cons index
  if(cudaHostGetDevicePointer(&dev_cons_index, &cq->cons_index, 0) != cudaSuccess)
      exit(0);
  // printf("Function: %s line number: %d\n",__func__, __LINE__);
  // register cons index in host memory 
  cudaStatus = cudaHostRegister(&total_wc, sizeof(total_wc), cudaHostRegisterMapped);
  if(cudaStatus != cudaErrorHostMemoryAlreadyRegistered && cudaStatus !=  cudaSuccess)
      exit(0);
      // printf("Function name: %s, line number: %d cqe64->op_own >> 4: %d\n", __func__, __LINE__, cqe64->op_own >> 4);
  // get GPU pointer for cons index
  if(cudaHostGetDevicePointer(&dev_total_wc, &total_wc, 0) != cudaSuccess)
      exit(0);
  // printf("Function: %s line number: %d\n",__func__, __LINE__);
  // register ret in host memory 
  cudaStatus = cudaHostRegister(&ret, sizeof(ret), cudaHostRegisterMapped);
  if(cudaStatus != cudaErrorHostMemoryAlreadyRegistered && cudaStatus !=  cudaSuccess)
      exit(0);
  // get GPU pointer for ret
  if(cudaHostGetDevicePointer(&dev_ret, &ret, 0) != cudaSuccess)
      exit(0);
      // printf("Function name: %s, line number: %d cqe64->op_own >> 4: %d\n", __func__, __LINE__, cqe64->op_own >> 4);
  // printf("Function: %s line number: %d\n",__func__, __LINE__);
  cudaStatus = cudaHostRegister(cq->dbrec, sizeof(cq->dbrec), cudaHostRegisterMapped);
  if(cudaStatus != cudaErrorHostMemoryAlreadyRegistered && cudaStatus !=  cudaSuccess) 
      exit(0);
  if(cudaHostGetDevicePointer(&dev_cq_dbrec, cq->dbrec, 0) != cudaSuccess)
      exit(0);
  // printf("Function: %s line number: %d\n",__func__, __LINE__);
  cudaStatus = cudaHostRegister(mctx, sizeof(mctx), cudaHostRegisterMapped);
  if(cudaStatus != cudaErrorHostMemoryAlreadyRegistered && cudaStatus !=  cudaSuccess) 
      exit(0);
      // printf("Function name: %s, line number: %d cqe64->op_own >> 4: %d\n", __func__, __LINE__, cqe64->op_own >> 4);
  if(cudaHostGetDevicePointer(&dev_mctx, mctx, 0) != cudaSuccess)
      exit(0);
  // printf("Function: %s line number: %d\n",__func__, __LINE__);
  cudaStatus = cudaHostRegister(rsc, sizeof(rsc), cudaHostRegisterMapped);
  if(cudaStatus != cudaErrorHostMemoryAlreadyRegistered && cudaStatus !=  cudaSuccess) 
      exit(0);
  if(cudaHostGetDevicePointer(&dev_rsc, rsc, 0) != cudaSuccess)
      exit(0);
      // printf("Function name: %s, line number: %d cqe64->op_own >> 4: %d\n", __func__, __LINE__, cqe64->op_own >> 4);
      // qp->buf.buf
  // printf("Function: %s line number: %d\n",__func__, __LINE__);
  // cudaStatus = cudaHostRegister(qp->buf.buf, sizeof(qp->buf.buf), cudaHostRegisterMapped);
  // fprintf(stderr, "cudaHostRegister(qp->buf.buf failed with error code %d: %s\n", cudaStatus, cudaGetErrorString(cudaStatus));

  // if(!wq_buffer_gpu){  
  //   // comment the below lines when gpu buffer used for qp->buf.buf
  //   cudaError_t cuda_success = cudaHostRegister(qp->buf.buf, sizeof(qp->buf.buf), cudaHostRegisterMapped);
  //   if(cuda_success !=  cudaSuccess && cuda_success != cudaErrorHostMemoryAlreadyRegistered) 
  //       exit(0);
  //   if(cudaHostGetDevicePointer(&dev_qp_buf, qp->buf.buf, 0) != cudaSuccess)
  //       exit(0);
  // }
  // else{
  //   dev_qp_buf = qp->buf.buf;
  // }

  // printf("Function: %s line number: %d &qp->rq: 0x%llx\n",__func__, __LINE__, &qp->rq);
      // &qp->rq
  void *dev_rq, *dev_wrid;
  // printf("cudaHostRegister(&qp->rq, sizeof(qp->rq), cudaHostRegisterMapped): %d\n", cudaHostRegister(&qp->rq, sizeof(qp->rq), cudaHostRegisterMapped));
  cudaStatus = cudaHostRegister(&qp->rq, sizeof(qp->rq), cudaHostRegisterMapped);
  if(cudaStatus != cudaErrorHostMemoryAlreadyRegistered && cudaStatus !=  cudaSuccess) 
      exit(0);
  if(cudaHostGetDevicePointer(&dev_rq, &qp->rq, 0) != cudaSuccess)
      exit(0);
      // printf("Function name: %s, line number: %d cqe64->op_own >> 4: %d\n", __func__, __LINE__, cqe64->op_own >> 4);
  struct mlx5_wq *qp_rq = &qp->rq;
  // printf("Function: %s line number: %d &qp->rq: 0x%llx\n",__func__, __LINE__, &qp->rq);

  // cudaStatus = cudaHostRegister(qp_ctx, sizeof(qp_ctx), cudaHostRegisterMapped);
  // fprintf(stderr, "cudaHostRegister(qp_ctx failed with error code %d: %s\n", cudaStatus, cudaGetErrorString(cudaStatus));
  struct mlx5_wq * wq_sq = &qp->sq;
  void *dev_wq;
  // printf("Function: %s line number: %d wq_sq->wqe_head[0]: %d\n", __func__, __LINE__, wq_sq->wqe_head[0]);
  // printf("Function: %s line number: %d wq_sq->tail: %d\n", __func__, __LINE__, wq_sq->tail);
  // wq_sq->wrid[0];
  // wq_sq->wqe_head[0];
  cudaError_t cudaState = cudaHostRegister(wq_sq, sizeof(qp->sq), cudaHostRegisterMapped);
  if(cudaState !=  cudaSuccess && cudaState != cudaErrorHostMemoryAlreadyRegistered) 
      exit(0);
  if(cudaHostGetDevicePointer(&dev_wq, wq_sq, 0) != cudaSuccess)
      exit(0);
      // printf("Function name: %s, line number: %d cqe64->op_own >> 4: %d\n", __func__, __LINE__, cqe64->op_own >> 4);

  cudaState = cudaHostRegister(qp_rq->wrid, sizeof(qp_rq->wrid), cudaHostRegisterMapped);
  // printf("Function: %s line number: %d cudaState: %d\n",__func__, __LINE__, cudaState);
  // comment the below lines when gpu buffer used for qp->buf.buf
  printf("Function: %s line number: %d\n",__func__, __LINE__);
  if(cudaState != cudaSuccess && cudaState != cudaErrorHostMemoryAlreadyRegistered) 
      exit(0);
  // printf("Function: %s line number: %d\n",__func__, __LINE__);
  if(cudaHostGetDevicePointer(&dev_wrid, qp_rq->wrid, 0) != cudaSuccess)
      exit(0);
  // printf("Function: %s line number: %d\n",__func__, __LINE__);
  // printf("Function: %s line number: %d\n",__func__, __LINE__);
    // if(cudaHostRegister(qp_ctx, sizeof(qp_ctx), cudaHostRegisterMapped) !=  cudaSuccess) {
    //     printf("cudaHostRegister(qp_ctx, sizeof(qp_ctx), cudaHostRegisterMapped) : %d\n", cudaHostRegister(qp_ctx, sizeof(qp_ctx), cudaHostRegisterMapped) );
    //     exit(0);
    // }
  // printf("Function: %s line number: %d\n",__func__, __LINE__);
  // printf("Function name: %s, line number: %d cqe64->op_own >> 4: %d\n", __func__, __LINE__, cqe64->op_own >> 4);
  printf("Function: %s line number: %d\n",__func__, __LINE__);
	cudaStatus = cudaHostGetDevicePointer(&dev_qp_context, qp_ctx, 0);
	// fprintf(stderr, "cudaHostRegister failed with error code %d: %s\n", cudaStatus, cudaGetErrorString(cudaStatus));
  if(cudaHostGetDevicePointer(&dev_qp_context, qp_ctx, 0) != cudaSuccess)
      exit(0);
  // printf("Function: %s line number: %d\n",__func__, __LINE__);
  // printf("Function name: %s, line number: %d cqe64->op_own >> 4: %d\n", __func__, __LINE__, cqe64->op_own >> 4);
  void *cqe_dev;
  printf("Function: %s line number: %d\n",__func__, __LINE__);
	// if (!registered)
	//   printf("Function: %s line number: %d cudaHostRegister(cqe, sizeof(cqe), cudaHostRegisterMapped): %d\n",__func__, __LINE__,
  //         cudaHostRegister(cqe, sizeof(cqe), cudaHostRegisterMapped));
  if (cq_buffer_gpu != 1){
    cqe = cq->active_buf->buf + (cq->cons_index & cq->verbs_cq.cq.cqe) * cq->cqe_sz;
    // cqe = (struct mlx5_cqe64 *)((cq->cqe_sz == 64) ? cqe : cqe + 64);
    cudaStatus = cudaHostRegister(cqe, sizeof(cqe), cudaHostRegisterMapped);
    if(cudaStatus != cudaErrorHostMemoryAlreadyRegistered && cudaStatus !=  cudaSuccess) 
        exit(0);
    // printf("Function: %s line number: %d cudaHostGetDevicePointer(&cqe_dev, cqe, 0): %d\n",__func__, __LINE__,
          // cudaHostGetDevicePointer(&cqe_dev, cqe, 0));
    if(cudaHostGetDevicePointer(&cqe_dev, cqe, 0) != cudaSuccess)
        exit(0);
  }
  else {
    cqe_dev = cq->active_buf->buf;
  }
  
  // printf("Function name: %s, line number: %d cqe64->op_own >> 4: %d\n", __func__, __LINE__, cqe64->op_own >> 4);

	// printf("Function: %s line number: %d\n",__func__, __LINE__);

  uint16_t	wqe_ctr;
  struct mlx5_wq *wq = (struct mlx5_wq *) &qp->rq;
  wqe_ctr = wq->tail & (wq->wqe_cnt - 1);
  // struct mlx5_wqe_data_seg *scat = (struct mlx5_wqe_data_seg *)(qp->buf.buf + qp->rq.offset + (wqe_ctr << qp->rq.wqe_shift));
  // void *scat_address = (void *)(unsigned long)htonl64(scat->addr);
  void *dev_scat_address = NULL; 

  // if(!registered && cudaHostRegister(scat_address, sizeof(scat_address), cudaHostRegisterMapped) !=  cudaSuccess) 
  //     exit(0);
  // if(cudaHostGetDevicePointer(&dev_scat_address, scat_address, 0) != cudaSuccess)
  //     exit(0);
  // printf("Function: %s line number: %d\n",__func__, __LINE__);
    


	int cuda_device_id = 0;
	int cuda_pci_bus_id;
	int cuda_pci_device_id;
	int index;
	CUdevice cu_device;
	CUdevice cuDevice_selected;

	// printf("initializing CUDA\n");
	CUresult error = cuInit(0);
	if (error != CUDA_SUCCESS) {
		// printf("cuInit(0) returned %d\n", error);
		return -1;
	}
  // printf("Function name: %s, line number: %d cqe64->op_own >> 4: %d\n", __func__, __LINE__, cqe64->op_own >> 4);
	int deviceCount = 0;
	error = cuDeviceGetCount(&deviceCount);
	if (error != CUDA_SUCCESS) {
		// printf("cuDeviceGetCount() returned %d\n", error);
		return -1;
	}
	/* This function call returns 0 if there are no CUDA capable devices. */
	if (deviceCount == 0) {
		// printf("There are no available device(s) that support CUDA\n");
		return -1;
	}
  // printf("Function name: %s, line number: %d cqe64->op_own >> 4: %d\n", __func__, __LINE__, cqe64->op_own >> 4);
	// printf("Listing all CUDA devices in system:\n");
	for (int index = 0; index < deviceCount; index++) {
		if(cuDeviceGet(&cu_device, index) != CUDA_SUCCESS) exit(0);
		cuDeviceGetAttribute(&cuda_pci_bus_id, CU_DEVICE_ATTRIBUTE_PCI_BUS_ID , cu_device);
		cuDeviceGetAttribute(&cuda_pci_device_id, CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID , cu_device);
		// printf("CUDA device %d: PCIe address is %02X:%02X\n", index, (unsigned int)cuda_pci_bus_id, (unsigned int)cuda_pci_device_id);
	}
  // printf("Function name: %s, line number: %d cqe64->op_own >> 4: %d\n", __func__, __LINE__, cqe64->op_own >> 4);
	// printf("\nPicking device No. %d\n", cuda_device_id);

	if(cuDeviceGet(&cuDevice_selected, cuda_device_id) != cudaSuccess){
		// printf("error on cuDeviceGet\n");
		exit(0);
	}
  // printf("Function name: %s, line number: %d cqe64->op_own >> 4: %d\n", __func__, __LINE__, cqe64->op_own >> 4);
	char name[128];
	if(cuDeviceGetName(name, sizeof(name), cuda_device_id) != cudaSuccess){
		// printf("error on cuDeviceGetName\n");
		exit(0);
	}
	// printf("[pid = %d, dev = %d] device name = [%s]\n", getpid(), cuDevice_selected, name);

    struct timespec res;
    long nano1,nano2;

    clock_t start, end;
    double cpu_time_used;
    struct timeval cpu_timer[2];

    size_t free_memory, total_memory;
    cudaMemGetInfo(&free_memory, &total_memory);
    // printf("free memory: %zu, total_memory: %zu\n", free_memory/(1024 * 1024), total_memory/(1024 * 1024));

	clock_t *dtimer = NULL;
	clock_t timer[2];
  // printf("Function name: %s, line number: %d cqe64->op_own >> 4: %d\n", __func__, __LINE__, cqe64->op_own >> 4);
	if (cudaSuccess != cudaMalloc((void **)&dtimer, sizeof(clock_t) * 2)) 
		exit(0);
  
	cudaDeviceProp devProp;
    cudaGetDeviceProperties(&devProp, 0);
    // printf("Cuda device clock rate = %d\n", devProp.clockRate);

    int a = cudaDeviceSynchronize();
    printf("cudasynchronize: %d\n", a);
    // start = clock();
    // gettimeofday(&cpu_timer[0], NULL);
    clock_gettime(CLOCK_REALTIME,&res);
    nano1 = res.tv_nsec;
    // printf("Function name: %s, line number: %d cqe64->op_own: %d\n", __func__, __LINE__, cqe64->op_own);
	// printf("Function: %s line number: %d cqe_sz: %d\n",__func__, __LINE__, cq->cqe_sz);
    global_gpu_poll_cq<<<1,1>>>(dev_cq_ptr, dev_wc, dev_cons_index, cq->verbs_cq.cq.cqe, cq->cqe_sz, num_entries, 
                    (int *) dev_total_wc, (int *) dev_ret, dev_cq_dbrec, dev_mctx, dev_rsc, mctx->uidx_table[0].refcnt,
                    dev_qp_context /*qp_ctx*/, qp_ctx->dump_fill_mkey_be, dev_rq, dev_wrid,
                    qp->rq.wrid[0], cqe_dev, 1, dev_scat_address, dtimer, wq_sq->wrid[0], wq_sq->wqe_head[0],
                    dev_wq);
    // printf("Function: %s line number: %d\n",__func__, __LINE__);
    clock_gettime(CLOCK_REALTIME,&res);
    nano2 = res.tv_nsec;

	cudaMemcpy(timer, dtimer, sizeof(clock_t) * (2), cudaMemcpyDeviceToHost);
    cudaFree(dtimer);
	// printf("Function: %s line number: %d\n",__func__, __LINE__);
    //gettimeofday(&cpu_timer[1], NULL);
    a = cudaDeviceSynchronize();
    printf("cudasynchronize: %d\n", a);
    // if (a != 0){
    //     printf("cudasynchronize: %d\n", a); 
    //     exit(0);
    // }

	float freq = (float)1/(devProp.clockRate*1000);
	float g_usec = (float)((float)1/(devProp.clockRate*1000))*((timer[1]-timer[0])) * 1000000;
	printf("POLLING - INTERNAL MEASUREMENT: %f useconds to execute \n", g_usec);
    // float timer_usec = (cpu_timer[1].tv_nsec - cpu_timer[0].tv_usec);
    // cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
    // printf("gpu_send took %lu useconds to execute \n", nano2-nano1);
    // printf("Function: %s line number: %d\n",__func__, __LINE__);
    //cudaMemcpy(&host_ret, &dev_ret, sizeof(int), cudaMemcpyDeviceToHost);
    // cudaMemcpy(&total_wc, &dev_total_wc, sizeof(int), cudaMemcpyDeviceToHost);

    
    // printf("Function: %s line number: %d\n",__func__, __LINE__);
//    do {
// 	printf("polling\n");
//        ret = /*ibv_poll_cq*/cpu_poll_cq(cq_ptr /* the CQ, we got notification for */, 
// 	       max_wc - total_wc /* number of remaining WC elements*/,
// 	       wc + total_wc/* where to store */);

//        if (ret < 0) {
// 	       printf("Failed to poll cq for wc due to %d \n", ret);
// 	       continue;
// 		   /* ret is errno here */
// 	       // return ret;
//        }
//        total_wc += ret;
//    } while (total_wc < max_wc); 
//    printf("%d WC are completed \n", total_wc);
    // printf("Function: %s line number: %d\n",__func__, __LINE__);
    /* Now we check validity and status of I/O work completions */
    // struct ibv_wc *wc_list = wc;
    // int i;
    // for( i = 0 ; i < total_wc ; i++) {
    //     if (wc_list[i].status != IBV_WC_SUCCESS) {
    //         printf("Work completion (WC) has error status: %s at index %d", 
    //                 ibv_wc_status_str(wc_list[i].status), i);
    //         /* return negative value */
    //         return -(wc_list[i].status);
    //     }
    // }
    // printf("Function: %s line number: %d\n",__func__, __LINE__);
    //    /* Similar to connection management events, we need to acknowledge CQ events */
    // ibv_ack_cq_events(cq_ptr, 
    //         1 /* we received one event notification. This is not 
    //         number of WC elements */);
    // printf("Function: %s line number: %d\n",__func__, __LINE__);
    return total_wc; 
}


int host_gpu_poll_cq2 (struct ibv_cq *cq_ptr, int num_entries, struct ibv_wc *wc)
{
  struct mlx5_cq *cq = to_mcq(cq_ptr);
  // printf("Function: %s line number: %d cq->buf.length: %d\n",__func__, __LINE__, cq->buf_a.length);
  struct mlx5_cqe64 *cqe64;
  int cond = 0;
  // void *cqe2 = cq->active_buf->buf + (cq->cons_index & cq->verbs_cq.cq.cqe) * cq->cqe_sz;
  // cqe64 = (struct mlx5_cqe64 *)((cq->cqe_sz == 64) ? cqe2 : cqe2 + 64);
  // cond = (cqe64->op_own >> 4 != MLX5_CQE_INVALID) &&
  //   !((cqe64->op_own & MLX5_CQE_OWNER_MASK) ^ !!(num_entries & (cq->verbs_cq.cq.cqe + 1)));
  // if (cond) {
  //   // ++cq->cons_index;
  //   printf("Function: %s line number: %d\n",__func__, __LINE__);
  //   printf("opcode: %d\n", cqe64->op_own >> 4);
  // }

  void *dev_cq_ptr; // cq pointer for GPU
  void *dev_wc; // wc pointer for GPU memory
  void *dev_cons_index; // 
  void *dev_ret; // 
  void *dev_total_wc; // 
  void *dev_cq_dbrec;
  struct mlx5_context *mctx = container_of(cq->verbs_cq.cq.context, struct mlx5_context, ibv_ctx.context);
  // mctx->uidx_table[0].table
  void *dev_mctx;
  void **dev_table;
  void *dev_rsc;
  void *dev_qp_context;
  void *dev_qp_buf;
  int total_wc = 0, ret = -1;
  struct mlx5_resource *rsc = mctx->uidx_table[0].table[0];
  if(!rsc) printf("rsc is null\n\n\n");
  struct mlx5_qp *qp = (struct mlx5_qp *)(rsc);
  // printf("qp buf address: 0x%llx\n\n\n", qp->buf.buf);
  struct mlx5_context *qp_ctx = to_mctx(qp->ibv_qp->pd->context); 
  // container_of(qp->ibv_qp->pd->context, struct mlx5_context, ibv_ctx.context);
  // printf("qp_ctx address: 0x%llx\n\n\n", qp_ctx);
  // printf("Function: %s line number: %d ctx->dump_fill_mkey_be:%d\n",__func__, __LINE__, qp_ctx->dump_fill_mkey);
  // printf("Function: %s line number: %d\n",__func__, __LINE__);
  //struct ibv_wc *twc = wc;
  cudaError_t crc = cudaSuccess;
  cudaError_t cudaStatus;
  //register cq in host memory
  if (cq_buffer_gpu != 1){
    cudaStatus = cudaHostRegister(cq->active_buf->buf /*cqbuf*/, cq->active_buf->length /*cqbuf_size*/, cudaHostRegisterMapped);
    if(cudaStatus != cudaErrorHostMemoryAlreadyRegistered && cudaStatus != cudaSuccess)
        exit(0);
    // get GPU pointer for cq
    if(cudaHostGetDevicePointer(&dev_cq_ptr, /*cqbuf*/ cq->active_buf->buf, 0) !=  cudaSuccess)
        exit(0);
  }
  else{
    dev_cq_ptr = cq->active_buf->buf;
  }
  // printf("Function: %s line number: %d\n",__func__, __LINE__);
  // register wc in host memory 
  cudaStatus = cudaHostRegister(wc, sizeof(wc), cudaHostRegisterMapped);
  if(cudaStatus != cudaErrorHostMemoryAlreadyRegistered && cudaStatus !=  cudaSuccess)
      exit(0);
  // get GPU pointer for wc
  if(cudaHostGetDevicePointer(&dev_wc, wc, 0) != cudaSuccess)
      exit(0);
  // printf("Function: %s line number: %d cq->cons_index: %d\n ",__func__, __LINE__, cq->cons_index);
  // register cons index in host memory 
  cudaStatus = cudaHostRegister(&cq->cons_index, sizeof(cq->cons_index), cudaHostRegisterMapped);
  if(cudaStatus != cudaSuccess && cudaStatus != cudaErrorHostMemoryAlreadyRegistered)
      exit(0);
  // get GPU pointer for cons index
  if(cudaHostGetDevicePointer(&dev_cons_index, &cq->cons_index, 0) != cudaSuccess)
      exit(0);
  // printf("Function: %s line number: %d\n",__func__, __LINE__);
  // register cons index in host memory 
  cudaStatus = cudaHostRegister(&total_wc, sizeof(total_wc), cudaHostRegisterMapped);
  if(cudaStatus != cudaErrorHostMemoryAlreadyRegistered && cudaStatus !=  cudaSuccess)
      exit(0);
  // get GPU pointer for cons index
  if(cudaHostGetDevicePointer(&dev_total_wc, &total_wc, 0) != cudaSuccess)
      exit(0);
  // printf("Function: %s line number: %d\n",__func__, __LINE__);
  // register ret in host memory 
  cudaStatus = cudaHostRegister(&ret, sizeof(ret), cudaHostRegisterMapped);
  if(cudaStatus != cudaErrorHostMemoryAlreadyRegistered && cudaStatus !=  cudaSuccess)
      exit(0);
  // get GPU pointer for ret
  if(cudaHostGetDevicePointer(&dev_ret, &ret, 0) != cudaSuccess)
      exit(0);
  // printf("Function: %s line number: %d\n",__func__, __LINE__);
  cudaStatus = cudaHostRegister(cq->dbrec, sizeof(cq->dbrec), cudaHostRegisterMapped);
  if(cudaStatus != cudaErrorHostMemoryAlreadyRegistered && cudaStatus !=  cudaSuccess) 
      exit(0);
  if(cudaHostGetDevicePointer(&dev_cq_dbrec, cq->dbrec, 0) != cudaSuccess)
      exit(0);
  // printf("Function: %s line number: %d\n",__func__, __LINE__);
  cudaStatus = cudaHostRegister(mctx, sizeof(mctx), cudaHostRegisterMapped);
  if(cudaStatus != cudaErrorHostMemoryAlreadyRegistered && cudaStatus !=  cudaSuccess) 
      exit(0);
  if(cudaHostGetDevicePointer(&dev_mctx, mctx, 0) != cudaSuccess)
      exit(0);
  // printf("Function: %s line number: %d\n",__func__, __LINE__);
  cudaStatus = cudaHostRegister(rsc, sizeof(rsc), cudaHostRegisterMapped);
  if(cudaStatus != cudaErrorHostMemoryAlreadyRegistered && cudaStatus !=  cudaSuccess) 
      exit(0);
  if(cudaHostGetDevicePointer(&dev_rsc, rsc, 0) != cudaSuccess)
      exit(0);
      // qp->buf.buf
  // printf("Function: %s line number: %d\n",__func__, __LINE__);
  // cudaStatus = cudaHostRegister(qp->buf.buf, sizeof(qp->buf.buf), cudaHostRegisterMapped);
  // fprintf(stderr, "cudaHostRegister(qp->buf.buf failed with error code %d: %s\n", cudaStatus, cudaGetErrorString(cudaStatus));

  // if(!wq_buffer_gpu){  
  //   // comment the below lines when gpu buffer used for qp->buf.buf
  //   cudaError_t cuda_success = cudaHostRegister(qp->buf.buf, sizeof(qp->buf.buf), cudaHostRegisterMapped);
  //   if(cuda_success !=  cudaSuccess && cuda_success != cudaErrorHostMemoryAlreadyRegistered) 
  //       exit(0);
  //   if(cudaHostGetDevicePointer(&dev_qp_buf, qp->buf.buf, 0) != cudaSuccess)
  //       exit(0);
  // }
  // else{
  //   dev_qp_buf = qp->buf.buf;
  // }

  // printf("Function: %s line number: %d &qp->rq: 0x%llx\n",__func__, __LINE__, &qp->rq);
      // &qp->rq
  void *dev_rq, *dev_wrid;
  // printf("cudaHostRegister(&qp->rq, sizeof(qp->rq), cudaHostRegisterMapped): %d\n", cudaHostRegister(&qp->rq, sizeof(qp->rq), cudaHostRegisterMapped));
  cudaStatus = cudaHostRegister(&qp->rq, sizeof(qp->rq), cudaHostRegisterMapped);
  if(cudaStatus != cudaErrorHostMemoryAlreadyRegistered && cudaStatus !=  cudaSuccess) 
      exit(0);
  if(cudaHostGetDevicePointer(&dev_rq, &qp->rq, 0) != cudaSuccess)
      exit(0);
  struct mlx5_wq *qp_rq = &qp->rq;
  // printf("Function: %s line number: %d &qp->rq: 0x%llx\n",__func__, __LINE__, &qp->rq);

  // cudaStatus = cudaHostRegister(qp_ctx, sizeof(qp_ctx), cudaHostRegisterMapped);
  // fprintf(stderr, "cudaHostRegister(qp_ctx failed with error code %d: %s\n", cudaStatus, cudaGetErrorString(cudaStatus));
  struct mlx5_wq * wq_sq = &qp->sq;
  void *dev_wq;
  // printf("Function: %s line number: %d wq_sq->wqe_head[0]: %d\n", __func__, __LINE__, wq_sq->wqe_head[0]);
  // printf("Function: %s line number: %d wq_sq->tail: %d\n", __func__, __LINE__, wq_sq->tail);
  // wq_sq->wrid[0];
  // wq_sq->wqe_head[0];
  cudaError_t cudaState = cudaHostRegister(wq_sq, sizeof(qp->sq), cudaHostRegisterMapped);
  if(cudaState !=  cudaSuccess && cudaState != cudaErrorHostMemoryAlreadyRegistered) 
      exit(0);
  if(cudaHostGetDevicePointer(&dev_wq, wq_sq, 0) != cudaSuccess)
      exit(0);

  cudaState = cudaHostRegister(qp_rq->wrid, sizeof(qp_rq->wrid), cudaHostRegisterMapped);
  // printf("Function: %s line number: %d cudaState: %d\n",__func__, __LINE__, cudaState);
  // comment the below lines when gpu buffer used for qp->buf.buf
  if(cudaState != cudaSuccess && cudaState != cudaErrorHostMemoryAlreadyRegistered) 
      exit(0);
  // printf("Function: %s line number: %d\n",__func__, __LINE__);
  if(cudaHostGetDevicePointer(&dev_wrid, qp_rq->wrid, 0) != cudaSuccess)
      exit(0);
  // printf("Function: %s line number: %d\n",__func__, __LINE__);
  // printf("Function: %s line number: %d\n",__func__, __LINE__);
    // if(cudaHostRegister(qp_ctx, sizeof(qp_ctx), cudaHostRegisterMapped) !=  cudaSuccess) {
    //     printf("cudaHostRegister(qp_ctx, sizeof(qp_ctx), cudaHostRegisterMapped) : %d\n", cudaHostRegister(qp_ctx, sizeof(qp_ctx), cudaHostRegisterMapped) );
    //     exit(0);
    // }
  // printf("Function: %s line number: %d\n",__func__, __LINE__);
	cudaStatus = cudaHostGetDevicePointer(&dev_qp_context, qp_ctx, 0);
	// fprintf(stderr, "cudaHostRegister failed with error code %d: %s\n", cudaStatus, cudaGetErrorString(cudaStatus));
  if(cudaHostGetDevicePointer(&dev_qp_context, qp_ctx, 0) != cudaSuccess)
      exit(0);
  // printf("Function: %s line number: %d\n",__func__, __LINE__);

  void *cqe, *cqe_dev;
	
	// if (!registered)
	//   printf("Function: %s line number: %d cudaHostRegister(cqe, sizeof(cqe), cudaHostRegisterMapped): %d\n",__func__, __LINE__,
  //         cudaHostRegister(cqe, sizeof(cqe), cudaHostRegisterMapped));
  if (cq_buffer_gpu != 1){
    cqe = cq->active_buf->buf + (cq->cons_index & cq->verbs_cq.cq.cqe) * cq->cqe_sz;
    cudaStatus = cudaHostRegister(cqe, sizeof(cqe), cudaHostRegisterMapped);
    if(cudaStatus != cudaErrorHostMemoryAlreadyRegistered && cudaStatus !=  cudaSuccess) 
        exit(0);
    // printf("Function: %s line number: %d cudaHostGetDevicePointer(&cqe_dev, cqe, 0): %d\n",__func__, __LINE__,
          // cudaHostGetDevicePointer(&cqe_dev, cqe, 0));
    if(cudaHostGetDevicePointer(&cqe_dev, cqe, 0) != cudaSuccess)
        exit(0);
  }
  else {
    cqe_dev = NULL;
  }

	// printf("Function: %s line number: %d\n",__func__, __LINE__);

  uint16_t	wqe_ctr;
  struct mlx5_wq *wq = (struct mlx5_wq *) &qp->rq;
  wqe_ctr = wq->tail & (wq->wqe_cnt - 1);
  // struct mlx5_wqe_data_seg *scat = (struct mlx5_wqe_data_seg *)(qp->buf.buf + qp->rq.offset + (wqe_ctr << qp->rq.wqe_shift));
  // void *scat_address = (void *)(unsigned long)htonl64(scat->addr);
  void *dev_scat_address = NULL; 

  // if(!registered && cudaHostRegister(scat_address, sizeof(scat_address), cudaHostRegisterMapped) !=  cudaSuccess) 
  //     exit(0);
  // if(cudaHostGetDevicePointer(&dev_scat_address, scat_address, 0) != cudaSuccess)
  //     exit(0);
  // printf("Function: %s line number: %d\n",__func__, __LINE__);
    


	int cuda_device_id = 0;
	int cuda_pci_bus_id;
	int cuda_pci_device_id;
	int index;
	CUdevice cu_device;
	CUdevice cuDevice_selected;

	// printf("initializing CUDA\n");
	CUresult error = cuInit(0);
	if (error != CUDA_SUCCESS) {
		printf("cuInit(0) returned %d\n", error);
		return -1;
	}

	int deviceCount = 0;
	error = cuDeviceGetCount(&deviceCount);
	if (error != CUDA_SUCCESS) {
		printf("cuDeviceGetCount() returned %d\n", error);
		return -1;
	}
	/* This function call returns 0 if there are no CUDA capable devices. */
	if (deviceCount == 0) {
		printf("There are no available device(s) that support CUDA\n");
		return -1;
	}

	printf("Listing all CUDA devices in system:\n");
	for (int index = 0; index < deviceCount; index++) {
		if(cuDeviceGet(&cu_device, index) != CUDA_SUCCESS) exit(0);
		cuDeviceGetAttribute(&cuda_pci_bus_id, CU_DEVICE_ATTRIBUTE_PCI_BUS_ID , cu_device);
		cuDeviceGetAttribute(&cuda_pci_device_id, CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID , cu_device);
		printf("CUDA device %d: PCIe address is %02X:%02X\n", index, (unsigned int)cuda_pci_bus_id, (unsigned int)cuda_pci_device_id);
	}

	printf("\nPicking device No. %d\n", cuda_device_id);

	if(cuDeviceGet(&cuDevice_selected, cuda_device_id) != cudaSuccess){
		printf("error on cuDeviceGet\n");
		exit(0);
	}

	char name[128];
	if(cuDeviceGetName(name, sizeof(name), cuda_device_id) != cudaSuccess){
		printf("error on cuDeviceGetName\n");
		exit(0);
	}
	printf("[pid = %d, dev = %d] device name = [%s]\n", getpid(), cuDevice_selected, name);

    struct timespec res;
    long nano1,nano2;

    clock_t start, end;
    double cpu_time_used;
    struct timeval cpu_timer[2];

    size_t free_memory, total_memory;
    cudaMemGetInfo(&free_memory, &total_memory);
    printf("free memory: %zu, total_memory: %zu\n", free_memory/(1024 * 1024), total_memory/(1024 * 1024));

	clock_t *dtimer = NULL;
	clock_t timer[2];

	if (cudaSuccess != cudaMalloc((void **)&dtimer, sizeof(clock_t) * 2)) 
		exit(0);

	cudaDeviceProp devProp;
    cudaGetDeviceProperties(&devProp, 0);
    printf("Cuda device clock rate = %d\n", devProp.clockRate);

    if (cudaDeviceSynchronize() != 0) exit(0);
    // start = clock();
    // gettimeofday(&cpu_timer[0], NULL);
    clock_gettime(CLOCK_REALTIME,&res);
    nano1 = res.tv_nsec;
	// printf("Function: %s line number: %d cqe_sz: %d\n",__func__, __LINE__, cq->cqe_sz);
    global_gpu_poll_cq2<<<1,1>>>(dev_cq_ptr, dev_wc, dev_cons_index, cq->verbs_cq.cq.cqe, cq->cqe_sz, num_entries, 
                    (int *) dev_total_wc, (int *) dev_ret, dev_cq_dbrec, dev_mctx, dev_rsc, mctx->uidx_table[0].refcnt,
                    dev_qp_context /*qp_ctx*/, qp_ctx->dump_fill_mkey_be, dev_rq, dev_wrid,
                    qp->rq.wrid[0], cqe_dev, cond, dev_scat_address, dtimer, wq_sq->wrid[0], wq_sq->wqe_head[0],
                    dev_wq);
    // printf("Function: %s line number: %d\n",__func__, __LINE__);
    clock_gettime(CLOCK_REALTIME,&res);
    nano2 = res.tv_nsec;

	cudaMemcpy(timer, dtimer, sizeof(clock_t) * (2), cudaMemcpyDeviceToHost);
    cudaFree(dtimer);
	// printf("Function: %s line number: %d\n",__func__, __LINE__);
    //gettimeofday(&cpu_timer[1], NULL);
    int a = cudaDeviceSynchronize();
    if (a != 0){
        printf("cudasynchronize: %d\n", a); 
        exit(0);
    }

	float freq = (float)1/(devProp.clockRate*1000);
	float g_usec = (float)((float)1/(devProp.clockRate*1000))*((timer[1]-timer[0])) * 1000000;
	printf("POLLING - INTERNAL MEASUREMENT: %f useconds to execute \n", g_usec);
    // float timer_usec = (cpu_timer[1].tv_nsec - cpu_timer[0].tv_usec);
    // cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
    // printf("gpu_send took %lu useconds to execute \n", nano2-nano1);
    // printf("Function: %s line number: %d\n",__func__, __LINE__);
    //cudaMemcpy(&host_ret, &dev_ret, sizeof(int), cudaMemcpyDeviceToHost);
    // cudaMemcpy(&total_wc, &dev_total_wc, sizeof(int), cudaMemcpyDeviceToHost);

    
    // printf("Function: %s line number: %d\n",__func__, __LINE__);
//    do {
// 	printf("polling\n");
//        ret = /*ibv_poll_cq*/cpu_poll_cq(cq_ptr /* the CQ, we got notification for */, 
// 	       max_wc - total_wc /* number of remaining WC elements*/,
// 	       wc + total_wc/* where to store */);

//        if (ret < 0) {
// 	       printf("Failed to poll cq for wc due to %d \n", ret);
// 	       continue;
// 		   /* ret is errno here */
// 	       // return ret;
//        }
//        total_wc += ret;
//    } while (total_wc < max_wc); 
//    printf("%d WC are completed \n", total_wc);
    // printf("Function: %s line number: %d\n",__func__, __LINE__);
    /* Now we check validity and status of I/O work completions */
    // struct ibv_wc *wc_list = wc;
    // int i;
    // for( i = 0 ; i < total_wc ; i++) {
    //     if (wc_list[i].status != IBV_WC_SUCCESS) {
    //         printf("Work completion (WC) has error status: %s at index %d", 
    //                 ibv_wc_status_str(wc_list[i].status), i);
    //         /* return negative value */
    //         return -(wc_list[i].status);
    //     }
    // }
    // printf("Function: %s line number: %d\n",__func__, __LINE__);
    //    /* Similar to connection management events, we need to acknowledge CQ events */
    // ibv_ack_cq_events(cq_ptr, 
    //         1 /* we received one event notification. This is not 
    //         number of WC elements */);
    // printf("Function: %s line number: %d\n",__func__, __LINE__);
    return total_wc; 
}

__device__ int gpu_post_recv(
		   unsigned int rq_head, unsigned int rq_wqe_cnt, 
		   int rq_offset, int rq_wqe_shift, uint32_t qp_flags, int wr_num_sge,
		   uint64_t wr_id, int dev_qp_type, int ibqp_state,
       uint32_t sg_length, uint32_t sg_lkey, uint64_t sg_addr
       , void *dev_rq_wrid
		   , void *dev_qp_db
      , void *dev_qp_buf
       )
{
	uint64_t *rq_wrid = (uint64_t *) dev_rq_wrid;
	unsigned int *qp_db = (unsigned int *) dev_qp_db;
	   // struct ibv_sge *wr_sg_list = (struct ibv_sge *) dev_wr_sg_list;
	    // struct ibv_recv_wr *wr = (struct ibv_recv_wr *) dev_wr;
	ibv_qp_type qp_type = (ibv_qp_type) dev_qp_type;
	    // struct mlx5_qp *qp = to_mqp(ibqp);
	struct mlx5_wqe_data_seg *scat;
	int err = 0;
	int nreq;
	int ind;
	int i, j;
	struct mlx5_rwqe_sig *sig;
	// printf("Function: %s line number: %d\n",__func__, __LINE__);

	// unsigned int rq_head = qp->rq.head
	// int rq_offset = qp->rq.offset
	// unsigned int rq_wqe_cnt = qp->rq.wqe_cnt
	// ind = ind
	// unsigned int rq_head = qp->rq.head
	// int rq_wqe_shift = qp->rq.wqe_shift
	// uint64_t *rq_wrid qp->rq.wrid
	// unsigned int *qp_db = qp->db[0]
	// ibv_sge *wr_sg_list = wr->sg_list
	// void *dev_qp_buf = qp->buf.buf
	// int wr_num_sge = wr->num_sge
	// ibv_qp_type qp_type = ibqp->qp_type
	// ibv_qp_state ibqp_state = ibqp->state
	// uint32_t qp_flags = qp->flags
	// uint64_t wr_id = wr->wr_id
	
	
	// printf("Function: %s line number: %d\n",__func__, __LINE__);
	ind = rq_head & (rq_wqe_cnt - 1);
  // printf("Function: %s line number: %d ind: %d\n",__func__, __LINE__, ind);
	for (nreq = 0; nreq < 1; ++nreq) {
    // printf("Function: %s line number: %d\n",__func__, __LINE__);
		scat = (struct mlx5_wqe_data_seg *) (dev_qp_buf + rq_offset + (ind << rq_wqe_shift)); 
    // printf("qp->rq.offset: %d, ind: %d, qp->rq.wqe_shift: %d\n", rq_offset, ind, rq_wqe_shift);
    // sig = (struct mlx5_rwqe_sig *)scat;
    // printf("Function: %s line number: %d wr_num_sge: %d\n",__func__, __LINE__, wr_num_sge);
		for (i = 0, j = 0; i < wr_num_sge; ++i) {
			// if ((!wr_sg_list->length))
			// 	continue;
			struct mlx5_wqe_data_seg *dseg = scat;
      
			// struct ibv_sge *sg = wr_sg_list;
			int offset = 0;
			dseg->byte_count = htonl(sg_length - offset);
			dseg->lkey       = htonl(sg_lkey);
			dseg->addr       = htonl64(sg_addr + offset);
		}
    // printf("Function: %s line number: %d\n",__func__, __LINE__);
		rq_wrid[ind] = wr_id;
    // printf("Function: %s line number: %d\n",__func__, __LINE__);
		ind = (ind + 1) & (rq_wqe_cnt - 1);
	}

out:
	if ((nreq)) {
		rq_head += nreq;

    // printf("Function: %s line number: %d\n",__func__, __LINE__);
		if ((!((dev_qp_type == 8 ||
			      qp_flags & 1) &&
			     ibqp_state < 2))){
			qp_db[0] = htonl(rq_head & 0xffff);
      // printf("Function: %s line number: %d\n",__func__, __LINE__);
    }
	}
  // printf("Function: %s line number: %d err: %d\n",__func__, __LINE__, err);
	return err;
}

__global__ void global_gpu_post_recv(
			unsigned int rq_head, unsigned int rq_wqe_cnt, 
		   int rq_offset, int rq_wqe_shift, uint32_t qp_flags, int wr_num_sge,
		   uint64_t wr_id, int dev_qp_type, int ibqp_state, 
       uint32_t sg_length, uint32_t sg_lkey, uint64_t sg_addr
       , void *dev_rq_wrid
		   , void *dev_qp_db
      , void *dev_qp_buf
      , int *ret
      ){
	// printf("Function: %s line number: %d\n",__func__, __LINE__);
	int local_ret;
	local_ret = gpu_post_recv(
		   rq_head, rq_wqe_cnt, 
		   rq_offset, rq_wqe_shift, qp_flags, wr_num_sge,
		   wr_id, dev_qp_type, ibqp_state,
       sg_length, sg_lkey, sg_addr
       , dev_rq_wrid
		   , dev_qp_db
      , dev_qp_buf
      );
	*ret = local_ret;
}

int host_gpu_post_recv(struct ibv_qp *ibqp, struct ibv_recv_wr *wr,
		   struct ibv_recv_wr **bad_wr){
	// printf("Function: %s line number: %d\n",__func__, __LINE__);
	struct mlx5_qp *qp = to_mqp(ibqp);

	unsigned int rq_head = qp->rq.head;
	int rq_offset = qp->rq.offset;
	unsigned int rq_wqe_cnt = qp->rq.wqe_cnt;
	int rq_wqe_shift = qp->rq.wqe_shift;
	uint64_t *rq_wrid = qp->rq.wrid;
	// unsigned int *qp_db = qp->db;
	// struct ibv_sge *wr_sg_list = wr->sg_list;
	void *dev_qp_buf; // = qp->buf.buf;
	int wr_num_sge = wr->num_sge;
	int qp_type = (int) ibqp->qp_type;
	int ibqp_state = (int) ibqp->state;
	uint32_t qp_flags = qp->flags;
	uint64_t wr_id = wr->wr_id;
	
	// bad_wr = NULL;
	void *dev_rq_wrid;
	void *dev_qp_db; 
	void *dev_wr_sg_list;

  cudaError_t cudasuccess;
  if (wq_buffer_gpu != 1){
    cudasuccess = cudaHostRegister(qp->buf.buf, sizeof(qp->buf.buf), cudaHostRegisterMapped);
    // printf("Function: %s line number: %d cudasuccess: %d\n",__func__, __LINE__, cudasuccess);
    if(cudasuccess !=  cudaSuccess && cudasuccess != cudaErrorHostMemoryAlreadyRegistered)
          exit(0);
    // get GPU pointer for qp->buf.buf
    if(cudaHostGetDevicePointer(&dev_qp_buf, qp->buf.buf, 0) != cudaSuccess)
        exit(0);
  }
  else {
    dev_qp_buf = qp->buf.buf;
  }
	// printf("Function: %s line number: %d cudaHostRegister(qp->rq.wrid, sizeof(qp->rq.wrid), cudaHostRegisterMapped): %d\n",
  //         __func__, __LINE__, cudaHostRegister(qp->rq.wrid, sizeof(qp->rq.wrid), cudaHostRegisterMapped));
	// register qp->rq.wrid in host memory 

  // comment the below lines when gpu buffer used for qp->buf.buf
  cudasuccess = cudaHostRegister(qp->rq.wrid, sizeof(qp->rq.wrid), cudaHostRegisterMapped);
  if(cudasuccess !=  cudaSuccess && cudasuccess != cudaErrorHostMemoryAlreadyRegistered)
      exit(0);

  // get GPU pointer for qp->rq.wrid
  cudasuccess = cudaHostGetDevicePointer(&dev_rq_wrid, qp->rq.wrid, 0);
  // printf("Function: %s line number: %d cudasuccess: %d\n",__func__, __LINE__, cudasuccess);
  if(cudasuccess != cudaSuccess)
      exit(0);
	// printf("Function: %s line number: %d cudaHostRegister(qp->db, sizeof(qp->db), cudaHostRegisterMapped): %d\n",
  //         __func__, __LINE__, cudaHostRegister(qp->db, sizeof(qp->db), cudaHostRegisterMapped));
	// register qp->rq.wrid in host memory 

  // comment the below lines when gpu buffer used for qp->buf.buf
  cudasuccess = cudaHostRegister(qp->db, sizeof(qp->db), cudaHostRegisterMapped);
    if(cudasuccess !=  cudaSuccess && cudasuccess != cudaErrorHostMemoryAlreadyRegistered)
        exit(0);

    // get GPU pointer for qp->rq.wrid
    if(cudaHostGetDevicePointer(&dev_qp_db, qp->db, 0) != cudaSuccess)
        exit(0);
	// printf("Function: %s line number: %d\n",__func__, __LINE__);
	// register qp->rq.wrid in host memory 
    if(cudaHostRegister(wr->sg_list, sizeof(wr->sg_list), cudaHostRegisterMapped) !=  cudaSuccess)
        exit(0);
    // get GPU pointer for qp->rq.wrid
    if(cudaHostGetDevicePointer(&dev_wr_sg_list, wr->sg_list, 0) != cudaSuccess)
        exit(0);
	
	int ret, *dev_ret;
  cudaError_t cudaSuccess = cudaHostRegister(&ret, sizeof(ret), cudaHostRegisterMapped);
  // printf("Function: %s line number: %d cudaHostRegister(ret, sizeof(ret), cudaHostRegisterMapped): %d\n",
          // __func__, __LINE__, cudaSuccess);
	// register ret in host memory 
    if(cudaSuccess !=  cudaSuccess)
        exit(0);
    // get GPU pointer for ret
    if(cudaHostGetDevicePointer(&dev_ret, &ret, 0) != cudaSuccess)
        exit(0);
	// printf("Function: %s line number: %d\n",__func__, __LINE__);
  cudaError_t success = cudaDeviceSynchronize();
	if (success != 0) exit(0);
  // printf("Function: %s line number: %d success: %d\n",__func__, __LINE__, success);
	global_gpu_post_recv<<<1,1>>>(rq_head, rq_wqe_cnt, 
						rq_offset, rq_wqe_shift, qp_flags, wr_num_sge,
						wr_id, qp_type, ibqp_state, 
            wr->sg_list->length, wr->sg_list->lkey, wr->sg_list->addr
            , dev_rq_wrid
						, dev_qp_db
            , dev_qp_buf
            , dev_ret
            );
	
	success = cudaDeviceSynchronize();

	// if (success != 0) exit(0);
  // printf("Function: %s line number: %d success: %d\n",__func__, __LINE__, success);
  if(success != 0){
    exit(0);
  }
  
		// 	(void * dev_wr,
		//    struct ibv_recv_wr **bad_wr, unsigned int rq_head, unsigned int rq_wqe_cnt, 
		//    int rq_offset, int rq_wqe_shift, uint32_t qp_flags, int wr_num_sge,
		//    uint64_t wr_id, int dev_qp_type, int ibqp_state, void *dev_rq_wrid,
		//    void *dev_qp_db, void *dev_wr_sg_list, void *dev_qp_buf)

	return ret;

}

__device__ int device_gpu_post_send(unsigned int qpbf_bufsize,/* unsigned int qpsq_cur_post,
          unsigned int qpsq_wqe_cnt,*/ uint64_t wr_rdma_remote_addr, uint32_t wr_rdma_rkey,
          uint32_t wr_sg_length, uint32_t wr_sg_lkey, uint64_t wr_sg_addr, int wr_opcode, 
          unsigned int bf_offset, uint32_t qp_num, uint64_t wr_id, 
          void *qp_buf, /*void *dev_qpsq_wr_data,*/ void *dev_qpsq_wqe_head, 
          void *dev_qp_sq, void *dev_qp_db,/*void *dev_wr_sg,*/ void *dev_wrid,
          uint32_t *first_dword, uint32_t *second_dword, void *bf_reg, clock_t *timer)
{
  // timer[0] = clock();
  
	// struct mlx5_qp *qp = to_mqp(ibqp);
	void *seg;
	struct mlx5_wqe_eth_seg *eseg;
	struct mlx5_wqe_ctrl_seg *ctrl = NULL;
	struct mlx5_wqe_data_seg *dpseg;
	struct mlx5_sg_copy_ptr sg_copy_ptr = {.index = 0, .offset = 0};
	int nreq = 0;
	int inl = 0;
	int err = 0;
	int size = 0;
	int i;
	unsigned idx;
	uint8_t opmod = 0;

	uint32_t mlx5_opcode;
	struct mlx5_wqe_xrc_seg *xrc;

    // input variables:
    // unsigned int qpbf_bufsize = qp->bf->buf_size; // 256
    // unsigned int qpsq_cur_post = qp->sq.cur_post;
    // unsigned int qpsq_wqe_cnt = qp->sq.wqe_cnt;
    // uint64_t wr_rdma_remote_addr = wr->wr.rdma.remote_addr;
    // uint32_t wr_rdma_rkey = wr->wr.rdma.rkey;
    // uint32_t wr_sg_length = wr->sg_list->length;
    // uint32_t wr_sg_lkey = wr->sg_list->lkey;
    // uint64_t wr_sg_addr = wr->sg_list->addr;
    // int wr_opcode = wr->opcode;
    // uint32_t qp_num = ibqp->qp_num;
    // unsigned int bf_offset = bf->offset; 
    // uint64_t wr_id = wr->wr_id;
    // first_dword
    // second_dword

    // pointers
    // void* qp_buf = qp->buf.buf;
    // uint32_t *qpsq_wr_data = (uint32_t *) dev_qpsq_wr_data;//qp->sq.wr_data;
    unsigned int *qpsq_wqe_head = (unsigned int *) dev_qpsq_wqe_head;// qp->sq.wqe_head;
    struct mlx5_wq *qp_sq = (struct mlx5_wq *) dev_qp_sq; // &qp->sq;

    unsigned int *qp_db = (unsigned int *) dev_qp_db; // qp->db

    uint64_t * qpsq_wrid = (uint64_t *) dev_wrid;// qp_sq->wrid[idx] = wr_id;
    idx = qp_sq->cur_post & (qp_sq->wqe_cnt - 1);
    seg = (qp_buf + 256 + (idx * 64)); // mlx5_get_send_wqe(qp, idx);
    ctrl = (struct mlx5_wqe_ctrl_seg *) seg;
    *(uint32_t *)(seg + 8) = 0;
    ctrl->imm = 0; // send_ieth(wr);
    ctrl->fm_ce_se = 8; // MLX5_WQE_CTRL_CQ_UPDATE;

    seg = seg + 16; // sizeof(*ctrl);
    size = 1;
    // printf("sizeof(*ctrl): %d",sizeof(*ctrl) );
    // qpsq_wr_data[idx] = 0; // qp_sq->wr_data[idx] = 0;
  
    if(wr_opcode == 4 || wr_opcode == 0){
      ((struct mlx5_wqe_raddr_seg *) seg)->raddr    = htonl64(wr_rdma_remote_addr);
      ((struct mlx5_wqe_raddr_seg *) seg)->rkey     = htonl(wr_rdma_rkey);
      ((struct mlx5_wqe_raddr_seg *) seg)->reserved = 0;
      seg = seg + 16; // sizeof(struct mlx5_wqe_raddr_seg);
      size += 1; // sizeof(struct mlx5_wqe_raddr_seg) / 16;
    }

    dpseg = (struct mlx5_wqe_data_seg *) seg;
   
    ((struct mlx5_wqe_data_seg *) seg)->byte_count = htonl(wr_sg_length); // htonl(wr_sg_list->length);
    ((struct mlx5_wqe_data_seg *) seg)->lkey       = htonl(wr_sg_lkey); // htonl(wr_sg_list->lkey);
    ((struct mlx5_wqe_data_seg *) seg)->addr       = htonl64(wr_sg_addr); // htonl64(wr_sg_list->addr);
 
    size += 1; // sizeof(struct mlx5_wqe_data_seg) / 16;
    // read: 4 -> 16, write: 0 -> 8, send: 2 -> 10
    mlx5_opcode = wr_opcode*2 + 8 - 2*(wr_opcode == 2); // mlx5_ib_opcode[wr->opcode];
    ctrl->opmod_idx_opcode = htonl(((qp_sq->cur_post & 0xffff) << 8) | mlx5_opcode);
    ctrl->qpn_ds = htonl(size | (qp_num << 8));
    
    qpsq_wrid[idx] = wr_id;
    qpsq_wqe_head[idx] = qp_sq->head;
    int tmp = (size * 16 + 64 - 1) / 64;
    qp_sq->cur_post += tmp;
    // printf("Function name: %s, line number: %d\n", __func__, __LINE__); 
    // db update:
    // qp_sq->head += 1;
    qp_db[MLX5_SND_DBR] = htonl(tmp & 0xffff);

    void *addr = bf_reg; // + 0xa00;
    uint64_t val;
    /* Do 64 bytes at a time */
    // addr = bf_reg; // + bf_offset;
    val = *(uint64_t *) ctrl;
    // *first_dword = htonl(htonl64(val) >> 32);
    // *second_dword = htonl(htonl64(val));
    *(volatile uint32_t *)addr = htonl(htonl64(val) >> 32);
    *(volatile uint32_t *)(addr+4) = htonl(htonl64(val));
    // bf->offset ^= bf->buf_size;
    // printf("Function name: %s, line number: %d\n", __func__, __LINE__);
    // timer[1] = clock();
	return 0;
}

__device__ int device_gpu_post_write(unsigned int qpbf_bufsize,/* unsigned int qpsq_cur_post,
          unsigned int qpsq_wqe_cnt,*/ uint64_t wr_rdma_remote_addr, uint32_t wr_rdma_rkey,
          uint32_t wr_sg_length, uint32_t wr_sg_lkey, uint64_t wr_sg_addr, int wr_opcode, 
          unsigned int bf_offset, uint32_t qp_num, uint64_t wr_id, 
          void *qp_buf, /*void *dev_qpsq_wr_data,*/ void *dev_qpsq_wqe_head, 
          void *dev_qp_sq, void *dev_qp_db,/*void *dev_wr_sg,*/ void *dev_wrid,
          uint32_t *first_dword, uint32_t *second_dword, clock_t *timer)
{
  // timer[0] = clock();
  
	// struct mlx5_qp *qp = to_mqp(ibqp);
	void *seg;
	struct mlx5_wqe_eth_seg *eseg;
	struct mlx5_wqe_ctrl_seg *ctrl = NULL;
	struct mlx5_wqe_data_seg *dpseg;
	struct mlx5_sg_copy_ptr sg_copy_ptr = {.index = 0, .offset = 0};
	int nreq = 0;
	int inl = 0;
	int err = 0;
	int size = 0;
	int i;
	unsigned idx;
	uint8_t opmod = 0;

	uint32_t mlx5_opcode;
	struct mlx5_wqe_xrc_seg *xrc;

  unsigned int *qpsq_wqe_head = (unsigned int *) dev_qpsq_wqe_head;// qp->sq.wqe_head;
  struct mlx5_wq *qp_sq = (struct mlx5_wq *) dev_qp_sq; // &qp->sq;

  unsigned int *qp_db = (unsigned int *) dev_qp_db; // qp->db

  uint64_t * qpsq_wrid = (uint64_t *) dev_wrid;// qp_sq->wrid[idx] = wr_id;
  idx = qp_sq->cur_post & (qp_sq->wqe_cnt - 1);
  seg = (qp_buf + qpbf_bufsize + (idx * 64)); // mlx5_get_send_wqe(qp, idx);
  ctrl = (struct mlx5_wqe_ctrl_seg *) seg;
  *(uint32_t *)(seg + 8) = 0;
  ctrl->imm = 0; // send_ieth(wr);
  ctrl->fm_ce_se = 8; // MLX5_WQE_CTRL_CQ_UPDATE;

  seg = seg + sizeof(*ctrl);
  size = sizeof(*ctrl) / 16;
  printf("sizeof(*ctrl): %d",sizeof(*ctrl) );
  // qpsq_wr_data[idx] = 0; // qp_sq->wr_data[idx] = 0;


  ((struct mlx5_wqe_raddr_seg *) seg)->raddr    = htonl64(wr_rdma_remote_addr);
  ((struct mlx5_wqe_raddr_seg *) seg)->rkey     = htonl(wr_rdma_rkey);
  ((struct mlx5_wqe_raddr_seg *) seg)->reserved = 0;
  seg = seg + 16; 
  size += 1; 
  

  dpseg = (struct mlx5_wqe_data_seg *) seg;
  
  ((struct mlx5_wqe_data_seg *) seg)->byte_count = htonl(wr_sg_length); 
  ((struct mlx5_wqe_data_seg *) seg)->lkey       = htonl(wr_sg_lkey); // htonl(wr_sg_list->lkey);
  ((struct mlx5_wqe_data_seg *) seg)->addr       = htonl64(wr_sg_addr);

  size += 1; 
  // read: 4 -> 16, write: 0 -> 8, send: 2 -> 10
  mlx5_opcode = 8;
  ctrl->opmod_idx_opcode = htonl(((qp_sq->cur_post & 0xffff) << 8) | mlx5_opcode);
  ctrl->qpn_ds = htonl(size | (qp_num << 8));
  
  qpsq_wrid[idx] = wr_id;
  qpsq_wqe_head[idx] = qp_sq->head;
  int tmp = (size * 16 + 64 - 1) / 64;
  qp_sq->cur_post += tmp;
  // printf("Function name: %s, line number: %d\n", __func__, __LINE__); 
  // db update:
  qp_sq->head += 1;
  qp_db[MLX5_SND_DBR] = htonl(tmp & 0xffff);

  void *addr;
  uint64_t val;
  /* Do 64 bytes at a time */
  // addr = bf_reg + 0x800 + bf_offset;
  val = *(uint64_t *) ctrl;
  *first_dword = htonl(htonl64(val) >> 32);
  *second_dword = htonl(htonl64(val));

  //  *(volatile uint32_t *)addr = (uint32_t)( *first_dword);
  //   *(volatile uint32_t *)(addr + 4) = (uint32_t)(*second_dword);
  
	return 0;
}

__global__ void global_gpu_post_send(
          unsigned int qpbf_bufsize, uint64_t wr_rdma_remote_addr, uint32_t wr_rdma_rkey,
          uint32_t wr_sg_length, uint32_t wr_sg_lkey, uint64_t wr_sg_addr,
          int wr_opcode, unsigned int bf_offset, uint32_t qp_num, uint64_t wr_id,
          void *qp_buf, /*void *dev_qpsq_wr_data,*/ void *dev_qpsq_wqe_head, 
          void *dev_qp_sq, void *dev_qp_db,/*void *dev_wr_sg,*/ void *dev_wrid,
          uint32_t *first_dword, uint32_t *second_dword, void *bf_reg, clock_t *timer, int *ret){
  // printf("Function name: %s, line number: %d\n", __func__, __LINE__);
  int start = 0;
  timer[0] = clock();
  start = start + timer[0];
  int err = device_gpu_post_send(qpbf_bufsize, wr_rdma_remote_addr, wr_rdma_rkey,
          wr_sg_length, wr_sg_lkey, wr_sg_addr, wr_opcode, 
          bf_offset, qp_num, wr_id, qp_buf, /*dev_qpsq_wr_data,*/ 
          dev_qpsq_wqe_head, dev_qp_sq, dev_qp_db, /*dev_wr_sg,*/ dev_wrid, first_dword, second_dword, bf_reg, timer);
  timer[1] = clock();
  start = timer[1] - start;
  // printf("Function name: %s, line number: %d\n", __func__, __LINE__);

  // printf("timer[0]: %d, timer[1]: %d\n", timer[0], timer[1]);
  // printf("timer[1] - timer[0]: %d\n", timer[1] - timer[0]);
  // printf("start: %d\n", start);

  *ret = err;
}

__global__ void global_gpu_post_send2(
          unsigned int qpbf_bufsize, uint64_t wr_rdma_remote_addr, uint32_t wr_rdma_rkey,
          uint32_t wr_sg_length, uint32_t wr_sg_lkey, uint64_t wr_sg_addr,
          int wr_opcode, unsigned int bf_offset, uint32_t qp_num, uint64_t wr_id,
          void *qp_buf, /*void *dev_qpsq_wr_data,*/ void *dev_qpsq_wqe_head, 
          void *dev_qp_sq, void *dev_qp_db,/*void *dev_wr_sg,*/ void *dev_wrid,
          uint32_t *first_dword, uint32_t *second_dword, void *bf_reg, clock_t *timer, int *ret){
  // printf("Function name: %s, line number: %d\n", __func__, __LINE__);
  int start = 0;
  timer[0] = clock();
  start = start + timer[0];
  // int err = device_gpu_post_send(qpbf_bufsize, wr_rdma_remote_addr, wr_rdma_rkey,
  //         wr_sg_length, wr_sg_lkey, wr_sg_addr, wr_opcode, 
  //         bf_offset, qp_num, wr_id, qp_buf, /*dev_qpsq_wr_data,*/ 
  //         dev_qpsq_wqe_head, dev_qp_sq, dev_qp_db, /*dev_wr_sg,*/ dev_wrid, first_dword, second_dword, bf_reg, timer);

  int err = read_write_post(
          wr_rdma_remote_addr, wr_rdma_rkey,
          wr_sg_length, wr_sg_lkey, wr_sg_addr, wr_opcode, 
          qp_num, wr_id, 
          qp_buf, dev_qpsq_wqe_head, 
          dev_qp_sq, dev_qp_db, dev_wrid,
          bf_reg);
  timer[1] = clock();
  start = timer[1] - start;
  // printf("Function name: %s, line number: %d\n", __func__, __LINE__);

  // printf("timer[0]: %d, timer[1]: %d\n", timer[0], timer[1]);
  // printf("timer[1] - timer[0]: %d\n", timer[1] - timer[0]);
  // printf("start: %d\n", start);

  *ret = err;
}

__global__ void global_gpu_post_write(unsigned int qpbf_bufsize, uint64_t wr_rdma_remote_addr, uint32_t wr_rdma_rkey,
          uint32_t wr_sg_length, uint32_t wr_sg_lkey, uint64_t wr_sg_addr,
          int wr_opcode, unsigned int bf_offset, uint32_t qp_num, uint64_t wr_id,
          void *qp_buf, /*void *dev_qpsq_wr_data,*/ void *dev_qpsq_wqe_head, 
          void *dev_qp_sq, void *bf_reg, void *dev_qp_db,/*void *dev_wr_sg,*/ void *dev_wrid,
          uint32_t *first_dword, uint32_t *second_dword, clock_t *timer, int *ret){

  int start = 0;
  timer[0] = clock();
  start = start + timer[0];
  int err = device_gpu_post_write(qpbf_bufsize, wr_rdma_remote_addr, wr_rdma_rkey,
          wr_sg_length, wr_sg_lkey, wr_sg_addr, wr_opcode, 
          bf_offset, qp_num, wr_id, qp_buf, /*dev_qpsq_wr_data,*/
          dev_qpsq_wqe_head, dev_qp_sq, dev_qp_db, /*dev_wr_sg,*/ dev_wrid, first_dword, second_dword, timer);
  timer[1] = clock();
  start = timer[1] - start;
  *ret = err;
}

#include <stddef.h>

// Function to align an address to a specified byte boundary
void* align_address(void* addr, size_t alignment) {
    uintptr_t aligned_addr = ((uintptr_t)addr + alignment - 1) & ~(alignment - 1);
    return (void*)aligned_addr;
}

int host_gpu_post_send(struct ibv_qp *ibqp, struct ibv_send_wr *wr,
                       struct ibv_send_wr **bad_wr, unsigned long offset){

  int local_ret;
  struct mlx5_qp *qp = to_mqp(ibqp);

  struct mlx5_bf *bf = qp->bf;
  // printf("Function name: %s, line number: %d, qp->buf.length: %d\n", __func__, __LINE__, qp->buf.length);
  // input variables:
  unsigned int qpbf_bufsize = qp->bf->buf_size; // 256
  unsigned int qpsq_cur_post = qp->sq.cur_post;
  unsigned int qpsq_wqe_cnt = qp->sq.wqe_cnt;
  uint64_t wr_rdma_remote_addr = wr->wr.rdma.remote_addr;
  uint32_t wr_rdma_rkey = wr->wr.rdma.rkey;
  uint32_t wr_sg_length = wr->sg_list->length;
  uint32_t wr_sg_lkey = wr->sg_list->lkey;
  uint64_t wr_sg_addr = wr->sg_list->addr;
  int wr_opcode = wr->opcode;
  uint32_t qp_num = ibqp->qp_num;
  unsigned int bf_offset = bf->offset; 
  uint64_t wr_id = wr->wr_id;
  // printf("Function name: %s, line number: %d\n", __func__, __LINE__);
  // pointers
  void *qp_buf;
  void *dev_qpsq_wr_data; //qp->sq.wr_data;
  void *dev_qpsq_wqe_head;// qp->sq.wqe_head;
  void *dev_qp_sq; // &qp->sq;
  int *bf_reg = (int *) bf->reg; // bf->reg;
  void *dev_qp_db; // qp->db
  void *dev_wr_sg; // wr->sg_list
  int *dev_ret; // &ret
  void *dev_bf_reg; 
  void *dev_wrid; // qp->sq.wrid
  void *dev_wqe_head; // qp->sq.wqe_head
  
  // printf("Function name: %s, line number: %d cudaHostRegister(qp->buf.buf, sizeof(qp->buf.buf), cudaHostRegisterMapped): %d\n", 
  //         __func__, __LINE__, cudaHostRegister(qp->buf.buf, sizeof(qp->buf.buf), cudaHostRegisterMapped));
  cudaError_t success;
  // pin host memory and get device pointer:
  if (wq_buffer_gpu != 1){
    success = cudaHostRegister(qp->buf.buf, qp->buf.length, cudaHostRegisterMapped);
    // comment the below lines when gpu buffer used for qp->buf.buf
    if(success !=  cudaSuccess && success != cudaErrorHostMemoryAlreadyRegistered)
          exit(0);
    // get GPU pointer for qp_buf
    // printf("Function name: %s, line number: %d cudaHostGetDevicePointer(&qp_buf, qp->buf.buf, 0): %d\n", __func__, __LINE__,
    //         cudaHostGetDevicePointer(&qp_buf, qp->buf.buf, 0));
    if(cudaHostGetDevicePointer(&qp_buf, qp->buf.buf, 0) != cudaSuccess)
        exit(0);
  }
  else {
    qp_buf = qp->buf.buf;
  }
  // printf("Function name: %s, line number: %d\n", __func__, __LINE__);
  success = cudaHostRegister(qp->sq.wr_data, sizeof(qp->sq.wr_data), cudaHostRegisterMapped);
  if(success != cudaErrorHostMemoryAlreadyRegistered && success != cudaSuccess)
        exit(0);
  // get GPU pointer for qp->sq.wr_data
  if(cudaHostGetDevicePointer(&dev_qpsq_wr_data, qp->sq.wr_data, 0) != cudaSuccess)
      exit(0);
  success = cudaHostRegister(qp->sq.wqe_head, sizeof(qp->sq.wqe_head), cudaHostRegisterMapped);
  if(success != cudaErrorHostMemoryAlreadyRegistered && success !=  cudaSuccess)
        exit(0);
  // get GPU pointer for qp->sq.wqe_head
  if(cudaHostGetDevicePointer(&dev_qpsq_wqe_head, qp->sq.wqe_head, 0) != cudaSuccess)
      exit(0);

  // printf("Function name: %s, line number: %d\n", __func__, __LINE__);
  success = cudaHostRegister(&qp->sq, sizeof(qp->sq), cudaHostRegisterMapped);
  if(success != cudaErrorHostMemoryAlreadyRegistered && success !=  cudaSuccess)
        exit(0);
  // get GPU pointer for qp->sq
  if(cudaHostGetDevicePointer(&dev_qp_sq, &qp->sq, 0) != cudaSuccess)
      exit(0);

  

  // printf("Function name: %s, line number: %d cudaHostRegister(bf_reg, sizeof(bf_reg), cudaHostRegisterMapped): %d\n", 
  //         __func__, __LINE__, cudaHostRegister(bf->reg, sizeof(bf), cudaHostRegisterMapped));
 
  // printf("Function name: %s, line number: %d bf->offset: %d\n", __func__, __LINE__, bf->offset);
  // printf("Function name: %s, line number: %d bf->reg: 0x%llx\n", __func__, __LINE__, bf->reg);
  printf("Function name: %s, line number: %d bf->uar: 0x%llx\n", __func__, __LINE__, bf->uar);
  // printf("Function name: %s, line number: %d bf->buf_size: %d\n", __func__, __LINE__, bf->buf_size);

  cudaError_t cudaStatus1;

  void *device_db;
  cudaStatus1 = cudaHostRegister(bf->reg,  8, cudaHostRegisterIoMemory);
    if (cudaStatus1 == cudaSuccess || cudaStatus1 == cudaErrorHostMemoryAlreadyRegistered) {
      printf("cudaHostRegister successful for address 0x%llx: %s and i: %d\n", bf->uar, cudaGetErrorString(cudaStatus1), bf->length);
      // exit(0);
    }
    else {
      printf("cudaHostRegister not success for address 0x%llx: %s and i: %d\n", bf->uar, cudaGetErrorString(cudaStatus1), bf->length);
      exit(0);
    }
    cudaStatus1 = cudaHostGetDevicePointer(&device_db, bf->reg, 0);
    if (cudaStatus1 == cudaSuccess) {
      printf("cudaHostGetDevicePointer successful with no error: %s\n", cudaGetErrorString(cudaStatus1));
      // exit(0);
    }
    else {
      printf("cudaHostGetDevicePointer failed with  error: %s\n", cudaGetErrorString(cudaStatus1));
      exit(0);
    }

  // unsigned long tmp_flag = (unsigned long)(offset)*0x1000;
  
  // printf("Function name: %s, line number: %d bf->offset: %lu\n", __func__, __LINE__, bf->offset);
  // printf("Function name: %s, line number: %d bf->uar_mmap_offset: 0x%lx\n", __func__, __LINE__, bf->uar_mmap_offset);
  // printf("Function name: %s, line number: %d bf->uuarn: 0x%lx\n", __func__, __LINE__, bf->uuarn);
  // // uar_mmap_offset
  // printf("Function name: %s, line number: %d ibqp->uuar: %lu\n", __func__, __LINE__, ibqp->uuar);
  // printf("Function name: %s, line number: %d tmp_flag: 0x%lx\n", __func__, __LINE__, tmp_flag);

  // void *nvidia_buf;

	// /* Allocate fake memory alligned to 64K */
	// if(posix_memalign(&nvidia_buf, 64*1024, 32*4096)) {
	// 	printf("Failed to allocated memory for DB register\n");
	// 		return 1;
	// 	}
	// memset(nvidia_buf, 0, 32*4096);

	// /* Update DoorBell uuar in driver */
	// /* 1- Send wrong registeration address */
	// /* 2- Update the uuar address in the driver */
	// cudaHostRegister((void *)(0x1000), 4096, cudaHostRegisterMapped);
  // printf("Function name: %s, line number: %d cudaStatus1: %d\n",  __func__, __LINE__, cudaStatus1);
	// cudaHostRegister((void *)tmp_flag, 4096, cudaHostRegisterMapped);
  // printf("Function name: %s, line number: %d cudaStatus1: %d\n",  __func__, __LINE__, cudaStatus1);
  
  // printf("Function name: %s, line number: %d nvidia_buf: 0x%llx\n",  __func__, __LINE__, nvidia_buf);
	// /* Register DoorBell in driver */
	// cudaStatus1 = cudaHostRegister(nvidia_buf, 32*4096, cudaHostRegisterMapped);
  // printf("Function name: %s, line number: %d cudaStatus1: %d\n",  __func__, __LINE__, cudaStatus1);
	// cudaStatus1 = cudaHostGetDevicePointer(&device_db, nvidia_buf, 0);
  // printf("Function name: %s, line number: %d cudaStatus1: %d\n",  __func__, __LINE__, cudaStatus1);
  // printf("Function name: %s, line number: %d nvidia_buf: 0x%llx\n",  __func__, __LINE__, nvidia_buf);

  // if(mlock(bf->uar, bf->length) != 0) {
  //       printf("mlock failed");
  //       return -1;
  // }

  // bf->reg = nvidia_buf;
  // cudaStatus1 = cudaHostRegister(bf->uar, bf->length, cudaHostRegisterMapped);
  // if (cudaStatus1 != cudaSuccess) {
  //   printf("cudaHostRegister success for address 0x%llx: %s and i: %d\n", (unsigned long long)bf->uar, cudaGetErrorString(cudaStatus1), bf->length);
  //   // exit(0);
  // }

  // cudaStatus1 = cudaHostGetDevicePointer(&device_db, bf->uar, 0);
  // if (cudaStatus1 != cudaSuccess) {
  //   printf("Function name: %s, line number: %d cudaStatus1: %d\n",  __func__, __LINE__, cudaStatus1);
  //   // exit(0);
  // }


  // bf->reg = nvidia_buf;
  // for(int i = 1; i < 1024*1024; i++){
  //   cudaError_t cudaStatus1 = cudaHostRegister(bf->uar, i, cudaHostRegisterMapped);
  //   if (cudaStatus1 == cudaSuccess) {
  //     printf("cudaHostRegister success for address 0x%llx: %s and i: %d\n", (unsigned long long)nvidia_buf, cudaGetErrorString(cudaStatus1), sizeof(nvidia_buf));
  //     exit(0);
  //   }
  // }
  
	/* Allocate fake memory alligned to 64K */
	// if(posix_memalign(&nvidia_buf, 64*1024, 32*4096)) {
	// 	printf("Failed to allocated memory for DB register\n");
	// 		return 1;
	// 	}
	// memset(nvidia_buf, 0, 32*4096);
  // printf("Aligned Address: 0x%llx\n", (unsigned long long)nvidia_buf);
  // cudaStatus = cudaHostRegister(nvidia_buf, 1*4096, cudaHostRegisterMapped);
  // if (cudaStatus != cudaSuccess) {
  //     printf("cudaHostRegister failed for address: 0x%llx, :%s\n", nvidia_buf, cudaGetErrorString(cudaStatus));
  //     // Additional error handling...
  // }

  // for(int i = 1; i < 1024*1024; i++){
  //   cudaError_t cudaStatus1 = cudaHostRegister(nvidia_buf, i, cudaHostRegisterMapped);
  //   if (cudaStatus1 == cudaSuccess) {
  //     printf("cudaHostRegister success for address 0x%llx: %s and i: %d\n", (unsigned long long)nvidia_buf, cudaGetErrorString(cudaStatus1), i);
  //     exit(0);
  //   }
  // }

  // if(cudaHostRegister(bf_reg, sizeof(bf_reg), cudaHostRegisterMapped) !=  cudaSuccess)
  //       exit(0);
  // get GPU pointer for bf->reg 0x7f43b65f8900 0x7f746c3e9800
  // printf("Function name: %s, line number: %d cudaHostGetDevicePointer(&dev_bf_reg, bf_reg, 0): %d\n", 
  //         __func__, __LINE__, cudaHostGetDevicePointer(&dev_bf_reg, nvidia_buf, 0));
  // if(cudaHostGetDevicePointer(&dev_bf_reg, nvidia_buf, 0) != cudaSuccess)
  //     exit(0);

  // comment the below lines when gpu buffer used for qp->buf.buf
  // printf("Function name: %s, line number: %d qp_registered: %d cudaHostRegister(qp->db, sizeof(qp->db), cudaHostRegisterMapped): %d\n",
  //          __func__, __LINE__, qp_registered, cudaHostRegister(qp->db, sizeof(qp->db), cudaHostRegisterMapped));
  
  // void *nvidia_buf;
  // void *dev_bf;
  // if(posix_memalign(&nvidia_buf, 64*1024, 32*4096)) {
	// 	printf("Failed to allocated memory for DB register\n");
	// 		return 1;
	// 	}
	// memset(nvidia_buf, 0, 32*4096);
  // bf->uar = nvidia_buf; 

  cudaError_t cudaState;//= cudaHostRegister(nvidia_buf, 32*4096, cudaHostRegisterMapped);
  // if(cudaState != cudaSuccess && cudaState != cudaErrorHostMemoryAlreadyRegistered)
  //       exit(0);
  // // printf("Function name: %s, line number: %d\n", __func__, __LINE__);
  // // get GPU pointer for qp->db
  // if(cudaHostGetDevicePointer(&dev_bf, nvidia_buf, 0) != cudaSuccess)
  //     exit(0);

  cudaState = cudaHostRegister(qp->db, sizeof(qp->db), cudaHostRegisterMapped);

  if(cudaState != cudaSuccess && cudaState != cudaErrorHostMemoryAlreadyRegistered)
        exit(0);
  // printf("Function name: %s, line number: %d\n", __func__, __LINE__);
  // get GPU pointer for qp->db
  if(cudaHostGetDevicePointer(&dev_qp_db, qp->db, 0) != cudaSuccess)
      exit(0);
  cudaError_t cudaStatus = cudaHostRegister(&local_ret, sizeof(local_ret), cudaHostRegisterMapped);
  // printf("Function name: %s, line number: %d cudaHostRegister(&ret, sizeof(ret), cudaHostRegisterMapped): %d\n", 
          // __func__, __LINE__, cudaStatus);
  if(cudaStatus != cudaErrorHostMemoryAlreadyRegistered && cudaStatus !=  cudaSuccess)
        exit(0);
  // printf("Function name: %s, line number: %d\n", __func__, __LINE__);
  cudaStatus = cudaHostGetDevicePointer(&dev_ret, &local_ret, 0) ;
  // printf("Function name: %s, line number: %d cudaStatus: %d\n",
          // __func__, __LINE__, cudaStatus);
  // get GPU pointer for qp->db
  if(cudaStatus != cudaSuccess)
      exit(0);
  // printf("Function name: %s, line number: %d cudaHostRegister(wr->sg_list, sizeof(wr->sg_list), cudaHostRegisterMapped): %d\n",
          // __func__, __LINE__, 1, sizeof(wr->sg_list), cudaHostRegisterMapped));

  
  cudaStatus = cudaHostRegister(wr->sg_list, sizeof(wr->sg_list), cudaHostRegisterMapped);
  // printf("Function name: %s, line number: %d cudaSuccess: %d\n",
  //         __func__, __LINE__, cudaSuccess);
  
  cudaStatus = cudaHostRegister(wr->sg_list, sizeof(wr->sg_list), cudaHostRegisterMapped);
  // printf("Function name: %s, line number: %d cudaSuccess: %d\n",
  //         __func__, __LINE__, cudaSuccess);

  // comment the below lines when gpu buffer used for qp->buf.buf
  if(cudaStatus != cudaErrorHostMemoryAlreadyRegistered && cudaSuccess !=  cudaSuccess)
        exit(0);
  // get GPU pointer for qp->db
  if(cudaHostGetDevicePointer(&dev_wr_sg, wr->sg_list, 0) != cudaSuccess)
      exit(0);
  // printf("Function name: %s, line number: %d\n", __func__, __LINE__);


  printf("Function name: %s, line number: %d &qp->sq: 0x%llx\n", __func__, __LINE__, &qp->sq);
  cudaStatus = cudaHostRegister(qp->sq.wrid, sizeof(qp->sq.wrid), cudaHostRegisterMapped);
  printf("Function name: %s, line number: %d cudaStatus: %d\n", __func__, __LINE__, cudaStatus);
  if(cudaStatus != cudaErrorHostMemoryAlreadyRegistered && cudaStatus != cudaSuccess)
        exit(0);
  // get GPU pointer for qp->sq
  if(cudaHostGetDevicePointer(&dev_wrid, qp->sq.wrid, 0) != cudaSuccess)
      exit(0);
  printf("Function name: %s, line number: %d cudaStatus: %d\n", __func__, __LINE__, cudaStatus);
  // printf("Function name: %s, line number: %d\n", __func__, __LINE__);
  // printf("Function name: %s, line number: %d cudaHostGetDevicePointer(&dev_wrid, qp->sq.wrid, 0): %d\n", __func__, __LINE__, cudaHostGetDevicePointer(&dev_wrid, qp->sq.wrid, 0));
  // if(cudaHostRegister(qp->sq.wqe_head, sizeof(qp->sq.wqe_head), cudaHostRegisterMapped) !=  cudaSuccess)
  //       exit(0);
  // // get GPU pointer for qp->sq
  // if(cudaHostGetDevicePointer(&dev_wqe_head, qp->sq.wqe_head, 0) != cudaSuccess)
  //     exit(0);
  printf("Function name: %s, line number: %d cudaStatus: %d\n", __func__, __LINE__, cudaStatus);
  uint32_t first_dword, second_dword, *dev_first, *dev_second;
  cudaError_t cuda_success = cudaHostRegister(&first_dword, sizeof(first_dword), cudaHostRegisterMapped);
  if(cuda_success != cudaErrorHostMemoryAlreadyRegistered &&  cuda_success !=  cudaSuccess)
        exit(0);
  // get GPU pointer for qp->sq
  if(cudaHostGetDevicePointer(&dev_first, &first_dword, 0) != cudaSuccess)
      exit(0);
  // printf("Function name: %s, line number: %d\n", __func__, __LINE__);
  printf("Function name: %s, line number: %d cudaStatus: %d\n", __func__, __LINE__, cudaStatus);
  cuda_success = cudaHostRegister(&second_dword, sizeof(second_dword), cudaHostRegisterMapped);
  // printf("Function name: %s, line number: %d cuda_success: %d\n", __func__, __LINE__, cuda_success);
  if(cuda_success != cudaSuccess && cuda_success != cudaErrorHostMemoryAlreadyRegistered)
        exit(0);
  // get GPU pointer for qp->sq
  printf("Function name: %s, line number: %d cudaStatus: %d\n", __func__, __LINE__, cudaStatus);
  if(cudaHostGetDevicePointer(&dev_second, &second_dword, 0) != cudaSuccess)
      exit(0);
  // printf("Function name: %s, line number: %d\n", __func__, __LINE__);
  printf("Function name: %s, line number: %d cudaStatus: %d\n", __func__, __LINE__, cudaStatus);
  clock_t *dtimer = NULL;
	clock_t timer[2];

	if (cudaSuccess != cudaMalloc((void **)&dtimer, sizeof(clock_t) * 2)) 
		exit(0);

  cuda_success = cudaDeviceSynchronize();
  printf("Function name: %s, line number: %d cuda_success: %d\n", __func__, __LINE__, cuda_success);
  if (cuda_success != 0) exit(0);

  global_gpu_post_send<<<1,1>>>(
          qpbf_bufsize, /*qpsq_cur_post, qpsq_wqe_cnt,*/
          wr_rdma_remote_addr, wr_rdma_rkey,
          wr_sg_length, wr_sg_lkey, wr_sg_addr,
          wr_opcode, 
          bf_offset, qp_num, wr_id, /*qp->buf.buf*/ qp_buf, /*dev_qpsq_wr_data,*/ 
          dev_qpsq_wqe_head, dev_qp_sq, dev_qp_db, /*dev_wr_sg,*/ dev_wrid, 
          dev_first, dev_second, device_db, dtimer, dev_ret);

  // printf("Function name: %s, line number: %d cudaDeviceSynchronize(): %d\n", __func__, __LINE__, cudaDeviceSynchronize());
  cuda_success = cudaDeviceSynchronize();
  printf("Function name: %s, line number: %d cuda_success: %d\n", __func__, __LINE__, cuda_success);
  if (cuda_success != 0) exit(0);

  cudaMemcpy(timer, dtimer, sizeof(clock_t) * (2), cudaMemcpyDeviceToHost);
  cudaFree(dtimer);

  cudaDeviceProp devProp;
  cudaGetDeviceProperties(&devProp, 0);
  // printf("Cuda device clock rate = %d\n", devProp.clockRate);

  float freq = (float)1/((float)devProp.clockRate*1000);
	float g_usec = (float)((float)1/(devProp.clockRate*1000))*((timer[1]-timer[0])) * 1000000;
	printf("POST - INTERNAL MEASUREMENT: %f useconds to execute \n", g_usec);

  void *addr;

    /* Do 64 bytes at a time */
    addr = bf->uar + 0xa00; // +  bf->offset;

    // *(volatile uint32_t *)addr = (uint32_t)( first_dword);
    // *(volatile uint32_t *)(addr + 4) = (uint32_t)(second_dword);

  return local_ret;

}

int host_gpu_post_send2(struct ibv_qp *ibqp, struct ibv_send_wr *wr,
                       struct ibv_send_wr **bad_wr, unsigned long offset){

  int local_ret;
  struct mlx5_qp *qp = to_mqp(ibqp);

  struct mlx5_bf *bf = qp->bf;
  // printf("Function name: %s, line number: %d, qp->buf.length: %d\n", __func__, __LINE__, qp->buf.length);
  // input variables:
  unsigned int qpbf_bufsize = qp->bf->buf_size; // 256
  unsigned int qpsq_cur_post = qp->sq.cur_post;
  unsigned int qpsq_wqe_cnt = qp->sq.wqe_cnt;
  uint64_t wr_rdma_remote_addr = wr->wr.rdma.remote_addr;
  uint32_t wr_rdma_rkey = wr->wr.rdma.rkey;
  uint32_t wr_sg_length = wr->sg_list->length;
  uint32_t wr_sg_lkey = wr->sg_list->lkey;
  uint64_t wr_sg_addr = wr->sg_list->addr;
  int wr_opcode = wr->opcode;
  uint32_t qp_num = ibqp->qp_num;
  unsigned int bf_offset = bf->offset; 
  uint64_t wr_id = wr->wr_id;
  // printf("Function name: %s, line number: %d\n", __func__, __LINE__);
  // pointers
  void *qp_buf;
  void *dev_qpsq_wr_data; //qp->sq.wr_data;
  void *dev_qpsq_wqe_head;// qp->sq.wqe_head;
  void *dev_qp_sq; // &qp->sq;
  int *bf_reg = (int *) bf->reg; // bf->reg;
  void *dev_qp_db; // qp->db
  void *dev_wr_sg; // wr->sg_list
  int *dev_ret; // &ret
  void *dev_bf_reg; 
  void *dev_wrid; // qp->sq.wrid
  void *dev_wqe_head; // qp->sq.wqe_head
  
  // printf("Function name: %s, line number: %d cudaHostRegister(qp->buf.buf, sizeof(qp->buf.buf), cudaHostRegisterMapped): %d\n", 
  //         __func__, __LINE__, cudaHostRegister(qp->buf.buf, sizeof(qp->buf.buf), cudaHostRegisterMapped));
  cudaError_t success;
  // pin host memory and get device pointer:
  if (wq_buffer_gpu != 1){
    success = cudaHostRegister(qp->buf.buf, qp->buf.length, cudaHostRegisterMapped);
    // comment the below lines when gpu buffer used for qp->buf.buf
    if(success !=  cudaSuccess && success != cudaErrorHostMemoryAlreadyRegistered)
          exit(0);
    // get GPU pointer for qp_buf
    // printf("Function name: %s, line number: %d cudaHostGetDevicePointer(&qp_buf, qp->buf.buf, 0): %d\n", __func__, __LINE__,
    //         cudaHostGetDevicePointer(&qp_buf, qp->buf.buf, 0));
    if(cudaHostGetDevicePointer(&qp_buf, qp->buf.buf, 0) != cudaSuccess)
        exit(0);
  }
  else {
    qp_buf = qp->buf.buf;
  }
  // printf("Function name: %s, line number: %d\n", __func__, __LINE__);
  success = cudaHostRegister(qp->sq.wr_data, sizeof(qp->sq.wr_data), cudaHostRegisterMapped);
  if(success != cudaErrorHostMemoryAlreadyRegistered && success != cudaSuccess)
        exit(0);
  // get GPU pointer for qp->sq.wr_data
  if(cudaHostGetDevicePointer(&dev_qpsq_wr_data, qp->sq.wr_data, 0) != cudaSuccess)
      exit(0);
  success = cudaHostRegister(qp->sq.wqe_head, sizeof(qp->sq.wqe_head), cudaHostRegisterMapped);
  if(success != cudaErrorHostMemoryAlreadyRegistered && success !=  cudaSuccess)
        exit(0);
  // get GPU pointer for qp->sq.wqe_head
  if(cudaHostGetDevicePointer(&dev_qpsq_wqe_head, qp->sq.wqe_head, 0) != cudaSuccess)
      exit(0);

  // printf("Function name: %s, line number: %d\n", __func__, __LINE__);
  success = cudaHostRegister(&qp->sq, sizeof(qp->sq), cudaHostRegisterMapped);
  if(success != cudaErrorHostMemoryAlreadyRegistered && success !=  cudaSuccess)
        exit(0);
  // get GPU pointer for qp->sq
  if(cudaHostGetDevicePointer(&dev_qp_sq, &qp->sq, 0) != cudaSuccess)
      exit(0);


  cudaError_t cudaStatus1;

  void *device_db;
  cudaStatus1 = cudaHostRegister(bf->reg,  8, cudaHostRegisterIoMemory);
    if (cudaStatus1 == cudaSuccess || cudaStatus1 == cudaErrorHostMemoryAlreadyRegistered) {
      printf("cudaHostRegister successful for address 0x%llx: %s and i: %d\n", bf->uar, cudaGetErrorString(cudaStatus1), bf->length);
      // exit(0);
    }
    else {
      printf("cudaHostRegister not success for address 0x%llx: %s and i: %d\n", bf->uar, cudaGetErrorString(cudaStatus1), bf->length);
      exit(0);
    }
    cudaStatus1 = cudaHostGetDevicePointer(&device_db, bf->reg, 0);
    if (cudaStatus1 == cudaSuccess) {
      printf("cudaHostGetDevicePointer successful with no error: %s\n", cudaGetErrorString(cudaStatus1));
      // exit(0);
    }
    else {
      printf("cudaHostGetDevicePointer failed with  error: %s\n", cudaGetErrorString(cudaStatus1));
      exit(0);
    }

  cudaError_t cudaState;
  cudaState = cudaHostRegister(qp->db, sizeof(qp->db), cudaHostRegisterMapped);

  if(cudaState != cudaSuccess && cudaState != cudaErrorHostMemoryAlreadyRegistered)
        exit(0);
  // printf("Function name: %s, line number: %d\n", __func__, __LINE__);
  // get GPU pointer for qp->db
  if(cudaHostGetDevicePointer(&dev_qp_db, qp->db, 0) != cudaSuccess)
      exit(0);
  cudaError_t cudaStatus = cudaHostRegister(&local_ret, sizeof(local_ret), cudaHostRegisterMapped);
  // printf("Function name: %s, line number: %d cudaHostRegister(&ret, sizeof(ret), cudaHostRegisterMapped): %d\n", 
          // __func__, __LINE__, cudaStatus);
  if(cudaStatus != cudaErrorHostMemoryAlreadyRegistered && cudaStatus !=  cudaSuccess)
        exit(0);
  // printf("Function name: %s, line number: %d\n", __func__, __LINE__);
  cudaStatus = cudaHostGetDevicePointer(&dev_ret, &local_ret, 0) ;
  // printf("Function name: %s, line number: %d cudaStatus: %d\n",
          // __func__, __LINE__, cudaStatus);
  // get GPU pointer for qp->db
  if(cudaStatus != cudaSuccess)
      exit(0);
  // printf("Function name: %s, line number: %d cudaHostRegister(wr->sg_list, sizeof(wr->sg_list), cudaHostRegisterMapped): %d\n",
          // __func__, __LINE__, 1, sizeof(wr->sg_list), cudaHostRegisterMapped));

  
  cudaStatus = cudaHostRegister(wr->sg_list, sizeof(wr->sg_list), cudaHostRegisterMapped);
  // printf("Function name: %s, line number: %d cudaSuccess: %d\n",
  //         __func__, __LINE__, cudaSuccess);
  
  cudaStatus = cudaHostRegister(wr->sg_list, sizeof(wr->sg_list), cudaHostRegisterMapped);
  // printf("Function name: %s, line number: %d cudaSuccess: %d\n",
  //         __func__, __LINE__, cudaSuccess);

  // comment the below lines when gpu buffer used for qp->buf.buf
  if(cudaStatus != cudaErrorHostMemoryAlreadyRegistered && cudaSuccess !=  cudaSuccess)
        exit(0);
  // get GPU pointer for qp->db
  if(cudaHostGetDevicePointer(&dev_wr_sg, wr->sg_list, 0) != cudaSuccess)
      exit(0);
  // printf("Function name: %s, line number: %d\n", __func__, __LINE__);


  printf("Function name: %s, line number: %d &qp->sq: 0x%llx\n", __func__, __LINE__, &qp->sq);
  cudaStatus = cudaHostRegister(qp->sq.wrid, sizeof(qp->sq.wrid), cudaHostRegisterMapped);
  printf("Function name: %s, line number: %d cudaStatus: %d\n", __func__, __LINE__, cudaStatus);
  if(cudaStatus != cudaErrorHostMemoryAlreadyRegistered && cudaStatus != cudaSuccess)
        exit(0);
  // get GPU pointer for qp->sq
  if(cudaHostGetDevicePointer(&dev_wrid, qp->sq.wrid, 0) != cudaSuccess)
      exit(0);
  printf("Function name: %s, line number: %d cudaStatus: %d\n", __func__, __LINE__, cudaStatus);

  printf("Function name: %s, line number: %d cudaStatus: %d\n", __func__, __LINE__, cudaStatus);
  uint32_t first_dword, second_dword, *dev_first, *dev_second;
  cudaError_t cuda_success = cudaHostRegister(&first_dword, sizeof(first_dword), cudaHostRegisterMapped);
  if(cuda_success != cudaErrorHostMemoryAlreadyRegistered &&  cuda_success !=  cudaSuccess)
        exit(0);
  // get GPU pointer for qp->sq
  if(cudaHostGetDevicePointer(&dev_first, &first_dword, 0) != cudaSuccess)
      exit(0);
  // printf("Function name: %s, line number: %d\n", __func__, __LINE__);
  printf("Function name: %s, line number: %d cudaStatus: %d\n", __func__, __LINE__, cudaStatus);
  cuda_success = cudaHostRegister(&second_dword, sizeof(second_dword), cudaHostRegisterMapped);
  // printf("Function name: %s, line number: %d cuda_success: %d\n", __func__, __LINE__, cuda_success);
  if(cuda_success != cudaSuccess && cuda_success != cudaErrorHostMemoryAlreadyRegistered)
        exit(0);
  // get GPU pointer for qp->sq
  printf("Function name: %s, line number: %d cudaStatus: %d\n", __func__, __LINE__, cudaStatus);
  if(cudaHostGetDevicePointer(&dev_second, &second_dword, 0) != cudaSuccess)
      exit(0);
  // printf("Function name: %s, line number: %d\n", __func__, __LINE__);
  printf("Function name: %s, line number: %d cudaStatus: %d\n", __func__, __LINE__, cudaStatus);
  clock_t *dtimer = NULL;
	clock_t timer[2];

	if (cudaSuccess != cudaMalloc((void **)&dtimer, sizeof(clock_t) * 2)) 
		exit(0);

  cuda_success = cudaDeviceSynchronize();
  printf("Function name: %s, line number: %d cuda_success: %d\n", __func__, __LINE__, cuda_success);
  if (cuda_success != 0) exit(0);

  global_gpu_post_send2<<<1,1>>>(
          qpbf_bufsize, /*qpsq_cur_post, qpsq_wqe_cnt,*/
          wr_rdma_remote_addr, wr_rdma_rkey,
          wr_sg_length, wr_sg_lkey, wr_sg_addr,
          wr_opcode, 
          bf_offset, qp_num, wr_id, /*qp->buf.buf*/ qp_buf, /*dev_qpsq_wr_data,*/ 
          dev_qpsq_wqe_head, dev_qp_sq, dev_qp_db, /*dev_wr_sg,*/ dev_wrid, 
          dev_first, dev_second, device_db, dtimer, dev_ret);

  // printf("Function name: %s, line number: %d cudaDeviceSynchronize(): %d\n", __func__, __LINE__, cudaDeviceSynchronize());
  cuda_success = cudaDeviceSynchronize();
  printf("Function name: %s, line number: %d cuda_success: %d\n", __func__, __LINE__, cuda_success);
  if (cuda_success != 0) exit(0);

  cudaMemcpy(timer, dtimer, sizeof(clock_t) * (2), cudaMemcpyDeviceToHost);
  cudaFree(dtimer);

  cudaDeviceProp devProp;
  cudaGetDeviceProperties(&devProp, 0);
  // printf("Cuda device clock rate = %d\n", devProp.clockRate);

  float freq = (float)1/((float)devProp.clockRate*1000);
	float g_usec = (float)((float)1/(devProp.clockRate*1000))*((timer[1]-timer[0])) * 1000000;
	printf("POST - INTERNAL MEASUREMENT: %f useconds to execute \n", g_usec);

  void *addr;

    /* Do 64 bytes at a time */
    addr = bf->uar + 0xa00; // +  bf->offset;

  return local_ret;
}

int host_gpu_post_write(struct ibv_qp *ibqp, struct ibv_send_wr *wr, struct ibv_send_wr **bad_wr){

  int local_ret;
  struct mlx5_qp *qp = to_mqp(ibqp);

  struct mlx5_bf *bf = qp->bf;
  // printf("Function name: %s, line number: %d, qp->buf.length: %d\n", __func__, __LINE__, qp->buf.length);
  // input variables:
  unsigned int qpbf_bufsize = qp->bf->buf_size; // 256
  unsigned int qpsq_cur_post = qp->sq.cur_post;
  unsigned int qpsq_wqe_cnt = qp->sq.wqe_cnt;
  uint64_t wr_rdma_remote_addr = wr->wr.rdma.remote_addr;
  uint32_t wr_rdma_rkey = wr->wr.rdma.rkey;
  uint32_t wr_sg_length = wr->sg_list->length;
  uint32_t wr_sg_lkey = wr->sg_list->lkey;
  uint64_t wr_sg_addr = wr->sg_list->addr;
  int wr_opcode = wr->opcode;
  uint32_t qp_num = ibqp->qp_num;
  unsigned int bf_offset = bf->offset; 
  uint64_t wr_id = wr->wr_id;
  // printf("Function name: %s, line number: %d\n", __func__, __LINE__);
  // pointers
  void *qp_buf;
  void *dev_qpsq_wr_data; //qp->sq.wr_data;
  void *dev_qpsq_wqe_head;// qp->sq.wqe_head;
  void *dev_qp_sq; // &qp->sq;
  int *bf_reg = (int *) bf->reg; // bf->reg;
  void *dev_qp_db; // qp->db
  void *dev_wr_sg; // wr->sg_list
  int *dev_ret; // &ret
  void *dev_bf_reg; 
  void *dev_wrid; // qp->sq.wrid
  void *dev_wqe_head; // qp->sq.wqe_head
  
  // printf("Function name: %s, line number: %d cudaHostRegister(qp->buf.buf, sizeof(qp->buf.buf), cudaHostRegisterMapped): %d\n", 
  //         __func__, __LINE__, cudaHostRegister(qp->buf.buf, sizeof(qp->buf.buf), cudaHostRegisterMapped));
  cudaError_t success;
  // pin host memory and get device pointer:
  if (wq_buffer_gpu != 1){
    success = cudaHostRegister(qp->buf.buf, qp->buf.length, cudaHostRegisterMapped);
    // comment the below lines when gpu buffer used for qp->buf.buf
    if(success !=  cudaSuccess && success != cudaErrorHostMemoryAlreadyRegistered)
          exit(0);
    // get GPU pointer for qp_buf
    // printf("Function name: %s, line number: %d cudaHostGetDevicePointer(&qp_buf, qp->buf.buf, 0): %d\n", __func__, __LINE__,
    //         cudaHostGetDevicePointer(&qp_buf, qp->buf.buf, 0));
    if(cudaHostGetDevicePointer(&qp_buf, qp->buf.buf, 0) != cudaSuccess)
        exit(0);
  }
  else {
    qp_buf = qp->buf.buf;
  }
  // printf("Function name: %s, line number: %d\n", __func__, __LINE__);
  success = cudaHostRegister(qp->sq.wr_data, sizeof(qp->sq.wr_data), cudaHostRegisterMapped);
  if(success != cudaErrorHostMemoryAlreadyRegistered && success != cudaSuccess)
        exit(0);
  // get GPU pointer for qp->sq.wr_data
  if(cudaHostGetDevicePointer(&dev_qpsq_wr_data, qp->sq.wr_data, 0) != cudaSuccess)
      exit(0);
  success = cudaHostRegister(qp->sq.wqe_head, sizeof(qp->sq.wqe_head), cudaHostRegisterMapped);
  if(success != cudaErrorHostMemoryAlreadyRegistered && success !=  cudaSuccess)
        exit(0);
  // get GPU pointer for qp->sq.wqe_head
  if(cudaHostGetDevicePointer(&dev_qpsq_wqe_head, qp->sq.wqe_head, 0) != cudaSuccess)
      exit(0);

  // printf("Function name: %s, line number: %d\n", __func__, __LINE__);
  success = cudaHostRegister(&qp->sq, sizeof(qp->sq), cudaHostRegisterMapped);
  if(success != cudaErrorHostMemoryAlreadyRegistered && success !=  cudaSuccess)
        exit(0);
  // get GPU pointer for qp->sq
  if(cudaHostGetDevicePointer(&dev_qp_sq, &qp->sq, 0) != cudaSuccess)
      exit(0);

  void *nvidia_buf;
  void *dev_bf;
  if(posix_memalign(&nvidia_buf, 64*1024, 32*4096)) {
		printf("Failed to allocated memory for DB register\n");
			return 1;
		}
	// memset(nvidia_buf, 0, 32*4096);
  // bf->uar = nvidia_buf; 

  cudaError_t cudaState = cudaHostRegister(nvidia_buf, 32*4096, cudaHostRegisterMapped);
  if(cudaState != cudaSuccess && cudaState != cudaErrorHostMemoryAlreadyRegistered)
        exit(0);
  // printf("Function name: %s, line number: %d\n", __func__, __LINE__);
  // get GPU pointer for qp->db
  if(cudaHostGetDevicePointer(&dev_bf, nvidia_buf, 0) != cudaSuccess)
      exit(0);

  cudaState = cudaHostRegister(qp->db, sizeof(qp->db), cudaHostRegisterMapped);

  if(cudaState != cudaSuccess && cudaState != cudaErrorHostMemoryAlreadyRegistered)
        exit(0);
  // printf("Function name: %s, line number: %d\n", __func__, __LINE__);
  // get GPU pointer for qp->db
  if(cudaHostGetDevicePointer(&dev_qp_db, qp->db, 0) != cudaSuccess)
      exit(0);
  cudaError_t cudaStatus = cudaHostRegister(&local_ret, sizeof(local_ret), cudaHostRegisterMapped);
  // printf("Function name: %s, line number: %d cudaHostRegister(&ret, sizeof(ret), cudaHostRegisterMapped): %d\n", 
          // __func__, __LINE__, cudaStatus);
  if(cudaStatus != cudaErrorHostMemoryAlreadyRegistered && cudaStatus !=  cudaSuccess)
        exit(0);
  // printf("Function name: %s, line number: %d\n", __func__, __LINE__);
  cudaStatus = cudaHostGetDevicePointer(&dev_ret, &local_ret, 0) ;
  // printf("Function name: %s, line number: %d cudaStatus: %d\n",
          // __func__, __LINE__, cudaStatus);
  // get GPU pointer for qp->db
  if(cudaStatus != cudaSuccess)
      exit(0);
  // printf("Function name: %s, line number: %d cudaHostRegister(wr->sg_list, sizeof(wr->sg_list), cudaHostRegisterMapped): %d\n",
          // __func__, __LINE__, 1, sizeof(wr->sg_list), cudaHostRegisterMapped));

  
  cudaStatus = cudaHostRegister(wr->sg_list, sizeof(wr->sg_list), cudaHostRegisterMapped);
  // printf("Function name: %s, line number: %d cudaSuccess: %d\n",
  //         __func__, __LINE__, cudaSuccess);
  
  cudaStatus = cudaHostRegister(wr->sg_list, sizeof(wr->sg_list), cudaHostRegisterMapped);
  // printf("Function name: %s, line number: %d cudaSuccess: %d\n",
  //         __func__, __LINE__, cudaSuccess);

  // comment the below lines when gpu buffer used for qp->buf.buf
  if(cudaStatus != cudaErrorHostMemoryAlreadyRegistered && cudaSuccess !=  cudaSuccess)
        exit(0);
  // get GPU pointer for qp->db
  if(cudaHostGetDevicePointer(&dev_wr_sg, wr->sg_list, 0) != cudaSuccess)
      exit(0);
  // printf("Function name: %s, line number: %d\n", __func__, __LINE__);


  printf("Function name: %s, line number: %d &qp->sq: 0x%llx\n", __func__, __LINE__, &qp->sq);
  cudaStatus = cudaHostRegister(qp->sq.wrid, sizeof(qp->sq.wrid), cudaHostRegisterMapped);
  if(cudaStatus != cudaErrorHostMemoryAlreadyRegistered && cudaStatus !=  cudaSuccess)
        exit(0);
  // get GPU pointer for qp->sq
  if(cudaHostGetDevicePointer(&dev_wrid, qp->sq.wrid, 0) != cudaSuccess)
      exit(0);
  // printf("Function name: %s, line number: %d\n", __func__, __LINE__);
  // printf("Function name: %s, line number: %d cudaHostGetDevicePointer(&dev_wrid, qp->sq.wrid, 0): %d\n", __func__, __LINE__, cudaHostGetDevicePointer(&dev_wrid, qp->sq.wrid, 0));
  // if(cudaHostRegister(qp->sq.wqe_head, sizeof(qp->sq.wqe_head), cudaHostRegisterMapped) !=  cudaSuccess)
  //       exit(0);
  // // get GPU pointer for qp->sq
  // if(cudaHostGetDevicePointer(&dev_wqe_head, qp->sq.wqe_head, 0) != cudaSuccess)
  //     exit(0);

  uint32_t first_dword, second_dword, *dev_first, *dev_second;
  cudaError_t cuda_success = cudaHostRegister(&first_dword, sizeof(first_dword), cudaHostRegisterMapped);
  if(cuda_success != cudaErrorHostMemoryAlreadyRegistered &&  cuda_success !=  cudaSuccess)
        exit(0);
  // get GPU pointer for qp->sq
  if(cudaHostGetDevicePointer(&dev_first, &first_dword, 0) != cudaSuccess)
      exit(0);
  // printf("Function name: %s, line number: %d\n", __func__, __LINE__);

  cuda_success = cudaHostRegister(&second_dword, sizeof(second_dword), cudaHostRegisterMapped);
  // printf("Function name: %s, line number: %d cuda_success: %d\n", __func__, __LINE__, cuda_success);
  if(cuda_success != cudaSuccess && cuda_success != cudaErrorHostMemoryAlreadyRegistered)
        exit(0);
  // get GPU pointer for qp->sq
  
  if(cudaHostGetDevicePointer(&dev_second, &second_dword, 0) != cudaSuccess)
      exit(0);
  // printf("Function name: %s, line number: %d\n", __func__, __LINE__);

  clock_t *dtimer = NULL;
	clock_t timer[2];

	if (cudaSuccess != cudaMalloc((void **)&dtimer, sizeof(clock_t) * 2)) 
		exit(0);

  if (cudaDeviceSynchronize() != 0) exit(0);
  global_gpu_post_write<<<1,1>>>(
          qpbf_bufsize, /*qpsq_cur_post, qpsq_wqe_cnt,*/
          wr_rdma_remote_addr, wr_rdma_rkey,
          wr_sg_length, wr_sg_lkey, wr_sg_addr,
           wr_opcode, 
          bf_offset, qp_num, wr_id, /*qp->buf.buf*/ qp_buf, /*dev_qpsq_wr_data,*/ 
          dev_qpsq_wqe_head, dev_qp_sq, bf_reg, dev_qp_db, /*dev_wr_sg,*/ dev_wrid, 
          dev_first, dev_second, dtimer, dev_ret);

  // printf("Function name: %s, line number: %d cudaDeviceSynchronize(): %d\n", __func__, __LINE__, cudaDeviceSynchronize());
  if (cudaDeviceSynchronize() != 0) exit(0);

  cudaMemcpy(timer, dtimer, sizeof(clock_t) * (2), cudaMemcpyDeviceToHost);
  cudaFree(dtimer);

  cudaDeviceProp devProp;
  cudaGetDeviceProperties(&devProp, 0);
  // printf("Cuda device clock rate = %d\n", devProp.clockRate);

  float freq = (float)1/((float)devProp.clockRate*1000);
	float g_usec = (float)((float)1/(devProp.clockRate*1000))*((timer[1]-timer[0])) * 1000000;
	printf("Write POST - INTERNAL MEASUREMENT: %f useconds to execute \n", g_usec);

  void *addr;

    /* Do 64 bytes at a time */
    addr = bf->reg + bf->offset;

    *(volatile uint32_t *)addr = (uint32_t)( first_dword);
    *(volatile uint32_t *)(addr + 4) = (uint32_t)(second_dword);

  return local_ret;

}


int cpu_poll_cq(struct ibv_cq *ibcq, int n, struct ibv_wc *wc) 
{
    struct mlx5_cq *cq = to_mcq(ibcq);
	// ((struct mlx5_cq *)(ibcq - offsetof(struct mlx5_cq, verbs_cq.cq)));
    int npolled=0;
	int err = CQ_OK;
	struct mlx5_resource *rsc = NULL;
	struct mlx5_srq *srq = NULL;
	// printf("cq stall enable: %d\n", cq->stall_enable);
	void *cqe;
	struct mlx5_cqe64 *cqe64;
	int cqe_ver = 1;
	struct mlx5_wq *wq;
	uint16_t wqe_ctr;
	uint32_t qpn;
	uint32_t srqn_uidx;
	int idx;
	uint8_t opcode;
	struct mlx5_err_cqe *ecqe;
	struct mlx5_sigerr_cqe *sigerr_cqe;
	struct mlx5_mkey *mkey;
	struct mlx5_qp *mqp;
	struct mlx5_context *mctx;
	uint8_t is_srq;
  npolled = 1;
    // for (npolled = 0 ; npolled < n; ++npolled) {
	
		
		
		cqe = cq->active_buf->buf + (cq->cons_index & cq->verbs_cq.cq.cqe) * cq->cqe_sz;
		cqe64 = (struct mlx5_cqe64 *) ((cq->cqe_sz == 64) ? cqe : cqe + 64);
    // printf("cqe64->op_own >> 4: %d\n", cqe64->op_own >> 4);
		if ((cqe64->op_own >> 4 != MLX5_CQE_INVALID) &&
			!((cqe64->op_own & MLX5_CQE_OWNER_MASK) ^ !!(1 & (cq->verbs_cq.cq.cqe + 1)))) {
			++cq->cons_index;
		} else {
			err = CQ_EMPTY;
			// break;
      goto out;
		}
		is_srq = 0;
		err = 0; 
		//((struct mlx5_context *)(cq->verbs_cq.cq.context - offsetof(struct mlx5_context, ibv_ctx.context)));//
		mctx = to_mctx(cq->verbs_cq.cq.context);
		qpn = be32toh(cqe64->sop_drop_qpn) & 0xffffff;
			(wc)->wc_flags = 0;
			(wc)->qp_num = qpn;
		opcode = cqe64->op_own >> 4;
    // printf("cqe64->op_own >> 4: %d\n", cqe64->op_own >> 4);
		switch (opcode) {
		case MLX5_CQE_REQ:
		{
			uint32_t rsn = (cqe_ver ? (be32toh(cqe64->srqn_uidx) & 0xffffff) : qpn);
			if (!rsc || (rsn != rsc->rsn)){
				if(cqe_ver) {
					int tind = rsn >> MLX5_UIDX_TABLE_SHIFT;
					// printf("Function: %s line number: %d rsn & MLX5_UIDX_TABLE_MASK: %d tind: %d\n",__func__, __LINE__, rsn & MLX5_UIDX_TABLE_MASK, tind);
          //           printf("mctx->uidx_table[tind=%d].refcnt: %d\n", tind, mctx->uidx_table[tind].refcnt);
					if (likely(mctx->uidx_table[tind].refcnt))
						rsc = mctx->uidx_table[tind].table[rsn & MLX5_UIDX_TABLE_MASK];
					else rsc = NULL;
				}
			}
			mqp = (struct mlx5_qp *) rsc; 
			if (unlikely(!mqp)){
				err = CQ_POLL_ERR;
				break;
			}
			wq = &mqp->sq;
			wqe_ctr = htons(cqe64->wqe_counter);
			idx = wqe_ctr & (wq->wqe_cnt - 1);
            if(htonl(cqe64->sop_drop_qpn) >> 24 == MLX5_OPCODE_RDMA_WRITE){
				wc->opcode    = IBV_WC_RDMA_WRITE;
            }
            else {
                wc->opcode    = IBV_WC_RDMA_READ;
                wc->byte_len  = htonl(cqe64->byte_cnt);
            }
				
				wc->wr_id = wq->wrid[idx];
				wc->status = (enum ibv_wc_status) err;
			wq->tail = wq->wqe_head[idx] + 1;

      break;
		}
		case MLX5_CQE_RESP_SEND:
		{
			srqn_uidx = be32toh(cqe64->srqn_uidx) & 0xffffff;
			struct mlx5_qp *mqp;

			if (!rsc || (srqn_uidx != rsc->rsn)) {
				int tind = srqn_uidx >> MLX5_UIDX_TABLE_SHIFT;
				if ((mctx->uidx_table[tind].refcnt))
					rsc = mctx->uidx_table[tind].table[srqn_uidx & MLX5_UIDX_TABLE_MASK];
				if ((!rsc)){
					err = CQ_POLL_ERR;
					break;
				}
			}
			
			// mqp = (struct mlx5_qp *) rsc;
			// // printf("Function: %s line number: %d mqp->verbs_qp.qp.srq: %d\n",__func__, __LINE__, mqp->verbs_qp.qp.srq);
			// if (mqp->verbs_qp.qp.srq) {
			// 	// printf("Function: %s line number: %d \n",__func__, __LINE__);
			// 	srq = to_msrq(mqp->verbs_qp.qp.srq);
			// 	is_srq = 1;
			// }
			// err = CQ_OK;
			// printf("Function: %s line number: %d \n",__func__, __LINE__);
			
			uint16_t	wqe_ctr;
			struct mlx5_wq *wq;
			struct mlx5_qp *qp = rsc_to_mqp(rsc);
			uint8_t g;
			int err = 0;

			wc->byte_len = be32toh(cqe64->byte_cnt);
			if ((rsc->type == MLX5_RSC_TYPE_QP)) {
				wq = &qp->rq;
			} 
			
			wqe_ctr = wq->tail & (wq->wqe_cnt - 1);
			wc->wr_id = wq->wrid[wqe_ctr];
			++wq->tail;
				
			if (err){
				(wc)->status = (enum ibv_wc_status) err;
				break;
			}
			wc->opcode   = IBV_WC_RECV;
			(wc+npolled)->status = IBV_WC_SUCCESS;
			
			break;	
		}
		}
		if (err != CQ_OK){
			// break;
      goto out;
		}
		
	// }

out:
	/* Update cons index */
	cq->dbrec[0] = htonl(cq->cons_index & 0xffffff);
    return err; //  == CQ_POLL_ERR ? err : 0;
}

__device__ void device_process_gpu_mr(int *addr, int size){
  printf("\n\nGpu memory read: %d, %d\n\n", addr[2], addr[size-1]);
}

__global__ void global_process_gpu_mr(int *addr, int size){
  
  device_process_gpu_mr(addr, size);
}

void process_gpu_mr(int *addr, int size){
  cudaError_t success = cudaDeviceSynchronize();
  if(success != 0) exit(-1);

  global_process_gpu_mr<<<1, 1>>> (addr, size);

  success = cudaDeviceSynchronize();
  if(success != 0) exit(-1);
}

int mlx5_cq_event(struct ibv_comp_channel *channel, struct ibv_cq **cq)
{
  struct ib_uverbs_comp_event_desc ev;

  if (read(channel->fd, &ev, sizeof(ev)) != sizeof(ev))
		return -1;

	// if (read(channel->fd, &ev, sizeof ev) != sizeof ev)
	// 	return -1;

	*cq         = (struct ibv_cq *) (uintptr_t) ev.cq_handle;

	to_mcq(*cq)->arm_sn++;
  return 0;
}

__global__ void global_gpu_rdma_read(){

}
void gpu_rdma_read(){}





__device__ int read_write_post(
          uint64_t wr_rdma_remote_addr, uint32_t wr_rdma_rkey,
          uint32_t wr_sg_length, uint32_t wr_sg_lkey, uint64_t wr_sg_addr, int wr_opcode, 
          uint32_t qp_num, uint64_t wr_id, 
          void *qp_buf, void *dev_qpsq_wqe_head, 
          void *dev_qp_sq, void *dev_qp_db, void *dev_wrid,
          void *bf_reg)
{
  
	void *seg;
	struct mlx5_wqe_ctrl_seg *ctrl = NULL;
	// struct mlx5_wqe_data_seg *dpseg;
	// struct mlx5_sg_copy_ptr sg_copy_ptr = {.index = 0, .offset = 0};
	// int nreq = 0;
	// int inl = 0;
	// int err = 0;
	int size = 0;
	// int i;
	unsigned idx;
	uint8_t opmod = 0;

	uint32_t mlx5_opcode;
	
    unsigned int *qpsq_wqe_head = (unsigned int *) dev_qpsq_wqe_head;// qp->sq.wqe_head;
    struct mlx5_wq *qp_sq = (struct mlx5_wq *) dev_qp_sq; // &qp->sq;

    unsigned int *qp_db = (unsigned int *) dev_qp_db; // qp->db

    uint64_t * qpsq_wrid = (uint64_t *) dev_wrid;// qp_sq->wrid[idx] = wr_id;
    idx = qp_sq->cur_post & (qp_sq->wqe_cnt - 1);
    seg = (qp_buf + 256 + (idx * 64)); // mlx5_get_send_wqe(qp, idx);
    ctrl = (struct mlx5_wqe_ctrl_seg *) seg;
    *(uint32_t *)(seg + 8) = 0;
    ctrl->imm = 0; // send_ieth(wr);
    ctrl->fm_ce_se = 8; // MLX5_WQE_CTRL_CQ_UPDATE;

    seg = seg + 16; // sizeof(*ctrl);
    size = 1;
    // printf("sizeof(*ctrl): %d",sizeof(*ctrl) );
    // qpsq_wr_data[idx] = 0; // qp_sq->wr_data[idx] = 0;
  
    if(wr_opcode == 4 || wr_opcode == 0){
      ((struct mlx5_wqe_raddr_seg *) seg)->raddr    = htonl64(wr_rdma_remote_addr);
      ((struct mlx5_wqe_raddr_seg *) seg)->rkey     = htonl(wr_rdma_rkey);
      ((struct mlx5_wqe_raddr_seg *) seg)->reserved = 0;
      seg = seg + 16; // sizeof(struct mlx5_wqe_raddr_seg);
      size += 1; // sizeof(struct mlx5_wqe_raddr_seg) / 16;
    }

    // dpseg = (struct mlx5_wqe_data_seg *) seg;
   
    ((struct mlx5_wqe_data_seg *) seg)->byte_count = htonl(wr_sg_length); // htonl(wr_sg_list->length);
    ((struct mlx5_wqe_data_seg *) seg)->lkey       = htonl(wr_sg_lkey); // htonl(wr_sg_list->lkey);
    ((struct mlx5_wqe_data_seg *) seg)->addr       = htonl64(wr_sg_addr); // htonl64(wr_sg_list->addr);
 
    size += 1; // sizeof(struct mlx5_wqe_data_seg) / 16;
    // read: 4 -> 16, write: 0 -> 8, send: 2 -> 10
    mlx5_opcode = wr_opcode*2 + 8 - 2*(wr_opcode == 2); // mlx5_ib_opcode[wr->opcode];
    ctrl->opmod_idx_opcode = htonl(((qp_sq->cur_post & 0xffff) << 8) | mlx5_opcode);
    ctrl->qpn_ds = htonl(size | (qp_num << 8));
    
    qpsq_wrid[idx] = wr_id;
    qpsq_wqe_head[idx] = qp_sq->head;
    int tmp = (size * 16 + 64 - 1) / 64;
    qp_sq->cur_post += tmp;
    // printf("Function name: %s, line number: %d\n", __func__, __LINE__); 
    // db update:
    // qp_sq->head += 1;
    qp_db[MLX5_SND_DBR] = htonl(tmp & 0xffff);

    void *addr = bf_reg; // + 0xa00;
    uint64_t val;
    /* Do 64 bytes at a time */
    // addr = bf_reg; // + bf_offset;
    val = *(uint64_t *) ctrl;

    *(volatile uint32_t *)addr = htonl(htonl64(val) >> 32);
    *(volatile uint32_t *)(addr+4) = htonl(htonl64(val));
    // // bf->offset ^= bf->buf_size;
    
	return 0;
}



__device__ int poll_read_write( void *cq_buf, void *twc, uint32_t *cons_index,
                            int ibv_cqe, uint32_t cqe_sz, int n, void *cq_dbrec,
                            void *dev_rsc, 
                            // void *dev_wrid,
							              uint64_t wrid_0, unsigned int wqe_head_0,
                            void *dev_wq) 
{
   
  uint32_t *gpu_dbrec = (uint32_t *) cq_dbrec;
  // timer[1] = clock();
  struct ibv_wc *wc = (struct ibv_wc *)twc;
  int npolled=0;
	int err = CQ_OK;
	struct mlx5_resource *rsc = NULL;
	struct mlx5_srq *srq = NULL;
	void *cqe;
	struct mlx5_cqe64 *cqe64;
	int cqe_ver = 1;
	struct mlx5_wq *wq;
	uint16_t wqe_ctr;
	uint32_t qpn;
	;
	int idx;
	uint8_t opcode;
	struct mlx5_err_cqe *ecqe;
	struct mlx5_sigerr_cqe *sigerr_cqe;

	struct mlx5_qp *mqp;

	uint8_t is_srq;
	uint32_t cons_index_dev = *cons_index;

		cqe = cq_buf + (cons_index_dev & ibv_cqe) * cqe_sz;
   
		cqe64 = (struct mlx5_cqe64 *)((cqe_sz == 64) ? cqe : cqe + 64);
    
		int cond1 = (cqe64->op_own >> 4 != MLX5_CQE_INVALID) &&
			!((cqe64->op_own & MLX5_CQE_OWNER_MASK) ^ !!(1 & (ibv_cqe + 1)));
    
    printf("cqe64->op_own: %d\n",cqe64->op_own);
		if (cond1) {
            (*cons_index)++;
		} else {
			err = CQ_EMPTY;
			goto out1;
      // return err;
		}
		is_srq = 0;
		err = 0; 
		qpn = htonl(cqe64->sop_drop_qpn) & 0xffffff;
        wc->wc_flags = 0;
        wc->qp_num = qpn;
		opcode = cqe64->op_own >> 4;
	
			mqp = (struct mlx5_qp *) dev_rsc;
			if ((!mqp)){
				err = CQ_POLL_ERR;
			  goto out1;
			}
			wq = (struct mlx5_wq *) dev_wq; // &mqp->sq;
			wqe_ctr = htons (cqe64->wqe_counter);
			idx = wqe_ctr & (wq->wqe_cnt - 1);
     
      if(htonl(cqe64->sop_drop_qpn) >> 24 == MLX5_OPCODE_RDMA_WRITE){
				wc->opcode    = IBV_WC_RDMA_WRITE;
      }
      else {
        wc->opcode    = IBV_WC_RDMA_READ;
        wc->byte_len  = htonl(cqe64->byte_cnt);
      }
      wc->wr_id = wrid_0; // wq->wrid[idx];
      wc->status = (ibv_wc_status) err;
			wq->tail = wqe_head_0 + 1; // wq->wqe_head[idx] + 1;
      // printf("wqe_head_0: %d\n", wqe_head_0);

out1:
  gpu_dbrec[0] = htonl(*cons_index & 0xffffff);
  return err; // == CQ_POLL_ERR ? err : 1;
}


__global__ void gpu_benchmark(uint32_t mesg_size, uint64_t wr_id, uint32_t peer_rkey, uintptr_t peer_addr, uintptr_t local_address, uint32_t lkey,
              /* post*/       uint32_t qpn,
              /* post*/       void *dev_qpsq_wqe_head, void *qp_buf,void *dev_qp_sq, void *dev_qp_db,
              /* post*/       void *qp_sq_wrid, void *bf_reg,
              /*poll*/        int ibv_cqe, uint32_t cqe_sz, int n, uint64_t wrid_0, unsigned int wqe_head_0,
                              void *dev_wq, 
              /*poll*/        void *cq_buf, uint32_t *cons_index, void *cq_dbrec, void *dev_rsc, 
                              void *dev_wrid, clock_t *dtimer, int result){

  int ret;
  printf("benchmark started\n");
  struct ibv_send_wr wr, *bad_wr = NULL;
  struct ibv_sge sge;
  int offset = 19; // any positive number > 2 

  struct ibv_wc wc1;
  printf("benchmark started\n");
  struct connection *conn;
  conn = (struct connection *)(uintptr_t) wr_id;
  bad_wr = NULL;
  memset(&wr, 0, sizeof(wr));
  wr.wr_id = (uintptr_t)conn;
  wr.opcode = (ibv_wr_opcode) 4; // IBV_WR_RDMA_READ;
  wr.sg_list = &sge;
  wr.num_sge = 1;
  wr.send_flags = 2; // IBV_SEND_SIGNALED;
  wr.wr.rdma.remote_addr = (uintptr_t)peer_addr; // conn->peer_mr.addr;
  wr.wr.rdma.rkey = peer_rkey; // conn->peer_mr.rkey;
  sge.addr = (uintptr_t) local_address; // (uintptr_t)conn->rdma_local_region;
  sge.length = (uint32_t) mesg_size*sizeof(int)- 4; //RDMA_BUFFER_SIZE;
  sge.lkey = lkey; // conn->rdma_local_mr->lkey;
  printf("Posting started\n");
  dtimer[0] = clock64();
  ret = read_write_post(wr.wr.rdma.remote_addr, wr.wr.rdma.rkey,
        sge.length, sge.lkey, sge.addr, wr.opcode, 
        qpn, wr_id, qp_buf, /*dev_qpsq_wr_data,*/ 
        dev_qpsq_wqe_head, dev_qp_sq, dev_qp_db, /*dev_wr_sg,*/ qp_sq_wrid, bf_reg);
  dtimer[1] = clock64();

  printf("Polling started\n");
  dtimer[2] = clock64();
  while (poll_read_write(cq_buf /* the CQ, we got notification for */, 
            &wc1/* where to store */,
            (uint32_t *) cons_index,
            ibv_cqe,
            cqe_sz,
            1, // max_wc /* number of remaining WC elements*/,
            (uint32_t *) cq_dbrec,
            // (struct mlx5_context *) mctx,
            dev_rsc,
            wrid_0, wqe_head_0,
            dev_wq) < 0);
    // do {
        
    //     ret = poll_read_write(cq_buf /* the CQ, we got notification for */, 
    //         &wc1/* where to store */,
    //         (uint32_t *) cons_index,
    //         ibv_cqe,
    //         cqe_sz,
    //         1, // max_wc /* number of remaining WC elements*/,
    //         (uint32_t *) cq_dbrec,
    //         // (struct mlx5_context *) mctx,
    //         dev_rsc,
    //         wrid_0, wqe_head_0,
    //         dev_wq);
    //     // printf("polling\n");
    // } while (ret < 0); 
    dtimer[3] = clock64();
  
  if (wc1.status != IBV_WC_SUCCESS){
    printf("wc->status: %d\n", wc1.status);
    // return;
  }
    // device_process_gpu_mr((char *)local_address);
    // printf("\n\nGpu memory read: %s\n\n", (char *)local_address);
  // calculcate sum and multiply by 2:
  printf("wc->status: %d\n", wc1.status);
  long int sum = 0;
  int *array = (int *) local_address;
  // printf("delay\n");
  if(array[mesg_size-9] == 2)
    printf("Data Retrieved!\n");
  else printf("Data not retrieved! :%d\n", array[mesg_size-9]);
  // printf("mesg_size: %d \n", mesg_size);
  // for (int i = 0; i < mesg_size; i++)
  //   printf("%d ", array[i]);
  printf("Computation started\n");
  for (int i = 0; i < mesg_size; i++){
    sum += array[i];
    if (i == 0) printf("index 0: %d\n", sum);
  }
  // printf("\n %d \n", sum);
  // result = result * 2;
  printf("End started\n");
  if(sum == result-6) {
    printf("equal - sum: %d\n", sum);
    printf("equal - result: %d\n", result);
  }
  else {
    printf("unequal - sum: %d\n", sum);
    printf("unequal - result: %d\n", result);
  }
  __syncthreads();
}

int start_polling_on_cpu(struct ibv_cq *ibcq, int n, struct ibv_wc *wc){
  struct mlx5_cq *cq = to_mcq(ibcq);
  struct mlx5_context *mctx = container_of(cq->verbs_cq.cq.context, struct mlx5_context, ibv_ctx.context);
  struct mlx5_resource *rsc = mctx->uidx_table[0].table[0];
  struct mlx5_qp *qp = (struct mlx5_qp *)(rsc);
  // printf("qp buf address: 0x%llx\n\n\n", qp->buf.buf);
  struct mlx5_context *qp_ctx = to_mctx(qp->ibv_qp->pd->context);
  struct mlx5_wq *qp_rq = &qp->rq;
  void *cqe = cq->buf_a.buf + (cq->cons_index & cq->verbs_cq.cq.cqe) * cq->cqe_sz;
  struct mlx5_wq * wq_sq = &qp->sq;
  int ret;
  printf("Function name: %s, line: %d\n", __func__, __LINE__);
  do {
    ret = /*ibv_poll_cq host_gpu*/cpu_poll_cq2(cq->buf_a.buf, 
      wc, &(cq->cons_index), ibcq->cqe, cq->cqe_sz, n, cq->dbrec,
      mctx, rsc, mctx->uidx_table[0].refcnt, qp_ctx, qp_ctx->dump_fill_mkey_be,
      &qp->rq, qp->rq.wrid, qp->rq.wrid[0], cqe, 1/*cond*/, NULL,
      NULL, qp->sq.wrid[0], qp->sq.wqe_head[0], wq_sq
      );
      printf("polling\n");
  } while (ret < 0);


}

int cpu_poll_cq2(void *cq_buf, void *twc, uint32_t *cons_index,
                 int ibv_cqe, uint32_t cqe_sz, int n, void *cq_dbrec,
                 void *mctx_t, void *dev_rsc, int refcnt,
                 void *qp_context, int dump_fill_mkey_be,
                 void *dev_rq, void *dev_wrid, uint64_t wrid_1,
                 void *cqe_dev, int cond, void *dev_scat_address,
                 clock_t *timer, uint64_t wrid_0, unsigned int wqe_head_0,
                 void *dev_wq) 
{
  printf("Function name: %s, line: %d\n", __func__, __LINE__);
	
  uint32_t *gpu_dbrec = (uint32_t *) cq_dbrec;
  // timer[1] = clock();
  struct ibv_wc *wc = (struct ibv_wc *)twc;
  int npolled=0;
	int err = CQ_OK;
	struct mlx5_resource *rsc = NULL;
	struct mlx5_srq *srq = NULL;
	void *cqe;
	struct mlx5_cqe64 *cqe64;
	int cqe_ver = 1;
	struct mlx5_wq *wq;
	uint16_t wqe_ctr;
	uint32_t qpn;
	uint32_t srqn_uidx;
	int idx;
	uint8_t opcode;
	struct mlx5_err_cqe *ecqe;
	struct mlx5_sigerr_cqe *sigerr_cqe;
	struct mlx5_mkey *mkey;
	struct mlx5_qp *mqp;
	struct mlx5_context *mctx;
	uint8_t is_srq;
	uint32_t cons_index_dev = *cons_index;
  printf("Function name: %s, line: %d\n", __func__, __LINE__);
  // for (npolled = 0 ; npolled < 1; ++npolled) {

    // timer[0] = clock();
    
    
		cqe = cq_buf + (cons_index_dev & ibv_cqe) * cqe_sz;
    
    // printf("Function: %s line number: %d cq_buf: 0x%llx\n",__func__, __LINE__, cq_buf);
    // cqe = cqe_dev;// cq_buf + ((*cons_index) & ibv_cqe) * cqe_sz;
		cqe64 = (struct mlx5_cqe64 *)((cqe_sz == 64) ? cqe : cqe + 64);
    // timer[1] = clock();
		
        // (cqe64->op_own >> 4 != MLX5_CQE_INVALID) &&
		// 	!((cqe64->op_own & MLX5_CQE_OWNER_MASK) ^ !!(n & (ibv_cqe + 1)))
    printf("Function name: %s, line: %d\n", __func__, __LINE__);
		int cond1 = (cqe64->op_own >> 4 != MLX5_CQE_INVALID) &&
			!((cqe64->op_own & MLX5_CQE_OWNER_MASK) ^ !!(1 & (ibv_cqe + 1)));
      printf("cqe64->op_own: %d\n", cqe64->op_own >> 4);
      printf("Function name: %s, line: %d\n", __func__, __LINE__);
		if (cond1) {
			// printf("Function: %s line number: %d\n",__func__, __LINE__);
            (*cons_index)++;
            printf("cq not empty: %d\n", cond1);

		} else {
      printf("cq empty: %d\n", cond1 );
			err = CQ_EMPTY;
			// break;
			goto out1;
		}
    printf("Function name: %s, line: %d\n", __func__, __LINE__);
		// printf("Function: %s line number: %d\n",__func__, __LINE__);
		is_srq = 0;
		err = 0; 
    
		// printf("Function: %s line number: %d\n",__func__, __LINE__);
        // mctx = (struct mlx5_context *) mctx_t;
        
		qpn = htonl(cqe64->sop_drop_qpn) & 0xffffff;
    // printf("Function: %s line number: %d\n",__func__, __LINE__);  
        wc->wc_flags = 0;
        wc->qp_num = qpn;
		opcode = cqe64->op_own >> 4;
		// printf("Function: %s line number: %d opcode: %d\n",__func__, __LINE__, opcode);
		if(opcode == MLX5_CQE_REQ)
		{
			// uint32_t rsn = cqe_ver ? (htonl(cqe64->srqn_uidx) & 0xffffff) : qpn;
      // printf("Function: %s line number: %d\n",__func__, __LINE__);
			// if (!rsc || (rsn != rsc->rsn)){
                // printf("Function: %s line number: %d\n",__func__, __LINE__);
				// if(cqe_ver) {
                    // printf("Function: %s line number: %d\n",__func__, __LINE__);
					// int tind = rsn >> MLX5_UIDX_TABLE_SHIFT;
                    // printf("Function: %s line number: %d rsn & MLX5_UIDX_TABLE_MASK: %d tind: %d\n",__func__, __LINE__, rsn & MLX5_UIDX_TABLE_MASK, tind);
                    // printf("mctx->uidx_table[tind].refcnt: %d\n", /*mctx->uidx_table[tind].*/refcnt);
					// if (/*(mctx->uidx_table[tind].*/refcnt){
          //               // printf("Function: %s line number: %d\n",__func__, __LINE__);
          //               rsc = (struct mlx5_resource *) dev_rsc;// mctx->uidx_table[tind].table[rsn & MLX5_UIDX_TABLE_MASK];
          //               // printf("Function: %s line number: %d\n",__func__, __LINE__);
          //           }
          //           else rsc = NULL;
			    // }
            // }
            // printf("Function: %s line number: %d\n",__func__, __LINE__);
			mqp = (struct mlx5_qp *) dev_rsc;
			if ((!mqp)){
                // printf("Function: %s line number: %d\n",__func__, __LINE__);
				err = CQ_POLL_ERR;
				// break;
			  goto out1;
			}
			wq = (struct mlx5_wq *) dev_wq; // &mqp->sq;
			wqe_ctr = htons (cqe64->wqe_counter);
			idx = wqe_ctr & (wq->wqe_cnt - 1);
     
      if(htonl(cqe64->sop_drop_qpn) >> 24 == MLX5_OPCODE_RDMA_WRITE){
				wc->opcode    = IBV_WC_RDMA_WRITE;
      }
      else {
        wc->opcode    = IBV_WC_RDMA_READ;
        wc->byte_len  = htonl(cqe64->byte_cnt);
      }
      // printf("wq->tail: %d\n", wq->tail);
      wc->wr_id = wrid_0; // wq->wrid[idx];
      wc->status = (ibv_wc_status) err;
			wq->tail = wqe_head_0 + 1; // wq->wqe_head[idx] + 1;
      // printf("wqe_head_0: %d\n", wqe_head_0);
      // printf("wq->tail: %d\n", wq->tail);
      // printf("wc->status: %d\n", wc->status);
    }
	// else // MLX5_CQE_RESP_SEND:
  //     {
  //       // printf("Function: %s line number: %d\n",__func__, __LINE__);
  //       uint16_t	wqe_ctr;
  //       struct mlx5_wq *wq;
  //       struct mlx5_qp *qp = (struct mlx5_qp *)(dev_rsc);

  //       wc->byte_len = htonl(cqe64->byte_cnt);
  //       wq = (struct mlx5_wq *) dev_rq; // &qp->rq;
                
  //       wqe_ctr = wq->tail & (wq->wqe_cnt - 1);
  //       wc->wr_id = wrid_1;// wq->wrid[wqe_ctr];
  //       ++wq->tail;
            
  //       int size = wc->byte_len;
        
  //       int copy;

  //       err = IBV_WC_SUCCESS;
  //       wc->opcode   = IBV_WC_RECV;
  //       wc->status = IBV_WC_SUCCESS;
  //     }
		
        	
        // if (err != CQ_OK){
		// 	break;
		// }
        // printf("Function: %s line number: %d\n",__func__, __LINE__);
//  }
	/* Update cons index */
	// cq->dbrec[0] = htonl(*cons_index & 0xffffff);
    // printf("Function: %s line number: %d\n",__func__, __LINE__);
out1:
  gpu_dbrec[0] = htonl((*cons_index) & 0xffffff);
    
  return err; // == CQ_POLL_ERR ? err : 1;
}


// void host_benchmark(){



//   uint64_t wr_id, uint32_t peer_rkey, uintptr_t peer_addr, uintptr_t local_address, uint32_t lkey,
//               /* post*/       uint32_t qpn,
//               /* post*/       void *dev_qpsq_wqe_head, void *qp_buf,void *dev_qp_sq, void *dev_qp_db,
//               /* post*/       void *qp_sq_wrid, void *bf_reg,
//               /*poll*/        int ibv_cqe, uint32_t cqe_sz, int n, uint64_t wrid_0, unsigned int wqe_head_0,
//                               void *dev_wq, 
//               /*poll*/        void *cq_buf, uint32_t *cons_index, void *cq_dbrec, void *dev_rsc, 
//                               void *dev_wrid
// }

int cpu_benchmark(struct ibv_cq *cq_ptr, int num_entries, struct ibv_wc *wc,
                   struct ibv_qp *ibqp, struct ibv_send_wr *wr,
                   struct ibv_send_wr **bad_wr){


  int local_ret;
  struct mlx5_qp *qp = to_mqp(ibqp);

  struct mlx5_bf *bf = qp->bf;
  // printf("Function name: %s, line number: %d, qp->buf.length: %d\n", __func__, __LINE__, qp->buf.length);
  // input variables:
  unsigned int qpbf_bufsize = qp->bf->buf_size; // 256
  unsigned int qpsq_cur_post = qp->sq.cur_post;
  unsigned int qpsq_wqe_cnt = qp->sq.wqe_cnt;
  uint64_t wr_rdma_remote_addr = wr->wr.rdma.remote_addr;
  uint32_t wr_rdma_rkey = wr->wr.rdma.rkey;
  uint32_t wr_sg_length = wr->sg_list->length;
  uint32_t wr_sg_lkey = wr->sg_list->lkey;
  uint64_t wr_sg_addr = wr->sg_list->addr;
  int wr_opcode = wr->opcode;
  uint32_t qp_num = ibqp->qp_num;
  unsigned int bf_offset = bf->offset; 
  uint64_t wr_id = wr->wr_id;
  // printf("Function name: %s, line number: %d\n", __func__, __LINE__);
  // pointers
  void *qp_buf;
  void *dev_qpsq_wr_data; //qp->sq.wr_data;
  void *dev_qpsq_wqe_head;// qp->sq.wqe_head;
  void *dev_qp_sq; // &qp->sq;
  int *bf_reg = (int *) bf->reg; // bf->reg;
  void *dev_qp_db; // qp->db
  void *dev_wr_sg; // wr->sg_list
  int *dev_ret; // &ret
  void *dev_bf_reg; 
  void *dev_wrid, *dev_wrid1; // qp->sq.wrid
  void *dev_wqe_head; // qp->sq.wqe_head
  
  // printf("Function name: %s, line number: %d cudaHostRegister(qp->buf.buf, sizeof(qp->buf.buf), cudaHostRegisterMapped): %d\n", 
  //         __func__, __LINE__, cudaHostRegister(qp->buf.buf, sizeof(qp->buf.buf), cudaHostRegisterMapped));
  cudaError_t success;
  // pin host memory and get device pointer:
  if (wq_buffer_gpu != 1){
    success = cudaHostRegister(qp->buf.buf, qp->buf.length, cudaHostRegisterMapped);
    // comment the below lines when gpu buffer used for qp->buf.buf
    if(success !=  cudaSuccess && success != cudaErrorHostMemoryAlreadyRegistered)
          exit(0);
    // get GPU pointer for qp_buf
    // printf("Function name: %s, line number: %d cudaHostGetDevicePointer(&qp_buf, qp->buf.buf, 0): %d\n", __func__, __LINE__,
    //         cudaHostGetDevicePointer(&qp_buf, qp->buf.buf, 0));
    if(cudaHostGetDevicePointer(&qp_buf, qp->buf.buf, 0) != cudaSuccess)
        exit(0);
  }
  else {
    qp_buf = qp->buf.buf;
  }
  // printf("Function name: %s, line number: %d\n", __func__, __LINE__);
  success = cudaHostRegister(qp->sq.wr_data, sizeof(qp->sq.wr_data), cudaHostRegisterMapped);
  if(success != cudaErrorHostMemoryAlreadyRegistered && success != cudaSuccess)
        exit(0);
  // get GPU pointer for qp->sq.wr_data
  if(cudaHostGetDevicePointer(&dev_qpsq_wr_data, qp->sq.wr_data, 0) != cudaSuccess)
      exit(0);
  success = cudaHostRegister(qp->sq.wqe_head, sizeof(qp->sq.wqe_head), cudaHostRegisterMapped);
  if(success != cudaErrorHostMemoryAlreadyRegistered && success !=  cudaSuccess)
        exit(0);
  // get GPU pointer for qp->sq.wqe_head
  if(cudaHostGetDevicePointer(&dev_qpsq_wqe_head, qp->sq.wqe_head, 0) != cudaSuccess)
      exit(0);

  // printf("Function name: %s, line number: %d\n", __func__, __LINE__);
  success = cudaHostRegister(&qp->sq, sizeof(qp->sq), cudaHostRegisterMapped);
  if(success != cudaErrorHostMemoryAlreadyRegistered && success !=  cudaSuccess)
        exit(0);
  // get GPU pointer for qp->sq
  if(cudaHostGetDevicePointer(&dev_qp_sq, &qp->sq, 0) != cudaSuccess)
      exit(0);

  printf("Function name: %s, line number: %d bf->uar: 0x%llx\n", __func__, __LINE__, bf->uar);
  cudaError_t cudaStatus1;

  void *device_db;
  cudaStatus1 = cudaHostRegister(bf->reg,  8, cudaHostRegisterIoMemory);
    if (cudaStatus1 == cudaSuccess || cudaStatus1 == cudaErrorHostMemoryAlreadyRegistered) {
      printf("cudaHostRegister successful for address 0x%llx: %s and i: %d\n", bf->uar, cudaGetErrorString(cudaStatus1), bf->length);
      // exit(0);
    }
    else {
      printf("cudaHostRegister not success for address 0x%llx: %s and i: %d\n", bf->uar, cudaGetErrorString(cudaStatus1), bf->length);
      exit(0);
    }
    cudaStatus1 = cudaHostGetDevicePointer(&device_db, bf->reg, 0);
    if (cudaStatus1 == cudaSuccess) {
      printf("cudaHostGetDevicePointer successful with no error: %s\n", cudaGetErrorString(cudaStatus1));
      // exit(0);
    }
    else {
      printf("cudaHostGetDevicePointer failed with  error: %s\n", cudaGetErrorString(cudaStatus1));
      exit(0);
    }

  cudaError_t cudaState;

  cudaState = cudaHostRegister(qp->db, sizeof(qp->db), cudaHostRegisterMapped);

  if(cudaState != cudaSuccess && cudaState != cudaErrorHostMemoryAlreadyRegistered)
        exit(0);
  // printf("Function name: %s, line number: %d\n", __func__, __LINE__);
  // get GPU pointer for qp->db
  if(cudaHostGetDevicePointer(&dev_qp_db, qp->db, 0) != cudaSuccess)
      exit(0);
  cudaError_t cudaStatus = cudaHostRegister(&local_ret, sizeof(local_ret), cudaHostRegisterMapped);
  // printf("Function name: %s, line number: %d cudaHostRegister(&ret, sizeof(ret), cudaHostRegisterMapped): %d\n", 
          // __func__, __LINE__, cudaStatus);
  if(cudaStatus != cudaErrorHostMemoryAlreadyRegistered && cudaStatus !=  cudaSuccess)
        exit(0);
  // printf("Function name: %s, line number: %d\n", __func__, __LINE__);
  cudaStatus = cudaHostGetDevicePointer(&dev_ret, &local_ret, 0) ;
  // printf("Function name: %s, line number: %d cudaStatus: %d\n",
          // __func__, __LINE__, cudaStatus);
  // get GPU pointer for qp->db
  if(cudaStatus != cudaSuccess)
      exit(0);
  // printf("Function name: %s, line number: %d cudaHostRegister(wr->sg_list, sizeof(wr->sg_list), cudaHostRegisterMapped): %d\n",
          // __func__, __LINE__, 1, sizeof(wr->sg_list), cudaHostRegisterMapped));

  
  cudaStatus = cudaHostRegister(wr->sg_list, sizeof(wr->sg_list), cudaHostRegisterMapped);
  // printf("Function name: %s, line number: %d cudaSuccess: %d\n",
  //         __func__, __LINE__, cudaSuccess);
  
  cudaStatus = cudaHostRegister(wr->sg_list, sizeof(wr->sg_list), cudaHostRegisterMapped);
  if(cudaStatus != cudaErrorHostMemoryAlreadyRegistered && cudaSuccess !=  cudaSuccess)
        exit(0);
  // get GPU pointer for qp->db
  if(cudaHostGetDevicePointer(&dev_wr_sg, wr->sg_list, 0) != cudaSuccess)
      exit(0);
  // printf("Function name: %s, line number: %d\n", __func__, __LINE__);


  printf("Function name: %s, line number: %d &qp->sq: 0x%llx\n", __func__, __LINE__, &qp->sq);
  cudaStatus = cudaHostRegister(qp->sq.wrid, sizeof(qp->sq.wrid), cudaHostRegisterMapped);
  printf("Function name: %s, line number: %d cudaStatus: %d\n", __func__, __LINE__, cudaStatus);
  if(cudaStatus != cudaErrorHostMemoryAlreadyRegistered && cudaStatus != cudaSuccess)
        exit(0);
  // get GPU pointer for qp->sq
  if(cudaHostGetDevicePointer(&dev_wrid1, qp->sq.wrid, 0) != cudaSuccess)
      exit(0);
  uint32_t first_dword, second_dword, *dev_first, *dev_second;
  cudaError_t cuda_success = cudaHostRegister(&first_dword, sizeof(first_dword), cudaHostRegisterMapped);
  if(cuda_success != cudaErrorHostMemoryAlreadyRegistered &&  cuda_success !=  cudaSuccess)
        exit(0);
  // get GPU pointer for qp->sq
  if(cudaHostGetDevicePointer(&dev_first, &first_dword, 0) != cudaSuccess)
      exit(0);
  // printf("Function name: %s, line number: %d\n", __func__, __LINE__);
  printf("Function name: %s, line number: %d cudaStatus: %d\n", __func__, __LINE__, cudaStatus);
  cuda_success = cudaHostRegister(&second_dword, sizeof(second_dword), cudaHostRegisterMapped);
  // printf("Function name: %s, line number: %d cuda_success: %d\n", __func__, __LINE__, cuda_success);
  if(cuda_success != cudaSuccess && cuda_success != cudaErrorHostMemoryAlreadyRegistered)
        exit(0);
  // get GPU pointer for qp->sq
  printf("Function name: %s, line number: %d cudaStatus: %d\n", __func__, __LINE__, cudaStatus);
  if(cudaHostGetDevicePointer(&dev_second, &second_dword, 0) != cudaSuccess)
      exit(0);
  // printf("Function name: %s, line number: %d\n", __func__, __LINE__);
  printf("Function name: %s, line number: %d cudaStatus: %d\n", __func__, __LINE__, cudaStatus);
  



  struct mlx5_cq *cq = to_mcq(cq_ptr);
  void *cqe;
  // printf("Function: %s line number: %d cq->buf.length: %d\n",__func__, __LINE__, cq->buf_a.length);
  struct mlx5_cqe64 *cqe64;
  // int cond = 0;
  uint32_t cons_index = cq->cons_index;
  int cq_cqe = cq->verbs_cq.cq.cqe;
  int cq_cqe_sz = cq->cqe_sz;
  void *cq_buf_a = cq->buf_a.buf; 
  cqe = cq_buf_a ;//+ (cons_index & cq_cqe) * cq_cqe_sz;

  printf("cons_index: %d\n", cons_index);
  printf("cq_cqe: %d\n", cq_cqe);
  printf("cq_cqe_sz: %d\n", cq_cqe_sz);
  
 
  void *dev_cq_ptr; // cq pointer for GPU
  void *dev_wc; // wc pointer for GPU memory
  void *dev_cons_index; // 
  // void *dev_ret; // 
  void *dev_total_wc; // 
  void *dev_cq_dbrec;
  struct mlx5_context *mctx = container_of(cq->verbs_cq.cq.context, struct mlx5_context, ibv_ctx.context);
  // mctx->uidx_table[0].table
  void *dev_mctx;
  void **dev_table;
  void *dev_rsc;
  void *dev_qp_context;
  void *dev_qp_buf;
  int total_wc = 0, ret = -1;
  struct mlx5_resource *rsc = mctx->uidx_table[0].table[0];
  if(!rsc) printf("rsc is null\n\n\n");
  qp = (struct mlx5_qp *)(rsc);
  // printf("qp buf address: 0x%llx\n\n\n", qp->buf.buf);
  struct mlx5_context *qp_ctx = to_mctx(qp->ibv_qp->pd->context); 
  
  cudaError_t crc = cudaSuccess;
  // cudaError_t cudaStatus;
 
  if (cq_buffer_gpu != 1){
    cudaStatus = cudaHostRegister(cq->active_buf->buf /*cqbuf*/, cq->active_buf->length /*cqbuf_size*/, cudaHostRegisterMapped);
    if(cudaStatus != cudaErrorHostMemoryAlreadyRegistered && cudaStatus != cudaSuccess)
        exit(0);
    // get GPU pointer for cq
    if(cudaHostGetDevicePointer(&dev_cq_ptr, /*cqbuf*/ cq->active_buf->buf, 0) !=  cudaSuccess)
        exit(0);
  }
  else{
    dev_cq_ptr = cq->active_buf->buf;
  }
 
  cudaStatus = cudaHostRegister(wc, sizeof(wc), cudaHostRegisterMapped);
  if(cudaStatus != cudaErrorHostMemoryAlreadyRegistered && cudaStatus !=  cudaSuccess)
      exit(0);
  // get GPU pointer for wc
  if(cudaHostGetDevicePointer(&dev_wc, wc, 0) != cudaSuccess)
      exit(0);
   
  cudaStatus = cudaHostRegister(&cq->cons_index, sizeof(cq->cons_index), cudaHostRegisterMapped);
  if(cudaStatus != cudaSuccess && cudaStatus != cudaErrorHostMemoryAlreadyRegistered)
      exit(0);
  // get GPU pointer for cons index
  if(cudaHostGetDevicePointer(&dev_cons_index, &cq->cons_index, 0) != cudaSuccess)
      exit(0);
  // printf("Function: %s line number: %d\n",__func__, __LINE__);
  // register cons index in host memory 
  cudaStatus = cudaHostRegister(&total_wc, sizeof(total_wc), cudaHostRegisterMapped);
  if(cudaStatus != cudaErrorHostMemoryAlreadyRegistered && cudaStatus !=  cudaSuccess)
      exit(0);
  // get GPU pointer for cons index
  if(cudaHostGetDevicePointer(&dev_total_wc, &total_wc, 0) != cudaSuccess)
      exit(0);
  // printf("Function: %s line number: %d\n",__func__, __LINE__);
  // register ret in host memory 
  cudaStatus = cudaHostRegister(&ret, sizeof(ret), cudaHostRegisterMapped);
  if(cudaStatus != cudaErrorHostMemoryAlreadyRegistered && cudaStatus !=  cudaSuccess)
      exit(0);
  // get GPU pointer for ret
  if(cudaHostGetDevicePointer(&dev_ret, &ret, 0) != cudaSuccess)
      exit(0);
      // printf("Function name: %s, line number: %d cqe64->op_own >> 4: %d\n", __func__, __LINE__, cqe64->op_own >> 4);
  // printf("Function: %s line number: %d\n",__func__, __LINE__);
  cudaStatus = cudaHostRegister(cq->dbrec, sizeof(cq->dbrec), cudaHostRegisterMapped);
  if(cudaStatus != cudaErrorHostMemoryAlreadyRegistered && cudaStatus !=  cudaSuccess) 
      exit(0);
  if(cudaHostGetDevicePointer(&dev_cq_dbrec, cq->dbrec, 0) != cudaSuccess)
      exit(0);
  // printf("Function: %s line number: %d\n",__func__, __LINE__);
  cudaStatus = cudaHostRegister(mctx, sizeof(mctx), cudaHostRegisterMapped);
  if(cudaStatus != cudaErrorHostMemoryAlreadyRegistered && cudaStatus !=  cudaSuccess) 
      exit(0);
      // printf("Function name: %s, line number: %d cqe64->op_own >> 4: %d\n", __func__, __LINE__, cqe64->op_own >> 4);
  if(cudaHostGetDevicePointer(&dev_mctx, mctx, 0) != cudaSuccess)
      exit(0);
  // printf("Function: %s line number: %d\n",__func__, __LINE__);
  cudaStatus = cudaHostRegister(rsc, sizeof(rsc), cudaHostRegisterMapped);
  if(cudaStatus != cudaErrorHostMemoryAlreadyRegistered && cudaStatus !=  cudaSuccess) 
      exit(0);
  if(cudaHostGetDevicePointer(&dev_rsc, rsc, 0) != cudaSuccess)
      exit(0);

  void *dev_rq;
  // printf("cudaHostRegister(&qp->rq, sizeof(qp->rq), cudaHostRegisterMapped): %d\n", cudaHostRegister(&qp->rq, sizeof(qp->rq), cudaHostRegisterMapped));
  cudaStatus = cudaHostRegister(&qp->rq, sizeof(qp->rq), cudaHostRegisterMapped);
  if(cudaStatus != cudaErrorHostMemoryAlreadyRegistered && cudaStatus !=  cudaSuccess) 
      exit(0);
  if(cudaHostGetDevicePointer(&dev_rq, &qp->rq, 0) != cudaSuccess)
      exit(0);
      // printf("Function name: %s, line number: %d cqe64->op_own >> 4: %d\n", __func__, __LINE__, cqe64->op_own >> 4);
  struct mlx5_wq *qp_rq = &qp->rq;
 
  struct mlx5_wq * wq_sq = &qp->sq;
  void *dev_wq;
 
  cudaState = cudaHostRegister(wq_sq, sizeof(qp->sq), cudaHostRegisterMapped);
  if(cudaState !=  cudaSuccess && cudaState != cudaErrorHostMemoryAlreadyRegistered) 
      exit(0);
  if(cudaHostGetDevicePointer(&dev_wq, wq_sq, 0) != cudaSuccess)
      exit(0);
      // printf("Function name: %s, line number: %d cqe64->op_own >> 4: %d\n", __func__, __LINE__, cqe64->op_own >> 4);

  cudaState = cudaHostRegister(qp_rq->wrid, sizeof(qp_rq->wrid), cudaHostRegisterMapped);
  // printf("Function: %s line number: %d cudaState: %d\n",__func__, __LINE__, cudaState);
  // comment the below lines when gpu buffer used for qp->buf.buf
  printf("Function: %s line number: %d\n",__func__, __LINE__);
  if(cudaState != cudaSuccess && cudaState != cudaErrorHostMemoryAlreadyRegistered) 
      exit(0);
  // printf("Function: %s line number: %d\n",__func__, __LINE__);
  if(cudaHostGetDevicePointer(&dev_wrid, qp_rq->wrid, 0) != cudaSuccess)
      exit(0);
  
  printf("Function: %s line number: %d\n",__func__, __LINE__);
	cudaStatus = cudaHostGetDevicePointer(&dev_qp_context, qp_ctx, 0);
	// fprintf(stderr, "cudaHostRegister failed with error code %d: %s\n", cudaStatus, cudaGetErrorString(cudaStatus));
  if(cudaHostGetDevicePointer(&dev_qp_context, qp_ctx, 0) != cudaSuccess)
      exit(0);
  // printf("Function: %s line number: %d\n",__func__, __LINE__);
  // printf("Function name: %s, line number: %d cqe64->op_own >> 4: %d\n", __func__, __LINE__, cqe64->op_own >> 4);
  void *cqe_dev;
  printf("Function: %s line number: %d\n",__func__, __LINE__);

  if (cq_buffer_gpu != 1){
    cqe = cq->active_buf->buf + (cq->cons_index & cq->verbs_cq.cq.cqe) * cq->cqe_sz;
    // cqe = (struct mlx5_cqe64 *)((cq->cqe_sz == 64) ? cqe : cqe + 64);
    cudaStatus = cudaHostRegister(cqe, sizeof(cqe), cudaHostRegisterMapped);
    if(cudaStatus != cudaErrorHostMemoryAlreadyRegistered && cudaStatus !=  cudaSuccess) 
        exit(0);
    // printf("Function: %s line number: %d cudaHostGetDevicePointer(&cqe_dev, cqe, 0): %d\n",__func__, __LINE__,
          // cudaHostGetDevicePointer(&cqe_dev, cqe, 0));
    if(cudaHostGetDevicePointer(&cqe_dev, cqe, 0) != cudaSuccess)
        exit(0);
  }
  else {
    cqe_dev = cq->active_buf->buf;
  }
 

  uint16_t	wqe_ctr;
  struct mlx5_wq *wq = (struct mlx5_wq *) &qp->rq;
  wqe_ctr = wq->tail & (wq->wqe_cnt - 1);
  // struct mlx5_wqe_data_seg *scat = (struct mlx5_wqe_data_seg *)(qp->buf.buf + qp->rq.offset + (wqe_ctr << qp->rq.wqe_shift));
  // void *scat_address = (void *)(unsigned long)htonl64(scat->addr);
  void *dev_scat_address = NULL; 



	int cuda_device_id = 0;
	int cuda_pci_bus_id;
	int cuda_pci_device_id;
	int index;
	CUdevice cu_device;
	CUdevice cuDevice_selected;

	// printf("initializing CUDA\n");
	CUresult error = cuInit(0);
	if (error != CUDA_SUCCESS) {
		// printf("cuInit(0) returned %d\n", error);
		return -1;
	}
  // printf("Function name: %s, line number: %d cqe64->op_own >> 4: %d\n", __func__, __LINE__, cqe64->op_own >> 4);
	int deviceCount = 0;
	error = cuDeviceGetCount(&deviceCount);
	if (error != CUDA_SUCCESS) {
		// printf("cuDeviceGetCount() returned %d\n", error);
		return -1;
	}
	/* This function call returns 0 if there are no CUDA capable devices. */
	if (deviceCount == 0) {
		// printf("There are no available device(s) that support CUDA\n");
		return -1;
	}
  
	for (int index = 0; index < deviceCount; index++) {
		if(cuDeviceGet(&cu_device, index) != CUDA_SUCCESS) exit(0);
		cuDeviceGetAttribute(&cuda_pci_bus_id, CU_DEVICE_ATTRIBUTE_PCI_BUS_ID , cu_device);
		cuDeviceGetAttribute(&cuda_pci_device_id, CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID , cu_device);
		// printf("CUDA device %d: PCIe address is %02X:%02X\n", index, (unsigned int)cuda_pci_bus_id, (unsigned int)cuda_pci_device_id);
	}


	if(cuDeviceGet(&cuDevice_selected, cuda_device_id) != cudaSuccess){
		// printf("error on cuDeviceGet\n");
		exit(0);
	}
  // printf("Function name: %s, line number: %d cqe64->op_own >> 4: %d\n", __func__, __LINE__, cqe64->op_own >> 4);
	char name[128];
	if(cuDeviceGetName(name, sizeof(name), cuda_device_id) != cudaSuccess){
		// printf("error on cuDeviceGetName\n");
		exit(0);
	}
	// printf("[pid = %d, dev = %d] device name = [%s]\n", getpid(), cuDevice_selected, name);

    struct timespec res;
    long nano1,nano2;

    clock_t start, end;
    double cpu_time_used;
    struct timeval cpu_timer[2];

    size_t free_memory, total_memory;
    cudaMemGetInfo(&free_memory, &total_memory);
    // printf("free memory: %zu, total_memory: %zu\n", free_memory/(1024 * 1024), total_memory/(1024 * 1024));

  clock_t *dtimer = NULL;
	clock_t timer[2];

	if (cudaSuccess != cudaMalloc((void **)&dtimer, sizeof(clock_t) * 2)) 
		exit(0);
  
	cudaDeviceProp devProp;
    cudaGetDeviceProperties(&devProp, 0);
    // printf("Cuda device clock rate = %d\n", devProp.clockRate);

  cuda_success = cudaDeviceSynchronize();
  printf("Function name: %s, line number: %d cuda_success: %d\n", __func__, __LINE__, cuda_success);
  if (cuda_success != 0) exit(0);

  global_gpu_post_send<<<1,1>>>(
          qpbf_bufsize, /*qpsq_cur_post, qpsq_wqe_cnt,*/
          wr_rdma_remote_addr, wr_rdma_rkey,
          wr_sg_length, wr_sg_lkey, wr_sg_addr,
          wr_opcode, 
          bf_offset, qp_num, wr_id, /*qp->buf.buf*/ qp_buf, /*dev_qpsq_wr_data,*/ 
          dev_qpsq_wqe_head, dev_qp_sq, dev_qp_db, /*dev_wr_sg,*/ dev_wrid1, 
          dev_first, dev_second, device_db, dtimer, dev_ret);

  // printf("Function name: %s, line number: %d cudaDeviceSynchronize(): %d\n", __func__, __LINE__, cudaDeviceSynchronize());
  cuda_success = cudaDeviceSynchronize();
  printf("Function name: %s, line number: %d cuda_success: %d\n", __func__, __LINE__, cuda_success);
  if (cuda_success != 0) exit(0);

  cudaMemcpy(timer, dtimer, sizeof(clock_t) * (2), cudaMemcpyDeviceToHost);
  cudaFree(dtimer);

  // cudaDeviceProp devProp;
  cudaGetDeviceProperties(&devProp, 0);
  // printf("Cuda device clock rate = %d\n", devProp.clockRate);

  float freq = (float)1/((float)devProp.clockRate*1000);
	float g_usec = (float)((float)1/(devProp.clockRate*1000))*((timer[1]-timer[0])) * 1000000;
	printf("POST - INTERNAL MEASUREMENT: %f useconds to execute \n", g_usec);

  dtimer = NULL;
	// clock_t timer[2];
  // printf("Function name: %s, line number: %d cqe64->op_own >> 4: %d\n", __func__, __LINE__, cqe64->op_own >> 4);
	if (cudaSuccess != cudaMalloc((void **)&dtimer, sizeof(clock_t) * 2)) 
		exit(0);

  clock_gettime(CLOCK_REALTIME,&res);
    nano1 = res.tv_nsec;
    int a = cudaDeviceSynchronize();
    printf("cudasynchronize: %d\n", a);
    // start = clock();
    // gettimeofday(&cpu_timer[0], NULL);
    
    global_gpu_poll_cq<<<1,1>>>(dev_cq_ptr, dev_wc, dev_cons_index, cq->verbs_cq.cq.cqe, cq->cqe_sz, num_entries, 
                    (int *) dev_total_wc, (int *) dev_ret, dev_cq_dbrec, dev_mctx, dev_rsc, mctx->uidx_table[0].refcnt,
                    dev_qp_context /*qp_ctx*/, qp_ctx->dump_fill_mkey_be, dev_rq, dev_wrid,
                    qp->rq.wrid[0], cqe_dev, 1, dev_scat_address, dtimer, wq_sq->wrid[0], wq_sq->wqe_head[0],
                    dev_wq);
    a = cudaDeviceSynchronize();
    printf("cudasynchronize: %d\n", a);
    clock_gettime(CLOCK_REALTIME,&res);
    nano2 = res.tv_nsec;

	cudaMemcpy(timer, dtimer, sizeof(clock_t) * (2), cudaMemcpyDeviceToHost);
    cudaFree(dtimer);
    

  
  
  freq = (float)1/(devProp.clockRate*1000);
	g_usec = (float)((float)1/(devProp.clockRate*1000))*((timer[1]-timer[0])) * 1000000;
	printf("POLLING - INTERNAL MEASUREMENT: %f useconds to execute \n", g_usec);

}



int cpu_benchmark_whole(struct ibv_cq *cq_ptr, int num_entries, struct ibv_wc *wc,
                   struct ibv_qp *ibqp, struct ibv_send_wr *wr,
                   struct ibv_send_wr **bad_wr, int num_packets, int mesg_size, float *bandwidth){


  int local_ret;
  struct mlx5_qp *qp = to_mqp(ibqp);

  struct mlx5_bf *bf = qp->bf;
  // printf("Function name: %s, line number: %d, qp->buf.length: %d\n", __func__, __LINE__, qp->buf.length);
  // input variables:
  unsigned int qpbf_bufsize = qp->bf->buf_size; // 256
  unsigned int qpsq_cur_post = qp->sq.cur_post;
  unsigned int qpsq_wqe_cnt = qp->sq.wqe_cnt;
  uint64_t wr_rdma_remote_addr = wr->wr.rdma.remote_addr;
  uint32_t wr_rdma_rkey = wr->wr.rdma.rkey;
  uint32_t wr_sg_length = wr->sg_list->length;
  uint32_t wr_sg_lkey = wr->sg_list->lkey;
  uint64_t wr_sg_addr = wr->sg_list->addr;
  int wr_opcode = wr->opcode;
  uint32_t qp_num = ibqp->qp_num;
  unsigned int bf_offset = bf->offset; 
  uint64_t wr_id = wr->wr_id;
  // printf("Function name: %s, line number: %d\n", __func__, __LINE__);
  // pointers
  void *qp_buf;
  void *dev_qpsq_wr_data; //qp->sq.wr_data;
  void *dev_qpsq_wqe_head;// qp->sq.wqe_head;
  void *dev_qp_sq; // &qp->sq;
  int *bf_reg = (int *) bf->reg; // bf->reg;
  void *dev_qp_db; // qp->db
  void *dev_wr_sg; // wr->sg_list
  int *dev_ret; // &ret
  void *dev_bf_reg; 
  void *dev_wrid, *dev_wrid1; // qp->sq.wrid
  void *dev_wqe_head; // qp->sq.wqe_head
  
  // printf("Function name: %s, line number: %d cudaHostRegister(qp->buf.buf, sizeof(qp->buf.buf), cudaHostRegisterMapped): %d\n", 
  //         __func__, __LINE__, cudaHostRegister(qp->buf.buf, sizeof(qp->buf.buf), cudaHostRegisterMapped));
  cudaError_t success;
  // pin host memory and get device pointer:
  if (wq_buffer_gpu != 1){
    success = cudaHostRegister(qp->buf.buf, qp->buf.length, cudaHostRegisterMapped);
    if(success !=  cudaSuccess && success != cudaErrorHostMemoryAlreadyRegistered) exit(0);
    if(cudaHostGetDevicePointer(&qp_buf, qp->buf.buf, 0) != cudaSuccess) exit(0);
  }
  else qp_buf = qp->buf.buf;
  
  success = cudaHostRegister(qp->sq.wqe_head, sizeof(qp->sq.wqe_head), cudaHostRegisterMapped);
  if(success != cudaErrorHostMemoryAlreadyRegistered && success !=  cudaSuccess)
        exit(0);
  // get GPU pointer for qp->sq.wqe_head
  if(cudaHostGetDevicePointer(&dev_qpsq_wqe_head, qp->sq.wqe_head, 0) != cudaSuccess)
      exit(0);

  // printf("Function name: %s, line number: %d\n", __func__, __LINE__);
  success = cudaHostRegister(&qp->sq, sizeof(qp->sq), cudaHostRegisterMapped);
  if(success != cudaErrorHostMemoryAlreadyRegistered && success !=  cudaSuccess)
        exit(0);
  // get GPU pointer for qp->sq
  if(cudaHostGetDevicePointer(&dev_qp_sq, &qp->sq, 0) != cudaSuccess)
      exit(0);

  printf("Function name: %s, line number: %d bf->uar: 0x%llx\n", __func__, __LINE__, bf->uar);
  cudaError_t cudaStatus1;

  void *device_db;
  cudaStatus1 = cudaHostRegister(bf->reg,  8, cudaHostRegisterIoMemory);
  if (cudaStatus1 != cudaSuccess && cudaStatus1 != cudaErrorHostMemoryAlreadyRegistered) exit(0);
  
  cudaStatus1 = cudaHostGetDevicePointer(&device_db, bf->reg, 0);
  if (cudaStatus1 != cudaSuccess) exit(0);


  cudaError_t cudaState;
  cudaState = cudaHostRegister(qp->db, sizeof(qp->db), cudaHostRegisterMapped);

  if(cudaState != cudaSuccess && cudaState != cudaErrorHostMemoryAlreadyRegistered)
        exit(0);
  if(cudaHostGetDevicePointer(&dev_qp_db, qp->db, 0) != cudaSuccess)
      exit(0);
  cudaError_t cudaStatus;
  

  cudaStatus = cudaHostRegister(qp->sq.wrid, sizeof(qp->sq.wrid), cudaHostRegisterMapped);
  if(cudaStatus != cudaErrorHostMemoryAlreadyRegistered && cudaStatus != cudaSuccess)
        exit(0);
  // get GPU pointer for qp->sq
  if(cudaHostGetDevicePointer(&dev_wrid1, qp->sq.wrid, 0) != cudaSuccess)
      exit(0);
  
 

  struct mlx5_cq *cq = to_mcq(cq_ptr);
  void *cqe;
  // printf("Function: %s line number: %d cq->buf.length: %d\n",__func__, __LINE__, cq->buf_a.length);
  struct mlx5_cqe64 *cqe64;
  // int cond = 0;
  uint32_t cons_index = cq->cons_index;
 
  void *dev_cq_ptr; // cq pointer for GPU
  void *dev_cons_index; // 
  // void *dev_ret; // 
  void *dev_total_wc; // 
  void *dev_cq_dbrec;
  struct mlx5_context *mctx = container_of(cq->verbs_cq.cq.context, struct mlx5_context, ibv_ctx.context);
  // mctx->uidx_table[0].table
  void *dev_mctx;
  void **dev_table;
  void *dev_rsc;
  void *dev_qp_context;
  void *dev_qp_buf;
  int total_wc = 0, ret = -1;
  struct mlx5_resource *rsc = mctx->uidx_table[0].table[0];
  if(!rsc) printf("rsc is null\n\n\n");
  qp = (struct mlx5_qp *)(rsc);
  // printf("qp buf address: 0x%llx\n\n\n", qp->buf.buf);
  struct mlx5_context *qp_ctx = to_mctx(qp->ibv_qp->pd->context); 
  
  cudaError_t crc = cudaSuccess;
  // cudaError_t cudaStatus;
 
  if (cq_buffer_gpu != 1){
    cudaStatus = cudaHostRegister(cq->active_buf->buf /*cqbuf*/, cq->active_buf->length /*cqbuf_size*/, cudaHostRegisterMapped);
    if(cudaStatus != cudaErrorHostMemoryAlreadyRegistered && cudaStatus != cudaSuccess) exit(0);
    if(cudaHostGetDevicePointer(&dev_cq_ptr, /*cqbuf*/ cq->active_buf->buf, 0) !=  cudaSuccess) exit(0);
  }
  else dev_cq_ptr = cq->active_buf->buf;
  
   
  cudaStatus = cudaHostRegister(&cq->cons_index, sizeof(cq->cons_index), cudaHostRegisterMapped);
  if(cudaStatus != cudaSuccess && cudaStatus != cudaErrorHostMemoryAlreadyRegistered) exit(0);
  if(cudaHostGetDevicePointer(&dev_cons_index, &cq->cons_index, 0) != cudaSuccess) exit(0);
 
  cudaStatus = cudaHostRegister(&ret, sizeof(ret), cudaHostRegisterMapped);
  if(cudaStatus != cudaErrorHostMemoryAlreadyRegistered && cudaStatus !=  cudaSuccess) exit(0);
  if(cudaHostGetDevicePointer(&dev_ret, &ret, 0) != cudaSuccess) exit(0);
      
  cudaStatus = cudaHostRegister(cq->dbrec, sizeof(cq->dbrec), cudaHostRegisterMapped);
  if(cudaStatus != cudaErrorHostMemoryAlreadyRegistered && cudaStatus !=  cudaSuccess) exit(0);
  if(cudaHostGetDevicePointer(&dev_cq_dbrec, cq->dbrec, 0) != cudaSuccess)
      exit(0);


  cudaStatus = cudaHostRegister(rsc, sizeof(rsc), cudaHostRegisterMapped);
  if(cudaStatus != cudaErrorHostMemoryAlreadyRegistered && cudaStatus !=  cudaSuccess) exit(0);
  if(cudaHostGetDevicePointer(&dev_rsc, rsc, 0) != cudaSuccess) exit(0);

  struct mlx5_wq *qp_rq = &qp->rq;
 
  struct mlx5_wq * wq_sq = &qp->sq;
  void *dev_wq;
 
  cudaState = cudaHostRegister(wq_sq, sizeof(qp->sq), cudaHostRegisterMapped);
  if(cudaState !=  cudaSuccess && cudaState != cudaErrorHostMemoryAlreadyRegistered)  exit(0);
  if(cudaHostGetDevicePointer(&dev_wq, wq_sq, 0) != cudaSuccess) exit(0);

  cudaState = cudaHostRegister(qp_rq->wrid, sizeof(qp_rq->wrid), cudaHostRegisterMapped);
  if(cudaState != cudaSuccess && cudaState != cudaErrorHostMemoryAlreadyRegistered) exit(0);
  if(cudaHostGetDevicePointer(&dev_wrid, qp_rq->wrid, 0) != cudaSuccess) exit(0);

	int cuda_device_id = 0;
	int cuda_pci_bus_id;
	int cuda_pci_device_id;
	int index;
	CUdevice cu_device;
	CUdevice cuDevice_selected;

	// printf("initializing CUDA\n");
	CUresult error = cuInit(0);
	if (error != CUDA_SUCCESS) {
		// printf("cuInit(0) returned %d\n", error);
		return -1;
	}
  // printf("Function name: %s, line number: %d cqe64->op_own >> 4: %d\n", __func__, __LINE__, cqe64->op_own >> 4);
	int deviceCount = 0;
	error = cuDeviceGetCount(&deviceCount);
	if (error != CUDA_SUCCESS) {
		// printf("cuDeviceGetCount() returned %d\n", error);
		return -1;
	}
	/* This function call returns 0 if there are no CUDA capable devices. */
	if (deviceCount == 0) {
		// printf("There are no available device(s) that support CUDA\n");
		return -1;
	}
  
	for (int index = 0; index < deviceCount; index++) {
		if(cuDeviceGet(&cu_device, index) != CUDA_SUCCESS) exit(0);
		cuDeviceGetAttribute(&cuda_pci_bus_id, CU_DEVICE_ATTRIBUTE_PCI_BUS_ID , cu_device);
		cuDeviceGetAttribute(&cuda_pci_device_id, CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID , cu_device);
		// printf("CUDA device %d: PCIe address is %02X:%02X\n", index, (unsigned int)cuda_pci_bus_id, (unsigned int)cuda_pci_device_id);
	}


	if(cuDeviceGet(&cuDevice_selected, cuda_device_id) != cudaSuccess){
		// printf("error on cuDeviceGet\n");
		exit(0);
	}
  // printf("Function name: %s, line number: %d cqe64->op_own >> 4: %d\n", __func__, __LINE__, cqe64->op_own >> 4);
	char name[128];
	if(cuDeviceGetName(name, sizeof(name), cuda_device_id) != cudaSuccess){
		// printf("error on cuDeviceGetName\n");
		exit(0);
	}
	// printf("[pid = %d, dev = %d] device name = [%s]\n", getpid(), cuDevice_selected, name);

    struct timespec res;
    long nano1,nano2;

    clock_t start, end;
    double cpu_time_used;
    struct timeval cpu_timer[2];

    size_t free_memory, total_memory;
    cudaMemGetInfo(&free_memory, &total_memory);
    // printf("free memory: %zu, total_memory: %zu\n", free_memory/(1024 * 1024), total_memory/(1024 * 1024));

  int num_of_packets = num_packets;
  int size_timer = num_of_packets*4;
  clock_t *dtimer1 = NULL;
	clock_t timer[size_timer], timer1[2];

	if (cudaSuccess != cudaMalloc((void **)&dtimer1, sizeof(clock_t) * 2)) 
		exit(0);
  
	cudaDeviceProp devProp;
    cudaGetDeviceProperties(&devProp, 0);

  struct ibv_send_wr wr1 = *wr;
  clock_t *dtimer = NULL;
  
	if (cudaSuccess != cudaMalloc((void **)&dtimer, sizeof(clock_t) * size_timer)) 
		exit(0);

    //   int a = cudaDeviceSynchronize();
    // printf("cudasynchronize: %d\n", a);
    // gpu_whole<<<1, 1>>>(qpbf_bufsize, wr1, /*qpsq_cur_post, qpsq_wqe_cnt,*/
          
    //       wr_sg_length, wr_sg_lkey, wr_sg_addr,
    //       wr_opcode, 
    //       qp_num, wr_id, /*qp->buf.buf*/ qp_buf, 
    //       dev_qpsq_wqe_head, dev_qp_sq, dev_qp_db, dev_wrid1, 
    //       device_db,
          
    //       dev_cq_ptr, dev_cons_index, cq->verbs_cq.cq.cqe, cq->cqe_sz, 1, 
    //       dev_cq_dbrec, dev_rsc,
    //       dev_wrid,
    //       dtimer, wq_sq->wrid[0], wq_sq->wqe_head[0],
    //       dev_wq
    //       );
    int a = cudaDeviceSynchronize();
    printf("cudasynchronize: %d\n", a);
    
    multiple_packets<<<1, 1>>>(num_of_packets,
          qpbf_bufsize, wr1, /*qpsq_cur_post, qpsq_wqe_cnt,*/
          mesg_size,
          wr_sg_length, wr_sg_lkey, wr_sg_addr,
          wr_opcode, 
          qp_num, wr_id, /*qp->buf.buf*/ qp_buf, 
          dev_qpsq_wqe_head, dev_qp_sq, dev_qp_db, dev_wrid1, 
          device_db,
          
          dev_cq_ptr, dev_cons_index, cq->verbs_cq.cq.cqe, cq->cqe_sz, 1, 
          dev_cq_dbrec, dev_rsc,
          dev_wrid,
          dtimer, wq_sq->wrid[0], wq_sq->wqe_head[0],
          dev_wq);
        
    a = cudaDeviceSynchronize();
    printf("cudasynchronize: %d\n", a);
    clock_gettime(CLOCK_REALTIME,&res);
    nano2 = res.tv_nsec;

  cudaGetDeviceProperties(&devProp, 0);

	cudaMemcpy(timer, dtimer, sizeof(clock_t) * size_timer, cudaMemcpyDeviceToHost);
  cudaFree(dtimer);

  clock_t poll = 0, post = 0;
  for (int i = 0; i < num_of_packets; i++){
  
      poll += timer[4*i+3] - timer[4*i+2];
      post += timer[4*i+1] - timer[4*i];
  
  }
  printf("/*************************************************************/\n");
  printf("Test results for %d packets with %d bytes\n", num_of_packets, mesg_size*4);

  float freq_post = (float)1/((float)devProp.clockRate*1000);
	float g_usec_post = (float)((float)1/(devProp.clockRate*1000))*((post)) * 1000000;
	printf("POST - INTERNAL MEASUREMENT: %f useconds to execute \n", g_usec_post / num_of_packets);
  
  float freq_poll = (float)1/(devProp.clockRate*1000);
	float g_usec__poll = (float)((float)1/(devProp.clockRate*1000))*((poll)) * 1000000;
	printf("POLLING - INTERNAL MEASUREMENT: %f useconds to execute \n", g_usec__poll / num_of_packets);

  float total_usec = g_usec_post + g_usec__poll;
  printf("Total time: %f useconds for %d bytes data\n", total_usec, num_of_packets*mesg_size*4);

  float throughtput = (float)(num_of_packets*mesg_size*4*8)/(total_usec*1e-6*1e9);
  *bandwidth = throughtput;
  printf("Throughput: %f Gbps\n", throughtput);
  printf("/*************************************************************/\n");
}

__device__ void create_wr(struct ibv_send_wr *wr, uint64_t wr_id,
                          uint64_t wr_rdma_remote_addr, uint32_t wr_rdma_rkey,
                          uint32_t wr_sg_length, uint32_t wr_sg_lkey, uint64_t wr_sg_addr,
                          int wr_opcode){
  struct ibv_sge sge;
  struct connection *conn;
  conn = (struct connection *)(uintptr_t) wr_id;
  memset(&wr, 0, sizeof(wr));
  wr->wr_id = (uintptr_t)conn;
  wr->opcode = (ibv_wr_opcode) 4; // IBV_WR_RDMA_READ;
  wr->sg_list = &sge;
  wr->num_sge = 1;
  wr->send_flags = 2; // IBV_SEND_SIGNALED;
  wr->wr.rdma.remote_addr = (uintptr_t)wr_rdma_remote_addr; // conn->peer_mr.addr;
  wr->wr.rdma.rkey = wr_rdma_rkey; // conn->peer_mr.rkey;
  sge.addr = (uintptr_t) wr_sg_addr; // (uintptr_t)conn->rdma_local_region;
  sge.length = (uint32_t) RDMA_BUFFER_SIZE*sizeof(int)- 4; //RDMA_BUFFER_SIZE;
  sge.lkey = wr_sg_lkey; // conn->rdma_local_mr->lkey;
  int *addr = (int *) wr_sg_addr;
}

__global__ void gpu_whole(
          unsigned int qpbf_bufsize, struct ibv_send_wr wr1,
          uint32_t wr_sg_length, uint32_t wr_sg_lkey, uint64_t wr_sg_addr,
          int wr_opcode, uint32_t qp_num, uint64_t wr_id, int mesg_size,
          void *qp_buf, void *dev_qpsq_wqe_head, 
          void *dev_qp_sq, void *dev_qp_db, void *dev_wrid1,
          void *bf_reg,

          void *cq_buf, void *cons_index,
          int ibv_cqe, uint32_t cqe_sz, int max_wc, 
          void *dev_cq_dbrec,
          void* dev_rsc,
          void *dev_wrid,
          clock_t *timer, uint64_t wrid_0, 
          unsigned int wqe_head_0, void *dev_wq    
){

  struct ibv_send_wr wr, *bad_wr = NULL;
  struct ibv_sge sge;
  struct ibv_wc wc1;
  
  printf("benchmark started\n");
  struct connection *conn;
  conn = (struct connection *)(uintptr_t) wr1.wr_id;
  memset(&wr, 0, sizeof(wr));
  wr.wr_id = (uintptr_t)conn;
  wr.opcode = (ibv_wr_opcode) 4; // IBV_WR_RDMA_READ;
  wr.sg_list = &sge;
  wr.num_sge = 1;
  wr.send_flags = 2; // IBV_SEND_SIGNALED;
  wr.wr.rdma.remote_addr = (uintptr_t) wr1.wr.rdma.remote_addr; // wr_rdma_remote_addr; // conn->peer_mr.addr;
  wr.wr.rdma.rkey = wr1.wr.rdma.rkey; // wr_rdma_rkey; // conn->peer_mr.rkey;
  sge.addr = (uintptr_t) wr_sg_addr; // (uintptr_t)conn->rdma_local_region;
  sge.length = (uint32_t) RDMA_BUFFER_SIZE*sizeof(int); //RDMA_BUFFER_SIZE;
  sge.lkey = wr_sg_lkey; // conn->rdma_local_mr->lkey;
  int *addr = (int *) wr_sg_addr;

  timer[0] = clock();
  int ret = post(qpbf_bufsize, wr.wr.rdma.remote_addr, wr.wr.rdma.rkey,
          wr_sg_length, wr_sg_lkey, wr_sg_addr, wr_opcode, 
          qp_num, wr_id, &wr, qp_buf,
          dev_qpsq_wqe_head, dev_qp_sq, dev_qp_db, dev_wrid1, bf_reg);
  timer[1] = clock();

  
  timer[2] = clock();
      while(poll(cq_buf /* the CQ, we got notification for */, 
          &wc1, // twc/* where to store */,
          (uint32_t *) cons_index,
          ibv_cqe,
          cqe_sz,
          max_wc /* number of remaining WC elements*/,
          (uint32_t *) dev_cq_dbrec,
          
          dev_rsc, 
          dev_wrid, wrid_0, wqe_head_0,
          dev_wq) < 0);
      // printf("gpu polling\n");
  // while (addr[RDMA_BUFFER_SIZE-2] == 0); 
  
  timer[3] = clock();
  printf("addr[RDMA_BUFFER_SIZE-2]: %d\n", addr[RDMA_BUFFER_SIZE-2]);
  printf("wr_rdma_remote_addr 0x%llx\n", &wr);
  if(wc1.status != IBV_WC_SUCCESS){
    printf("WC1 status is not success: %d\n", wc1.status);
    return;
  }
  // 
  // for(int i = 0; i < RDMA_BUFFER_SIZE)
}

__device__ int poll( void *cq_buf, struct ibv_wc *wc, uint32_t *cons_index,
                            int ibv_cqe, uint32_t cqe_sz, int n, void *cq_dbrec,
                            void *dev_rsc, void *dev_wrid, 
							              uint64_t wrid_0, unsigned int wqe_head_0,
                            void *dev_wq) 
{
   
	// timer[0] = clock();
  uint32_t *gpu_dbrec = (uint32_t *) cq_dbrec;
  int npolled=0;
	int err = 0;
	void *cqe;
	struct mlx5_cqe64 *cqe64;
	int cqe_ver = 1;
	struct mlx5_wq *wq;
	uint16_t wqe_ctr;
	uint32_t qpn;
	int idx;
	uint8_t opcode;
	uint32_t cons_index_dev = *cons_index;
    
  cqe = cq_buf + (cons_index_dev & ibv_cqe) * cqe_sz;
  cqe64 = (struct mlx5_cqe64 *)((cqe_sz == 64) ? cqe : cqe + 64);
  int cond1 = (cqe64->op_own != 240) &&
    !((cqe64->op_own & 1) ^ !!(1 & (ibv_cqe + 1)));
    // printf("cqe64->op_own: %d\n", cqe64->op_own);
    
    // printf("cons_index_dev: %d\n", cons_index_dev);
    // printf("ibv_cqe: %d\n", ibv_cqe);
    // printf("cqe_sz: %d\n", cqe_sz);
  if (!cond1) {
    err = CQ_EMPTY;
    gpu_dbrec[0] = htonl(cons_index_dev & 0xffffff);
    // printf("cond1: %d\n", cond1);
    return err; 
  } 
  
  (*cons_index)++;


  qpn = htonl(cqe64->sop_drop_qpn) & 0xffffff;
  wc->wc_flags = 0;
  wc->qp_num = qpn;
  opcode = cqe64->op_own >> 4;

  wq = (struct mlx5_wq *) dev_wq; // &mqp->sq;
  wqe_ctr = htons (cqe64->wqe_counter);
  idx = wqe_ctr & (wq->wqe_cnt - 1);
  // if(htonl(cqe64->sop_drop_qpn) >> 24 == MLX5_OPCODE_RDMA_WRITE){
  //   wc->opcode    = (ibv_wc_opcode) (MLX5_OPCODE_RDMA_WRITE >> 3); // IBV_WC_RDMA_WRITE;
  // }
  // else {
    wc->opcode    = (ibv_wc_opcode) ((htonl(cqe64->sop_drop_qpn) >> 24) >> 3); // IBV_WC_RDMA_READ;
    wc->byte_len  = htonl(cqe64->byte_cnt);
  // }
  wc->wr_id = wrid_0; // wq->wrid[idx];
  wc->status = (ibv_wc_status) err;
  wq->tail = wqe_head_0 + 1; // wq->wqe_head[idx] + 1;
  
out1:
  gpu_dbrec[0] = htonl((*cons_index) & 0xffffff);
  return err; 
}

__device__ int post(unsigned int qpbf_bufsize,
          uint64_t wr_rdma_remote_addr, uint32_t wr_rdma_rkey,
          uint32_t wr_sg_length, uint32_t wr_sg_lkey, uint64_t wr_sg_addr, int wr_opcode, 
          uint32_t qp_num, uint64_t wr_id, struct ibv_send_wr *wr, 
          void *qp_buf,  void *dev_qpsq_wqe_head, 
          void *dev_qp_sq, void *dev_qp_db, void *dev_wrid,
          void *bf_reg)
{
  
	void *seg;
	struct mlx5_wqe_ctrl_seg *ctrl = NULL;
	int err = 0;
	unsigned idx;
	uint32_t mlx5_opcode;
  unsigned int *qpsq_wqe_head = (unsigned int *) dev_qpsq_wqe_head;// qp->sq.wqe_head;
  struct mlx5_wq *qp_sq = (struct mlx5_wq *) dev_qp_sq; // &qp->sq;
  unsigned int *qp_db = (unsigned int *) dev_qp_db; // qp->db
  uint64_t * qpsq_wrid = (uint64_t *) dev_wrid;// qp_sq->wrid[idx] = wr_id;
  idx = qp_sq->cur_post & (qp_sq->wqe_cnt - 1);
  // printf("idx: %d\n", idx);
  seg = (qp_buf + 256 + (idx * 64)); // mlx5_get_send_wqe(qp, idx);
  ctrl = (struct mlx5_wqe_ctrl_seg *) (qp_buf + 256 + (idx * 64));
  
  
  mlx5_opcode = wr_opcode*2 + 8 - 2*(wr_opcode == 2); // mlx5_ib_opcode[wr->opcode];
  
  ctrl->opmod_idx_opcode = htonl(((qp_sq->cur_post & 0xffff) << 8) | mlx5_opcode);
  ctrl->qpn_ds = htonl(3 | (qp_num << 8));
  ctrl->signature = 0;
  ctrl->fm_ce_se = 8; // MLX5_WQE_CTRL_CQ_UPDATE;
  ctrl->imm = 0; // 
  struct mlx5_wqe_raddr_seg *rdma = (struct mlx5_wqe_raddr_seg *)(seg + 16); // seg + 16; // sizeof(*ctrl);
  // clock_t start = clock64();
  rdma->raddr    = htonl64(wr_rdma_remote_addr);
  rdma->rkey     = htonl(wr_rdma_rkey);
  rdma->reserved = 0;
  // clock_t end = clock64();
  struct mlx5_wqe_data_seg *data = (struct mlx5_wqe_data_seg *) (seg + 32);
  data->byte_count = htonl(wr_sg_length); // htonl(wr_sg_list->length);
  data->lkey       = htonl(wr_sg_lkey); // htonl(wr_sg_list->lkey);
  data->addr       = htonl64(wr_sg_addr); // htonl64(wr_sg_list->addr);

  
  qpsq_wrid[idx] = wr_id;
  qpsq_wqe_head[idx] = qp_sq->head;
  
  // int tmp = (size * 16 + 64 - 1) / 64;
  qp_sq->cur_post += 1; // tmp;
  
  qp_db[1] = htonl(1 & 0xffff);
  
  // void *addr = bf_reg; // + 0xa00;
  uint64_t val;
  /* Do 64 bytes at a time */
  // addr = bf_reg; // + bf_offset;
  val = *(uint64_t *) ctrl;
  
  *(volatile uint32_t *)bf_reg = htonl(htonl64(val) >> 32);
  *(volatile uint32_t *)(bf_reg+4) = htonl(htonl64(val));
  
  // printf("dif: %d\n",(int) end-start);
	return 0;
}

__global__ void multiple_packets(int num_of_packets,
          unsigned int qpbf_bufsize, struct ibv_send_wr wr1, int mesg_size,
          uint32_t wr_sg_length, uint32_t wr_sg_lkey, uint64_t wr_sg_addr,
          int wr_opcode, uint32_t qp_num, uint64_t wr_id,
          void *qp_buf, void *dev_qpsq_wqe_head, 
          void *dev_qp_sq, void *dev_qp_db, void *dev_wrid1,
          void *bf_reg,

          void *cq_buf, void *cons_index,
          int ibv_cqe, uint32_t cqe_sz, int max_wc, 
          void *dev_cq_dbrec,
          void* dev_rsc,
          void *dev_wrid,
          clock_t *timer, uint64_t wrid_0, 
          unsigned int wqe_head_0, void *dev_wq    
){

  struct ibv_send_wr wr, *bad_wr = NULL;
  struct ibv_sge sge;
  struct ibv_wc wc1;
  
  printf("benchmark started\n");
  struct connection *conn;
  conn = (struct connection *)(uintptr_t) wr1.wr_id;
  memset(&wr, 0, sizeof(wr));
  wr.wr_id = (uintptr_t)conn;
  wr.opcode = (ibv_wr_opcode) 4; // IBV_WR_RDMA_READ;
  wr.sg_list = &sge;
  wr.num_sge = 1;
  wr.send_flags = 2; // IBV_SEND_SIGNALED;
  wr.wr.rdma.remote_addr = (uintptr_t) wr1.wr.rdma.remote_addr; // wr_rdma_remote_addr; // conn->peer_mr.addr;
  wr.wr.rdma.rkey = wr1.wr.rdma.rkey; // wr_rdma_rkey; // conn->peer_mr.rkey;
  sge.addr = (uintptr_t) wr_sg_addr; // (uintptr_t)conn->rdma_local_region;
  sge.length = (uint32_t) mesg_size*sizeof(int); // RDMA_BUFFER_SIZE*sizeof(int); //RDMA_BUFFER_SIZE;
  printf("mesg_size: %d\n", mesg_size*sizeof(int));
  printf("wr_sg_addr: 0x%llx\n", wr_sg_addr);
  // printf("read_after_write_buffer: 0x%llx\n", read_after_write_buffer);
  sge.lkey = wr_sg_lkey; // conn->rdma_local_mr->lkey;
  int *addr = (int *) wr_sg_addr;

  for(int p_num=0 ;  p_num < num_of_packets ; p_num++)
  {

    timer[4*p_num] = clock();
    int ret = post(qpbf_bufsize, wr.wr.rdma.remote_addr, wr.wr.rdma.rkey,
            sge.length, wr_sg_lkey, wr_sg_addr, wr_opcode, 
            qp_num, wr_id, &wr, qp_buf,
            dev_qpsq_wqe_head, dev_qp_sq, dev_qp_db, dev_wrid1, bf_reg);
    timer[4*p_num+1] = clock();

    
    timer[4*p_num+2] = clock();
    // while(poll(cq_buf /* the CQ, we got notification for */, 
    //     &wc1, // twc/* where to store */,
    //     (uint32_t *) cons_index,
    //     ibv_cqe,
    //     cqe_sz,
    //     max_wc /* number of remaining WC elements*/,
    //     (uint32_t *) dev_cq_dbrec,
        
    //     dev_rsc, 
    //     dev_wrid, wrid_0, wqe_head_0,
    //     dev_wq) < 0);
        // printf("gpu polling\n");
    while (addr[mesg_size-1] == 0); 
    
    timer[4*p_num+3] = clock();

    if(wc1.status != IBV_WC_SUCCESS){
      printf("WC1 status is not success: %d\n", wc1.status);
      return;
    }
  }
  
    // int p_num = 0;
    // for(p_num=0 ;  p_num<num_of_packets ; p_num++)
    // {
    //     timer[4*p_num] = clock();
    //     /* GPU Post send */
    //     // gpu_post_send(buf, db_record, p_num, qpn, raddr, rkey, len,(uintptr_t)(laddr + p_num*len), lkey);
    //     // timer[4*p_num+1] = clock();

    //     // /* Ring door bell  */
    //     // gpu_ring_doorbell(bf_addr + 0x800, db_val);
    //     // timer[4*p_num+2] = clock();
    
    //     // /* GPU Poll a completion queue */
    //     // while(gpu_poll_cq(cq_buf, n, wc + p_num*sizeof(struct ibv_wc), (uint32_t *)cons_index_p, ibv_cqe, cqe_sz, cq_dbrec) == 0);
    //     // timer[4*p_num+3] = clock();

    //     /* GPU check WC status */
        
    // }
    __syncthreads();

    timer[4*num_of_packets] = clock();
}