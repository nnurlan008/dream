
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
#include <linux/types.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stddef.h>
#include <endian.h>
#include <sys/time.h>

extern "C"{
#include "gpu-utils.h"
}

const int TIMEOUT_IN_MS = 500; /* ms */

static int on_addr_resolved(struct rdma_cm_id *id);
static int on_connection(struct rdma_cm_id *id);
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
                           void *qp_context, int dump_fill_mkey_be, void *qp_buf,
                           void *dev_rq, void *dev_wrid, uint64_t wrid_1,
                           void *cqe_dev, int cond, void *dev_scat_address,
						               clock_t *timer, uint64_t wrid_0, unsigned int wqe_head_0
                           /*, void **table table, int refcnt*/);

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

void send_message(struct connection *conn)
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

  while (!conn->connected);

  TEST_NZ(ibv_post_send(conn->qp, &wr, &bad_wr));
}

void send_mr(void *context)
{
  struct connection *conn = (struct connection *)context;

  // conn->send_msg->type = MSG_MR;
  memcpy(&conn->send_msg->data.mr, conn->rdma_remote_mr, sizeof(struct ibv_mr));

  send_message(conn);
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

int host_gpu_post_recv(struct ibv_qp *ibqp, struct ibv_recv_wr *wr,
		   struct ibv_recv_wr **bad_wr);

char *read_after_write_buffer = NULL;
struct ibv_mr *read_after_write_mr;
// void *cqbuf, *wqbuf;
void *cqbuf = NULL;
void *wqbuf = NULL;
int wqbuf_size = 5124, cqbuf_size = 32768;

int registered = 0;

int main(int argc, char **argv)
{
  struct addrinfo *addr;
  struct rdma_cm_event *event = NULL;
  struct rdma_cm_event event_copy;
  struct rdma_cm_id *conn_id= NULL;
  struct rdma_event_channel *ec = NULL;
  printf("Hello\n");
  if (argc != 4)
    usage(argv[0]);
  printf("Hello\n");
  if (strcmp(argv[1], "write") == 0)
    set_mode(M_WRITE);
  else if (strcmp(argv[1], "read") == 0)
    set_mode(M_READ);
  else
    usage(argv[0]);
  printf("Hello\n");
  TEST_NZ(getaddrinfo(argv[2], "9700", NULL, &addr));

  TEST_Z(ec = rdma_create_event_channel());
  TEST_NZ(rdma_create_id(ec, &conn_id, NULL, RDMA_PS_TCP));
  TEST_NZ(rdma_resolve_addr(conn_id, NULL, addr->ai_addr, TIMEOUT_IN_MS));

  freeaddrinfo(addr);

  /*Allocate control buffers - CQ and WQ*/
  int ret = cudaMalloc(&cqbuf, cqbuf_size);
  if (cudaSuccess != ret) {
    printf("error on cudaMalloc for cqbuf: %d\n", ret);
    exit(0);
  }
  ret = cudaMemset(cqbuf,0,cqbuf_size);
  if (cudaSuccess != ret) {
    printf("error on cudaMemset for cqbuf: %d\n", ret);
    exit(0);
  }

  ret = cudaMalloc(&wqbuf, wqbuf_size);
  if (cudaSuccess != ret) {
    printf("error on cudaMalloc for wqbuf: %d\n", ret);
    exit(0);
  }
  ret = cudaMemset(cqbuf,0,cqbuf_size);
  if (cudaSuccess != ret) {
    printf("error on cudaMemset for wqbuf: %d\n", ret);
    exit(0);
  }
  printf("Function: %s line number: %d\n",__func__, __LINE__);
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
  if(on_connection(event_copy.id)){
    printf("error on on_connection!\n");
    return -1;
  }
  printf("Function: %s line number: %d\n",__func__, __LINE__);
  struct ibv_wc wc;
  process_work_completion_events (s_ctx->comp_channel, &wc, 1);
  printf("Function: %s line number: %d\n",__func__, __LINE__);
  process_work_completion_events (s_ctx->comp_channel, &wc, 1);
  struct connection *conn = (struct connection *)(uintptr_t)wc.wr_id;
  if (wc.opcode & IBV_WC_RECV){
    printf("receive completed\n");
    printf("Function: %s line number: %d\n",__func__, __LINE__);
    memcpy(&conn->peer_mr, &conn->recv_msg->data.mr, sizeof(conn->peer_mr));
    printf("Function: %s line number: %d\n",__func__, __LINE__);
    post_receives(conn);
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
  sge.addr = (uintptr_t)conn->rdma_local_region;
  sge.length = RDMA_BUFFER_SIZE;
  sge.lkey = conn->rdma_local_mr->lkey;
  TEST_NZ(ibv_post_send(conn->qp, &wr, &bad_wr));
  process_work_completion_events (s_ctx->comp_channel, &wc, 1);
  printf("remote buffer: %s\n", conn->rdma_local_region);
  printf("Function name: %s, line number: %d\n", __func__, __LINE__);

  conn = (struct connection *)(uintptr_t)wc.wr_id;
  bad_wr = NULL;
  memset(&wr, 0, sizeof(wr));
  wr.wr_id = (uintptr_t)conn;
  wr.opcode = IBV_WR_RDMA_WRITE;
  wr.sg_list = &sge;
  wr.num_sge = 1;
  wr.send_flags = IBV_SEND_SIGNALED;
  wr.wr.rdma.remote_addr = (uintptr_t)conn->peer_mr.addr;
  wr.wr.rdma.rkey = conn->peer_mr.rkey;
  sge.addr = (uintptr_t)conn->rdma_remote_region;
  sge.length = RDMA_BUFFER_SIZE;
  sge.lkey = conn->rdma_remote_mr->lkey;
  TEST_NZ(ibv_post_send(conn->qp, &wr, &bad_wr));
  process_work_completion_events (s_ctx->comp_channel, &wc, 1);
  printf("remote buffer: %s\n", conn->rdma_remote_region);
  printf("Function name: %s, line number: %d\n", __func__, __LINE__);

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
  sge.addr = (uintptr_t)read_after_write_mr->addr;
  sge.length = RDMA_BUFFER_SIZE;
  sge.lkey = read_after_write_mr->lkey;
  TEST_NZ(ibv_post_send(conn->qp, &wr, &bad_wr));
  process_work_completion_events (s_ctx->comp_channel, &wc, 1);
  printf("remote buffer: %s\n", read_after_write_buffer);
  printf("Function name: %s, line number: %d\n", __func__, __LINE__);

  sprintf(conn->rdma_remote_region, "I'm doing well. Hbu? with %d", getpid());
  conn = (struct connection *)(uintptr_t)wc.wr_id;
  bad_wr = NULL;
  memset(&wr, 0, sizeof(wr));
  wr.wr_id = (uintptr_t)conn;
  wr.opcode = IBV_WR_RDMA_WRITE;
  wr.sg_list = &sge;
  wr.num_sge = 1;
  wr.send_flags = IBV_SEND_SIGNALED;
  wr.wr.rdma.remote_addr = (uintptr_t)conn->peer_mr.addr;
  wr.wr.rdma.rkey = conn->peer_mr.rkey;
  sge.addr = (uintptr_t)conn->rdma_remote_region;
  sge.length = RDMA_BUFFER_SIZE;
  sge.lkey = conn->rdma_remote_mr->lkey;
  TEST_NZ(ibv_post_send(conn->qp, &wr, &bad_wr));
  process_work_completion_events (s_ctx->comp_channel, &wc, 1);
  printf("remote buffer: %s\n", conn->rdma_remote_region);
  printf("Function name: %s, line number: %d\n", __func__, __LINE__);

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
  sge.addr = (uintptr_t)read_after_write_mr->addr;
  sge.length = RDMA_BUFFER_SIZE;
  sge.lkey = read_after_write_mr->lkey;
  TEST_NZ(ibv_post_send(conn->qp, &wr, &bad_wr));
  process_work_completion_events (s_ctx->comp_channel, &wc, 1);
  printf("remote buffer: %s\n", read_after_write_buffer);
  printf("Function name: %s, line number: %d\n", __func__, __LINE__);
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

int process_work_completion_events (struct ibv_comp_channel *comp_channel, 
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
    int ret = /*ibv_poll_cq*/ host_gpu_poll_cq(cq /* the CQ, we got notification for */, 
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
  TEST_NZ(rdma_create_qp(id, s_ctx->pd, &qp_attr));
  printf("Function: %s line number: %d\n",__func__, __LINE__);
  // TEST_NZ(rdmax_create_qp(id, s_ctx->pd, &qp_attr, wqbuf, wqbuf_size));
  // printf("Function: %s line number: %d\n",__func__, __LINE__);
  
  id->context = conn = (struct connection *)malloc(sizeof(struct connection));

  conn->id = id;
  conn->qp = id->qp;

//   conn->send_state = SS_INIT;
//   conn->recv_state = RS_INIT;

  conn->connected = 0;
  printf("Function: %s line number: %d\n",__func__, __LINE__);
  register_memory(conn);
  post_receives(conn);
}

void post_receives(struct connection *conn)
{
  printf("Function: %s line number: %d\n",__func__, __LINE__);
  struct ibv_recv_wr wr, *bad_wr = NULL;
  struct ibv_sge sge;

  wr.wr_id = (uintptr_t)conn;
  wr.next = NULL;
  wr.sg_list = &sge;
  wr.num_sge = 1;

  sge.addr = (uintptr_t)conn->recv_msg;
  sge.length = sizeof(struct message);
  sge.lkey = conn->recv_mr->lkey;
  printf("Function: %s line number: %d\n",__func__, __LINE__);
  TEST_NZ(ibv_post_recv(conn->qp, &wr, &bad_wr));
}

void register_memory(struct connection *conn)
{
  conn->send_msg = (struct message *) malloc(sizeof(struct message));
  conn->recv_msg = (struct message *) malloc(sizeof(struct message));

  conn->rdma_local_region = (char *) malloc(RDMA_BUFFER_SIZE);
  conn->rdma_remote_region = (char *) malloc(RDMA_BUFFER_SIZE);

  read_after_write_buffer = (char *) malloc(RDMA_BUFFER_SIZE);

  TEST_Z(read_after_write_mr = ibv_reg_mr(
    s_ctx->pd, read_after_write_buffer, RDMA_BUFFER_SIZE,
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
  TEST_Z(s_ctx->cq = ibvx_create_cq(s_ctx->ctx, 10, NULL, s_ctx->comp_channel, 0, cqbuf, cqbuf_size)); /* cqe=10 is arbitrary */
  TEST_NZ(ibv_req_notify_cq(s_ctx->cq, 0));

 

  // TEST_NZ(pthread_create(&s_ctx->cq_poller_thread, NULL, poll_cq, NULL));
}

int on_addr_resolved(struct rdma_cm_id *id)
{
  printf("address resolved.\n");

  build_connection(id);
  sprintf((char *) get_local_message_region(id->context), "Hello my friend.  with pid %d", getpid());
  TEST_NZ(rdma_resolve_route(id, TIMEOUT_IN_MS));

  return 0;
}

int on_connection(struct rdma_cm_id *id)
{
  on_connect(id->context);
  send_mr(id->context);

  return 0;
}

int on_disconnect(struct rdma_cm_id *id)
{
  printf("disconnected.\n");

  destroy_connection(id->context);
  return 1; /* exit event loop */
}

int on_event(struct rdma_cm_event *event)
{
  int r = 0;

  if (event->event == RDMA_CM_EVENT_ADDR_RESOLVED)
    r = on_addr_resolved(event->id);
  else if (event->event == RDMA_CM_EVENT_ROUTE_RESOLVED)
    r = on_route_resolved(event->id);
  else if (event->event == RDMA_CM_EVENT_ESTABLISHED)
    r = on_connection(event->id);
  else if (event->event == RDMA_CM_EVENT_DISCONNECTED)
    r = on_disconnect(event->id);
  else {
    fprintf(stderr, "on_event: %d\n", event->event);
    die("on_event: unknown event.");
  }

  return r;
}

int on_route_resolved(struct rdma_cm_id *id)
{
  struct rdma_conn_param cm_params;

  printf("route resolved.\n");
  build_params(&cm_params);
  printf("Function: %s line number: %d\n",__func__, __LINE__);
  TEST_NZ(rdma_connect(id, &cm_params));
  printf("Function: %s line number: %d\n",__func__, __LINE__);
  return 0;
}

void usage(const char *argv0)
{
  fprintf(stderr, "usage: %s <mode> <server-address> <server-port>\n  mode = \"read\", \"write\"\n", argv0);
  exit(1);
}

__global__ void gpu_send(void *cq_buf, void *twc, void *cons_index,
                            int ibv_cqe, uint32_t cqe_sz, int max_wc, 
                            int *total_wc, int *dev_ret, void *dev_cq_dbrec,
                            void *mctx, void* dev_rsc, int refcnt, void *qp_context,
                            int dump_fill_mkey_be, void *qp_buf, void *dev_rq,
                            void *dev_wrid, uint64_t wrid, void *cqe_dev, int cond,
                            void *dev_scat_address, clock_t *timer, uint64_t wrid_0, unsigned int wqe_head_0) 
{  
	printf("Function: %s line number: %d\n",__func__, __LINE__);
    int ret;
    *total_wc = 0;
    // printf("\n\n\ngpu polling\n\n\n");
    do {
    __syncthreads();
    __threadfence();
		// timer[0] = clock();
        ret = /*ibv_poll_cq*/gpu_poll_cq(cq_buf /* the CQ, we got notification for */, 
            twc + (*total_wc)/* where to store */,
            (uint32_t *) cons_index,
            ibv_cqe,
            cqe_sz,
            max_wc - (*total_wc) /* number of remaining WC elements*/,
            (uint32_t *) dev_cq_dbrec,
            (struct mlx5_context *) mctx,
            dev_rsc, refcnt, qp_context, dump_fill_mkey_be, qp_buf, dev_rq,
            dev_wrid, wrid, cqe_dev, cond, dev_scat_address, timer, wrid_0, wqe_head_0);
    
		// timer[1] = clock();
    __syncthreads();
    __threadfence();
        // printf("ret: %d total_wc: %d\n", ret, *total_wc);
        if (ret < 0) {
            // printf("Failed to poll cq for wc due to %d \n", ret);
            continue;
            /* ret is errno here */
            // return ret;
        }
        (*total_wc) += ret;
        (*dev_ret) = ret;
    } while (*total_wc < max_wc); 
    // printf("%d WC are completed \n", *total_wc);
    
    // __syncthreads();

   
}

__device__ int gpu_poll_cq( void *cq_buf, void *twc, uint32_t *cons_index,
                            int ibv_cqe, uint32_t cqe_sz, int n, void *cq_dbrec,
                            void *mctx_t, void *dev_rsc, int refcnt,
                            void *qp_context, int dump_fill_mkey_be, void *qp_buf,
                            void *dev_rq, void *dev_wrid, uint64_t wrid_1,
                            void *cqe_dev, int cond, void *dev_scat_address,
							              clock_t *timer, uint64_t wrid_0, unsigned int wqe_head_0
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
		cqe = cq_buf + (cons_index_dev & ibv_cqe) * cqe_sz;
		cqe64 = (struct mlx5_cqe64 *)(cqe + 64*(cqe_sz != 64));
    timer[0] = clock64();
    double start = clock64();
		int cond1 = (cqe64->op_own >> 4 != MLX5_CQE_INVALID) &&
			!((cqe64->op_own & MLX5_CQE_OWNER_MASK) ^ !!(1 & (ibv_cqe + 1)));
    double end = clock64();
    printf("timer[1] - timer[0]: %f\n",end - start);
    timer[1] = clock64();
		if (cond1) {
            (*cons_index)++;

		} else {
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
			mqp = (struct mlx5_qp *) dev_rsc; 
			if ((!mqp)){
                // printf("Function: %s line number: %d\n",__func__, __LINE__);
				err = CQ_POLL_ERR;
				// break;
			  goto out1;
			}
			wq = &mqp->sq;
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
			wq->tail = wqe_head_0 + 1;// wq->wqe_head[idx] + 1;
      
    }
	else // MLX5_CQE_RESP_SEND:
      {
        printf("Function: %s line number: %d\n",__func__, __LINE__);
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
		
        	
out1:
  gpu_dbrec[0] = htonl(*cons_index & 0xffffff);
    
	// timer[1] = clock();
  return err == CQ_POLL_ERR ? err : 1;
}

int host_gpu_poll_cq (struct ibv_cq *cq_ptr, int num_entries, struct ibv_wc *wc)
{
  struct mlx5_cq *cq = to_mcq(cq_ptr);
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
  printf("qp buf address: 0x%llx\n\n\n", qp->buf.buf);
  struct mlx5_context *qp_ctx = to_mctx(qp->ibv_qp->pd->context); 
  // container_of(qp->ibv_qp->pd->context, struct mlx5_context, ibv_ctx.context);
  printf("qp_ctx address: 0x%llx\n\n\n", qp_ctx);
  // printf("Function: %s line number: %d ctx->dump_fill_mkey_be:%d\n",__func__, __LINE__, qp_ctx->dump_fill_mkey);
  printf("Function: %s line number: %d\n",__func__, __LINE__);
  //struct ibv_wc *twc = wc;
  cudaError_t crc = cudaSuccess;
  cudaError_t cudaStatus;
  //register cq in host memory
  if (!cqbuf){
    if(!registered && cudaHostRegister(cq->active_buf->buf /*cqbuf*/, sizeof(cq->active_buf->buf) /*cqbuf_size*/, cudaHostRegisterMapped) !=  cudaSuccess)
        exit(0);
    // get GPU pointer for cq
    if(cudaHostGetDevicePointer(&dev_cq_ptr, /*cqbuf*/ cq->active_buf->buf, 0) !=  cudaSuccess)
        exit(0);
  }
  printf("Function: %s line number: %d\n",__func__, __LINE__);
  // register wc in host memory 
  if(!registered && cudaHostRegister(wc, sizeof(wc), cudaHostRegisterMapped) !=  cudaSuccess)
      exit(0);
  // get GPU pointer for wc
  if(cudaHostGetDevicePointer(&dev_wc, wc, 0) != cudaSuccess)
      exit(0);
  printf("Function: %s line number: %d cq->cons_index: %d\n ",__func__, __LINE__, cq->cons_index);
  // register cons index in host memory 
  if(!registered && cudaHostRegister(&cq->cons_index, sizeof(cq->cons_index), cudaHostRegisterMapped) !=  cudaSuccess)
      exit(0);
  // get GPU pointer for cons index
  if(cudaHostGetDevicePointer(&dev_cons_index, &cq->cons_index, 0) != cudaSuccess)
      exit(0);
  printf("Function: %s line number: %d\n",__func__, __LINE__);
  // register cons index in host memory 
  if(!registered && cudaHostRegister(&total_wc, sizeof(total_wc), cudaHostRegisterMapped) !=  cudaSuccess)
      exit(0);
  // get GPU pointer for cons index
  if(cudaHostGetDevicePointer(&dev_total_wc, &total_wc, 0) != cudaSuccess)
      exit(0);
  printf("Function: %s line number: %d\n",__func__, __LINE__);
  // register ret in host memory 
  if(!registered && cudaHostRegister(&ret, sizeof(ret), cudaHostRegisterMapped) !=  cudaSuccess)
      exit(0);
  // get GPU pointer for ret
  if(cudaHostGetDevicePointer(&dev_ret, &ret, 0) != cudaSuccess)
      exit(0);
  printf("Function: %s line number: %d\n",__func__, __LINE__);
  if(!registered && cudaHostRegister(cq->dbrec, sizeof(cq->dbrec), cudaHostRegisterMapped) !=  cudaSuccess) 
      exit(0);
  if(cudaHostGetDevicePointer(&dev_cq_dbrec, cq->dbrec, 0) != cudaSuccess)
      exit(0);
  printf("Function: %s line number: %d\n",__func__, __LINE__);
  if(!registered && cudaHostRegister(mctx, sizeof(mctx), cudaHostRegisterMapped) !=  cudaSuccess) 
      exit(0);
  if(cudaHostGetDevicePointer(&dev_mctx, mctx, 0) != cudaSuccess)
      exit(0);
  printf("Function: %s line number: %d\n",__func__, __LINE__);
  if(!registered && cudaHostRegister(rsc, sizeof(rsc), cudaHostRegisterMapped) !=  cudaSuccess) 
      exit(0);
  if(cudaHostGetDevicePointer(&dev_rsc, rsc, 0) != cudaSuccess)
      exit(0);
      // qp->buf.buf
  printf("Function: %s line number: %d\n",__func__, __LINE__);
  // cudaStatus = cudaHostRegister(qp->buf.buf, sizeof(qp->buf.buf), cudaHostRegisterMapped);
  // fprintf(stderr, "cudaHostRegister(qp->buf.buf failed with error code %d: %s\n", cudaStatus, cudaGetErrorString(cudaStatus));
  if(!registered && cudaHostRegister(qp->buf.buf, sizeof(qp->buf.buf), cudaHostRegisterMapped) !=  cudaSuccess) 
      exit(0);
  if(cudaHostGetDevicePointer(&dev_qp_buf, qp->buf.buf, 0) != cudaSuccess)
      exit(0);
  printf("Function: %s line number: %d &qp->rq: 0x%llx\n",__func__, __LINE__, &qp->rq);
      // &qp->rq
  void *dev_rq, *dev_wrid;
  // printf("cudaHostRegister(&qp->rq, sizeof(qp->rq), cudaHostRegisterMapped): %d\n", cudaHostRegister(&qp->rq, sizeof(qp->rq), cudaHostRegisterMapped));
  if(!registered && cudaHostRegister(&qp->rq, sizeof(qp->rq), cudaHostRegisterMapped) !=  cudaSuccess) 
      exit(0);
  if(cudaHostGetDevicePointer(&dev_rq, &qp->rq, 0) != cudaSuccess)
      exit(0);
  struct mlx5_wq *qp_rq = &qp->rq;
  printf("Function: %s line number: %d &qp->rq: 0x%llx\n",__func__, __LINE__, &qp->rq);

  // cudaStatus = cudaHostRegister(qp_ctx, sizeof(qp_ctx), cudaHostRegisterMapped);
  // fprintf(stderr, "cudaHostRegister(qp_ctx failed with error code %d: %s\n", cudaStatus, cudaGetErrorString(cudaStatus));
  struct mlx5_wq * wq_sq = &qp->sq;
  // wq_sq->wrid[0];
  // wq_sq->wqe_head[0];
  if(!registered && cudaHostRegister(qp_rq->wrid, sizeof(qp_rq->wrid), cudaHostRegisterMapped) !=  cudaSuccess) 
      exit(0);
  printf("Function: %s line number: %d\n",__func__, __LINE__);
  if(cudaHostGetDevicePointer(&dev_wrid, qp_rq->wrid, 0) != cudaSuccess)
      exit(0);
  printf("Function: %s line number: %d\n",__func__, __LINE__);
  printf("Function: %s line number: %d\n",__func__, __LINE__);
    // if(cudaHostRegister(qp_ctx, sizeof(qp_ctx), cudaHostRegisterMapped) !=  cudaSuccess) {
    //     printf("cudaHostRegister(qp_ctx, sizeof(qp_ctx), cudaHostRegisterMapped) : %d\n", cudaHostRegister(qp_ctx, sizeof(qp_ctx), cudaHostRegisterMapped) );
    //     exit(0);
    // }
    printf("Function: %s line number: %d\n",__func__, __LINE__);
	cudaStatus = cudaHostGetDevicePointer(&dev_qp_context, qp_ctx, 0);
	fprintf(stderr, "cudaHostRegister failed with error code %d: %s\n", cudaStatus, cudaGetErrorString(cudaStatus));
    if(cudaHostGetDevicePointer(&dev_qp_context, qp_ctx, 0) != cudaSuccess)
        exit(0);
    printf("Function: %s line number: %d\n",__func__, __LINE__);

    void *cqe, *cqe_dev;
	
	if (!registered)
	  printf("Function: %s line number: %d cudaHostRegister(cqe, sizeof(cqe), cudaHostRegisterMapped): %d\n",__func__, __LINE__,
          cudaHostRegister(cqe, sizeof(cqe), cudaHostRegisterMapped));
  if (!cqbuf){
    cqe = cq->active_buf->buf + (cq->cons_index & cq->verbs_cq.cq.cqe) * cq->cqe_sz;
    if(registered && cudaHostRegister(cqe, sizeof(cqe), cudaHostRegisterMapped) !=  cudaSuccess) 
        exit(0);
    printf("Function: %s line number: %d cudaHostGetDevicePointer(&cqe_dev, cqe, 0): %d\n",__func__, __LINE__,
          cudaHostGetDevicePointer(&cqe_dev, cqe, 0));
    if(cudaHostGetDevicePointer(&cqe_dev, cqe, 0) != cudaSuccess)
        exit(0);
  }
	printf("Function: %s line number: %d\n",__func__, __LINE__);
    uint16_t	wqe_ctr;
    struct mlx5_wq *wq = (struct mlx5_wq *) &qp->rq;
    wqe_ctr = wq->tail & (wq->wqe_cnt - 1);
    struct mlx5_wqe_data_seg *scat = (struct mlx5_wqe_data_seg *)(qp->buf.buf + qp->rq.offset + (wqe_ctr << qp->rq.wqe_shift));
    void *scat_address = (void *)(unsigned long)htonl64(scat->addr);
    void *dev_scat_address; 

    // if(!registered && cudaHostRegister(scat_address, sizeof(scat_address), cudaHostRegisterMapped) !=  cudaSuccess) 
    //     exit(0);
    // if(cudaHostGetDevicePointer(&dev_scat_address, scat_address, 0) != cudaSuccess)
    //     exit(0);
	  printf("Function: %s line number: %d\n",__func__, __LINE__);
    


	int cuda_device_id = 0;
	int cuda_pci_bus_id;
	int cuda_pci_device_id;
	int index;
	CUdevice cu_device;
	CUdevice cuDevice_selected;

	printf("initializing CUDA\n");
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
	printf("Function: %s line number: %d cqe_sz: %d\n",__func__, __LINE__, cq->cqe_sz);
    gpu_send<<<1,1>>>(/*dev_cq_ptr*/ cqbuf, dev_wc, dev_cons_index, cq->verbs_cq.cq.cqe, cq->cqe_sz, num_entries, 
                    (int *) dev_total_wc, (int *) dev_ret, dev_cq_dbrec, dev_mctx, dev_rsc, mctx->uidx_table[0].refcnt,
                    dev_qp_context /*qp_ctx*/, qp_ctx->dump_fill_mkey_be, /*qp->buf.buf*/ dev_qp_buf, dev_rq, dev_wrid,
                    qp->rq.wrid[0], cqe_dev, cond, dev_scat_address, dtimer, wq_sq->wrid[0], wq_sq->wqe_head[0]);
    printf("Function: %s line number: %d\n",__func__, __LINE__);
    clock_gettime(CLOCK_REALTIME,&res);
    nano2 = res.tv_nsec;

	cudaMemcpy(timer, dtimer, sizeof(clock_t) * (2), cudaMemcpyDeviceToHost);
    cudaFree(dtimer);
	printf("Function: %s line number: %d\n",__func__, __LINE__);
    //gettimeofday(&cpu_timer[1], NULL);
    int a = cudaDeviceSynchronize();
    if (a != 0){
        printf("cudasynchronize: %d\n", a); 
        exit(0);
    }
  registered = 1;

	float freq = (float)1/(devProp.clockRate*1000);
	float g_usec = (float)((float)1/(devProp.clockRate*1000))*((timer[1]-timer[0])) * 1000000;
	printf("INTERNAL MEASUREMENT: %d useconds to execute \n", timer[1]-timer[0]);
  // float timer_usec = (cpu_timer[1].tv_nsec - cpu_timer[0].tv_usec);
  // cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
  printf("gpu_send took %lu useconds to execute \n", nano2-nano1);
  printf("Function: %s line number: %d\n",__func__, __LINE__);
  //cudaMemcpy(&host_ret, &dev_ret, sizeof(int), cudaMemcpyDeviceToHost);
  // cudaMemcpy(&total_wc, &dev_total_wc, sizeof(int), cudaMemcpyDeviceToHost);

    
    printf("Function: %s line number: %d\n", __func__, __LINE__);
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
    printf("Function: %s line number: %d\n",__func__, __LINE__);
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
    printf("Function: %s line number: %d\n",__func__, __LINE__);
    return total_wc; 
}


__device__ int gpu_post_recv(void * dev_wr,
		   unsigned int rq_head, unsigned int rq_wqe_cnt, 
		   int rq_offset, int rq_wqe_shift, uint32_t qp_flags, int wr_num_sge,
		   uint64_t wr_id, int dev_qp_type, int ibqp_state, void *dev_rq_wrid,
		   void *dev_qp_db, void *dev_wr_sg_list, void *dev_qp_buf)
{
	uint64_t *rq_wrid = (uint64_t *) dev_rq_wrid;
	unsigned int *qp_db = (unsigned int *) dev_qp_db;
	ibv_sge *wr_sg_list = (ibv_sge *) dev_wr_sg_list;
	struct ibv_recv_wr *wr = (struct ibv_recv_wr *) dev_wr;
	ibv_qp_type qp_type = (ibv_qp_type) dev_qp_type;
	// struct mlx5_qp *qp = to_mqp(ibqp);
	struct mlx5_wqe_data_seg *scat;
	int err = 0;
	int nreq;
	int ind;
	int i, j;
	struct mlx5_rwqe_sig *sig;
	printf("Function: %s line number: %d\n",__func__, __LINE__);
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
	
	
	printf("Function: %s line number: %d\n",__func__, __LINE__);
	ind = rq_head & (rq_wqe_cnt - 1);

	for (nreq = 0; nreq < 1; ++nreq) {
		
		scat = (struct mlx5_wqe_data_seg *) (dev_qp_buf + rq_offset + (ind << rq_wqe_cnt)); 
		sig = (struct mlx5_rwqe_sig *)scat;

		for (i = 0, j = 0; i < wr_num_sge; ++i) {
			if ((!wr_sg_list[i].length))
				continue;
			struct mlx5_wqe_data_seg *dseg = scat + j++;
			struct ibv_sge *sg = wr_sg_list + i;
			int offset = 0;
			dseg->byte_count = htonl(sg->length - offset);
			dseg->lkey       = htonl(sg->lkey);
			dseg->addr       = htonl64(sg->addr + offset);
		}

		rq_wrid[ind] = wr_id;

		ind = (ind + 1) & (rq_wqe_cnt - 1);
	}

out:
	if ((nreq)) {
		rq_head += nreq;

		
		if ((!((dev_qp_type == 8 ||
			      qp_flags & 1) &&
			     ibqp_state < 2)))
			*qp_db = htonl(rq_head & 0xffff);
	}

	return err;
}

__global__ void global_gpu_post_recv(void * dev_wr,
			unsigned int rq_head, unsigned int rq_wqe_cnt, 
		   int rq_offset, int rq_wqe_shift, uint32_t qp_flags, int wr_num_sge,
		   uint64_t wr_id, int dev_qp_type, int ibqp_state, void *dev_rq_wrid,
		   void *dev_qp_db, void *dev_wr_sg_list, void *dev_qp_buf, int *ret){
	printf("Function: %s line number: %d\n",__func__, __LINE__);
	int local_ret;
	local_ret = gpu_post_recv(dev_wr,
		   rq_head, rq_wqe_cnt, 
		   rq_offset, rq_wqe_shift, qp_flags, wr_num_sge,
		   wr_id, dev_qp_type, ibqp_state, dev_rq_wrid,
		   dev_qp_db, dev_wr_sg_list, dev_qp_buf);
	*ret = local_ret;

}

int host_gpu_post_recv(struct ibv_qp *ibqp, struct ibv_recv_wr *wr,
		   struct ibv_recv_wr **bad_wr){
	printf("Function: %s line number: %d\n",__func__, __LINE__);
	struct mlx5_qp *qp = to_mqp(ibqp);

	unsigned int rq_head = qp->rq.head;
	int rq_offset = qp->rq.offset;
	unsigned int rq_wqe_cnt = qp->rq.wqe_cnt;
	int rq_wqe_shift = qp->rq.wqe_shift;
	uint64_t *rq_wrid = qp->rq.wrid;
	// unsigned int *qp_db = qp->db;
	// struct ibv_sge *wr_sg_list = wr->sg_list;
	void *dev_qp_buf = qp->buf.buf;
	int wr_num_sge = wr->num_sge;
	int qp_type = (int) ibqp->qp_type;
	int ibqp_state = (int) ibqp->state;
	uint32_t qp_flags = qp->flags;
	uint64_t wr_id = wr->wr_id;
	printf("Function: %s line number: %d\n",__func__, __LINE__);
	// bad_wr = NULL;
	void *dev_rq_wrid;
	void *dev_qp_db; 
	void *dev_wr_sg_list;
	printf("Function: %s line number: %d\n",__func__, __LINE__);
	// register qp->rq.wrid in host memory 
    if(cudaHostRegister(qp->rq.wrid, sizeof(qp->rq.wrid), cudaHostRegisterMapped) !=  cudaSuccess)
        exit(0);
	printf("Function: %s line number: %d\n",__func__, __LINE__);
    // get GPU pointer for qp->rq.wrid
    if(cudaHostGetDevicePointer(&dev_rq_wrid, qp->rq.wrid, 0) != cudaSuccess)
        exit(0);
	printf("Function: %s line number: %d\n",__func__, __LINE__);
	// register qp->rq.wrid in host memory 
    if(cudaHostRegister(qp->db, sizeof(qp->db), cudaHostRegisterMapped) !=  cudaSuccess)
        exit(0);
    // get GPU pointer for qp->rq.wrid
    if(cudaHostGetDevicePointer(&dev_qp_db, qp->db, 0) != cudaSuccess)
        exit(0);
	printf("Function: %s line number: %d\n",__func__, __LINE__);
	// register qp->rq.wrid in host memory 
    if(cudaHostRegister(wr->sg_list, sizeof(wr->sg_list), cudaHostRegisterMapped) !=  cudaSuccess)
        exit(0);
    // get GPU pointer for qp->rq.wrid
    if(cudaHostGetDevicePointer(&dev_wr_sg_list, wr->sg_list, 0) != cudaSuccess)
        exit(0);
	printf("Function: %s line number: %d\n",__func__, __LINE__);
	int *ret, *dev_ret;
	// register ret in host memory 
    if(cudaHostRegister(ret, sizeof(ret), cudaHostRegisterMapped) !=  cudaSuccess)
        exit(0);
    // get GPU pointer for ret
    if(cudaHostGetDevicePointer(&dev_ret, ret, 0) != cudaSuccess)
        exit(0);
	printf("Function: %s line number: %d\n",__func__, __LINE__);
	if (cudaDeviceSynchronize() != 0) exit(0);
	global_gpu_post_recv<<<1,1>>>(NULL, rq_head, rq_wqe_cnt, 
						rq_offset, rq_wqe_shift, qp_flags, wr_num_sge,
						wr_id, qp_type, ibqp_state, dev_rq_wrid,
						dev_qp_db, dev_wr_sg_list, qp->buf.buf, dev_ret);
	
	if (cudaDeviceSynchronize() != 0) exit(0);

		// 	(void * dev_wr,
		//    struct ibv_recv_wr **bad_wr, unsigned int rq_head, unsigned int rq_wqe_cnt, 
		//    int rq_offset, int rq_wqe_shift, uint32_t qp_flags, int wr_num_sge,
		//    uint64_t wr_id, int dev_qp_type, int ibqp_state, void *dev_rq_wrid,
		//    void *dev_qp_db, void *dev_wr_sg_list, void *dev_qp_buf)

	return *ret;

}

