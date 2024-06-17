#include <infiniband/verbs.h>
#include <rdma/rdma_cma.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <stdio.h>
#include <stdlib.h>
#include <netdb.h>

#include "gpu-utils.h"

const int TIMEOUT_IN_MS = 500; /* ms */

static int on_addr_resolved(struct rdma_cm_id *id, struct context *s_ctx);
static int on_connection(struct rdma_cm_id *id);
static int on_disconnect(struct rdma_cm_id *id);
static int on_event(struct rdma_cm_event *event, struct context *s_ctx);
static int on_route_resolved(struct rdma_cm_id *id);
static void usage(const char *argv0);


// static const int RDMA_BUFFER_SIZE = 1024;

// struct message {
//   enum {
//     MSG_MR,
//     MSG_DONE
//   } type;

//   union {
//     struct ibv_mr mr;
//   } data;
// };

// struct context {
//   struct ibv_context *ctx;
//   struct ibv_pd *pd;
//   struct ibv_cq *cq;
//   struct ibv_comp_channel *comp_channel;

//   pthread_t cq_poller_thread;
// };

// struct connection {
//   struct rdma_cm_id *id;
//   struct ibv_qp *qp;

//   int connected;

//   struct ibv_mr *recv_mr;
//   struct ibv_mr *send_mr;
//   struct ibv_mr *rdma_local_mr;
//   struct ibv_mr *rdma_remote_mr;

//   struct ibv_mr peer_mr;

//   struct message *recv_msg;
//   struct message *send_msg;

//   char *rdma_local_region;
//   char *rdma_remote_region;

//   enum {
//     SS_INIT,
//     SS_MR_SENT,
//     SS_RDMA_SENT,
//     SS_DONE_SENT
//   } send_state;

//   enum {
//     RS_INIT,
//     RS_MR_RECV,
//     RS_DONE_RECV
//   } recv_state;
// };

// enum mode {
//   M_WRITE,
//   M_READ
// };


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
  printf("send message\n");
  // TEST_NZ(my_post_send(conn->qp, &wr, &bad_wr));
  // TEST_NZ(ibv_post_send(conn->qp, &wr, &bad_wr));
}

void send_mr(void *context)
{
  struct connection *conn = (struct connection *)context;

  conn->send_msg->type = MSG_MR;
  memcpy(&conn->send_msg->data.mr, conn->rdma_remote_mr, sizeof(struct ibv_mr));

  send_message(conn);
}

int process_cm_event(struct rdma_event_channel *cm_channel,
                     enum rdma_cm_event_type expected_event,
                     struct rdma_cm_event **cm_event,
                     struct rdma_cm_event *copy_event);

int process_work_completion_events (struct ibv_comp_channel *comp_channel, 
		struct ibv_wc *wc, int max_wc, struct context *s_ctx);

char *read_after_write_buffer = NULL;
struct ibv_mr *read_after_write_mr;

int get_myglid(struct ibv_context *ctx, int port, union ibv_gid *gid1, int *gid_entry1) {
    
    struct ibv_port_attr attr;
    TEST_NZ(ibv_query_port(ctx, port, &attr));
    int gid_tlb_len = attr.gid_tbl_len;
    union ibv_gid gid;
    int gid_entry;
    for (int i = 0; i < gid_tlb_len; i++)
    {
        if (ibv_query_gid(ctx, port, i, &gid) == 0)
        {
            printf("found at index %d \n", i);
            gid_entry = i;
            break;
        }
    }
    
    for (int i = 0; i < 16; i++)
    {
        printf("%d: ", gid.raw[i]);
    }
    *gid1 = gid;
    *gid_entry1 = gid_entry;
    printf("the entry is %d\n", gid_entry);

    return 0;
}

int get_mylid(struct ibv_context *context, int port, int *lid) {
    struct ibv_port_attr attr;
    TEST_NZ(ibv_query_port(context, port, &attr));
    *lid = attr.lid;
    return 0;
}

int main(int argc, char **argv)
{
  static struct context *s_ctx = NULL;
  s_ctx = (struct context *)malloc(sizeof(struct context));
  static struct context *s_ctx2 = NULL;
  s_ctx2 = (struct context *)malloc(sizeof(struct context));
  struct addrinfo *addr;
  struct rdma_cm_event *event = NULL;
  struct rdma_cm_event event_copy;
  struct rdma_cm_event *event2 = NULL;
  struct rdma_cm_event event_copy2;
  struct rdma_cm_id *conn_id = NULL;
  struct rdma_cm_id *conn_id2 = NULL;
  struct rdma_event_channel *ec = NULL;
  struct rdma_event_channel *ec2 = NULL;
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

  if (process_cm_event(ec, RDMA_CM_EVENT_ADDR_RESOLVED, &event, &event_copy))
    return -1;
  
  

  if (on_addr_resolved(event_copy.id, s_ctx)){
    printf("error on on_addr_resolved\n");
    return -1;
  }
  int ret;
  if (process_cm_event(ec, RDMA_CM_EVENT_ROUTE_RESOLVED, &event, &event_copy))
    return -1;

  if (on_route_resolved(event_copy.id)){
    printf("error on on_route_resolved!\n");
    return -1;
  }
  printf("Function name: %s, line number: %d\n", __func__, __LINE__);

  if (process_cm_event(ec, RDMA_CM_EVENT_ESTABLISHED, &event, &event_copy))
    return -1;
  printf("Function name: %s, line number: %d\n", __func__, __LINE__);
  if (on_connection(event_copy.id)){
    printf("error on on_connection!\n");
    return -1;
  }

  struct ibv_wc wc;
  // cpu_process_work_completion_events (s_ctx->comp_channel, &wc, 1, s_ctx);
  printf("Function name: %s, line number: %d\n", __func__, __LINE__);
  cpu_process_work_completion_events (s_ctx->comp_channel, &wc, 1, s_ctx);
  struct connection *conn = (struct connection *)(uintptr_t)wc.wr_id;
  if (wc.opcode & IBV_WC_RECV){
    printf("receive completed\n");
    memcpy(&conn->peer_mr, &conn->recv_msg->data.mr, sizeof(conn->peer_mr));
    // post_receives(conn);
    printf("Function name: %s, line number: %d\n", __func__, __LINE__);
  }

  // post for recv of qp info

  struct remote_qp_info{
    uint32_t target_qp_num[2];
    uint16_t target_lid;
    union ibv_gid target_gid;
  }; 

  struct remote_qp_info buffer; // = (struct remote_qp_info **) calloc(2, sizeof(struct remote_qp_info *));
  // for(int i = 0; i < 2; i++){
  //   buffer[i] = (struct remote_qp_info *) malloc(sizeof(struct remote_qp_info));
  // }
  struct ibv_mr *my_mr;
  TEST_Z(my_mr = ibv_reg_mr(
    s_ctx->pd, 
    &buffer, 
    sizeof(buffer), 
    IBV_ACCESS_LOCAL_WRITE));
  struct ibv_recv_wr wr1, *bad_wr1 = NULL;
  struct ibv_sge sge1;

  wr1.wr_id = (uintptr_t)conn;
  wr1.next = NULL;
  wr1.sg_list = &sge1;
  wr1.num_sge = 1;

  sge1.addr = (uintptr_t)&buffer;
  sge1.length = sizeof(buffer);
  sge1.lkey = my_mr->lkey;

  TEST_NZ(ibv_post_recv(conn->qp, &wr1, &bad_wr1));
  // post_receives(conn);
  cpu_process_work_completion_events (s_ctx->comp_channel, &wc, 1, s_ctx);

  if (wc.opcode & IBV_WC_RECV){
    printf("receive completed\n");
    // memcpy(&conn->peer_mr, &conn->recv_msg->data.mr, sizeof(conn->peer_mr));
    // post_receives(conn);
    printf("Function name: %s, line number: %d\n", __func__, __LINE__);
    printf("Function name: %s, line number: %d conn->peer_mr: %d buffer->target_lid: %d\n", __func__, __LINE__, buffer.target_qp_num[0], buffer.target_lid);
    // exit(0);
  }

  // we received server side info for qp
  // now we need to send the qp info from our side

  struct ibv_qp **g_qp = (struct ibv_qp **) calloc(2, sizeof(struct ibv_qp *));
  for(int i = 0; i < 2; i++){
    struct ibv_qp_init_attr qp_attr;
    memset(&qp_attr, 0, sizeof(qp_attr));
    printf("Function name: %s, line number: %d\n", __func__, __LINE__);
    qp_attr.send_cq = s_ctx->cq;
    qp_attr.recv_cq = s_ctx->cq;
    qp_attr.qp_type = IBV_QPT_RC;
    printf("Function name: %s, line number: %d\n", __func__, __LINE__);
    qp_attr.cap.max_send_wr = 10;
    qp_attr.cap.max_recv_wr = 10;
    qp_attr.cap.max_send_sge = 1;
    qp_attr.cap.max_recv_sge = 1;
    TEST_Z(g_qp[i] = ibv_create_qp(s_ctx->pd, &qp_attr));
    if (!g_qp[i]){
      printf("g_qp failed\n");
      exit(-1);
    }
  }

  union ibv_gid gid;; 
  int gid_entry;
  get_myglid(s_ctx->ctx, 1, &gid, &gid_entry);
   int mylid;
  get_mylid(s_ctx->ctx, 1, &mylid);

  struct remote_qp_info buffer1; // = (struct remote_qp_info **) calloc(2, sizeof(struct remote_qp_info *));
  for (int i = 0; i < 2; i++){
    buffer1.target_qp_num[i] = g_qp[i]->qp_num;
    // printf("client qp num: %d\n", g_qp->qp_num);
    
  }

  buffer1.target_gid = gid;
  buffer1.target_lid = mylid;

  struct ibv_mr *my_mr1;
  TEST_Z(my_mr1 = ibv_reg_mr(
    s_ctx->pd, 
    &buffer1, 
    sizeof(buffer1), 
    IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ));
  struct ibv_send_wr wr2, *bad_wr2 = NULL;
  struct ibv_sge sge2;
  memset(&wr2, 0, sizeof(wr2));
  wr2.wr_id = (uintptr_t)conn;
  wr2.opcode = IBV_WR_SEND;
  wr2.next = NULL;
  wr2.sg_list = &sge2;
  wr2.num_sge = 1;
  wr2.send_flags = IBV_SEND_SIGNALED;

  sge2.addr = (uintptr_t)&buffer1;
  sge2.length = sizeof(buffer1);
  sge2.lkey = my_mr1->lkey;

  TEST_NZ(ibv_post_send(conn->qp, &wr2, &bad_wr2));

  cpu_process_work_completion_events (s_ctx->comp_channel, &wc, 1, s_ctx);

  if (wc.opcode == IBV_WC_SEND){
    printf("QP info sent; qp_num: %d\n", buffer1.target_qp_num[0]);
  }
  // exit(0);

  for(int i = 0; i < 2; i++){
    struct ibv_qp_attr qp_attr1;
    memset(&qp_attr1, 0, sizeof(qp_attr1));

    qp_attr1.qp_state        = IBV_QPS_INIT;
    qp_attr1.pkey_index      = 0;
    qp_attr1.port_num = 1; // IB_PORT;
    qp_attr1.qp_access_flags = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ | IBV_ACCESS_REMOTE_ATOMIC | IBV_ACCESS_REMOTE_WRITE;

    ret = ibv_modify_qp (g_qp[i], &qp_attr1, IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT  | IBV_QP_ACCESS_FLAGS); 
    if(ret != 0)
    {
      printf("Failed to modify qp to INIT. ret: %d\n", ret);
      exit(-1);
    }
    

    memset(&qp_attr1, 0, sizeof(qp_attr1));
    qp_attr1.qp_state = IBV_QPS_RTR;
    qp_attr1.path_mtu = IBV_MTU_4096;
    qp_attr1.dest_qp_num = buffer.target_qp_num[i];
    qp_attr1.rq_psn = 0;

    qp_attr1.min_rnr_timer = 12;
    qp_attr1.ah_attr.is_global = 1;
    qp_attr1.ah_attr.dlid = buffer.target_lid;
    qp_attr1.ah_attr.sl = 0;
    qp_attr1.max_dest_rd_atomic = 1;
    qp_attr1.ah_attr.port_num = 1;
    qp_attr1.ah_attr.grh.dgid = buffer.target_gid;
    qp_attr1.ah_attr.grh.sgid_index = gid_entry;
    qp_attr1.ah_attr.grh.hop_limit = 0xFF;
    qp_attr1.ah_attr.grh.traffic_class = 0;

    ret = ibv_modify_qp(g_qp[i], &qp_attr1, IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU | IBV_QP_DEST_QPN | IBV_QP_RQ_PSN | IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER);
    if(ret != 0) {
      printf("Failed to modify qp to RTR. ret: %d\n", ret);
      exit(-1);
    }

    struct ibv_qp_attr qp_attr2 = {
              .qp_state = IBV_QPS_RTS,
              .timeout = 14,
              .retry_cnt = 7,
              .rnr_retry = 7,
              .sq_psn = 0,
              .max_rd_atomic = 1,
          };

    ret = ibv_modify_qp(g_qp[i], &qp_attr2,
                        IBV_QP_STATE | IBV_QP_TIMEOUT |
                            IBV_QP_RETRY_CNT | IBV_QP_RNR_RETRY |
                            IBV_QP_SQ_PSN | IBV_QP_MAX_QP_RD_ATOMIC);
                      
    // ret = ibv_modify_qp(g_qp, &qp_attr2, IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU | IBV_QP_DEST_QPN | IBV_QP_RQ_PSN | IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER);
    if(ret != 0) {
      printf("Failed to modify qp to RTS. ret: %d\n", ret);
      exit(-1);
    }
  }





  /*Post some requests*/
  // read request:
  // conn = (struct connection *)(uintptr_t)wc.wr_id;
  struct ibv_send_wr wr, *bad_wr = NULL;
  struct ibv_sge sge;
  printf("Function name: %s, line number: %d\n", __func__, __LINE__);
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
  printf("Function name: %s, line number: %d\n", __func__, __LINE__);
  struct ibv_wc wc1;
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
  printf("Function name: %s, line number: %d\n", __func__, __LINE__);
  // ret = ibv_post_send(g_qp, &wr, &bad_wr);
  // printf("ret: %d\n", ret);
  // exit(0);
  TEST_NZ(ibv_post_send(conn->qp, &wr, &bad_wr));
  cpu_process_work_completion_events (s_ctx->comp_channel, &wc1, 1, s_ctx);
  printf("read remote buffer: %s\n", conn->rdma_local_region);
  printf("Function name: %s, line number: %d\n", __func__, __LINE__);
  // exit(0);


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
  ret = ibv_post_send(g_qp[0], &wr, &bad_wr);
  printf("ret: %d\n", ret);
  
  // TEST_NZ(ret = ibv_post_send(conn->qp, &wr, &bad_wr));
  cpu_process_work_completion_events (s_ctx->comp_channel, &wc, 1, s_ctx);
  printf("write to remote buffer: %s\n", conn->rdma_remote_region);
  printf("Function name: %s, line number: %d\n", __func__, __LINE__);
  // exit(0);


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
  ret = ibv_post_send(g_qp[1], &wr, &bad_wr);
  printf("ret: %d\n", ret);
  // TEST_NZ(my_post_send(conn->qp, &wr, &bad_wr));
  cpu_process_work_completion_events (s_ctx->comp_channel, &wc, 1, s_ctx);
  printf("read remote buffer: %s\n", read_after_write_buffer);
  printf("Function name: %s, line number: %d\n", __func__, __LINE__);
  // exit(0);

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
  ret = ibv_post_send(g_qp[0], &wr, &bad_wr);
  printf("ret: %d\n", ret);
  // TEST_NZ(my_post_send(conn->qp, &wr, &bad_wr));
  cpu_process_work_completion_events (s_ctx->comp_channel, &wc, 1, s_ctx);
  printf("write to remote buffer: %s\n", conn->rdma_remote_region);
  printf("Function name: %s, line number: %d\n", __func__, __LINE__);
  // exit(0);


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
  // printf("ibv_post_send(conn->qp, &wr, &bad_wr): %d\n", ibv_post_send(conn->qp, &wr, &bad_wr));
  // ret = ibv_post_send(g_qp, &wr, &bad_wr);
  // printf("ret: %d\n", ret);
  TEST_NZ(my_post_send(conn->qp, &wr, &bad_wr));
  printf("polling... \n");
  cpu_process_work_completion_events (s_ctx->comp_channel, &wc, 1, s_ctx);
  printf("read remote buffer: %s\n", read_after_write_buffer);
  printf("final: Function name: %s, line number: %d\n", __func__, __LINE__);
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
                     struct rdma_cm_event *copy_event)
{

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
		struct ibv_wc *wc, int max_wc, struct context *s_ctx){
  
  struct ibv_cq *cq;
  void *context = NULL;
  // // printf("Function name: %s, line number: %d\n", __func__, __LINE__);
  // TEST_NZ(ibv_get_cq_event(comp_channel, &cq, &context));
  // // printf("Function name: %s, line number: %d\n", __func__, __LINE__);
  // ibv_ack_cq_events(cq, 1);
  // // printf("Function name: %s, line number: %d\n", __func__, __LINE__);
  // TEST_NZ(ibv_req_notify_cq(cq, 0));
  // // printf("Function name: %s, line number: %d\n", __func__, __LINE__);
  // // printf("Function name: %s, line number: %d\n", __func__, __LINE__);
  // // while (ibv_poll_cq(cq, 1, &wc))
  // //     on_completion(&wc);

  int total_wc = 0;
  do {
    int ret = ibv_poll_cq(s_ctx->cq /* the CQ, we got notification for */, 
      1 - total_wc /* number of remaining WC elements*/,
      wc + total_wc/* where to store */);
    printf("ret: %d\n", ret);
    if (ret < 0) {
      printf("Failed to poll cq for wc due to %d \n", ret);
      continue;
    /* ret is errno here */
      // return ret;
    }
    printf("polling\n");
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

int cpu_process_work_completion_events (struct ibv_comp_channel *comp_channel, 
		struct ibv_wc *wc, int max_wc, struct context *s_ctx){
  
  // struct ibv_cq *cq;
  // void *context = NULL;
  // printf("My function name: %s, line number: %d\n", __func__, __LINE__);
  // TEST_NZ(ibv_get_cq_event(comp_channel, &cq, &context));
  // printf("My function name: %s, line number: %d\n", __func__, __LINE__);
  // ibv_ack_cq_events(cq, 1);
  // printf("My function name: %s, line number: %d\n", __func__, __LINE__);
  // TEST_NZ(ibv_req_notify_cq(cq, 0));
  // printf("My function name: %s, line number: %d\n", __func__, __LINE__);
  // printf("Function name: %s, line number: %d\n", __func__, __LINE__);
  // while (ibv_poll_cq(cq, 1, &wc))
  //     on_completion(&wc);
  printf("polling starts here\n");
  int total_wc = 0;
  do {
    int ret = /*cpu_poll_cq*/cpu_poll_cq(s_ctx->cq /* the CQ, we got notification for */, 
      1 - total_wc /* number of remaining WC elements*/,
      wc + total_wc/* where to store */);

    if (ret < 0) {
      printf("Failed to poll cq for wc due to %d \n", ret);
      continue;
    /* ret is errno here */
      // return ret;
    }
    // printf("polling\n");
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

void build_connection(struct rdma_cm_id *id, struct context *s_ctx)
{
  struct connection *conn;
  struct ibv_qp_init_attr qp_attr;
  printf("Function name: %s, line number: %d\n", __func__, __LINE__);
  build_context(id->verbs, s_ctx);
  printf("Function name: %s, line number: %d\n", __func__, __LINE__);
  build_qp_attr(&qp_attr, s_ctx);
  // build_qp_attr(&qp_attr2);
  printf("Function name: %s, line number: %d\n", __func__, __LINE__);
  TEST_NZ(rdma_create_qp(id, s_ctx->pd, &qp_attr));
  // TEST_Z(g_qp = ibv_create_qp(s_ctx->pd, &qp_attr));
  
  id->context = conn = (struct connection *)malloc(sizeof(struct connection));
  printf("address resolved.\n");
  conn->id = id;
  conn->qp = id->qp;

  // conn->send_state = SS_INIT;
  // conn->recv_state = RS_INIT;

  conn->connected = 0;
  printf("address resolved.\n");
  register_memory(conn,s_ctx);
  post_receives(conn);
}

void post_receives(struct connection *conn)
{
  struct ibv_recv_wr wr, *bad_wr = NULL;
  struct ibv_sge sge;

  wr.wr_id = (uintptr_t)conn;
  wr.next = NULL;
  wr.sg_list = &sge;
  wr.num_sge = 1;

  sge.addr = (uintptr_t)conn->recv_msg;
  sge.length = sizeof(struct message);
  sge.lkey = conn->recv_mr->lkey;

  TEST_NZ(ibv_post_recv(conn->qp, &wr, &bad_wr));
}

void register_memory(struct connection *conn, struct context *s_ctx)
{
  conn->send_msg = malloc(sizeof(struct message));
  conn->recv_msg = malloc(sizeof(struct message));

  // conn->rdma_local_region = malloc(RDMA_BUFFER_SIZE);
  // conn->rdma_remote_region = malloc(RDMA_BUFFER_SIZE);

  read_after_write_buffer = malloc(RDMA_BUFFER_SIZE);

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

  // TEST_Z(conn->rdma_local_mr = ibv_reg_mr(
  //   s_ctx->pd, 
  //   conn->rdma_local_region, 
  //   RDMA_BUFFER_SIZE, 
  //   ((s_mode == M_WRITE) ? 0 :  IBV_ACCESS_LOCAL_WRITE)));

  // TEST_Z(conn->rdma_remote_mr = ibv_reg_mr(
  //   s_ctx->pd, 
  //   conn->rdma_remote_region, 
  //   RDMA_BUFFER_SIZE, 
  //   IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ));
  //   // ((s_mode == M_WRITE) ? (IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE) : IBV_ACCESS_REMOTE_READ)));
}

void build_qp_attr(struct ibv_qp_init_attr *qp_attr, struct context *s_ctx)
{
  memset(qp_attr, 0, sizeof(*qp_attr));
  printf("Function name: %s, line number: %d\n", __func__, __LINE__);
  qp_attr->send_cq = s_ctx->cq;
  qp_attr->recv_cq = s_ctx->cq;
  qp_attr->qp_type = IBV_QPT_RC;
  printf("Function name: %s, line number: %d\n", __func__, __LINE__);
  qp_attr->cap.max_send_wr = 10;
  qp_attr->cap.max_recv_wr = 10;
  qp_attr->cap.max_send_sge = 1;
  qp_attr->cap.max_recv_sge = 1;
}

void build_context(struct ibv_context *verbs, struct context *s_ctx)
{
  // if (s_ctx) {
  //   if (s_ctx->ctx != verbs)
  //     die("cannot handle events in more than one context.");

  //   return;
  // }
  printf("Function name: %s, line number: %d\n", __func__, __LINE__);
  

  s_ctx->ctx = verbs;
  printf("Function name: %s, line number: %d\n", __func__, __LINE__);
  TEST_Z(s_ctx->pd = ibv_alloc_pd(s_ctx->ctx));
  TEST_Z(s_ctx->comp_channel = ibv_create_comp_channel(s_ctx->ctx));
  TEST_Z(s_ctx->cq = ibv_create_cq(s_ctx->ctx, 10, NULL, s_ctx->comp_channel, 0)); /* cqe=10 is arbitrary */
  TEST_NZ(ibv_req_notify_cq(s_ctx->cq, 0));
  printf("Function name: %s, line number: %d s_ctx->cq: 0x%llx\n", __func__, __LINE__, s_ctx->cq);
 

  // TEST_NZ(pthread_create(&s_ctx->cq_poller_thread, NULL, poll_cq, NULL));
}

int on_addr_resolved(struct rdma_cm_id *id, struct context *s_ctx)
{
  printf("address resolved.\n");

  build_connection(id, s_ctx);
  // sprintf(get_local_message_region(id->context), "Hello my friend.  with pid %d", getpid());
  TEST_NZ(rdma_resolve_route(id, TIMEOUT_IN_MS));

  return 0;
}

int on_connection(struct rdma_cm_id *id)
{
  on_connect(id->context);
  printf("Function name: %s, line number: %d\n", __func__, __LINE__);
  // send_mr(id->context);

  return 0;
}

int on_disconnect(struct rdma_cm_id *id)
{
  printf("disconnected.\n");

  destroy_connection(id->context);
  return 1; /* exit event loop */
}

int on_event(struct rdma_cm_event *event, struct context *s_ctx)
{
  int r = 0;

  if (event->event == RDMA_CM_EVENT_ADDR_RESOLVED)
    r = on_addr_resolved(event->id, s_ctx);
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
  TEST_NZ(rdma_connect(id, &cm_params));

  return 0;
}

void usage(const char *argv0)
{
  fprintf(stderr, "usage: %s <mode> <server-address> <server-port>\n  mode = \"read\", \"write\"\n", argv0);
  exit(1);
}
