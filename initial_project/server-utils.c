#include "server-utils.h"



struct message {
  enum {
    MSG_MR,
    MSG_DONE
  } type;

  union {
    struct ibv_mr mr;
  } data;
};

struct context {
  struct ibv_context *ctx;
  struct ibv_pd *pd;
  struct ibv_cq *cq;
  struct ibv_comp_channel *comp_channel;

  pthread_t cq_poller_thread;
};

struct connection {
  struct rdma_cm_id *id;
  struct ibv_qp *qp;

  int connected;

  struct ibv_mr *recv_mr;
  struct ibv_mr *send_mr;
  struct ibv_mr *rdma_local_mr;
  struct ibv_mr *rdma_remote_mr;

  struct ibv_mr peer_mr;

  struct message *recv_msg;
  struct message *send_msg;

  char *rdma_local_region;
  int *rdma_remote_region;

  enum {
    SS_INIT,
    SS_MR_SENT,
    SS_RDMA_SENT,
    SS_DONE_SENT
  } send_state;

  enum {
    RS_INIT,
    RS_MR_RECV,
    RS_DONE_RECV
  } recv_state;
};

static void build_context(struct ibv_context *verbs);
static void build_qp_attr(struct ibv_qp_init_attr *qp_attr);
static char * get_peer_message_region(struct connection *conn);
static void on_completion(struct ibv_wc *);
static void * poll_cq(void *);
static void post_receives(struct connection *conn);
static void register_memory(struct connection *conn);
static void send_message(struct connection *conn);

static struct context *s_ctx = NULL;
static enum mode s_mode = M_WRITE;

void die(const char *reason)
{
  fprintf(stderr, "%s\n", reason);
  exit(EXIT_FAILURE);
}

void build_connection(struct rdma_cm_id *id)
{
  struct connection *conn;
  struct ibv_qp_init_attr qp_attr;

  build_context(id->verbs);
  build_qp_attr(&qp_attr);
  // printf("rdma_create_qp(id, s_ctx->pd, &qp_attr): %d\n", rdma_create_qp(id, s_ctx->pd, &qp_attr));
  TEST_NZ(rdma_create_qp(id, s_ctx->pd, &qp_attr));

  id->context = conn = (struct connection *)malloc(sizeof(struct connection));

  conn->id = id;
  conn->qp = id->qp;

  conn->send_state = SS_INIT;
  conn->recv_state = RS_INIT;

  conn->connected = 0;

  register_memory(conn);
  // post_receives(conn);
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
  TEST_Z(s_ctx->cq = ibv_create_cq(s_ctx->ctx, 10, NULL, s_ctx->comp_channel, 0)); /* cqe=10 is arbitrary */
  TEST_NZ(ibv_req_notify_cq(s_ctx->cq, 0));

  TEST_NZ(pthread_create(&s_ctx->cq_poller_thread, NULL, poll_cq, NULL));
}

void build_params(struct rdma_conn_param *params)
{
  memset(params, 0, sizeof(*params));

  params->initiator_depth = params->responder_resources = 1;
  params->rnr_retry_count = 7; /* infinite retry */
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
int mr_sent = 0;
void destroy_connection(void *context)
{
  mr_sent = 0;
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

void * get_local_message_region(void *context)
{
  if (s_mode == M_WRITE)
    return ((struct connection *)context)->rdma_local_region;
  else
    return ((struct connection *)context)->rdma_remote_region;
}

char * get_peer_message_region(struct connection *conn)
{
  if (s_mode == M_WRITE)
    return conn->rdma_remote_region;
  else
    return conn->rdma_local_region;
}

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


int cpu_process_work_completion_events (struct ibv_comp_channel *comp_channel, 
		struct ibv_wc *wc, int max_wc, struct context *s_ctx){

  printf("polling starts here\n");
  int total_wc = 0;
  do {
    int ret = /*cpu_poll_cq*/ibv_poll_cq(s_ctx->cq /* the CQ, we got notification for */, 
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
  if (wc->opcode & IBV_WC_RECV){
    printf("receive completed\n");
  }
  if(wc->opcode == IBV_WC_SEND){
    printf("send completed\n");
  }
  else printf("wc.opcode: %d\n", wc->opcode);
  return 0;
}

int a = 8;
struct remote_qp_info{
  uint32_t target_qp_num[2];
  uint16_t target_lid;
  union ibv_gid target_gid;
};
struct ibv_qp **g_qp;

void send_qp_info(void *context){

  struct connection *conn = (struct connection *)context;
  

  struct ibv_wc wc;
  
  
  printf("Function name: %s, line number: %d\n", __func__, __LINE__);
  g_qp = (struct ibv_qp **) calloc(N_QPs, sizeof(struct ibv_qp *));
  for(int i = 0; i < N_QPs; i++){
    struct ibv_qp_init_attr qp_attr;
    memset(&qp_attr, 0, sizeof(qp_attr));
    qp_attr.send_cq = s_ctx->cq;
    qp_attr.recv_cq = s_ctx->cq;
    qp_attr.qp_type = IBV_QPT_RC;
    qp_attr.cap.max_send_wr = 10;
    qp_attr.cap.max_recv_wr = 10;
    qp_attr.cap.max_send_sge = 1;
    qp_attr.cap.max_recv_sge = 1;
    TEST_Z(g_qp[i] = ibv_create_qp(s_ctx->pd, &qp_attr));
  }
  // TEST_Z(g_qp = ibv_create_qp(s_ctx->pd, &qp_attr));
  if (!g_qp){
    printf("g_qp failed\n");
    exit(-1);
  }
  printf("Function name: %s, line number: %d\n", __func__, __LINE__);
  struct remote_qp_info buffer; // = (struct remote_qp_info **) calloc(N_QPs, sizeof(struct remote_qp_info *));
  
  union ibv_gid gid;; 
  int gid_entry;
  int mylid;
  get_mylid(s_ctx->ctx, 1, &mylid);
  get_myglid(s_ctx->ctx, 1, &gid, &gid_entry);
  for(int i = 0; i < N_QPs; i++){
    // TEST_Z(buffer[i] = malloc(sizeof(struct remote_qp_info)));
    buffer.target_qp_num[i] = g_qp[i]->qp_num;
    
  }
  buffer.target_gid = gid;
  buffer.target_lid = mylid;

  // int f = g_qp->qp_num;
  // printf("need to send qp info! wc->opcode: %d g_qp->qp_num: %d\n", wc->opcode, f);

  // buffer->target_qp_num = g_qp->qp_num;
  struct ibv_mr *my_mr;
  TEST_Z(my_mr = ibv_reg_mr(
  s_ctx->pd, &buffer, 
  sizeof(buffer), 
  IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ));
  
  // buffer[0] = buffer[1] = a;
  printf("need to send qp info! g_qp->qp_num: %d\n", buffer.target_qp_num[0]);
  struct ibv_send_wr wr, *bad_wr = NULL;
  struct ibv_sge sge;

  memset(&wr, 0, sizeof(wr));

  wr.wr_id = (uintptr_t)conn;
  wr.opcode = IBV_WR_SEND;
  wr.sg_list = &sge;
  wr.num_sge = 1;
  wr.send_flags = IBV_SEND_SIGNALED;

  sge.addr = (uintptr_t)&buffer;
  sge.length = sizeof(buffer);
  sge.lkey = my_mr->lkey;

  // while (!conn->connected);

  TEST_NZ(ibv_post_send(conn->qp, &wr, &bad_wr));
  // a++;
  // exit(0);
  // struct ibv_wc wc;
  cpu_process_work_completion_events (s_ctx->comp_channel, &wc, 1, s_ctx);

  /********************************************************************/
  /*receive the other side's information*/
  printf("Function name: %s, line number: %d\n", __func__, __LINE__);
  struct remote_qp_info buffer1; // = (struct remote_qp_info **) calloc(N_QPs, sizeof(struct remote_qp_info *));
  struct ibv_mr *my_mr1;
  TEST_Z(my_mr1 = ibv_reg_mr(
    s_ctx->pd, 
    &buffer1, 
    sizeof(buffer1), 
    IBV_ACCESS_LOCAL_WRITE));
  struct ibv_recv_wr wr1, *bad_wr1 = NULL;
  struct ibv_sge sge1;
  printf("Function name: %s, line number: %d\n", __func__, __LINE__);
  memset(&wr1, 0, sizeof(wr1));
  wr1.wr_id = (uintptr_t)conn;
  wr1.next = NULL;
  wr1.sg_list = &sge1;
  wr1.num_sge = 1;
  printf("Function name: %s, line number: %d\n", __func__, __LINE__);
  sge1.addr = (uintptr_t)&buffer1;
  sge1.length = sizeof(buffer1);
  sge1.lkey = my_mr1->lkey;
  printf("Function name: %s, line number: %d\n", __func__, __LINE__);
  TEST_NZ(ibv_post_recv(conn->qp, &wr1, &bad_wr1));
  // post_receives(conn);
  cpu_process_work_completion_events (s_ctx->comp_channel, &wc, 1, s_ctx);

  if (wc.opcode & IBV_WC_RECV){
    printf("receive completed\n");
    // memcpy(&conn->peer_mr, &conn->recv_msg->data.mr, sizeof(conn->peer_mr));
    // post_receives(conn);
    printf("Function name: %s, line number: %d\n", __func__, __LINE__);
    printf("Function name: %s, line number: %d conn->peer_mr: %d buffer->target_lid: %d\n", __func__, __LINE__, buffer1.target_qp_num[0], buffer1.target_lid);
    // exit(0);
  }

  /*modify qp to init*/
  for(int i = 0; i < N_QPs; i++){
    struct ibv_qp_attr qp_attr1;
    memset(&qp_attr1, 0, sizeof(qp_attr1));

    qp_attr1.qp_state        = IBV_QPS_INIT;
    qp_attr1.pkey_index      = 0;
    qp_attr1.port_num = 1; // IB_PORT;
    qp_attr1.qp_access_flags = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ | IBV_ACCESS_REMOTE_ATOMIC | IBV_ACCESS_REMOTE_WRITE;

    int ret = ibv_modify_qp (g_qp[i], &qp_attr1, IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT  | IBV_QP_ACCESS_FLAGS); 
    if(ret != 0)
    {
      printf("Failed to modify qp to INIT. ret: %d\n", ret);
      exit(-1);
    }
    
    /*modify qp to rtr*/
    memset(&qp_attr1, 0, sizeof(qp_attr1));
    qp_attr1.qp_state = IBV_QPS_RTR;
    qp_attr1.path_mtu = IBV_MTU_4096;
    qp_attr1.dest_qp_num = buffer1.target_qp_num[i];
    qp_attr1.rq_psn = 0;

    qp_attr1.min_rnr_timer = 12;
    qp_attr1.ah_attr.is_global = 1;
    qp_attr1.ah_attr.dlid = buffer1.target_lid;
    qp_attr1.ah_attr.sl = 0;
    qp_attr1.max_dest_rd_atomic = 1;
    qp_attr1.ah_attr.port_num = 1;
    qp_attr1.ah_attr.grh.dgid = buffer1.target_gid;
    qp_attr1.ah_attr.grh.sgid_index = gid_entry;
    qp_attr1.ah_attr.grh.hop_limit = 0xFF;
    qp_attr1.ah_attr.grh.traffic_class = 0;

    ret = ibv_modify_qp(g_qp[i], &qp_attr1, IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU | IBV_QP_DEST_QPN | IBV_QP_RQ_PSN | IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER);
    if(ret != 0) {
      printf("Failed to modify qp to RTS. ret: %d\n", ret);
      exit(-1);
    }

    // struct ibv_qp_attr qp_attr2 = {
    //           .qp_state = IBV_QPS_RTS,
    //           .timeout = 14,
    //           .retry_cnt = 7,
    //           .rnr_retry = 7,
    //           .sq_psn = 0,
    //           .max_rd_atomic = 1,
    //       };

    // ret = ibv_modify_qp(g_qp, &qp_attr2,
    //                     IBV_QP_STATE | IBV_QP_TIMEOUT |
    //                         IBV_QP_RETRY_CNT | IBV_QP_RNR_RETRY |
    //                         IBV_QP_SQ_PSN | IBV_QP_MAX_QP_RD_ATOMIC);
                      
    // // ret = ibv_modify_qp(g_qp, &qp_attr2, IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU | IBV_QP_DEST_QPN | IBV_QP_RQ_PSN | IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER);
    // if(ret != 0) {
    //   printf("Failed to modify qp to RTS. ret: %d\n", ret);
    //   exit(-1);
    // }
  }

}


void on_completion(struct ibv_wc *wc)
{
  struct connection *conn = (struct connection *)(uintptr_t)wc->wr_id;
  
  if (wc->status != IBV_WC_SUCCESS)
    die("on_completion: status is not IBV_WC_SUCCESS.");

  // printf("need to send qp info! wc->opcode: %d\n", wc->opcode);

  // if ((wc->opcode & IBV_WC_RECV) && !mr_sent) {
  //   conn->recv_state++;
  //   mr_sent = 1;
  //   // if (conn->recv_msg->type == MSG_MR) {
  //   memcpy(&conn->peer_mr, &conn->recv_msg->data.mr, sizeof(conn->peer_mr));
  //   // post_receives(conn); /* only rearm for MSG_MR */
  //   printf("received peer's MR ");
  //   // if (conn->send_state == SS_INIT) {/* received peer's MR before sending ours, so send ours back */
  //     send_mr(conn);
  //     printf(" before sending ours\n");
  //     // }
  //     // else printf("\n");
  //   // }

  // }
  // else{
  //   struct ibv_qp *g_qp;
  //   struct ibv_qp_init_attr qp_attr;
  //   memset(&qp_attr, 0, sizeof(qp_attr));
  //   printf("Function name: %s, line number: %d\n", __func__, __LINE__);
  //   qp_attr.send_cq = s_ctx->cq;
  //   qp_attr.recv_cq = s_ctx->cq;
  //   qp_attr.qp_type = IBV_QPT_RC;
  //   printf("Function name: %s, line number: %d\n", __func__, __LINE__);
  //   qp_attr.cap.max_send_wr = 10;
  //   qp_attr.cap.max_recv_wr = 10;
  //   qp_attr.cap.max_send_sge = 1;
  //   qp_attr.cap.max_recv_sge = 1;
  //   TEST_Z(g_qp = ibv_create_qp(s_ctx->pd, &qp_attr));
  //   if (!g_qp){
  //     printf("g_qp failed\n");
  //     exit(-1);
  //   }
  //   int *buffer = malloc(sizeof(int));
  //   int f = g_qp->qp_num;
  //   printf("need to send qp info! wc->opcode: %d g_qp->qp_num: %d\n", wc->opcode, f);
  //   buffer[0] = f;
  //   // buffer->target_qp_num = g_qp->qp_num;
  //   struct ibv_mr *my_mr;
  //   TEST_Z(my_mr = ibv_reg_mr(
  //   s_ctx->pd, buffer, 
  //   sizeof(int), 
  //   IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ));
    
  //   // buffer[0] = buffer[1] = a;
  //   printf("need to send qp info! wc->opcode: %d g_qp->qp_num: %d\n", wc->opcode, buffer[0]);
  //   struct ibv_send_wr wr, *bad_wr = NULL;
  //   struct ibv_sge sge;

  //   memset(&wr, 0, sizeof(wr));

  //   wr.wr_id = (uintptr_t)conn;
  //   wr.opcode = IBV_WR_SEND;
  //   wr.sg_list = &sge;
  //   wr.num_sge = 1;
  //   wr.send_flags = IBV_SEND_SIGNALED;

  //   sge.addr = (uintptr_t)buffer;
  //   sge.length = sizeof(int);
  //   sge.lkey = my_mr->lkey;

  //   while (!conn->connected);

  //   TEST_NZ(ibv_post_send(conn->qp, &wr, &bad_wr));
  //   a++;
  //   // exit(0);
  // }

  if (wc->opcode & IBV_WC_RECV){
    printf("qp info sent!\n");
  }

  if (conn->send_state == SS_MR_SENT && conn->recv_state == RS_MR_RECV) {
    struct ibv_send_wr wr, *bad_wr = NULL;
    struct ibv_sge sge;

    if (s_mode == M_WRITE)
      printf("received MSG_MR. writing message to remote memory...\n");
    else // M_READ
      printf("received MSG_MR\n");

    memset(&wr, 0, sizeof(wr));

    wr.wr_id = (uintptr_t)conn;
    wr.opcode = (s_mode == M_WRITE) ? IBV_WR_RDMA_WRITE : IBV_WR_RDMA_READ;
    wr.sg_list = &sge;
    wr.num_sge = 1;
    wr.send_flags = IBV_SEND_SIGNALED;
    wr.wr.rdma.remote_addr = (uintptr_t)conn->peer_mr.addr;
    wr.wr.rdma.rkey = conn->peer_mr.rkey;

    sge.addr = (uintptr_t)conn->rdma_local_region;
    sge.length = RDMA_BUFFER_SIZE;
    sge.lkey = conn->rdma_local_mr->lkey;

    // TEST_NZ(ibv_post_send(conn->qp, &wr, &bad_wr));

    conn->send_msg->type = MSG_DONE;
    // send_message(conn);

  } else if (conn->send_state == SS_DONE_SENT && conn->recv_state == RS_DONE_RECV) {
    printf("remote buffer: %s\n", get_peer_message_region(conn));
    rdma_disconnect(conn->id);
  }
}

void on_connect(void *context)
{
  ((struct connection *)context)->connected = 1;
  // send_mr(conn);
}

void * poll_cq(void *ctx)
{
  struct ibv_cq *cq;
  struct ibv_wc wc;

  while (1) {
    TEST_NZ(ibv_get_cq_event(s_ctx->comp_channel, &cq, &ctx));
    ibv_ack_cq_events(cq, 1);
    TEST_NZ(ibv_req_notify_cq(cq, 0));

    while (ibv_poll_cq(cq, 1, &wc))
      on_completion(&wc);
  }

  return NULL;
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

void register_memory(struct connection *conn)
{
  conn->send_msg = malloc(sizeof(struct message));
  conn->recv_msg = malloc(sizeof(struct message));

  conn->rdma_local_region = malloc(RDMA_BUFFER_SIZE);
  conn->rdma_remote_region = malloc(RDMA_BUFFER_SIZE * sizeof(int));

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
    IBV_ACCESS_LOCAL_WRITE));
    // ((s_mode == M_WRITE) ? 0 : IBV_ACCESS_LOCAL_WRITE)));

  TEST_Z(conn->rdma_remote_mr = ibv_reg_mr(
    s_ctx->pd, 
    conn->rdma_remote_region, 
    RDMA_BUFFER_SIZE*sizeof(int), 
    IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ));
    /*((s_mode == M_WRITE) ? (IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE) : IBV_ACCESS_REMOTE_READ)));*/
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

  conn->send_msg->type = MSG_MR;
  memcpy(&conn->send_msg->data.mr, conn->rdma_remote_mr, sizeof(struct ibv_mr));

  send_message(conn);
}

void set_mode(enum mode m)
{
  s_mode = m;
}
