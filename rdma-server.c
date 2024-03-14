#include "server-utils.h"

static int on_connect_request(struct rdma_cm_id *id);
static int on_connection(struct rdma_cm_id *id);
static int on_disconnect(struct rdma_cm_id *id);
static int on_event(struct rdma_cm_event *event);
static void usage(const char *argv0);

int main(int argc, char **argv)
{
  struct sockaddr_in6 addr;
  struct rdma_cm_event *event = NULL;
  struct rdma_cm_id *listener = NULL;
  struct rdma_event_channel *ec = NULL;
  uint16_t port = 0;
  printf("Server side with pid %d\n", getpid());
  if (argc != 2)
    usage(argv[0]);

  if (strcmp(argv[1], "write") == 0)
    set_mode(M_WRITE);
  else if (strcmp(argv[1], "read") == 0)
    set_mode(M_READ);
  else
    usage(argv[0]);
  
  memset(&addr, 0, sizeof(addr));
  addr.sin6_family = AF_INET6;
  addr.sin6_port = htons(PORT);
  

  TEST_Z(ec = rdma_create_event_channel());
  TEST_NZ(rdma_create_id(ec, &listener, NULL, RDMA_PS_IB));
  // printf("rdma_bind_addr(listener, (struct sockaddr *)&addr): %d\n", rdma_bind_addr(listener, (struct sockaddr *)&addr));
  
  TEST_NZ(rdma_bind_addr(listener, (struct sockaddr *)&addr));
  printf("errno:%d \n", errno);
  TEST_NZ(rdma_listen(listener, 10)); /* backlog=10 is arbitrary */

  port = PORT; //ntohs(rdma_get_src_port(listener));

  printf("listening on port %d.\n", port);

  while (rdma_get_cm_event(ec, &event) == 0) {
    struct rdma_cm_event event_copy;

    memcpy(&event_copy, event, sizeof(*event));
    rdma_ack_cm_event(event);

    if (on_event(&event_copy))
      break;
  }

  rdma_destroy_id(listener);
  rdma_destroy_event_channel(ec);

  return 0;
}

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

int on_connect_request(struct rdma_cm_id *id)
{
  struct rdma_conn_param cm_params;

  printf("received connection request.\n");
  build_connection(id);
  struct connection *context = (struct connection *) id->context;
  printf("context)->rdma_remote_region: 0x%llx context)->rdma_local_region: 0x%llx\n", 
          context->rdma_remote_region, context->rdma_local_region);
  printf("rkey: %u\n", context->rdma_remote_mr->rkey);
  build_params(&cm_params);
  // printf("sizeof(get_local_message_region(id->context)): %d\n", sizeof(context->rdma_remote_region));
  // context->rdma_remote_region[RDMA_BUFFER_SIZE-1] = '\n';
  // printf("Function: %s line number: %d\n",__func__, __LINE__);
  for (int i = 0; i < RDMA_BUFFER_SIZE; i++){
    context->rdma_remote_region[i] = 2;
    // printf("Function: %s line number: %d i: %d\n",__func__, __LINE__, i);
  }
  // for (int i = 0; i < RDMA_BUFFER_SIZE; i++)
  //   printf("%d ", context->rdma_remote_region[i]);
  
  // memset(context->rdma_remote_region, 1, RDMA_BUFFER_SIZE*sizeof(int) );
  // printf("Function: %s line number: %d\n",__func__, __LINE__);
  // sprintf(get_local_message_region(id->context), "message from passive/server side with pid %d", getpid());
  TEST_NZ(rdma_accept(id, &cm_params));

  return 0;
}

int on_connection(struct rdma_cm_id *id)
{
  on_connect(id->context);

  return 0;
}

int on_disconnect(struct rdma_cm_id *id)
{
  printf("peer disconnected.\n");

  destroy_connection(id->context);
  return 0;
}

int on_event(struct rdma_cm_event *event)
{
  int r = 0;

  if (event->event == RDMA_CM_EVENT_CONNECT_REQUEST)
    r = on_connect_request(event->id);
  else if (event->event == RDMA_CM_EVENT_ESTABLISHED)
    r = on_connection(event->id);
  else if (event->event == RDMA_CM_EVENT_DISCONNECTED)
    r = on_disconnect(event->id);
  else
    die("on_event: unknown event.");

  return r;
}

void usage(const char *argv0)
{
  fprintf(stderr, "usage: %s <mode>\n  mode = \"read\", \"write\"\n", argv0);
  exit(1);
}
