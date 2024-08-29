
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
#include <time.h>
#include <sys/mman.h>
// #include "runtime.h"

#include <iostream>
using namespace std;

extern "C"{
  #include "rdma_utils.h"
}

#include "rdma_utils.cuh"

// #include <simt/atomic>

struct rdma_content main_content;
struct MemPool MemoryPool;
uint64_t remote_address;
uint64_t remote_address_2nic[2];

__device__ struct post_content *gpost_cont1;
__device__ struct poll_content *gpoll_cont1;

const int TIMEOUT_IN_MS = 500; /* ms */

void process_gpu_mr(int *addr, int size);
int host_gpu_post_send(struct ibv_qp *ibqp, struct ibv_send_wr *wr,
                       struct ibv_send_wr **bad_wr, unsigned long offset);
int gpu_process_work_completion_events (struct ibv_comp_channel *comp_channel, 
		struct ibv_wc *wc, int max_wc, struct ibv_cq *cq);

void die(const char *reason)
{
  fprintf(stderr, "%s\n", reason);
  exit(-1);
}

#define TEST_NZ(x) do { if ( (x)) die("error: " #x " failed (returned non-zero)." ); } while (0)
#define TEST_Z(x)  do { if (!(x)) die("error: " #x " failed (returned zero/null)."); } while (0)

#define PORT 9700

int init_gpu(int gpu){
    int cuda_device_id = 0;
	int cuda_pci_bus_id;
	int cuda_pci_device_id;
	int index;
	CUdevice cu_device;
	CUdevice cuDevice_selected;

	printf("initializing CUDA...\n");
	CUresult error = cuInit(gpu);
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
  
	for (int index = 0; index < deviceCount; index++) {
		if(cuDeviceGet(&cu_device, index) != CUDA_SUCCESS) return -1;
		cuDeviceGetAttribute(&cuda_pci_bus_id, CU_DEVICE_ATTRIBUTE_PCI_BUS_ID , cu_device);
		cuDeviceGetAttribute(&cuda_pci_device_id, CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID , cu_device);
		printf("CUDA device %d: PCIe address is %02X:%02X\n", index, (unsigned int)cuda_pci_bus_id, (unsigned int)cuda_pci_device_id);
	}


	if(cuDeviceGet(&cuDevice_selected, cuda_device_id) != cudaSuccess){
		printf("error on cuDeviceGet\n");
		return -1;
	}
	char name[128];
	if(cuDeviceGetName(name, sizeof(name), cuda_device_id) != cudaSuccess){
		printf("error on cuDeviceGetName\n");
		return -1;
	}
	printf("[pid = %d, dev = %d] device name = [%s]\n", getpid(), cuDevice_selected, name);

    size_t free_memory, total_memory;
    if(cudaSuccess != cudaMemGetInfo(&free_memory, &total_memory)){
        printf("error on cudaMemGetInfo\n");
        return -1;
    }
    printf("free memory: %zu, total_memory: %zu\n", free_memory/(1024 * 1024), total_memory/(1024 * 1024));

    cudaDeviceProp devProp;
    cudaGetDeviceProperties(&devProp, 0);
    printf("Cuda device clock rate = %d\n", devProp.clockRate);


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

void build_context(struct ibv_context *verbs, struct context *s_ctx)
{
    // if (s_ctx) {
    //     if (s_ctx->ctx != verbs)
    //     die("cannot handle events in more than one context.");

    //     return;
    // }
    s_ctx->ctx = verbs;
    TEST_Z(s_ctx->pd = ibv_alloc_pd(s_ctx->ctx));
    TEST_Z(s_ctx->comp_channel = ibv_create_comp_channel(s_ctx->ctx));
    //   if (cq_buffer_gpu == 1 || cq_buffer_gpu == 2){
    printf("Function: %s line number: %d\n",__func__, __LINE__);

    // for (int i = 0; i < s_ctx->n_bufs; i++){
    TEST_Z(s_ctx->main_cq = ibv_create_cq(s_ctx->ctx, 10, NULL, s_ctx->comp_channel, 0)); /* cqe=10 is arbitrary */
    TEST_NZ(ibv_req_notify_cq(s_ctx->main_cq, 0));
    // }

    s_ctx->gpu_cq = (struct ibv_cq **) calloc(s_ctx->n_bufs, sizeof(struct ibv_cq *));
    for (int i = 0; i < s_ctx->n_bufs; i++){
        TEST_Z(s_ctx->gpu_cq[i] = ibvx_create_cq(s_ctx->ctx, 10, NULL, s_ctx->comp_channel, 0, s_ctx->cqbuf[i], s_ctx->cqbuf_size)); /* cqe=10 is arbitrary */
        TEST_NZ(ibv_req_notify_cq(s_ctx->gpu_cq[i], 0));
    }


}

void build_qp_attr(struct ibv_qp_init_attr *qp_attr, struct context *s_ctx)
{
    // for(int i = 0; i < s_ctx->n_bufs; i++){
        memset(qp_attr, 0, sizeof(*qp_attr));
        
        qp_attr->send_cq = s_ctx->main_cq;
        qp_attr->recv_cq = s_ctx->main_cq;
        qp_attr->qp_type = IBV_QPT_RC;

        qp_attr->cap.max_send_wr = 10;
        qp_attr->cap.max_recv_wr = 10;
        qp_attr->cap.max_send_sge = 1;
        qp_attr->cap.max_recv_sge = 1;
    // }
}

void register_memory(struct connection *conn, struct context *s_ctx)
{
    conn->send_msg = (struct message *) malloc(sizeof(struct message));
    conn->recv_msg = (struct message *) malloc(sizeof(struct message));

    // conn->rdma_local_region = (char *) malloc(RDMA_BUFFER_SIZE);
    // conn->rdma_remote_region = (char *) malloc(RDMA_BUFFER_SIZE);
    
    cudaError_t state = cudaMalloc((void **) &s_ctx->gpu_buffer, s_ctx->gpu_buf_size);
    if(state != cudaSuccess)
        printf("Error on cudamalloc\n");
    printf("s_ctx->gpu_buf_size: %llu\n", s_ctx->gpu_buf_size);
    printf("s_ctx->gpu_buffer: 0x%llx\n", s_ctx->gpu_buffer);
    TEST_Z(s_ctx->gpu_mr = ibv_reg_mr(
        s_ctx->pd, s_ctx->gpu_buffer, s_ctx->gpu_buf_size,
        IBV_ACCESS_LOCAL_WRITE
    ));
    printf("s_ctx->gpu_mr->addr: 0x%llx\n", s_ctx->gpu_mr->addr);
    printf("s_ctx->gpu_mr->lkey: %d\n", s_ctx->gpu_mr->lkey);


    if(state != cudaSuccess)
        printf("Error on cudamalloc\n");

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

    TEST_Z(s_ctx->pool_mr = ibv_reg_mr(
        s_ctx->pd, 
        &MemoryPool, 
        sizeof(MemoryPool), 
        IBV_ACCESS_LOCAL_WRITE));

    // TEST_Z(conn->rdma_local_mr = ibv_reg_mr(
    //     s_ctx->pd, 
    //     conn->rdma_local_region, 
    //     RDMA_BUFFER_SIZE, 
    //     (/*(s_mode == M_WRITE) ? 0 :*/  IBV_ACCESS_LOCAL_WRITE)));

    // TEST_Z(conn->rdma_remote_mr = ibv_reg_mr(
    //     s_ctx->pd, 
    //     conn->rdma_remote_region, 
    //     RDMA_BUFFER_SIZE, 
    //     IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ));
    //     // ((s_mode == M_WRITE) ? (IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE) : IBV_ACCESS_REMOTE_READ)));
}

void post_receives(struct connection *conn, struct context *s_ctx)
{
        
    // TEST_Z(s_ctx->pool_mr = ibv_reg_mr(
    //     s_ctx->pd, 
    //     &MemoryPool, 
    //     sizeof(MemoryPool), 
    //     IBV_ACCESS_LOCAL_WRITE));

    // printf("Function: %s line number: %d\n",__func__, __LINE__);
    struct ibv_recv_wr wr, *bad_wr = NULL;
    struct ibv_sge sge;

    wr.wr_id = (uintptr_t)conn;
    wr.next = NULL;
    wr.sg_list = &sge;
    wr.num_sge = 1;

    sge.addr = (uintptr_t) &MemoryPool; // conn->recv_msg;
    sge.length = sizeof(struct MemPool);
    sge.lkey = s_ctx->pool_mr->lkey;
    // printf("Function: %s line number: %d sge.length: %d\n",__func__, __LINE__, sge.length);
    // TEST_NZ(ibv_post_recv(conn->qp, &wr, &bad_wr));
    TEST_NZ(ibv_post_recv(conn->qp, &wr, &bad_wr));
    // printf("Function: %s line number: %d\n",__func__, __LINE__);
}

// create new file for the following functions:

struct ibv_context* createContext(const std::string& device_name) {
    /* There is no way to directly open the device with its name; we should get the list of devices first. */
    struct ibv_context* context = nullptr;
    int num_devices;
    struct ibv_device** device_list = ibv_get_device_list(&num_devices);
    for (int i = 0; i < num_devices; i++){
        /* match device name. open the device and return it */
        if (device_name.compare(ibv_get_device_name(device_list[i])) == 0) {
        context = ibv_open_device(device_list[i]);
        break;
        }
    }

    /* it is important to free the device list; otherwise memory will be leaked. */
    ibv_free_device_list(device_list);
    if (context == nullptr) {
        std::cerr << "Unable to find the device " << device_name << std::endl;
        exit(-1);
    }
    return context;
}

void build_connection(struct rdma_cm_id *id, struct context *s_ctx)
{
    struct connection *conn;
    struct ibv_qp_init_attr qp_attr;
    build_context(id->verbs, s_ctx);
    build_qp_attr(&qp_attr, s_ctx);
    
    
    // for(int i = 0; i < s_ctx->n_bufs; i++){
    TEST_NZ(rdma_create_qp(id, s_ctx->pd, &qp_attr));
    // }
    printf("Function: %s line number: %d\n",__func__, __LINE__);

    // printf("Function: %s line number: %d\n",__func__, __LINE__);

    // printf("Function Name: %s, line number: %d\n", __func__, __LINE__);
    // struct mlx5_context *ctx = to_mctx(s_ctx->pd->context);
    // printf("Function Name: %s, line number: %d,  ctx->bfs: 0x%llx\n", __func__, __LINE__, ctx->bfs);
    
    id->context = conn = (struct connection *)malloc(sizeof(struct connection));

    conn->id = id;
    conn->qp = id->qp;
    s_ctx->main_qp = id->qp;

    //   conn->send_state = SS_INIT;
    //   conn->recv_state = RS_INIT;

    conn->connected = 0;
    // printf("Function: %s line number: %d\n",__func__, __LINE__);
    register_memory(conn, s_ctx);
    post_receives(conn, s_ctx);
    // printf("Function: %s line number: %d\n",__func__, __LINE__);
}

int on_addr_resolved(struct rdma_cm_id *id, struct context *s_ctx)
{
    printf("address resolved.\n");
    // s_ctx->ctx = id->verbs;
    build_connection(id, s_ctx);
    // printf("Function: %s line number: %d\n",__func__, __LINE__);
    // sprintf((char *) get_local_message_region(id->context), "Hello my friend.  with pid %d", getpid());
    TEST_NZ(rdma_resolve_route(id, TIMEOUT_IN_MS));

    return 0;
}

void build_params(struct rdma_conn_param *params)
{
  memset(params, 0, sizeof(*params));

  params->initiator_depth = params->responder_resources = 1;
  params->rnr_retry_count = 7; /* infinite retry */
}

int on_route_resolved(struct rdma_cm_id *id)
{
  struct rdma_conn_param cm_params;

  printf("route resolved.\n");
  build_params(&cm_params);
  // printf("Function: %s line number: %d\n",__func__, __LINE__);
  int ret;
  TEST_NZ(ret = rdma_connect(id, &cm_params));
  printf("Function: %s line number: %d ret: %d\n",__func__, __LINE__, ret);
  return 0;
}

void on_connect(void *context)
{
  ((struct connection *)context)->connected = 1;
}

int on_connection(struct rdma_cm_id *id)
{
  on_connect(id->context);
  printf("Function name: %s, line number: %d\n", __func__, __LINE__);
  // send_mr(id->context);

  return 0;
}

int process_work_completion_events (struct ibv_comp_channel *comp_channel, 
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
    int ret = /*cpu_poll_cq*/ibv_poll_cq(s_ctx->main_cq /* the CQ, we got notification for */, 
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

//   struct connection *conn = (struct connection *)(uintptr_t)wc->wr_id;
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

// TODO: check nent and cqe_sz in your system; otherwise values are assumed as below
__global__ void invalidate_cq(void *buf){
  int i;
  int nent = 64;
  int cqe_sz = 64;
  struct mlx5_cqe64 *cqe;
  
  for (i = 0; i < nent; ++i) {
    cqe = (struct mlx5_cqe64 *) (buf + i * cqe_sz);
    cqe += cqe_sz == 128 ? 1 : 0;
    volatile uint8_t *op_flag = &cqe->op_own;
    //cqe->op_own = MLX5_CQE_INVALID << 4;
    *op_flag = MLX5_CQE_INVALID << 4;
    // printf("cqe->op_own: %d\n", cqe->op_own);
  }
  printf("invalidate_cq - buf: 0x%llx cqe->op_own: %d\n", buf, cqe->op_own);
}

int connect(const char *ip, struct context *s_ctx){
    struct addrinfo *addr;
    struct rdma_cm_event *event = NULL;
    struct rdma_cm_event event_copy;
    struct rdma_cm_id *conn_id;
    struct rdma_event_channel *ec = NULL;
    int ret;

    TEST_NZ(getaddrinfo(ip, "9700", NULL, &addr));
    printf("ip: %s\n", ip);
    TEST_Z(ec = rdma_create_event_channel());
    TEST_NZ(rdma_create_id(ec, &conn_id, NULL, RDMA_PS_TCP));
    TEST_NZ(rdma_resolve_addr(conn_id, NULL, addr->ai_addr, TIMEOUT_IN_MS));

    freeaddrinfo(addr);

    // s_ctx->n_bufs = 256;
    s_ctx->cqbuf_size = 4096*2;
    s_ctx->wqbuf_size = 8192;
    // s_ctx->gpu_buf_size = 3*1024*1024;

    s_ctx->wqbuf = (void ** volatile) calloc(s_ctx->n_bufs, sizeof(void *));
    s_ctx->cqbuf = (void **) calloc(s_ctx->n_bufs, sizeof(void *));
    for(int i = 0; i < s_ctx->n_bufs; i++){
        void* volatile temp;
        ret = cudaMalloc((void **)&temp, s_ctx->wqbuf_size);
        if (cudaSuccess != ret) {
            printf("error on cudaMalloc for wqbuf: %d\n", ret);
            exit(0);
        }
        ret = cudaMemset(temp, 0, s_ctx->wqbuf_size);
        if (cudaSuccess != ret) {
            printf("error on cudaMemset for wqbuf: %d\n", ret);
            exit(0);
        }
        s_ctx->wqbuf[i] = temp;
    }
    for(int i = 0; i < s_ctx->n_bufs; i++){
        // printf("s_ctx->wqbuf[i]: 0x%llx\n", s_ctx->wqbuf[i]);
        void* volatile temp;
        ret = cudaMalloc((void **) &temp, s_ctx->cqbuf_size);
        if (cudaSuccess != ret) {
            printf("error on cudaMalloc for cqbuf: %d\n", ret);
            exit(0);
        }
        ret = cudaMemset(temp, 0, s_ctx->cqbuf_size);
        if (cudaSuccess != ret) {
            printf("error on cudaMemset for cqbuf: %d\n", ret);
            exit(0);
        }
        s_ctx->cqbuf[i] = temp;
        invalidate_cq<<<1,1>>>(s_ctx->cqbuf[i]);
        // printf("s_ctx->cqbuf[i]: 0x%llx\n", s_ctx->cqbuf[i]);
    }

    if (process_cm_event(ec, RDMA_CM_EVENT_ADDR_RESOLVED, &event, &event_copy))
    return -1;
 
    if (on_addr_resolved(event_copy.id, s_ctx)){
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
    // struct mlx5_qp *qp = to_mqp(event->id->qp);
    // printf("Function: %s line number: %d qp->buf.length: %d\n",__func__, __LINE__, qp->buf.length);

    // process_work_completion_events (s_ctx->comp_channel, &wc, 1, s_ctx);
    printf("Function: %s line number: %d wc.opcode: %d\n",__func__, __LINE__, wc.opcode);
    printf("Function: %s line number: %d\n",__func__, __LINE__);
    // exit(0);
    process_work_completion_events (s_ctx->comp_channel, &wc, 1, s_ctx);
    struct connection *conn = (struct connection *)(uintptr_t)wc.wr_id;
    if (wc.opcode & IBV_WC_RECV){
        printf("receive completed\n");
        printf("Function: %s line number: %d \n",__func__, __LINE__);
        for(int i = 0; i < N_8GB_Region; i++){
            printf("remote address: %p, rkey: %d\n", MemoryPool.addresses[i], MemoryPool.rkeys[i]);
        }
        // exit(0);
        memcpy(&s_ctx->server_memory, &MemoryPool, sizeof(struct MemPool));
        // memcpy(&s_ctx->server_mr, &conn->recv_msg->data.mr, sizeof(s_ctx->server_mr));
        printf("Function: %s line number: %d conn->recv_msg->data.mr.addr: 0x%llx\n",__func__, __LINE__, conn->recv_msg->data.mr.addr);
        printf("Function: %s line number: %d conn->recv_msg->data.mr.rkey: %u\n",__func__, __LINE__, conn->recv_msg->data.mr.rkey);
    //   conn->peer_mr.addr;
    // wr.wr.rdma.rkey = conn->peer_mr.rkey
        // post_receives(conn);
        // exit(0);
    }
    else {
        printf("not received: %d\n", wc.opcode);
        exit(-1);
    }

    // post for recv of qp info

   s_ctx->id = event_copy.id;

    struct remote_qp_info buffer; // = (struct remote_qp_info **) calloc(2, sizeof(struct remote_qp_info *));
    // buffer.target_qp_num = (uint32_t *) calloc(s_ctx->n_bufs, sizeof(uint32_t));
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
    process_work_completion_events (s_ctx->comp_channel, &wc, 1, s_ctx);

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

    s_ctx->gpu_qp = (struct ibv_qp **) calloc(s_ctx->n_bufs, sizeof(struct ibv_qp *));
    for(int i = 0; i < s_ctx->n_bufs; i++){
        struct ibv_qp_init_attr qp_attr;
        memset(&qp_attr, 0, sizeof(qp_attr));
        // printf("Function name: %s, line number: %d\n", __func__, __LINE__);
        qp_attr.send_cq = s_ctx->gpu_cq[i];
        qp_attr.recv_cq = s_ctx->gpu_cq[i];
        qp_attr.qp_type = IBV_QPT_RC;
        // printf("Function name: %s, line number: %d\n", __func__, __LINE__);
        qp_attr.cap.max_send_wr = 10;
        qp_attr.cap.max_recv_wr = 10;
        qp_attr.cap.max_send_sge = 1;
        qp_attr.cap.max_recv_sge = 1;
        // printf("Function name: %s, line number: %d\n", __func__, __LINE__);
        TEST_Z(s_ctx->gpu_qp[i] = ibvx_create_qp(s_ctx->pd, &qp_attr, s_ctx->wqbuf[i], s_ctx->wqbuf_size));
        if (!s_ctx->gpu_qp[i]){
        printf("g_qp failed\n");
        exit(-1);
        }
        // printf("gpu_qp[i]->qp_num: %d\n", s_ctx->gpu_qp[i]->qp_num);
    }
    printf("Function name: %s, line number: %d\n", __func__, __LINE__);
    union ibv_gid gid;; 
    int gid_entry;
    get_myglid(s_ctx->ctx, 1, &gid, &gid_entry);
    int mylid;
    get_mylid(s_ctx->ctx, 1, &mylid);
    printf("Function name: %s, line number: %d\n", __func__, __LINE__);
    struct remote_qp_info buffer1; // = (struct remote_qp_info **) calloc(2, sizeof(struct remote_qp_info *));
    // buffer.target_qp_num = (uint32_t *) calloc(s_ctx->n_bufs, sizeof(uint32_t));
    for (int i = 0; i < s_ctx->n_bufs; i++){
        buffer1.target_qp_num[i] = s_ctx->gpu_qp[i]->qp_num;
        // printf("client qp num: %d\n", s_ctx->gpu_qp[i]->qp_num);
        
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

    process_work_completion_events (s_ctx->comp_channel, &wc, 1, s_ctx);

    if (wc.opcode == IBV_WC_SEND){
        printf("QP info sent; qp_num: %d\n", buffer1.target_qp_num[0]);
    }
    // exit(0);

    for(int i = 0; i < s_ctx->n_bufs; i++){
        struct ibv_qp_attr qp_attr1;
        memset(&qp_attr1, 0, sizeof(qp_attr1));

        qp_attr1.qp_state        = IBV_QPS_INIT;
        qp_attr1.pkey_index      = 0;
        qp_attr1.port_num = 1; // IB_PORT;
        qp_attr1.qp_access_flags = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ | IBV_ACCESS_REMOTE_ATOMIC | IBV_ACCESS_REMOTE_WRITE;

        ret = ibv_modify_qp (s_ctx->gpu_qp[i], &qp_attr1, IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT  | IBV_QP_ACCESS_FLAGS); 
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

        ret = ibv_modify_qp(s_ctx->gpu_qp[i], &qp_attr1, IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU | IBV_QP_DEST_QPN | IBV_QP_RQ_PSN | IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER);
        if(ret != 0) {
        printf("Failed to modify qp to RTR. ret: %d\n", ret);
        exit(-1);
        }

        struct ibv_qp_attr qp_attr2; // = {
                qp_attr2.qp_state = IBV_QPS_RTS,
                qp_attr2.timeout = 14,
                qp_attr2.retry_cnt = 7,
                qp_attr2.rnr_retry = 7,
                qp_attr2.sq_psn = 0,
                qp_attr2.max_rd_atomic = 1,
            //   };

        ret = ibv_modify_qp(s_ctx->gpu_qp[i], &qp_attr2,
                            IBV_QP_STATE | IBV_QP_TIMEOUT |
                                IBV_QP_RETRY_CNT | IBV_QP_RNR_RETRY |
                                IBV_QP_SQ_PSN | IBV_QP_MAX_QP_RD_ATOMIC);
                        
        // ret = ibv_modify_qp(g_qp, &qp_attr2, IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU | IBV_QP_DEST_QPN | IBV_QP_RQ_PSN | IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER);
        if(ret != 0) {
        printf("Failed to modify qp to RTS. ret: %d\n", ret);
        exit(-1);
        }
    }

    struct ibv_send_wr wr, *bad_wr = NULL;
    struct ibv_sge sge;
    size_t srv_size = 10*1024*1024*1024llu;
    int *srv_buffer = (int *) malloc(srv_size*sizeof(int));
    struct ibv_mr *srv_mr;
    TEST_Z(srv_mr = ibv_reg_mr(
        s_ctx->pd, srv_buffer, srv_size*sizeof(int),
        IBV_ACCESS_LOCAL_WRITE
    ));

    printf("Function name: %s, line number: %d\n", __func__, __LINE__);
    struct ibv_wc wc1;
    conn = (struct connection *)(uintptr_t)wc.wr_id;
    bad_wr = NULL;
    memset(&wr, 0, sizeof(wr));
    // wr.wr_id = (uintptr_t)conn;
    wr.opcode = IBV_WR_RDMA_READ;
    wr.sg_list = &sge;
    wr.num_sge = 1;
    wr.send_flags = IBV_SEND_SIGNALED;
    wr.wr.rdma.remote_addr = MemoryPool.addresses[0] + 8*1024*1024*1024llu; 
    // (uintptr_t)s_ctx->server_mr.addr + 8*1024*1024*1024llu-1020;//10*1024*1024*1024llu;
    wr.wr.rdma.rkey = MemoryPool.rkeys[1];// s_ctx->server_mr.rkey;
    sge.addr = (uintptr_t)srv_buffer;
    sge.length = 1024;
    sge.lkey = srv_mr->lkey;
    printf("Function name: %s, line number: %d conn->peer_mr.addr: %p\n", __func__, __LINE__, conn->peer_mr.addr);
    printf("Function name: %s, line number: %d conn->peer_mr.rkey: %p\n", __func__, __LINE__, conn->peer_mr.rkey);
    // ret = ibv_post_send(s_ctx->main_qp, &wr, &bad_wr);
    // while(ibv_poll_cq(s_ctx->main_cq, 1, &wc) == 0);
    // printf("ret: %d\n", ret);

    // printf("sizeof(srv_buffer): %llu\n", srv_size);
    // bool flag = false;
    // for(int i = 0; i < sge.length/4; i++){
    //     if(srv_buffer[i] != 2) {
    //         printf("srv_buffer[%d]: %d\n", i, srv_buffer[i]);
    //         flag = true;
    //         break;
    //     }
    // }
    // if(flag) printf("problem\n");
    // else printf("no problem!\n");

    // exit(0);

    // TEST_NZ(host_gpu_post_send(s_ctx->gpu_qp[0], &wr, &bad_wr, 0));
    // // gpu_process_work_completion_events (s_ctx->comp_channel, &wc1, 1, s_ctx->gpu_cq[255]);
    // host_poll_fake(s_ctx->gpu_cq[0], &wc1);


    // printf("read remote buffer: %s\n", conn->rdma_local_region);

    // process_gpu_mr((int *)s_ctx->gpu_buffer, 4096);

    // TEST_NZ(host_gpu_post_send(s_ctx->gpu_qp[1], &wr, &bad_wr, 0));
    // // gpu_process_work_completion_events (s_ctx->comp_channel, &wc1, 1, s_ctx->gpu_cq[255]);
    // host_poll_fake(s_ctx->gpu_cq[1], &wc1);


    // printf("read remote buffer: %s\n", conn->rdma_local_region);

    // process_gpu_mr((int *)s_ctx->gpu_buffer, 4096);

    // benchmark(s_ctx, 1, 128);
    float bandwidth;
    // cpu_benchmark_whole(struct ibv_cq *cq_ptr, int num_entries, struct ibv_wc *wc,
    //                struct ibv_qp *ibqp, struct ibv_send_wr *wr,
    //                struct ibv_send_wr **bad_wr, int num_packets, int mesg_size, float *bandwidth)

    // cpu_benchmark_whole(s_ctx->gpu_cq[2], 1, &wc1, s_ctx->gpu_qp[2], &wr, &bad_wr, 1, 128, &bandwidth);

    // struct ibv_cq *cq_ptr, int num_entries, struct ibv_wc *wc,
    //                struct ibv_qp *ibqp, struct ibv_send_wr *wr,
    //                struct ibv_send_wr **bad_wr, int num_packets, int mesg_size, float *bandwidth
    float b1, b2;
    struct benchmark_content cont1, cont2, cont3, cont4, cont5, cont6, cont7;
    cont1 = {
        .cq_ptr = s_ctx->gpu_cq[0],
        .num_entries = 1,
        .wc = &wc1,
        .ibqp = s_ctx->gpu_qp[0],
        .wr = &wr,
        .bad_wr = &bad_wr,
        .num_packets = 2,
        .mesg_size = 128,
        .bandwidth = &b1,
    };

    wr.wr.rdma.remote_addr = (uintptr_t)conn->peer_mr.addr + 4096;
    sge.addr = (uintptr_t)s_ctx->gpu_buffer + 4096;

    cont2 = {
        .cq_ptr = s_ctx->gpu_cq[1],
        .num_entries = 1,
        .wc = &wc1,
        .ibqp = s_ctx->gpu_qp[1],
        .wr = &wr,
        .bad_wr = &bad_wr,
        .num_packets = 2,
        .mesg_size = 128,
        .bandwidth = &b2,
    };

    wr.wr.rdma.remote_addr = (uintptr_t)conn->peer_mr.addr + 4096*2;
    sge.addr = (uintptr_t)s_ctx->gpu_buffer + 4096*2;

    cont3 = {
        .cq_ptr = s_ctx->gpu_cq[2],
        .num_entries = 1,
        .wc = &wc1,
        .ibqp = s_ctx->gpu_qp[2],
        .wr = &wr,
        .bad_wr = &bad_wr,
        .num_packets = 2,
        .mesg_size = 128,
        .bandwidth = &b2,
    };

    wr.wr.rdma.remote_addr = (uintptr_t)conn->peer_mr.addr + 4096*3;
    sge.addr = (uintptr_t)s_ctx->gpu_buffer + 4096*3;

    cont4 = {
        .cq_ptr = s_ctx->gpu_cq[3],
        .num_entries = 1,
        .wc = &wc1,
        .ibqp = s_ctx->gpu_qp[3],
        .wr = &wr,
        .bad_wr = &bad_wr,
        .num_packets = 2,
        .mesg_size = 128,
        .bandwidth = &b2,
    };

    wr.wr.rdma.remote_addr = (uintptr_t)conn->peer_mr.addr + 4096*4;
    sge.addr = (uintptr_t)s_ctx->gpu_buffer + 4096*4;

    cont5 = {
        .cq_ptr = s_ctx->gpu_cq[4],
        .num_entries = 1,
        .wc = &wc1,
        .ibqp = s_ctx->gpu_qp[4],
        .wr = &wr,
        .bad_wr = &bad_wr,
        .num_packets = 2,
        .mesg_size = 128,
        .bandwidth = &b2,
    };

    wr.wr.rdma.remote_addr = (uintptr_t)conn->peer_mr.addr + 4096*5;
    sge.addr = (uintptr_t)s_ctx->gpu_buffer + 4096*5;

    cont6 = {
        .cq_ptr = s_ctx->gpu_cq[5],
        .num_entries = 1,
        .wc = &wc1,
        .ibqp = s_ctx->gpu_qp[5],
        .wr = &wr,
        .bad_wr = &bad_wr,
        .num_packets = 2,
        .mesg_size = 128,
        .bandwidth = &b2,
    };

    wr.wr.rdma.remote_addr = (uintptr_t)conn->peer_mr.addr + 4096*6;
    sge.addr = (uintptr_t)s_ctx->gpu_buffer + 4096*6;

    cont7 = {
        .cq_ptr = s_ctx->gpu_cq[6],
        .num_entries = 1,
        .wc = &wc1,
        .ibqp = s_ctx->gpu_qp[6],
        .wr = &wr,
        .bad_wr = &bad_wr,
        .num_packets = 2,
        .mesg_size = 128,
        .bandwidth = &b2,
    };

    // TEST_NZ(pthread_create(&s_ctx->cq_poller_thread, NULL, benchmark, &cont1));
    // TEST_NZ(pthread_create(&s_ctx->cq_poller_thread, NULL, benchmark, &cont2));
    // TEST_NZ(pthread_create(&s_ctx->cq_poller_thread, NULL, benchmark, &cont3));
    // TEST_NZ(pthread_create(&s_ctx->cq_poller_thread, NULL, benchmark, &cont4));
    // TEST_NZ(pthread_create(&s_ctx->cq_poller_thread, NULL, benchmark, &cont5));
    // TEST_NZ(pthread_create(&s_ctx->cq_poller_thread, NULL, benchmark, &cont6));
    // TEST_NZ(pthread_create(&s_ctx->cq_poller_thread, NULL, benchmark, &cont7));

    printf("Function name: %s, line number: %d\n", __func__, __LINE__);


    return 0;
}

int init_qp(struct ibv_qp *qp){
    struct ibv_qp_attr qp_attr1;
    memset(&qp_attr1, 0, sizeof(qp_attr1));

    qp_attr1.qp_state        = IBV_QPS_INIT;
    qp_attr1.pkey_index      = 0;
    qp_attr1.port_num = 1; // IB_PORT;
    qp_attr1.qp_access_flags = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ | IBV_ACCESS_REMOTE_ATOMIC | IBV_ACCESS_REMOTE_WRITE;

    int ret = ibv_modify_qp (qp, &qp_attr1, IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT  | IBV_QP_ACCESS_FLAGS); 

    return ret;
}

int rtr_qp(struct ibv_qp *qp, uint32_t qp_num, uint16_t lid, union ibv_gid gid, uint8_t gid_entry){
    struct ibv_qp_attr qp_attr1;
    memset(&qp_attr1, 0, sizeof(qp_attr1));
    qp_attr1.qp_state = IBV_QPS_RTR;
    qp_attr1.path_mtu = IBV_MTU_4096;
    qp_attr1.dest_qp_num = qp_num;// buffer.target_qp_num[i];
    qp_attr1.rq_psn = 0;

    qp_attr1.min_rnr_timer = 12;
    qp_attr1.ah_attr.is_global = 1;
    // qp_attr1.ah_attr.dlid = buffer.target_lid;
    qp_attr1.ah_attr.sl = 0;
    qp_attr1.max_dest_rd_atomic = 1;
    qp_attr1.ah_attr.port_num = 1;
    qp_attr1.ah_attr.src_path_bits = 0;
    qp_attr1.ah_attr.port_num = 1;
    // qp_attr1.dest_qp_num = s_ctx->gpu_qp[i]->qp_num;
    qp_attr1.ah_attr.dlid = lid;
    qp_attr1.ah_attr.grh.dgid = gid;
    qp_attr1.ah_attr.grh.sgid_index = gid_entry;
    qp_attr1.ah_attr.grh.hop_limit = 0xFF;
    qp_attr1.ah_attr.grh.traffic_class = 0;

    int ret = ibv_modify_qp(qp, &qp_attr1, IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU | IBV_QP_DEST_QPN | IBV_QP_RQ_PSN | IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER);
    return ret;
}

int rts_qp(struct ibv_qp *qp){
    struct ibv_qp_attr qp_attr;
    memset(&qp_attr, 0, sizeof(qp_attr));
    
    qp_attr.qp_state = IBV_QPS_RTS,
    qp_attr.timeout = 14,
    qp_attr.retry_cnt = 7,
    qp_attr.rnr_retry = 7,
    qp_attr.sq_psn = 0,
    qp_attr.max_rd_atomic = 1;

    int ret = ibv_modify_qp(qp, &qp_attr,
                        IBV_QP_STATE | IBV_QP_TIMEOUT |
                            IBV_QP_RETRY_CNT | IBV_QP_RNR_RETRY |
                            IBV_QP_SQ_PSN | IBV_QP_MAX_QP_RD_ATOMIC);

    return ret;
}

int local_connect(const char *mlx_name, struct context *s_ctx){
    int ret;

    s_ctx->ctx = createContext(mlx_name);
    TEST_Z(s_ctx->pd = ibv_alloc_pd(s_ctx->ctx));
    TEST_Z(s_ctx->comp_channel = ibv_create_comp_channel(s_ctx->ctx));

    TEST_Z(s_ctx->main_cq = ibv_create_cq(s_ctx->ctx, 10, NULL, s_ctx->comp_channel, 0)); /* cqe=10 is arbitrary */
    TEST_NZ(ibv_req_notify_cq(s_ctx->main_cq, 0));

    cudaError_t state = cudaMalloc((void **) &s_ctx->gpu_buffer, s_ctx->gpu_buf_size);
    if(state != cudaSuccess)
        printf("Error on cudamalloc\n");
    printf("s_ctx->gpu_buf_size: %llu\n", s_ctx->gpu_buf_size);
    printf("s_ctx->gpu_buffer: 0x%llx\n", s_ctx->gpu_buffer);
    TEST_Z(s_ctx->gpu_mr = ibv_reg_mr(
        s_ctx->pd, s_ctx->gpu_buffer, s_ctx->gpu_buf_size,
        IBV_ACCESS_LOCAL_WRITE
    ));

    printf("s_ctx->gpu_mr->addr: 0x%llx\n", s_ctx->gpu_mr->addr);
    printf("s_ctx->gpu_mr->lkey: %d\n", s_ctx->gpu_mr->lkey);

    void *memoryPool = (void *) malloc(RDMA_BUFFER_SIZE);
    if(memoryPool == NULL) {
        printf("Memory pool could not be created!\n");
        exit(-1);
    }
    remote_address = (uint64_t) memoryPool;
    struct ibv_mr *temp_mr;
    
    for(int index = 0; index < N_8GB_Region; index++){
        TEST_Z(temp_mr = ibv_reg_mr(
                s_ctx->pd, 
                memoryPool + index*Region_Size, 
                Region_Size, 
                IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ));
                
        printf("Registered server address: %p, server rkey: %d length: %llu\n\n\n", \
        temp_mr->addr, temp_mr->rkey, temp_mr->length);

        // s_ctx->server_memory.addresses[0]
        // s_ctx->server_memory.lkeys[0]
        // s_ctx->server_memory.rkeys[0]

        s_ctx->server_memory.addresses[index] = (uint64_t) (memoryPool + index*Region_Size);
        s_ctx->server_memory.rkeys[index] = temp_mr->rkey;
        s_ctx->server_memory.lkeys[index] = temp_mr->lkey;
    }  

    if(cudaSuccess != cudaDeviceSynchronize()) return -1;
    // s_ctx->n_bufs = 256;
    s_ctx->cqbuf_size = 4096*2;
    s_ctx->wqbuf_size = 8192;
    // s_ctx->gpu_buf_size = 3*1024*1024;

    s_ctx->wqbuf = (void ** volatile) calloc(s_ctx->n_bufs, sizeof(void *));
    s_ctx->cqbuf = (void **) calloc(s_ctx->n_bufs, sizeof(void *));
    for(int i = 0; i < s_ctx->n_bufs; i++){
        void* volatile temp;
        ret = cudaMalloc((void **)&temp, s_ctx->wqbuf_size);
        if (cudaSuccess != ret) {
            printf("error on cudaMalloc for wqbuf: %d\n", ret);
            exit(0);
        }
        ret = cudaMemset(temp, 0, s_ctx->wqbuf_size);
        if (cudaSuccess != ret) {
            printf("error on cudaMemset for wqbuf: %d\n", ret);
            exit(0);
        }
        s_ctx->wqbuf[i] = temp;
    }
    for(int i = 0; i < s_ctx->n_bufs; i++){
        // printf("s_ctx->wqbuf[i]: 0x%llx\n", s_ctx->wqbuf[i]);
        void* volatile temp;
        ret = cudaMalloc((void **) &temp, s_ctx->cqbuf_size);
        if (cudaSuccess != ret) {
            printf("error on cudaMalloc for cqbuf: %d\n", ret);
            exit(0);
        }
        ret = cudaMemset(temp, 0, s_ctx->cqbuf_size);
        if (cudaSuccess != ret) {
            printf("error on cudaMemset for cqbuf: %d\n", ret);
            exit(0);
        }
        s_ctx->cqbuf[i] = temp;
        if(cudaSuccess != cudaDeviceSynchronize()) return -1;
        invalidate_cq<<<1,1>>>(s_ctx->cqbuf[i]);
        if(cudaSuccess != cudaDeviceSynchronize()) return -1;
        printf("s_ctx->cqbuf[%d]: 0x%llx\n", i, s_ctx->cqbuf[i]);
    }
    // exit(0);
    s_ctx->gpu_cq = (struct ibv_cq **) calloc(s_ctx->n_bufs, sizeof(struct ibv_cq *));
    for (int i = 0; i < s_ctx->n_bufs; i++){
        TEST_Z(s_ctx->gpu_cq[i] = ibvx_create_cq(s_ctx->ctx, 10, NULL, s_ctx->comp_channel, 0, s_ctx->cqbuf[i], s_ctx->cqbuf_size)); /* cqe=10 is arbitrary */
        TEST_NZ(ibv_req_notify_cq(s_ctx->gpu_cq[i], 0));
    }

    printf("Function name: %s, line number: %d s_ctx->n_bufs: %d\n", __func__, __LINE__, s_ctx->n_bufs);
    s_ctx->gpu_qp = (struct ibv_qp **) calloc(s_ctx->n_bufs, sizeof(struct ibv_qp *));
    for(int i = 0; i < s_ctx->n_bufs; i++){
        struct ibv_qp_init_attr qp_attr;
        memset(&qp_attr, 0, sizeof(qp_attr));
        // printf("Function name: %s, line number: %d\n", __func__, __LINE__);
        qp_attr.send_cq = s_ctx->gpu_cq[i];
        qp_attr.recv_cq = s_ctx->gpu_cq[i];
        qp_attr.qp_type = IBV_QPT_RC;
        // printf("Function name: %s, line number: %d\n", __func__, __LINE__);
        qp_attr.cap.max_send_wr = 10;
        qp_attr.cap.max_recv_wr = 10;
        qp_attr.cap.max_send_sge = 1;
        qp_attr.cap.max_recv_sge = 1;
        // printf("Function name: %s, line number: %d\n", __func__, __LINE__);
        TEST_Z(s_ctx->gpu_qp[i] = ibvx_create_qp(s_ctx->pd, &qp_attr, s_ctx->wqbuf[i], s_ctx->wqbuf_size));
        if (!s_ctx->gpu_qp[i]){
            printf("g_qp failed\n");
            exit(-1);
        }
        // printf("gpu_qp[%d]->qp_num: %d\n", i, s_ctx->gpu_qp[i]->qp_num);
    }

    struct ibv_qp_init_attr main_qp_attr;
    memset(&main_qp_attr, 0, sizeof(main_qp_attr));
        
        main_qp_attr.send_cq = s_ctx->main_cq;
        main_qp_attr.recv_cq = s_ctx->main_cq;
        main_qp_attr.qp_type = IBV_QPT_RC;

        main_qp_attr.cap.max_send_wr = 10;
        main_qp_attr.cap.max_recv_wr = 10;
        main_qp_attr.cap.max_send_sge = 1;
        main_qp_attr.cap.max_recv_sge = 1;

    ibv_qp *temp_qp;
    TEST_Z(s_ctx->main_qp = ibv_create_qp(s_ctx->pd, &main_qp_attr));
    TEST_Z(temp_qp = ibv_create_qp(s_ctx->pd, &main_qp_attr));

    struct ibv_qp_attr qp_attr1;
    

    if(init_qp(s_ctx->main_qp) != 0)
    {
        printf("Failed to modify main qp to INIT. ret: %d\n", ret);
        exit(-1);
    }
    if(init_qp(temp_qp) != 0)
    {
        printf("Failed to modify main qp to INIT. ret: %d\n", ret);
        exit(-1);
    }

    ibv_port_attr main_port_attr, temp_port_attr;
    ibv_query_port(s_ctx->ctx, 1, &main_port_attr);
    ibv_query_port(s_ctx->ctx, 1, &temp_port_attr);

    union ibv_gid gid;; 
    int gid_entry;
    int mylid;
    get_myglid(s_ctx->ctx, 1, &gid, &gid_entry);
    get_mylid(s_ctx->ctx, 1, &mylid);

    ret = rtr_qp(s_ctx->main_qp, temp_qp->qp_num, temp_port_attr.lid, gid, gid_entry);
    if(ret != 0) {
        printf("Failed to modify main qp to RTR. ret: %d\n", ret);
        exit(-1);
    }

    ret = rtr_qp(temp_qp, s_ctx->main_qp->qp_num, main_port_attr.lid, gid, gid_entry);
    if(ret != 0) {
        printf("Failed to modify temp qp to RTR. ret: %d\n", ret);
        exit(-1);
    }

    ret = rts_qp(s_ctx->main_qp);
    if(ret != 0) {
        printf("Failed to modify qp to RTS. ret: %d\n", ret);
        exit(-1);
    }

    struct ibv_send_wr wr, *bad_wr = NULL;
    struct ibv_sge sge;
    struct ibv_wc wc;
    size_t srv_size = 1024;
    int *srv_buffer = (int *) malloc(srv_size*sizeof(int));
    struct ibv_mr *srv_mr;
    
    TEST_Z(srv_mr = ibv_reg_mr(
        s_ctx->pd, srv_buffer, srv_size*sizeof(int),
        IBV_ACCESS_LOCAL_WRITE
    ));

    // server QPs:
    struct ibv_qp **host_QPs = (struct ibv_qp **) calloc(s_ctx->n_bufs, sizeof(struct ibv_qp *));
    for(int i = 0; i < s_ctx->n_bufs; i++){
        struct ibv_qp_init_attr qp_attr;
        memset(&qp_attr, 0, sizeof(qp_attr));
        // printf("Function name: %s, line number: %d\n", __func__, __LINE__);
        qp_attr.send_cq = s_ctx->main_cq;
        qp_attr.recv_cq = s_ctx->main_cq;
        qp_attr.qp_type = IBV_QPT_RC;
        // printf("Function name: %s, line number: %d\n", __func__, __LINE__);
        qp_attr.cap.max_send_wr = 10;
        qp_attr.cap.max_recv_wr = 10;
        qp_attr.cap.max_send_sge = 1;
        qp_attr.cap.max_recv_sge = 1;
        // printf("Function name: %s, line number: %d\n", __func__, __LINE__);
        TEST_Z(host_QPs[i] = ibv_create_qp(s_ctx->pd, &qp_attr));
        if (!host_QPs[i]){
            printf("host_QPs[%d] failed\n", i);
            exit(-1);
        }
        // printf("gpu_qp[i]->qp_num: %d\n", s_ctx->gpu_qp[i]->qp_num);
    }
    struct remote_qp_info host_qp_info; 
    for (int i = 0; i < s_ctx->n_bufs; i++){
        host_qp_info.target_qp_num[i] = host_QPs[i]->qp_num;
        // printf("client qp num: %d\n", s_ctx->gpu_qp[i]->qp_num);
    }
    host_qp_info.target_gid = gid;
    host_qp_info.target_lid = mylid;

    // gpu QPs:
    printf("Function name: %s, line number: %d\n", __func__, __LINE__);
    struct remote_qp_info device_qp_info; // = (struct remote_qp_info **) calloc(2, sizeof(struct remote_qp_info *));
    for (int i = 0; i < s_ctx->n_bufs; i++){
        device_qp_info.target_qp_num[i] = s_ctx->gpu_qp[i]->qp_num;
        // printf("client qp num: %d\n", s_ctx->gpu_qp[i]->qp_num);
        
    }
    device_qp_info.target_gid = gid;
    device_qp_info.target_lid = mylid;

    for(int i = 0; i < s_ctx->n_bufs; i++){
        
        ret = init_qp(s_ctx->gpu_qp[i]); 
        if(ret != 0)
        {
            printf("Failed to modify gpu_qp[%d] to INIT. ret: %d\n", i, ret);
            exit(-1);
        }
        ret = init_qp(host_QPs[i]); 
        if(ret != 0)
        {
            printf("Failed to modify gpu_qp[%d] to INIT. ret: %d\n", i, ret);
            exit(-1);
        }
        

        ret = rtr_qp(s_ctx->gpu_qp[i], host_qp_info.target_qp_num[i], main_port_attr.lid, gid, gid_entry);
        if(ret != 0) {
            printf("Failed to modify gpu_qp[%d] to RTR. ret: %d\n", i, ret);
            exit(-1);
        }
        ret = rtr_qp(host_QPs[i], device_qp_info.target_qp_num[i], main_port_attr.lid, gid, gid_entry);
        if(ret != 0) {
            printf("Failed to modify host_QPs[%d] to RTR. ret: %d\n", i, ret);
            exit(-1);
        }

        rts_qp(s_ctx->gpu_qp[i]);
        if(ret != 0) {
            printf("Failed to modify gpu_qp[%d] to RTS. ret: %d\n", i, ret);
            exit(-1);
        }

        // rts_qp(host_QPs[i]);
        // if(ret != 0) {
        //     printf("Failed to modify host_QPs[%d] to RTS. ret: %d\n", i, ret);
        //     exit(-1);
        // }
    } 




    for (size_t i = 0; i < 1024; i++)
    {
        srv_buffer[i] = 5;
    }

    int *server_temp = (int *) s_ctx->server_memory.addresses[0]; // + 8*1024*1024*1024llu;
    for (size_t i = 0; i < 1024; i++)
    {
        server_temp[i] = 4;
    }
    

    printf("Function name: %s, line number: %d\n", __func__, __LINE__);
    struct ibv_wc wc1;
    // conn = (struct connection *)(uintptr_t)wc.wr_id;
    bad_wr = NULL;
    memset(&wr, 0, sizeof(wr));
    // wr.wr_id = (uintptr_t)conn;
    wr.opcode = IBV_WR_RDMA_READ;
    wr.sg_list = &sge;
    wr.num_sge = 1;
    wr.send_flags = IBV_SEND_SIGNALED;
    wr.wr.rdma.remote_addr = s_ctx->server_memory.addresses[0]; // + 8*1024*1024*1024llu; 
    // (uintptr_t)s_ctx->server_mr.addr + 8*1024*1024*1024llu-1020;//10*1024*1024*1024llu;
    wr.wr.rdma.rkey = s_ctx->server_memory.rkeys[0];// s_ctx->server_mr.rkey;
    sge.addr = (uintptr_t) s_ctx->gpu_mr->addr; // (uintptr_t)srv_buffer;
    sge.length = 1024*4;
    sge.lkey = s_ctx->gpu_mr->rkey; // srv_mr->lkey;
    printf("Function name: %s, line number: %d conn->peer_mr.addr: %p\n", __func__, __LINE__, s_ctx->server_memory.addresses[0]);
    printf("Function name: %s, line number: %d conn->peer_mr.rkey: %p\n", __func__, __LINE__, s_ctx->server_memory.rkeys[1]);
    // ret = ibv_post_send(s_ctx->main_qp, &wr, &bad_wr);
    // printf("post ret: %d\n", ret);
    // do{
    //     ret = ibv_poll_cq(s_ctx->main_cq, 1, &wc1);
    //     printf("poll ret: %d\n", ret);
    // }while(ret == 0);

    // process_gpu_mr((int *) s_ctx->gpu_mr->addr, 1024);
    // exit(0);

    // printf("sizeof(srv_buffer): %llu\n", srv_size);
    // bool flag = false;
    // for(int i = 0; i < sge.length/4; i++){
    //     printf(" srv_buffer[%d]: %d ", i, srv_buffer[i]);
    //     if(srv_buffer[i] != 2) {
    //         printf("srv_buffer[%d]: %d\n", i, srv_buffer[i]);
    //         flag = true;
    //         break;
    //     }
    // }
    // if(flag) printf("problem\n");
    // else printf("no problem!\n");

    // exit(0);


    return 0;
}


int local_connect_2gpu(const char *mlx_name, struct context_2gpu *s_ctx){
    int ret;

    s_ctx->ctx = createContext(mlx_name);
    TEST_Z(s_ctx->pd = ibv_alloc_pd(s_ctx->ctx));
    TEST_Z(s_ctx->comp_channel = ibv_create_comp_channel(s_ctx->ctx));

    TEST_Z(s_ctx->main_cq = ibv_create_cq(s_ctx->ctx, 10, NULL, s_ctx->comp_channel, 0)); /* cqe=10 is arbitrary */
    TEST_NZ(ibv_req_notify_cq(s_ctx->main_cq, 0));

    // this should be GPU 0
    cudaSetDevice(0);
    cudaError_t state = cudaMalloc((void **) &s_ctx->gpu_buffer1, s_ctx->gpu_buf1_size);
    if(state != cudaSuccess)
        printf("Error on cudamalloc\n");
    printf("s_ctx->gpu_buf_size: %llu\n", s_ctx->gpu_buf1_size);
    printf("s_ctx->gpu_buffer: 0x%llx\n", s_ctx->gpu_buffer1);
    TEST_Z(s_ctx->gpu1_mr = ibv_reg_mr(
        s_ctx->pd, s_ctx->gpu_buffer1, s_ctx->gpu_buf1_size,
        IBV_ACCESS_LOCAL_WRITE
    ));
    
    printf("s_ctx->gpu_mr->addr: 0x%llx\n", s_ctx->gpu1_mr->addr);
    printf("s_ctx->gpu_mr->lkey: %d\n", s_ctx->gpu1_mr->lkey);

    // this should be GPU 1
    cudaSetDevice(1);
    state = cudaMalloc((void **) &s_ctx->gpu_buffer2, s_ctx->gpu_buf2_size);
    if(state != cudaSuccess)
        printf("Error on cudamalloc\n");
    printf("s_ctx->gpu_buf_size: %llu\n", s_ctx->gpu_buf2_size);
    printf("s_ctx->gpu_buffer: 0x%llx\n", s_ctx->gpu_buffer2);
    TEST_Z(s_ctx->gpu2_mr = ibv_reg_mr(
        s_ctx->pd, s_ctx->gpu_buffer2, s_ctx->gpu_buf2_size,
        IBV_ACCESS_LOCAL_WRITE
    ));
    printf("s_ctx->gpu_mr->addr: 0x%llx\n", s_ctx->gpu2_mr->addr);
    printf("s_ctx->gpu_mr->lkey: %d\n", s_ctx->gpu2_mr->lkey);

    
    void *memoryPool = (void *) malloc(RDMA_BUFFER_SIZE);
    if(memoryPool == NULL) {
        printf("Memory pool could not be created!\n");
        exit(-1);
    }
    remote_address = (uint64_t) memoryPool;
    struct ibv_mr *temp_mr;
    
    for(int index = 0; index < N_8GB_Region; index++){
        TEST_Z(temp_mr = ibv_reg_mr(
                s_ctx->pd, 
                memoryPool + index*Region_Size, 
                Region_Size, 
                IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ));
                
        printf("Registered server address: %p, server rkey: %d length: %llu\n\n\n", \
        temp_mr->addr, temp_mr->rkey, temp_mr->length);

        // s_ctx->server_memory.addresses[0]
        // s_ctx->server_memory.lkeys[0]
        // s_ctx->server_memory.rkeys[0]

        s_ctx->server_memory.addresses[index] = (uint64_t) (memoryPool + index*Region_Size);
        s_ctx->server_memory.rkeys[index] = temp_mr->rkey;
        s_ctx->server_memory.lkeys[index] = temp_mr->lkey;
    }  

    if(cudaSuccess != cudaDeviceSynchronize()) return -1;
    // s_ctx->n_bufs = 256;
    s_ctx->cqbuf_size = 4096*2;
    s_ctx->wqbuf_size = 8192;
    // s_ctx->gpu_buf_size = 3*1024*1024;

    // multiply nbefs by 2 because we have 2 GPUs
    s_ctx->wqbuf = (void ** volatile) calloc(s_ctx->n_bufs*2, sizeof(void *));
    s_ctx->cqbuf = (void **) calloc(s_ctx->n_bufs*2, sizeof(void *));

    /**************** Allocate cq abd wq for GPU 0 *******************/
    for(int i = 0; i < s_ctx->n_bufs; i++){
        void* volatile temp;
        cudaSetDevice(0);
        ret = cudaMalloc((void **)&temp, s_ctx->wqbuf_size);
        if (cudaSuccess != ret) {
            printf("error on cudaMalloc for wqbuf: %d\n", ret);
            exit(0);
        }
        ret = cudaMemset(temp, 0, s_ctx->wqbuf_size);
        if (cudaSuccess != ret) {
            printf("error on cudaMemset for wqbuf: %d\n", ret);
            exit(0);
        }
        s_ctx->wqbuf[i] = temp;
    }
    for(int i = 0; i < s_ctx->n_bufs; i++){
        // printf("s_ctx->wqbuf[i]: 0x%llx\n", s_ctx->wqbuf[i]);
        void* volatile temp;
        cudaSetDevice(0);
        ret = cudaMalloc((void **) &temp, s_ctx->cqbuf_size);
        if (cudaSuccess != ret) {
            printf("error on cudaMalloc for cqbuf: %d\n", ret);
            exit(0);
        }
        ret = cudaMemset(temp, 0, s_ctx->cqbuf_size);
        if (cudaSuccess != ret) {
            printf("error on cudaMemset for cqbuf: %d\n", ret);
            exit(0);
        }
        s_ctx->cqbuf[i] = temp;
        if(cudaSuccess != cudaDeviceSynchronize()) return -1;
        invalidate_cq<<<1,1>>>(s_ctx->cqbuf[i]);
        if(cudaSuccess != cudaDeviceSynchronize()) return -1;
        printf("s_ctx->cqbuf[%d]: 0x%llx\n", i, s_ctx->cqbuf[i]);
    }
    /**************** Allocate cq abd wq for GPU 1 *******************/
    for(int i = s_ctx->n_bufs; i < s_ctx->n_bufs*2; i++){
        void* volatile temp;
        cudaSetDevice(1);
        ret = cudaMalloc((void **)&temp, s_ctx->wqbuf_size);
        if (cudaSuccess != ret) {
            printf("error on cudaMalloc for wqbuf: %d\n", ret);
            exit(0);
        }
        ret = cudaMemset(temp, 0, s_ctx->wqbuf_size);
        if (cudaSuccess != ret) {
            printf("error on cudaMemset for wqbuf: %d\n", ret);
            exit(0);
        }
        s_ctx->wqbuf[i] = temp;
    }
    for(int i = s_ctx->n_bufs; i < s_ctx->n_bufs*2; i++){
        printf("s_ctx->wqbuf[i]: 0x%llx\n", s_ctx->wqbuf[i]);
        void* volatile temp;
        cudaSetDevice(1);
        ret = cudaMalloc((void **) &temp, s_ctx->cqbuf_size);
        if (cudaSuccess != ret) {
            printf("error on cudaMalloc for cqbuf: %d\n", ret);
            exit(0);
        }
        ret = cudaMemset(temp, 0, s_ctx->cqbuf_size);
        if (cudaSuccess != ret) {
            printf("error on cudaMemset for cqbuf: %d\n", ret);
            exit(0);
        }
        s_ctx->cqbuf[i] = temp;
        if(cudaSuccess != cudaDeviceSynchronize()) return -1;
        invalidate_cq<<<1,1>>>(s_ctx->cqbuf[i]);
        if(cudaSuccess != cudaDeviceSynchronize()) return -1;
        printf("s_ctx->cqbuf[%d]: 0x%llx\n", i, s_ctx->cqbuf[i]);
    }
    /**************************************************************/

    // exit(0);
    s_ctx->gpu_cq = (struct ibv_cq **) calloc(s_ctx->n_bufs*2, sizeof(struct ibv_cq *));
    for(int i = 0; i < s_ctx->n_bufs*2; i++){
        TEST_Z(s_ctx->gpu_cq[i] = ibvx_create_cq(s_ctx->ctx, 10, NULL, s_ctx->comp_channel, 0, s_ctx->cqbuf[i], s_ctx->cqbuf_size)); /* cqe=10 is arbitrary */
        TEST_NZ(ibv_req_notify_cq(s_ctx->gpu_cq[i], 0));
    }

    printf("Function name: %s, line number: %d s_ctx->n_bufs: %d\n", __func__, __LINE__, s_ctx->n_bufs);
    s_ctx->gpu_qp = (struct ibv_qp **) calloc(s_ctx->n_bufs*2, sizeof(struct ibv_qp *));
    for(int i = 0; i < s_ctx->n_bufs*2; i++){
        struct ibv_qp_init_attr qp_attr;
        memset(&qp_attr, 0, sizeof(qp_attr));
        // printf("Function name: %s, line number: %d\n", __func__, __LINE__);
        qp_attr.send_cq = s_ctx->gpu_cq[i];
        qp_attr.recv_cq = s_ctx->gpu_cq[i];
        qp_attr.qp_type = IBV_QPT_RC;
        // printf("Function name: %s, line number: %d\n", __func__, __LINE__);
        qp_attr.cap.max_send_wr = 10;
        qp_attr.cap.max_recv_wr = 10;
        qp_attr.cap.max_send_sge = 1;
        qp_attr.cap.max_recv_sge = 1;
        // printf("Function name: %s, line number: %d\n", __func__, __LINE__);
        TEST_Z(s_ctx->gpu_qp[i] = ibvx_create_qp(s_ctx->pd, &qp_attr, s_ctx->wqbuf[i], s_ctx->wqbuf_size));
        if (!s_ctx->gpu_qp[i]){
            printf("g_qp failed\n");
            exit(-1);
        }
        // printf("gpu_qp[%d]->qp_num: %d\n", i, s_ctx->gpu_qp[i]->qp_num);
    }

    struct ibv_qp_init_attr main_qp_attr;
    memset(&main_qp_attr, 0, sizeof(main_qp_attr));
        
        main_qp_attr.send_cq = s_ctx->main_cq;
        main_qp_attr.recv_cq = s_ctx->main_cq;
        main_qp_attr.qp_type = IBV_QPT_RC;

        main_qp_attr.cap.max_send_wr = 10;
        main_qp_attr.cap.max_recv_wr = 10;
        main_qp_attr.cap.max_send_sge = 1;
        main_qp_attr.cap.max_recv_sge = 1;

    ibv_qp *temp_qp;
    TEST_Z(s_ctx->main_qp = ibv_create_qp(s_ctx->pd, &main_qp_attr));
    TEST_Z(temp_qp = ibv_create_qp(s_ctx->pd, &main_qp_attr));

    struct ibv_qp_attr qp_attr1;
    

    if(init_qp(s_ctx->main_qp) != 0)
    {
        printf("Failed to modify main qp to INIT. ret: %d\n", ret);
        exit(-1);
    }
    if(init_qp(temp_qp) != 0)
    {
        printf("Failed to modify main qp to INIT. ret: %d\n", ret);
        exit(-1);
    }

    ibv_port_attr main_port_attr, temp_port_attr;
    ibv_query_port(s_ctx->ctx, 1, &main_port_attr);
    ibv_query_port(s_ctx->ctx, 1, &temp_port_attr);

    union ibv_gid gid;; 
    int gid_entry;
    int mylid;
    get_myglid(s_ctx->ctx, 1, &gid, &gid_entry);
    get_mylid(s_ctx->ctx, 1, &mylid);

    ret = rtr_qp(s_ctx->main_qp, temp_qp->qp_num, temp_port_attr.lid, gid, gid_entry);
    if(ret != 0) {
        printf("Failed to modify main qp to RTR. ret: %d\n", ret);
        exit(-1);
    }

    ret = rtr_qp(temp_qp, s_ctx->main_qp->qp_num, main_port_attr.lid, gid, gid_entry);
    if(ret != 0) {
        printf("Failed to modify temp qp to RTR. ret: %d\n", ret);
        exit(-1);
    }

    ret = rts_qp(s_ctx->main_qp);
    if(ret != 0) {
        printf("Failed to modify qp to RTS. ret: %d\n", ret);
        exit(-1);
    }

    struct ibv_send_wr wr, *bad_wr = NULL;
    struct ibv_sge sge;
    struct ibv_wc wc;
    size_t srv_size = 1024;
    int *srv_buffer = (int *) malloc(srv_size*sizeof(int));
    struct ibv_mr *srv_mr;
    
    TEST_Z(srv_mr = ibv_reg_mr(
        s_ctx->pd, srv_buffer, srv_size*sizeof(int),
        IBV_ACCESS_LOCAL_WRITE
    ));

    // server QPs:
    struct ibv_qp **host_QPs = (struct ibv_qp **) calloc(s_ctx->n_bufs*2, sizeof(struct ibv_qp *));
    for(int i = 0; i < s_ctx->n_bufs*2; i++){
        struct ibv_qp_init_attr qp_attr;
        memset(&qp_attr, 0, sizeof(qp_attr));
        // printf("Function name: %s, line number: %d\n", __func__, __LINE__);
        qp_attr.send_cq = s_ctx->main_cq;
        qp_attr.recv_cq = s_ctx->main_cq;
        qp_attr.qp_type = IBV_QPT_RC;
        // printf("Function name: %s, line number: %d\n", __func__, __LINE__);
        qp_attr.cap.max_send_wr = 10;
        qp_attr.cap.max_recv_wr = 10;
        qp_attr.cap.max_send_sge = 1;
        qp_attr.cap.max_recv_sge = 1;
        // printf("Function name: %s, line number: %d\n", __func__, __LINE__);
        TEST_Z(host_QPs[i] = ibv_create_qp(s_ctx->pd, &qp_attr));
        if (!host_QPs[i]){
            printf("host_QPs[%d] failed\n", i);
            exit(-1);
        }
        // printf("gpu_qp[i]->qp_num: %d\n", s_ctx->gpu_qp[i]->qp_num);
    }
    struct remote_qp_info host_qp_info; 
    for (int i = 0; i < s_ctx->n_bufs*2; i++){
        host_qp_info.target_qp_num[i] = host_QPs[i]->qp_num;
        // printf("client qp num: %d\n", s_ctx->gpu_qp[i]->qp_num);
    }
    host_qp_info.target_gid = gid;
    host_qp_info.target_lid = mylid;

    // gpu QPs:
    printf("Function name: %s, line number: %d\n", __func__, __LINE__);
    struct remote_qp_info device_qp_info; // = (struct remote_qp_info **) calloc(2, sizeof(struct remote_qp_info *));
    for (int i = 0; i < s_ctx->n_bufs*2; i++){
        device_qp_info.target_qp_num[i] = s_ctx->gpu_qp[i]->qp_num;
        // printf("client qp num: %d\n", s_ctx->gpu_qp[i]->qp_num);
        
    }
    device_qp_info.target_gid = gid;
    device_qp_info.target_lid = mylid;

    for(int i = 0; i < s_ctx->n_bufs*2; i++){
        
        ret = init_qp(s_ctx->gpu_qp[i]); 
        if(ret != 0)
        {
            printf("Failed to modify gpu_qp[%d] to INIT. ret: %d\n", i, ret);
            exit(-1);
        }
        ret = init_qp(host_QPs[i]); 
        if(ret != 0)
        {
            printf("Failed to modify gpu_qp[%d] to INIT. ret: %d\n", i, ret);
            exit(-1);
        }
        

        ret = rtr_qp(s_ctx->gpu_qp[i], host_qp_info.target_qp_num[i], main_port_attr.lid, gid, gid_entry);
        if(ret != 0) {
            printf("Failed to modify gpu_qp[%d] to RTR. ret: %d\n", i, ret);
            exit(-1);
        }
        ret = rtr_qp(host_QPs[i], device_qp_info.target_qp_num[i], main_port_attr.lid, gid, gid_entry);
        if(ret != 0) {
            printf("Failed to modify host_QPs[%d] to RTR. ret: %d\n", i, ret);
            exit(-1);
        }

        rts_qp(s_ctx->gpu_qp[i]);
        if(ret != 0) {
            printf("Failed to modify gpu_qp[%d] to RTS. ret: %d\n", i, ret);
            exit(-1);
        }

        // rts_qp(host_QPs[i]);
        // if(ret != 0) {
        //     printf("Failed to modify host_QPs[%d] to RTS. ret: %d\n", i, ret);
        //     exit(-1);
        // }
    } 




    for (size_t i = 0; i < 1024; i++)
    {
        srv_buffer[i] = 5;
    }

    int *server_temp = (int *) s_ctx->server_memory.addresses[0]; // + 8*1024*1024*1024llu;
    for (size_t i = 0; i < 1024; i++)
    {
        server_temp[i] = 4;
    }
    

    printf("Function name: %s, line number: %d\n", __func__, __LINE__);
    struct ibv_wc wc1;
    // conn = (struct connection *)(uintptr_t)wc.wr_id;
    bad_wr = NULL;
    memset(&wr, 0, sizeof(wr));
    // wr.wr_id = (uintptr_t)conn;
    wr.opcode = IBV_WR_RDMA_READ;
    wr.sg_list = &sge;
    wr.num_sge = 1;
    wr.send_flags = IBV_SEND_SIGNALED;
    wr.wr.rdma.remote_addr = s_ctx->server_memory.addresses[0]; // + 8*1024*1024*1024llu; 
    // (uintptr_t)s_ctx->server_mr.addr + 8*1024*1024*1024llu-1020;//10*1024*1024*1024llu;
    wr.wr.rdma.rkey = s_ctx->server_memory.rkeys[0];// s_ctx->server_mr.rkey;
    sge.addr = (uintptr_t) s_ctx->gpu1_mr->addr; // (uintptr_t)srv_buffer;
    sge.length = 1024*4;
    sge.lkey = s_ctx->gpu1_mr->rkey; // srv_mr->lkey;
    printf("Function name: %s, line number: %d conn->peer_mr.addr: %p\n", __func__, __LINE__, s_ctx->server_memory.addresses[0]);
    printf("Function name: %s, line number: %d conn->peer_mr.rkey: %p\n", __func__, __LINE__, s_ctx->server_memory.rkeys[1]);
    // ret = ibv_post_send(s_ctx->main_qp, &wr, &bad_wr);
    // printf("post ret: %d\n", ret);
    // do{
    //     ret = ibv_poll_cq(s_ctx->main_cq, 1, &wc1);
    //     printf("poll ret: %d\n", ret);
    // }while(ret == 0);

    // process_gpu_mr((int *) s_ctx->gpu_mr->addr, 1024);
    // exit(0);

    // printf("sizeof(srv_buffer): %llu\n", srv_size);
    // bool flag = false;
    // for(int i = 0; i < sge.length/4; i++){
    //     printf(" srv_buffer[%d]: %d ", i, srv_buffer[i]);
    //     if(srv_buffer[i] != 2) {
    //         printf("srv_buffer[%d]: %d\n", i, srv_buffer[i]);
    //         flag = true;
    //         break;
    //     }
    // }
    // if(flag) printf("problem\n");
    // else printf("no problem!\n");

    // exit(0);


    return 0;
}

int local_connect_2gpu_2nic(const char *mlx_name, struct context_2gpu_2nic *s_ctx, int gpu){
    int ret;

    s_ctx->ctx[gpu] = createContext(mlx_name);
    TEST_Z(s_ctx->pd[gpu] = ibv_alloc_pd(s_ctx->ctx[gpu]));
    TEST_Z(s_ctx->comp_channel[gpu] = ibv_create_comp_channel(s_ctx->ctx[gpu]));

    TEST_Z(s_ctx->main_cq[gpu] = ibv_create_cq(s_ctx->ctx[gpu], 10, NULL, s_ctx->comp_channel[gpu], 0)); /* cqe=10 is arbitrary */
    TEST_NZ(ibv_req_notify_cq(s_ctx->main_cq[gpu], 0));

    // this should be GPU 0
    cudaSetDevice(gpu);
    cudaError_t state = cudaMalloc((void **) &s_ctx->gpu_buffer[gpu], s_ctx->gpu_buf_size[gpu]);
    if(state != cudaSuccess){
        printf("Error on cudamalloc\n");
        exit(-1);
    }
    printf("s_ctx->gpu_buf_size[%d]: %llu\n", gpu, s_ctx->gpu_buf_size[gpu]);
    printf("s_ctx->gpu_buffer[%d]: 0x%llx\n", gpu, s_ctx->gpu_buffer[gpu]);

    TEST_Z(s_ctx->gpu_mr[gpu] = ibv_reg_mr(
        s_ctx->pd[gpu], s_ctx->gpu_buffer[gpu], s_ctx->gpu_buf_size[gpu],
        IBV_ACCESS_LOCAL_WRITE
    ));
    
    printf("s_ctx->gpu_mr->addr: 0x%llx\n", s_ctx->gpu_mr[gpu]->addr);
    printf("s_ctx->gpu_mr->lkey: %d\n", s_ctx->gpu_mr[gpu]->lkey);
    
    void *memoryPool;

    if(gpu == 0)
        memoryPool = (void *) malloc(RDMA_BUFFER_SIZE/2);
    else
        memoryPool = (void *) remote_address_2nic[0];
    if(memoryPool == NULL) {
        printf("Memory pool could not be created!\n");
        exit(-1);
    }
    remote_address_2nic[gpu] = (uint64_t) memoryPool;
    struct ibv_mr *temp_mr;
    
    for(int index = 0; index < N_8GB_Region; index++){
        TEST_Z(temp_mr = ibv_reg_mr(
                s_ctx->pd[gpu], 
                memoryPool + index*Region_Size, 
                Region_Size, 
                IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ));
                
        printf("Registered server address: %p, server rkey: %d length: %llu\n\n\n", \
        temp_mr->addr, temp_mr->rkey, temp_mr->length);

        // s_ctx->server_memory.addresses[0]
        // s_ctx->server_memory.lkeys[0]
        // s_ctx->server_memory.rkeys[0]

        s_ctx->server_memory[gpu].addresses[index] = (uint64_t) (memoryPool + index*Region_Size);
        s_ctx->server_memory[gpu].rkeys[index] = temp_mr->rkey;
        s_ctx->server_memory[gpu].lkeys[index] = temp_mr->lkey;
    }  

    if(cudaSuccess != cudaDeviceSynchronize()) return -1;
    // s_ctx->n_bufs = 256;
    s_ctx->cqbuf_size = 4096*2;
    s_ctx->wqbuf_size = 8192;
    // s_ctx->gpu_buf_size = 3*1024*1024;

    // multiply nbefs by 2 because we have 2 GPUs and 2 NICs
    if(s_ctx->wqbuf == NULL)
        s_ctx->wqbuf = (void ** volatile) calloc(s_ctx->n_bufs*2, sizeof(void *));
    if(s_ctx->cqbuf == NULL)
        s_ctx->cqbuf = (void **) calloc(s_ctx->n_bufs*2, sizeof(void *));

    // /**************** Allocate cq abd wq for GPU 0 *******************/
    // for(int i = 0; i < s_ctx->n_bufs; i++){
    //     void* volatile temp;
    //     cudaSetDevice(0);
    //     ret = cudaMalloc((void **)&temp, s_ctx->wqbuf_size);
    //     if (cudaSuccess != ret) {
    //         printf("error on cudaMalloc for wqbuf: %d\n", ret);
    //         exit(0);
    //     }
    //     ret = cudaMemset(temp, 0, s_ctx->wqbuf_size);
    //     if (cudaSuccess != ret) {
    //         printf("error on cudaMemset for wqbuf: %d\n", ret);
    //         exit(0);
    //     }
    //     s_ctx->wqbuf[i] = temp;
    // }
    // for(int i = 0; i < s_ctx->n_bufs; i++){
    //     // printf("s_ctx->wqbuf[i]: 0x%llx\n", s_ctx->wqbuf[i]);
    //     void* volatile temp;
    //     cudaSetDevice(0);
    //     ret = cudaMalloc((void **) &temp, s_ctx->cqbuf_size);
    //     if (cudaSuccess != ret) {
    //         printf("error on cudaMalloc for cqbuf: %d\n", ret);
    //         exit(0);
    //     }
    //     ret = cudaMemset(temp, 0, s_ctx->cqbuf_size);
    //     if (cudaSuccess != ret) {
    //         printf("error on cudaMemset for cqbuf: %d\n", ret);
    //         exit(0);
    //     }
    //     s_ctx->cqbuf[i] = temp;
    //     if(cudaSuccess != cudaDeviceSynchronize()) return -1;
    //     invalidate_cq<<<1,1>>>(s_ctx->cqbuf[i]);
    //     if(cudaSuccess != cudaDeviceSynchronize()) return -1;
    //     printf("s_ctx->cqbuf[%d]: 0x%llx\n", i, s_ctx->cqbuf[i]);
    // }
    /**************** Allocate cq abd wq for GPU 1 *******************/
    for(int i = s_ctx->n_bufs*(gpu); i < s_ctx->n_bufs*(gpu + 1); i++){
        void* volatile temp;
        cudaSetDevice(gpu);
        ret = cudaMalloc((void **)&temp, s_ctx->wqbuf_size);
        if (cudaSuccess != ret) {
            printf("error on cudaMalloc for wqbuf: %d\n", ret);
            exit(0);
        }
        ret = cudaMemset(temp, 0, s_ctx->wqbuf_size);
        if (cudaSuccess != ret) {
            printf("error on cudaMemset for wqbuf: %d\n", ret);
            exit(0);
        }
        s_ctx->wqbuf[i] = temp;
    }
    for(int i = s_ctx->n_bufs*(gpu); i < s_ctx->n_bufs*(gpu+1); i++){
        printf("s_ctx->wqbuf[i]: 0x%llx\n", s_ctx->wqbuf[i]);
        void* volatile temp;
        cudaSetDevice(gpu);
        ret = cudaMalloc((void **) &temp, s_ctx->cqbuf_size);
        if (cudaSuccess != ret) {
            printf("error on cudaMalloc for cqbuf: %d\n", ret);
            exit(0);
        }
        ret = cudaMemset(temp, 0, s_ctx->cqbuf_size);
        if (cudaSuccess != ret) {
            printf("error on cudaMemset for cqbuf: %d\n", ret);
            exit(0);
        }
        s_ctx->cqbuf[i] = temp;
        if(cudaSuccess != cudaDeviceSynchronize()) return -1;
        invalidate_cq<<<1,1>>>(s_ctx->cqbuf[i]);
        if(cudaSuccess != cudaDeviceSynchronize()) return -1;
        printf("s_ctx->cqbuf[%d]: 0x%llx\n", i, s_ctx->cqbuf[i]);
    }
    /**************************************************************/
    
    // exit(0);
    if(s_ctx->gpu_cq == NULL)
        s_ctx->gpu_cq = (struct ibv_cq **) calloc(s_ctx->n_bufs*2, sizeof(struct ibv_cq *));
    for(int i = s_ctx->n_bufs*gpu; i < s_ctx->n_bufs*(gpu + 1); i++){
        TEST_Z(s_ctx->gpu_cq[i] = ibvx_create_cq(s_ctx->ctx[gpu], 10, NULL, s_ctx->comp_channel[gpu], 0, s_ctx->cqbuf[i], s_ctx->cqbuf_size)); /* cqe=10 is arbitrary */
        TEST_NZ(ibv_req_notify_cq(s_ctx->gpu_cq[i], 0));
    }

    struct mlx5_cq *cq1 = to_mcq(s_ctx->gpu_cq[0]);
    printf("gpu: %d s_ctx->gpu_cq[0]->buf: %p\n", gpu, cq1->active_buf->buf);

    printf("Function name: %s, line number: %d s_ctx->n_bufs: %d\n", __func__, __LINE__, s_ctx->n_bufs);
    if(s_ctx->gpu_qp == NULL)
        s_ctx->gpu_qp = (struct ibv_qp **) calloc(s_ctx->n_bufs*2, sizeof(struct ibv_qp *));
    for(int i = s_ctx->n_bufs*gpu; i < s_ctx->n_bufs*(gpu + 1); i++){
        struct ibv_qp_init_attr qp_attr;
        memset(&qp_attr, 0, sizeof(qp_attr));
        // printf("Function name: %s, line number: %d\n", __func__, __LINE__);
        qp_attr.send_cq = s_ctx->gpu_cq[i];
        qp_attr.recv_cq = s_ctx->gpu_cq[i];
        qp_attr.qp_type = IBV_QPT_RC;
        // printf("Function name: %s, line number: %d\n", __func__, __LINE__);
        qp_attr.cap.max_send_wr = 10;
        qp_attr.cap.max_recv_wr = 10;
        qp_attr.cap.max_send_sge = 1;
        qp_attr.cap.max_recv_sge = 1;
        // printf("Function name: %s, line number: %d\n", __func__, __LINE__);
        TEST_Z(s_ctx->gpu_qp[i] = ibvx_create_qp(s_ctx->pd[gpu], &qp_attr, s_ctx->wqbuf[i], s_ctx->wqbuf_size));
        if (!s_ctx->gpu_qp[i]){
            printf("g_qp failed\n");
            exit(-1);
        }
        // printf("gpu_qp[%d]->qp_num: %d\n", i, s_ctx->gpu_qp[i]->qp_num);
    }

    struct ibv_qp_init_attr main_qp_attr;
    memset(&main_qp_attr, 0, sizeof(main_qp_attr));
        
        main_qp_attr.send_cq = s_ctx->main_cq[gpu];
        main_qp_attr.recv_cq = s_ctx->main_cq[gpu];
        main_qp_attr.qp_type = IBV_QPT_RC;

        main_qp_attr.cap.max_send_wr = 10;
        main_qp_attr.cap.max_recv_wr = 10;
        main_qp_attr.cap.max_send_sge = 1;
        main_qp_attr.cap.max_recv_sge = 1;

    ibv_qp *temp_qp;
    TEST_Z(s_ctx->main_qp[gpu] = ibv_create_qp(s_ctx->pd[gpu], &main_qp_attr));
    TEST_Z(temp_qp = ibv_create_qp(s_ctx->pd[gpu], &main_qp_attr));

    struct ibv_qp_attr qp_attr1;
    

    if(init_qp(s_ctx->main_qp[gpu]) != 0)
    {
        printf("Failed to modify main qp to INIT. ret: %d\n", ret);
        exit(-1);
    }
    if(init_qp(temp_qp) != 0)
    {
        printf("Failed to modify main qp to INIT. ret: %d\n", ret);
        exit(-1);
    }

    ibv_port_attr main_port_attr, temp_port_attr;
    ibv_query_port(s_ctx->ctx[gpu], 1, &main_port_attr);
    ibv_query_port(s_ctx->ctx[gpu], 1, &temp_port_attr);

    union ibv_gid gid;; 
    int gid_entry;
    int mylid;
    get_myglid(s_ctx->ctx[gpu], 1, &gid, &gid_entry);
    get_mylid(s_ctx->ctx[gpu], 1, &mylid);

    ret = rtr_qp(s_ctx->main_qp[gpu], temp_qp->qp_num, temp_port_attr.lid, gid, gid_entry);
    if(ret != 0) {
        printf("Failed to modify main qp to RTR. ret: %d\n", ret);
        exit(-1);
    }

    ret = rtr_qp(temp_qp, s_ctx->main_qp[gpu]->qp_num, main_port_attr.lid, gid, gid_entry);
    if(ret != 0) {
        printf("Failed to modify temp qp to RTR. ret: %d\n", ret);
        exit(-1);
    }

    ret = rts_qp(s_ctx->main_qp[gpu]);
    if(ret != 0) {
        printf("Failed to modify qp to RTS. ret: %d\n", ret);
        exit(-1);
    }

    struct ibv_send_wr wr, *bad_wr = NULL;
    struct ibv_sge sge;
    struct ibv_wc wc;
    size_t srv_size = 1024;
    int *srv_buffer = (int *) malloc(srv_size*sizeof(int));
    struct ibv_mr *srv_mr;
    
    TEST_Z(srv_mr = ibv_reg_mr(
        s_ctx->pd[gpu], srv_buffer, srv_size*sizeof(int),
        IBV_ACCESS_LOCAL_WRITE
    ));

    // server QPs:
    
    struct ibv_qp **host_QPs = (struct ibv_qp **) calloc(s_ctx->n_bufs, sizeof(struct ibv_qp *));
    for(int i = s_ctx->n_bufs*gpu; i < s_ctx->n_bufs*(gpu+1); i++){
        struct ibv_qp_init_attr qp_attr;
        memset(&qp_attr, 0, sizeof(qp_attr));
        // printf("Function name: %s, line number: %d\n", __func__, __LINE__);
        qp_attr.send_cq = s_ctx->main_cq[gpu];
        qp_attr.recv_cq = s_ctx->main_cq[gpu];
        qp_attr.qp_type = IBV_QPT_RC;
        // printf("Function name: %s, line number: %d\n", __func__, __LINE__);
        qp_attr.cap.max_send_wr = 10;
        qp_attr.cap.max_recv_wr = 10;
        qp_attr.cap.max_send_sge = 1;
        qp_attr.cap.max_recv_sge = 1;
        // printf("Function name: %s, line number: %d\n", __func__, __LINE__);
        TEST_Z(host_QPs[i-s_ctx->n_bufs*gpu] = ibv_create_qp(s_ctx->pd[gpu], &qp_attr));
        if (!host_QPs[i-s_ctx->n_bufs*gpu]){
            printf("host_QPs[%d] failed gpu: %d\n", i, gpu);
            exit(-1);
        }
        // printf("gpu_qp[i]->qp_num: %d\n", s_ctx->gpu_qp[i]->qp_num);
    }
    struct remote_qp_info host_qp_info; 
    for (int i = 0; i < s_ctx->n_bufs; i++){
        host_qp_info.target_qp_num[i] = host_QPs[i]->qp_num;
        // printf("client qp num: %d\n", s_ctx->gpu_qp[i]->qp_num);
    }
    host_qp_info.target_gid = gid;
    host_qp_info.target_lid = mylid;

    // gpu QPs:
    printf("Function name: %s, line number: %d\n", __func__, __LINE__);
    struct remote_qp_info device_qp_info; // = (struct remote_qp_info **) calloc(2, sizeof(struct remote_qp_info *));
    for (int i = s_ctx->n_bufs*gpu; i < s_ctx->n_bufs*(gpu+1); i++){
        device_qp_info.target_qp_num[i - s_ctx->n_bufs*gpu] = s_ctx->gpu_qp[i]->qp_num;
        // printf("client qp num: %d\n", s_ctx->gpu_qp[i]->qp_num);
        
    }
    device_qp_info.target_gid = gid;
    device_qp_info.target_lid = mylid;

    for(int i = s_ctx->n_bufs*gpu; i < s_ctx->n_bufs*(gpu+1); i++){
        
        ret = init_qp(s_ctx->gpu_qp[i]); 
        if(ret != 0)
        {
            printf("Failed to modify gpu_qp[%d] to INIT. ret: %d\n", i, ret);
            exit(-1);
        }
        ret = init_qp(host_QPs[i - s_ctx->n_bufs*gpu]); 
        if(ret != 0)
        {
            printf("Failed to modify gpu_qp[%d] to INIT. ret: %d\n", i, ret);
            exit(-1);
        }
        

        ret = rtr_qp(s_ctx->gpu_qp[i], host_qp_info.target_qp_num[i - s_ctx->n_bufs*gpu], main_port_attr.lid, gid, gid_entry);
        if(ret != 0) {
            printf("Failed to modify gpu_qp[%d] to RTR. ret: %d\n", i, ret);
            exit(-1);
        }
        ret = rtr_qp(host_QPs[i - s_ctx->n_bufs*gpu], device_qp_info.target_qp_num[i - s_ctx->n_bufs*gpu], main_port_attr.lid, gid, gid_entry);
        if(ret != 0) {
            printf("Failed to modify host_QPs[%d] to RTR. ret: %d\n", i, ret);
            exit(-1);
        }

        rts_qp(s_ctx->gpu_qp[i]);
        if(ret != 0) {
            printf("Failed to modify gpu_qp[%d] to RTS. ret: %d\n", i, ret);
            exit(-1);
        }

        // rts_qp(host_QPs[i]);
        // if(ret != 0) {
        //     printf("Failed to modify host_QPs[%d] to RTS. ret: %d\n", i, ret);
        //     exit(-1);
        // }
    } 




    for (size_t i = 0; i < 1024; i++)
    {
        srv_buffer[i] = 5;
    }

    int *server_temp = (int *) s_ctx->server_memory[gpu].addresses[0]; // + 8*1024*1024*1024llu;
    for (size_t i = 0; i < 1024; i++)
    {
        server_temp[i] = 4;
    }
    

    printf("Function name: %s, line number: %d\n", __func__, __LINE__);
    struct ibv_wc wc1;
    // conn = (struct connection *)(uintptr_t)wc.wr_id;
    bad_wr = NULL;
    memset(&wr, 0, sizeof(wr));
    // wr.wr_id = (uintptr_t)conn;
    wr.opcode = IBV_WR_RDMA_READ;
    wr.sg_list = &sge;
    wr.num_sge = 1;
    wr.send_flags = IBV_SEND_SIGNALED;
    wr.wr.rdma.remote_addr = s_ctx->server_memory[gpu].addresses[0]; // + 8*1024*1024*1024llu; 
    // (uintptr_t)s_ctx->server_mr.addr + 8*1024*1024*1024llu-1020;//10*1024*1024*1024llu;
    wr.wr.rdma.rkey = s_ctx->server_memory[gpu].rkeys[0];// s_ctx->server_mr.rkey;
    sge.addr = (uintptr_t) s_ctx->gpu_mr[gpu]->addr; // (uintptr_t)srv_buffer;
    sge.length = 1024*4;
    sge.lkey = s_ctx->gpu_mr[gpu]->rkey; // srv_mr->lkey;
    printf("Function name: %s, line number: %d conn->peer_mr.addr: %p\n", __func__, __LINE__, s_ctx->server_memory[gpu].addresses[0]);
    printf("Function name: %s, line number: %d conn->peer_mr.rkey: %p\n", __func__, __LINE__, s_ctx->server_memory[gpu].rkeys[1]);
    // ret = ibv_post_send(s_ctx->main_qp, &wr, &bad_wr);
    // printf("post ret: %d\n", ret);
    // do{
    //     ret = ibv_poll_cq(s_ctx->main_cq, 1, &wc1);
    //     printf("poll ret: %d\n", ret);
    // }while(ret == 0);

    // process_gpu_mr((int *) s_ctx->gpu_mr->addr, 1024);
    // exit(0);

    // printf("sizeof(srv_buffer): %llu\n", srv_size);
    // bool flag = false;
    // for(int i = 0; i < sge.length/4; i++){
    //     printf(" srv_buffer[%d]: %d ", i, srv_buffer[i]);
    //     if(srv_buffer[i] != 2) {
    //         printf("srv_buffer[%d]: %d\n", i, srv_buffer[i]);
    //         flag = true;
    //         break;
    //     }
    // }
    // if(flag) printf("problem\n");
    // else printf("no problem!\n");

    // exit(0);


    return 0;
}

int local_connect_2nic(const char *mlx_name, struct context_2nic *s_ctx, int nic, int gpu){
    int ret;

    s_ctx->ctx[nic] = createContext(mlx_name);
    TEST_Z(s_ctx->pd[nic] = ibv_alloc_pd(s_ctx->ctx[nic]));
    TEST_Z(s_ctx->comp_channel[nic] = ibv_create_comp_channel(s_ctx->ctx[nic]));

    TEST_Z(s_ctx->main_cq[nic] = ibv_create_cq(s_ctx->ctx[nic], 10, NULL, s_ctx->comp_channel[nic], 0)); /* cqe=10 is arbitrary */
    TEST_NZ(ibv_req_notify_cq(s_ctx->main_cq[nic], 0));

    // this should be GPU 0
    cudaSetDevice(gpu);
    if(s_ctx->gpu_buffer == NULL){
        cudaError_t state = cudaMalloc((void **) &s_ctx->gpu_buffer, s_ctx->gpu_buf_size);
        if(state != cudaSuccess){
            printf("Error on cudamalloc\n");
            exit(-1);
        }
    }
    printf("nic: %d gpu: %d s_ctx->gpu_buf_size: %llu\n", nic, gpu, s_ctx->gpu_buf_size);
    printf("nic: %d gpu: %d s_ctx->gpu_buffer: 0x%llx\n", nic, gpu, s_ctx->gpu_buffer);

    TEST_Z(s_ctx->gpu_mr[nic] = ibv_reg_mr(
        s_ctx->pd[nic], s_ctx->gpu_buffer, s_ctx->gpu_buf_size,
        IBV_ACCESS_LOCAL_WRITE
    ));
    
    printf("s_ctx->gpu_mr->addr: 0x%llx\n", s_ctx->gpu_mr[nic]->addr);
    printf("s_ctx->gpu_mr->lkey: %d\n", s_ctx->gpu_mr[nic]->lkey);
    
    void *memoryPool;

    // I assume first nic=0 will be called
    if(nic == 0)
        memoryPool = (void *) malloc(RDMA_BUFFER_SIZE/2);
    else
        memoryPool = (void *) remote_address_2nic[0];
    if(memoryPool == NULL) {
        printf("Memory pool could not be created!\n");
        exit(-1);
    }
    remote_address_2nic[nic] = (uint64_t) memoryPool;
    struct ibv_mr *temp_mr;
    
    for(int index = 0; index < N_8GB_Region; index++){
        TEST_Z(temp_mr = ibv_reg_mr(
                s_ctx->pd[nic], 
                memoryPool + index*Region_Size, 
                Region_Size, 
                IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ));
                
        printf("Registered server address: %p, server rkey: %d length: %llu\n\n\n", \
        temp_mr->addr, temp_mr->rkey, temp_mr->length);

        // s_ctx->server_memory.addresses[0]
        // s_ctx->server_memory.lkeys[0]
        // s_ctx->server_memory.rkeys[0]

        s_ctx->server_memory[nic].addresses[index] = (uint64_t) (memoryPool + index*Region_Size);
        s_ctx->server_memory[nic].rkeys[index] = temp_mr->rkey;
        s_ctx->server_memory[nic].lkeys[index] = temp_mr->lkey;
    }  

    if(cudaSuccess != cudaDeviceSynchronize()) return -1;
    // s_ctx->n_bufs = 256;
    s_ctx->cqbuf_size = 4096*2;
    s_ctx->wqbuf_size = 8192;
    // s_ctx->gpu_buf_size = 3*1024*1024;

    // multiply nbefs by 2 because we have 2 GPUs and 2 NICs
    if(s_ctx->wqbuf == NULL)
        s_ctx->wqbuf = (void ** volatile) calloc(s_ctx->n_bufs*2, sizeof(void *));
    if(s_ctx->cqbuf == NULL)
        s_ctx->cqbuf = (void **) calloc(s_ctx->n_bufs*2, sizeof(void *));

    // /**************** Allocate cq abd wq for GPU 0 *******************/
    // for(int i = 0; i < s_ctx->n_bufs; i++){
    //     void* volatile temp;
    //     cudaSetDevice(0);
    //     ret = cudaMalloc((void **)&temp, s_ctx->wqbuf_size);
    //     if (cudaSuccess != ret) {
    //         printf("error on cudaMalloc for wqbuf: %d\n", ret);
    //         exit(0);
    //     }
    //     ret = cudaMemset(temp, 0, s_ctx->wqbuf_size);
    //     if (cudaSuccess != ret) {
    //         printf("error on cudaMemset for wqbuf: %d\n", ret);
    //         exit(0);
    //     }
    //     s_ctx->wqbuf[i] = temp;
    // }
    // for(int i = 0; i < s_ctx->n_bufs; i++){
    //     // printf("s_ctx->wqbuf[i]: 0x%llx\n", s_ctx->wqbuf[i]);
    //     void* volatile temp;
    //     cudaSetDevice(0);
    //     ret = cudaMalloc((void **) &temp, s_ctx->cqbuf_size);
    //     if (cudaSuccess != ret) {
    //         printf("error on cudaMalloc for cqbuf: %d\n", ret);
    //         exit(0);
    //     }
    //     ret = cudaMemset(temp, 0, s_ctx->cqbuf_size);
    //     if (cudaSuccess != ret) {
    //         printf("error on cudaMemset for cqbuf: %d\n", ret);
    //         exit(0);
    //     }
    //     s_ctx->cqbuf[i] = temp;
    //     if(cudaSuccess != cudaDeviceSynchronize()) return -1;
    //     invalidate_cq<<<1,1>>>(s_ctx->cqbuf[i]);
    //     if(cudaSuccess != cudaDeviceSynchronize()) return -1;
    //     printf("s_ctx->cqbuf[%d]: 0x%llx\n", i, s_ctx->cqbuf[i]);
    // }
    /**************** Allocate cq abd wq for GPU 1 *******************/
    for(int i = s_ctx->n_bufs*(nic); i < s_ctx->n_bufs*(nic + 1); i++){
        void* volatile temp;
        cudaSetDevice(gpu);
        ret = cudaMalloc((void **)&temp, s_ctx->wqbuf_size);
        if (cudaSuccess != ret) {
            printf("error on cudaMalloc for wqbuf: %d\n", ret);
            exit(0);
        }
        ret = cudaMemset(temp, 0, s_ctx->wqbuf_size);
        if (cudaSuccess != ret) {
            printf("error on cudaMemset for wqbuf: %d\n", ret);
            exit(0);
        }
        s_ctx->wqbuf[i] = temp;
    }
    // for(int i = s_ctx->n_bufs*(nic); i < s_ctx->n_bufs*(nic+1); i++){
    //     printf("s_ctx->wqbuf[i]: 0x%llx\n", s_ctx->wqbuf[i]);
    //     void* volatile temp;
    //     cudaSetDevice(gpu);
    //     ret = cudaMalloc((void **) &temp, s_ctx->cqbuf_size);
    //     if (cudaSuccess != ret) {
    //         printf("error on cudaMalloc for cqbuf: %d\n", ret);
    //         exit(0);
    //     }
    //     ret = cudaMemset(temp, 0, s_ctx->cqbuf_size);
    //     if (cudaSuccess != ret) {
    //         printf("error on cudaMemset for cqbuf: %d\n", ret);
    //         exit(0);
    //     }
    //     s_ctx->cqbuf[i] = temp;
    //     if(cudaSuccess != cudaDeviceSynchronize()) return -1;
    //     invalidate_cq<<<1,1>>>(s_ctx->cqbuf[i]);
    //     if(cudaSuccess != cudaDeviceSynchronize()) return -1;
    //     printf("s_ctx->cqbuf[%d]: 0x%llx\n", i, s_ctx->cqbuf[i]);
    // }
    void* volatile temp;
    cudaSetDevice(gpu);
    ret = cudaMalloc((void **) &temp, s_ctx->cqbuf_size*s_ctx->n_bufs);
    if (cudaSuccess != ret) {
        printf("error on cudaMalloc for cqbuf: %d\n", ret);
        exit(0);
    }
    ret = cudaMemset(temp, 0, s_ctx->cqbuf_size*s_ctx->n_bufs);
    if (cudaSuccess != ret) {
        printf("error on cudaMemset for cqbuf: %d\n", ret);
        exit(0);
    }
    for(int i = s_ctx->n_bufs*(nic); i < s_ctx->n_bufs*(nic+1); i++){
        printf("s_ctx->wqbuf[i]: 0x%llx\n", s_ctx->wqbuf[i]);
        
        s_ctx->cqbuf[i] = temp + (i - nic*s_ctx->n_bufs)*s_ctx->cqbuf_size;
        if(cudaSuccess != cudaDeviceSynchronize()) return -1;
        invalidate_cq<<<1,1>>>(s_ctx->cqbuf[i]);
        if(cudaSuccess != cudaDeviceSynchronize()) return -1;
        printf("s_ctx->cqbuf[%d]: 0x%llx\n", i, s_ctx->cqbuf[i]);
    }
    /**************************************************************/
    
    // exit(0);
    if(s_ctx->gpu_cq == NULL)
        s_ctx->gpu_cq = (struct ibv_cq **) calloc(s_ctx->n_bufs*2, sizeof(struct ibv_cq *));
    for(int i = s_ctx->n_bufs*nic; i < s_ctx->n_bufs*(nic + 1); i++){
        TEST_Z(s_ctx->gpu_cq[i] = ibvx_create_cq(s_ctx->ctx[nic], 10, NULL, s_ctx->comp_channel[nic], 0, s_ctx->cqbuf[i], s_ctx->cqbuf_size)); /* cqe=10 is arbitrary */
        TEST_NZ(ibv_req_notify_cq(s_ctx->gpu_cq[i], 0));
    }

    struct mlx5_cq *cq1 = to_mcq(s_ctx->gpu_cq[0]);
    printf("gpu: %d s_ctx->gpu_cq[0]->buf: %p\n", gpu, cq1->active_buf->buf);

    printf("Function name: %s, line number: %d s_ctx->n_bufs: %d\n", __func__, __LINE__, s_ctx->n_bufs);
    if(s_ctx->gpu_qp == NULL)
        s_ctx->gpu_qp = (struct ibv_qp **) calloc(s_ctx->n_bufs*2, sizeof(struct ibv_qp *));
    for(int i = s_ctx->n_bufs*nic; i < s_ctx->n_bufs*(nic + 1); i++){
        struct ibv_qp_init_attr qp_attr;
        memset(&qp_attr, 0, sizeof(qp_attr));
        // printf("Function name: %s, line number: %d\n", __func__, __LINE__);
        qp_attr.send_cq = s_ctx->gpu_cq[i];
        qp_attr.recv_cq = s_ctx->gpu_cq[i];
        qp_attr.qp_type = IBV_QPT_RC;
        // printf("Function name: %s, line number: %d\n", __func__, __LINE__);
        qp_attr.cap.max_send_wr = 10;
        qp_attr.cap.max_recv_wr = 10;
        qp_attr.cap.max_send_sge = 1;
        qp_attr.cap.max_recv_sge = 1;
        // printf("Function name: %s, line number: %d\n", __func__, __LINE__);
        TEST_Z(s_ctx->gpu_qp[i] = ibvx_create_qp(s_ctx->pd[nic], &qp_attr, s_ctx->wqbuf[i], s_ctx->wqbuf_size));
        if (!s_ctx->gpu_qp[i]){
            printf("g_qp failed\n");
            exit(-1);
        }
        // printf("gpu_qp[%d]->qp_num: %d\n", i, s_ctx->gpu_qp[i]->qp_num);
    }

    struct ibv_qp_init_attr main_qp_attr;
    memset(&main_qp_attr, 0, sizeof(main_qp_attr));
        
        main_qp_attr.send_cq = s_ctx->main_cq[nic];
        main_qp_attr.recv_cq = s_ctx->main_cq[nic];
        main_qp_attr.qp_type = IBV_QPT_RC;

        main_qp_attr.cap.max_send_wr = 10;
        main_qp_attr.cap.max_recv_wr = 10;
        main_qp_attr.cap.max_send_sge = 1;
        main_qp_attr.cap.max_recv_sge = 1;

    ibv_qp *temp_qp;
    TEST_Z(s_ctx->main_qp[nic] = ibv_create_qp(s_ctx->pd[nic], &main_qp_attr));
    TEST_Z(temp_qp = ibv_create_qp(s_ctx->pd[nic], &main_qp_attr));

    struct ibv_qp_attr qp_attr1;
    

    if(init_qp(s_ctx->main_qp[nic]) != 0)
    {
        printf("Failed to modify main qp to INIT. ret: %d\n", ret);
        exit(-1);
    }
    if(init_qp(temp_qp) != 0)
    {
        printf("Failed to modify main qp to INIT. ret: %d\n", ret);
        exit(-1);
    }

    ibv_port_attr main_port_attr, temp_port_attr;
    ibv_query_port(s_ctx->ctx[nic], 1, &main_port_attr);
    ibv_query_port(s_ctx->ctx[nic], 1, &temp_port_attr);

    union ibv_gid gid;; 
    int gid_entry;
    int mylid;
    get_myglid(s_ctx->ctx[nic], 1, &gid, &gid_entry);
    get_mylid(s_ctx->ctx[nic], 1, &mylid);

    ret = rtr_qp(s_ctx->main_qp[nic], temp_qp->qp_num, temp_port_attr.lid, gid, gid_entry);
    if(ret != 0) {
        printf("Failed to modify main qp to RTR. ret: %d\n", ret);
        exit(-1);
    }

    ret = rtr_qp(temp_qp, s_ctx->main_qp[nic]->qp_num, main_port_attr.lid, gid, gid_entry);
    if(ret != 0) {
        printf("Failed to modify temp qp to RTR. ret: %d\n", ret);
        exit(-1);
    }

    ret = rts_qp(s_ctx->main_qp[nic]);
    if(ret != 0) {
        printf("Failed to modify qp to RTS. ret: %d\n", ret);
        exit(-1);
    }

    struct ibv_send_wr wr, *bad_wr = NULL;
    struct ibv_sge sge;
    struct ibv_wc wc;
    size_t srv_size = 1024;
    int *srv_buffer = (int *) malloc(srv_size*sizeof(int));
    struct ibv_mr *srv_mr;
    
    TEST_Z(srv_mr = ibv_reg_mr(
        s_ctx->pd[nic], srv_buffer, srv_size*sizeof(int),
        IBV_ACCESS_LOCAL_WRITE
    ));

    // server QPs:
    
    struct ibv_qp **host_QPs = (struct ibv_qp **) calloc(s_ctx->n_bufs, sizeof(struct ibv_qp *));
    for(int i = s_ctx->n_bufs*nic; i < s_ctx->n_bufs*(nic+1); i++){
        struct ibv_qp_init_attr qp_attr;
        memset(&qp_attr, 0, sizeof(qp_attr));
        // printf("Function name: %s, line number: %d\n", __func__, __LINE__);
        qp_attr.send_cq = s_ctx->main_cq[nic];
        qp_attr.recv_cq = s_ctx->main_cq[nic];
        qp_attr.qp_type = IBV_QPT_RC;
        // printf("Function name: %s, line number: %d\n", __func__, __LINE__);
        qp_attr.cap.max_send_wr = 10;
        qp_attr.cap.max_recv_wr = 10;
        qp_attr.cap.max_send_sge = 1;
        qp_attr.cap.max_recv_sge = 1;
        // printf("Function name: %s, line number: %d\n", __func__, __LINE__);
        TEST_Z(host_QPs[i-s_ctx->n_bufs*nic] = ibv_create_qp(s_ctx->pd[nic], &qp_attr));
        if (!host_QPs[i-s_ctx->n_bufs*nic]){
            printf("host_QPs[%d] failed nic: %d\n", i, nic);
            exit(-1);
        }
        // printf("gpu_qp[i]->qp_num: %d\n", s_ctx->gpu_qp[i]->qp_num);
    }
    struct remote_qp_info host_qp_info; 
    for (int i = 0; i < s_ctx->n_bufs; i++){
        host_qp_info.target_qp_num[i] = host_QPs[i]->qp_num;
        // printf("client qp num: %d\n", s_ctx->gpu_qp[i]->qp_num);
    }
    host_qp_info.target_gid = gid;
    host_qp_info.target_lid = mylid;

    // gpu QPs:
    printf("Function name: %s, line number: %d\n", __func__, __LINE__);
    struct remote_qp_info device_qp_info; // = (struct remote_qp_info **) calloc(2, sizeof(struct remote_qp_info *));
    for (int i = s_ctx->n_bufs*nic; i < s_ctx->n_bufs*(nic+1); i++){
        device_qp_info.target_qp_num[i - s_ctx->n_bufs*nic] = s_ctx->gpu_qp[i]->qp_num;
        // printf("client qp num: %d\n", s_ctx->gpu_qp[i]->qp_num);
        
    }
    device_qp_info.target_gid = gid;
    device_qp_info.target_lid = mylid;

    for(int i = s_ctx->n_bufs*nic; i < s_ctx->n_bufs*(nic+1); i++){
        
        ret = init_qp(s_ctx->gpu_qp[i]); 
        if(ret != 0)
        {
            printf("Failed to modify gpu_qp[%d] to INIT. ret: %d\n", i, ret);
            exit(-1);
        }
        ret = init_qp(host_QPs[i - s_ctx->n_bufs*nic]); 
        if(ret != 0)
        {
            printf("Failed to modify gpu_qp[%d] to INIT. ret: %d\n", i, ret);
            exit(-1);
        }
        

        ret = rtr_qp(s_ctx->gpu_qp[i], host_qp_info.target_qp_num[i - s_ctx->n_bufs*nic], main_port_attr.lid, gid, gid_entry);
        if(ret != 0) {
            printf("Failed to modify gpu_qp[%d] to RTR. ret: %d\n", i, ret);
            exit(-1);
        }
        ret = rtr_qp(host_QPs[i - s_ctx->n_bufs*nic], device_qp_info.target_qp_num[i - s_ctx->n_bufs*nic], main_port_attr.lid, gid, gid_entry);
        if(ret != 0) {
            printf("Failed to modify host_QPs[%d] to RTR. ret: %d\n", i, ret);
            exit(-1);
        }

        rts_qp(s_ctx->gpu_qp[i]);
        if(ret != 0) {
            printf("Failed to modify gpu_qp[%d] to RTS. ret: %d\n", i, ret);
            exit(-1);
        }

        // rts_qp(host_QPs[i]);
        // if(ret != 0) {
        //     printf("Failed to modify host_QPs[%d] to RTS. ret: %d\n", i, ret);
        //     exit(-1);
        // }
    } 




    for (size_t i = 0; i < 1024; i++)
    {
        srv_buffer[i] = 5;
    }

    int *server_temp = (int *) s_ctx->server_memory[nic].addresses[0]; // + 8*1024*1024*1024llu;
    for (size_t i = 0; i < 1024; i++)
    {
        server_temp[i] = 4;
    }
    

    printf("Function name: %s, line number: %d\n", __func__, __LINE__);
    struct ibv_wc wc1;
    // conn = (struct connection *)(uintptr_t)wc.wr_id;
    bad_wr = NULL;
    memset(&wr, 0, sizeof(wr));
    // wr.wr_id = (uintptr_t)conn;
    wr.opcode = IBV_WR_RDMA_READ;
    wr.sg_list = &sge;
    wr.num_sge = 1;
    wr.send_flags = IBV_SEND_SIGNALED;
    wr.wr.rdma.remote_addr = s_ctx->server_memory[nic].addresses[0]; // + 8*1024*1024*1024llu; 
    // (uintptr_t)s_ctx->server_mr.addr + 8*1024*1024*1024llu-1020;//10*1024*1024*1024llu;
    wr.wr.rdma.rkey = s_ctx->server_memory[nic].rkeys[0];// s_ctx->server_mr.rkey;
    sge.addr = (uintptr_t) s_ctx->gpu_mr[nic]->addr; // (uintptr_t)srv_buffer;
    sge.length = 1024*4;
    sge.lkey = s_ctx->gpu_mr[nic]->rkey; // srv_mr->lkey;
    printf("Function name: %s, line number: %d conn->peer_mr.addr: %p\n", __func__, __LINE__, s_ctx->server_memory[nic].addresses[0]);
    printf("Function name: %s, line number: %d conn->peer_mr.rkey: %p\n", __func__, __LINE__, s_ctx->server_memory[nic].rkeys[1]);
    // ret = ibv_post_send(s_ctx->main_qp, &wr, &bad_wr);
    // printf("post ret: %d\n", ret);
    // do{
    //     ret = ibv_poll_cq(s_ctx->main_cq, 1, &wc1);
    //     printf("poll ret: %d\n", ret);
    // }while(ret == 0);

    // process_gpu_mr((int *) s_ctx->gpu_mr->addr, 1024);
    // exit(0);

    // printf("sizeof(srv_buffer): %llu\n", srv_size);
    // bool flag = false;
    // for(int i = 0; i < sge.length/4; i++){
    //     printf(" srv_buffer[%d]: %d ", i, srv_buffer[i]);
    //     if(srv_buffer[i] != 2) {
    //         printf("srv_buffer[%d]: %d\n", i, srv_buffer[i]);
    //         flag = true;
    //         break;
    //     }
    // }
    // if(flag) printf("problem\n");
    // else printf("no problem!\n");

    // exit(0);


    return 0;
}

void *benchmark(void *param1){
    struct benchmark_content *param = (struct benchmark_content *) param1;
    cpu_benchmark_whole(param->cq_ptr, 1, param->wc, param->ibqp, param->wr, \
                        param->bad_wr, param->num_packets, param->mesg_size, param->bandwidth);

}

__global__ void poll_fake(void *cq_buf, uint32_t *cons_index,
                    int ibv_cqe, uint32_t cqe_sz, int n, void *cq_dbrec){
    struct ibv_wc wc;                
    while(poll(cq_buf, &wc, cons_index, ibv_cqe, cqe_sz, n, cq_dbrec) == 0);
    printf("wc->status: %d\n", wc.status);
    if (wc.status != IBV_WC_SUCCESS){
        printf("wc->status: %d\n", wc.status);
        // die("on_completion: status is not IBV_WC_SUCCESS.");
    }
    
}

void host_poll_fake(struct ibv_cq *cq1, struct ibv_wc *wc){
    struct mlx5_cq *cq = to_mcq(cq1);
    void *cqe;
    // printf("Function: %s line number: %d cq->buf.length: %d\n",__func__, __LINE__, cq->buf_a.length);
    struct mlx5_cqe64 *cqe64;
    // int cond = 0;
    uint32_t cons_index = cq->cons_index;
    int cq_cqe = cq->verbs_cq.cq.cqe;
    int cq_cqe_sz = cq->cqe_sz;
    void *cq_buf_a = cq->buf_a.buf; 
    void *dev_cons_index, *dev_cq_dbrec;

    cudaError_t cudaStatus = cudaHostRegister(&cq->cons_index, sizeof(cq->cons_index), cudaHostRegisterMapped);
    if(cudaStatus != cudaSuccess && cudaStatus != cudaErrorHostMemoryAlreadyRegistered)
        exit(0);
    // get GPU pointer for cons index
    if(cudaHostGetDevicePointer(&dev_cons_index, &cq->cons_index, 0) != cudaSuccess)
        exit(0);

    cudaStatus = cudaHostRegister(cq->dbrec, sizeof(cq->dbrec), cudaHostRegisterMapped);
    if(cudaStatus != cudaErrorHostMemoryAlreadyRegistered && cudaStatus !=  cudaSuccess) 
        exit(0);
    if(cudaHostGetDevicePointer(&dev_cq_dbrec, cq->dbrec, 0) != cudaSuccess)
        exit(0);

    poll_fake<<<1,1>>>(cq_buf_a, (uint32_t *) dev_cons_index, cq_cqe, cq_cqe_sz, 1, dev_cq_dbrec);
}


__device__ void device_process_gpu_mr(int *addr, int size){
    printf("gpu code running dev\n");
    printf("\n\nGpu memory read: %d, %d\n\n", addr[2], addr[size-1]);
}

__global__ void global_process_gpu_mr(int *addr, int size){
    printf("gpu code running\n");
    device_process_gpu_mr(addr, size);
}

void process_gpu_mr(int *addr, int size){
    printf("gpu code running\n");
    cudaError_t success = cudaDeviceSynchronize();
    if(success != 0) exit(-1);
    printf("gpu code running\n");
    global_process_gpu_mr<<<1, 1>>> (addr, size);

    success = cudaDeviceSynchronize();
    if(success != 0) exit(-1);
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
    int cur_post = 0, err;
    err = device_gpu_post_send(qpbf_bufsize, wr_rdma_remote_addr, wr_rdma_rkey,
              wr_sg_length, wr_sg_lkey, wr_sg_addr, wr_opcode, 
              bf_offset, qp_num, wr_id, qp_buf, /*dev_qpsq_wr_data,*/ 
              dev_qpsq_wqe_head, dev_qp_sq, dev_qp_db, /*dev_wr_sg,*/ dev_wrid, first_dword, second_dword, bf_reg, timer);

    err = post(wr_rdma_remote_addr, wr_rdma_rkey,
            wr_sg_length, wr_sg_lkey, wr_sg_addr, wr_opcode, 
            qp_num, cur_post, qp_buf, /*dev_qpsq_wr_data,*/ 
            bf_reg, (unsigned int *)dev_qp_db);    
    cur_post++;
    timer[1] = clock();
    start = timer[1] - start;
    // printf("Function name: %s, line number: %d\n", __func__, __LINE__);

    // printf("timer[0]: %d, timer[1]: %d\n", timer[0], timer[1]);
    // printf("timer[1] - timer[0]: %d\n", timer[1] - timer[0]);
    // printf("start: %d\n", start);

    *ret = err;
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
  if (0){
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

    printf("device_db: 0x%llx \n", device_db);
    printf("qp_buf: 0x%llx \n", qp_buf);
    printf("qp: 0x%llx \n", qp_num);
    // printf();
    
    

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

int host_gpu_poll_cq (struct ibv_cq *cq_ptr, int num_entries, struct ibv_wc *wc);

int gpu_process_work_completion_events (struct ibv_comp_channel *comp_channel, 
		struct ibv_wc *wc, int max_wc, struct ibv_cq *cq){
  
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
  host_gpu_poll_cq(cq /* the CQ, we got notification for */, 
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
    
    if(1)
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
  if (0){
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
  if (0){
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

__device__ int post_s(struct post_wr wr, int cur_post, void *qp_buf, void *bf_reg)
{
    // printf("inside post\n");
    int wr_opcode = wr.wr_opcode;
    uint32_t qp_num = wr.qp_num;
    uint64_t wr_rdma_remote_addr = wr.wr_rdma_remote_addr;
    uint32_t wr_rdma_rkey = wr.wr_rdma_rkey;
    uint64_t wr_sg_addr = wr.wr_sg_addr;
    uint32_t wr_sg_length = wr.wr_sg_length;
    uint32_t wr_sg_lkey = wr.wr_sg_lkey;

	void *seg;
	struct mlx5_wqe_ctrl_seg *ctrl = NULL;
	int err = 0;
	unsigned idx = cur_post & 63;
	uint32_t mlx5_opcode;

    seg = (qp_buf + 256 + (idx * 64)); // mlx5_get_send_wqe(qp, idx);
    ctrl = (struct mlx5_wqe_ctrl_seg *) seg;

    mlx5_opcode = wr_opcode*2 + 8 - 2*(wr_opcode == 2); // mlx5_ib_opcode[wr->opcode];
    ctrl->opmod_idx_opcode = htonl(((cur_post & 0xffff) << 8) | mlx5_opcode);
    ctrl->qpn_ds = htonl(3 | (qp_num << 8));
    ctrl->signature = 0;
    ctrl->fm_ce_se = 8; // MLX5_WQE_CTRL_CQ_UPDATE;
    ctrl->imm = 0; // 

    struct mlx5_wqe_raddr_seg *rdma = (struct mlx5_wqe_raddr_seg *)(seg + 16); // seg + 16; // sizeof(*ctrl);
    rdma->raddr    = htonl64(wr_rdma_remote_addr);
    rdma->rkey     = htonl(wr_rdma_rkey);
    rdma->reserved = 0;

    struct mlx5_wqe_data_seg *data = (struct mlx5_wqe_data_seg *) (seg + 32);
    data->byte_count = htonl(wr_sg_length); // htonl(wr_sg_list->length);
    data->lkey       = htonl(wr_sg_lkey); // htonl(wr_sg_list->lkey);
    data->addr       = htonl64(wr_sg_addr); // htonl64(wr_sg_list->addr);
    __threadfence_system();
    uint64_t val;
    val = *(uint64_t *) ctrl;
    *(volatile uint32_t *)bf_reg = htonl(htonl64(val) >> 32);
    *(volatile uint32_t *)(bf_reg+4) = htonl(htonl64(val));
    __threadfence_system();
	return 0;
}

__device__ int update_db_spec(void *bf_reg, void *qp_buf, unsigned int cur_post)
{
	void *seg;
	unsigned int idx = cur_post & 63;
    seg = (qp_buf + 256 + (idx * 64)); // mlx5_get_send_wqe(qp, idx);
    struct mlx5_wqe_ctrl_seg *ctrl = (struct mlx5_wqe_ctrl_seg *) seg;
   
    __threadfence_system();
    *(volatile uint64_t *)bf_reg = *(uint64_t *) ctrl ;// 
	return 0;
}

__device__ int update_db(uint64_t *ctrl, void *bf_reg)
{
    __threadfence_system();
    *(volatile uint64_t *)bf_reg = *(uint64_t *) ctrl ;// 
	return 0;
}

__device__ int post_m(uint64_t wr_rdma_remote_addr, uint32_t wr_rdma_rkey,            
                      uint32_t wr_sg_length, uint32_t wr_sg_lkey, uint64_t wr_sg_addr,
                      int wr_opcode, uint32_t qp_num, int cur_post, uint64_t *value_ctrl, void *qp_buf,
                      void *bf_reg, unsigned int *qp_db, void *dev_qp_sq, int id)
{
    // printf("inside post\n");
    // int wr_opcode = wr.wr_opcode;
    // uint32_t qp_num = wr.qp_num;
    // uint64_t wr_rdma_remote_addr = wr.wr_rdma_remote_addr;
    // uint32_t wr_rdma_rkey = wr.wr_rdma_rkey;
    // uint64_t wr_sg_addr = wr.wr_sg_addr;
    // uint32_t wr_sg_length = wr.wr_sg_length;
    // uint32_t wr_sg_lkey = wr.wr_sg_lkey;
    
    struct mlx5_wq *qp_sq = (struct mlx5_wq *) dev_qp_sq; // &qp->sq;
	void *seg;
	// struct mlx5_wqe_ctrl_seg *ctrl = NULL;
	// int err = 0;
    // printf("qp_sq->wqe_cnt: %d\n", qp_sq->wqe_cnt);
	unsigned int idx = cur_post & 63;
	uint32_t mlx5_opcode;

    seg = (qp_buf + 256 + (idx * 64)); // mlx5_get_send_wqe(qp, idx);
    struct mlx5_wqe_ctrl_seg *ctrl = (struct mlx5_wqe_ctrl_seg *) seg;
    // wqe_segment_ctrl *ctrl = (struct wqe_segment_ctrl *) seg;



    // // ctrl->opmod      = 0;
    // ctrl->wqe_index  = htons(cur_post);
    // ctrl->opcode     = 8; //RDMA WRITE
    // ctrl->qpn_ds     = htonl((qp_num *256) | 3); //DS = 3
    // // ctrl->signature  = 0;
    // // ctrl->rsvd       = 0;
    // ctrl->fcs        = 8; //signaled
    // // ctrl->imm        = 0;

    // mlx5_opcode = 16;// wr_opcode*2 + 8 - 2*(wr_opcode == 2); // mlx5_ib_opcode[wr->opcode];
    // *(uint64_t *) seg = (uint64_t) (htonl((uint16_t) cur_post * 256 | 16)) |  ((uint64_t) htonl(3 | (qp_num *256)) << 32);
    
    ctrl->opmod_idx_opcode = htonl(((uint16_t) cur_post * 256) | 16);
    // __threadfence();
    // printf("id: %d, cur_post: %d qp_num: %d\n", id, cur_post, qp_num-gpost_cont1->qp_num);
    // printf("ctrl: %p, &ctrl->opmod_idx_opcode: %p\n", ctrl, &ctrl->opmod_idx_opcode);
    // printf("ctrl: %p, &ctrl->qpn_ds: %p\n", ctrl, &ctrl->qpn_ds);
    ctrl->qpn_ds = htonl(3 | (qp_num *256));
    // __threadfence();
    ctrl->signature = 0;
    ctrl->fm_ce_se = 8; // MLX5_WQE_CTRL_CQ_UPDATE;
    // __threadfence();
    ctrl->imm = 0; // 
    // ctrl->dci_stream_channel_id = 0;

    struct mlx5_wqe_raddr_seg *rdma = (struct mlx5_wqe_raddr_seg *)(seg + 16); // seg + 16; // sizeof(*ctrl);
    rdma->raddr    = htonl64(wr_rdma_remote_addr);
    // __threadfence();
    rdma->rkey     = htonl(wr_rdma_rkey);
    // __threadfence();
    rdma->reserved = 0;
    // __threadfence();

    struct mlx5_wqe_data_seg *data = (struct mlx5_wqe_data_seg *) (seg + 32);
    // *(unsigned long long *) (seg + 32) = ((unsigned long long)htonl(wr_sg_length) | (unsigned long long) htonl(wr_sg_lkey) << 32); 
    // *(unsigned long long *) (seg + 96) = (unsigned long long) htonl64(wr_sg_addr) << 64;
    // *(uint64_t *) (seg + 32) = (uint64_t) (htonl(wr_sg_length) | (uint64_t) htonl(wr_sg_lkey) << 32);
    data->byte_count = htonl(wr_sg_length); // htonl(wr_sg_list->length);
    // __threadfence();
    data->lkey       = htonl(wr_sg_lkey); // htonl(wr_sg_list->lkey);
    // __threadfence();
    data->addr       = htonl64(wr_sg_addr); // htonl64(wr_sg_list->addr);
    // __threadfence();
    // if(id == 0)
    // printf("data: %p, &data->addr: %p\n", data, &data->addr);

    // *(uint64_t *) (seg + 40) = (uint64_t) htonl64(wr_sg_addr);
    // printf("post_m id: %d, wr_opcode: %d cur_post: %d, qp_num: %d\n", id, wr_opcode, cur_post, qp_num);
    // printf("post_m id: %d, wr_sg_length: %d wr_sg_lkey: %d, wr_sg_addr: %p\n", id, wr_sg_length, wr_sg_lkey, wr_sg_addr);
    // printf("post_m id: %d, wr_rdma_remote_addr: 0x%llx wr_rdma_rkey: %d,\n", id, wr_rdma_remote_addr, wr_rdma_rkey);
    // printf("post_m id: %d, bf_reg: %p\n", id, bf_reg);
  
    // // 
    // printf("post_m id: %d, qp_buf: %p\n", id, qp_buf);
    // printf("post_m id: %d, *bf_reg: %d, *bf_reg+4: %d\n", id, *(uint32_t *)bf_reg, *(uint32_t *)(bf_reg+4));
    // cur_post++;
    // qp_sq->cur_post += 1;
    // qp_sq->head += 1;
    // if(cur_post == 0)
    // qp_db[1] = (uint16_t) (cur_post + 1) ; // htonl(cur_post & 0xffff);
    
    
    // __threadfence_system();
    // uint32_t val[2];
    // memcpy(val, ctrl, 2*sizeof(uint32_t));
    // *(uint64_t *) val = *(uint64_t *) ctrl;
    // uint64_t val;
    // val = *(uint64_t *) ctrl;
    // __threadfence_system();
    // if(id == 0){
    //     *(volatile uint32_t *)bf_reg = (uint32_t) htonl(htonl64(val) >> 32);
    // *(volatile uint32_t *)(bf_reg+4) = (uint32_t) htonl(htonl64(val));
        // *(volatile uint32_t *)(bf_reg+4) = (uint32_t) htonl(htonl64(val));
        // *value_ctrl = *(uint64_t *) ctrl;
        __threadfence_system();
        *(volatile uint64_t *)bf_reg = *(uint64_t *) ctrl; // 
        
        // *(volatile uint64_t *)bf_reg = ((uint64_t) val[1] << 32 | val[0]);
        // __threadfence_system();
    // }
    // __threadfence_system();
    // printf("after id: %d, *bf_reg: %lld, *bf_reg+4: %lld\n", id, *(uint32_t *)bf_reg, *(uint32_t *)(bf_reg+4));
    // printf("id: %d, htonl(htonl64(val) >> 32: %d, htonl(htonl64(val)): %d\n", id, htonl(htonl64(val) >> 32), htonl(htonl64(val)));
    // __threadfence_system()
    
	return 0;
}

__device__ int post_write(uint64_t wr_rdma_remote_addr, uint32_t wr_rdma_rkey,            
                      uint32_t wr_sg_length, uint32_t wr_sg_lkey, uint64_t wr_sg_addr,
                      int wr_opcode, uint32_t qp_num, int cur_post, void *qp_buf, void *bf_reg, unsigned int *qp_db, void *dev_qp_sq, int id)
{
    
    struct mlx5_wq *qp_sq = (struct mlx5_wq *) dev_qp_sq; // &qp->sq;
	void *seg;
	unsigned int idx = cur_post & 63;
	uint32_t mlx5_opcode;

    seg = (qp_buf + 256 + (idx * 64)); // mlx5_get_send_wqe(qp, idx);
    struct mlx5_wqe_ctrl_seg *ctrl = (struct mlx5_wqe_ctrl_seg *) seg;
    // mlx5_opcode = 16;// wr_opcode*2 + 8 - 2*(wr_opcode == 2); // mlx5_ib_opcode[wr->opcode];
    // *(uint64_t *) seg = (uint64_t) (htonl((uint16_t) cur_post * 256 | 16)) |  ((uint64_t) htonl(3 | (qp_num *256)) << 32);
    ctrl->opmod_idx_opcode = htonl(((uint16_t) cur_post * 256) | 8);
    ctrl->qpn_ds = htonl(3 | (qp_num *256));
    ctrl->signature = 0;
    ctrl->fm_ce_se = 8; // MLX5_WQE_CTRL_CQ_UPDATE;
    ctrl->imm = 0; // 
    struct mlx5_wqe_raddr_seg *rdma = (struct mlx5_wqe_raddr_seg *)(seg + 16); // seg + 16; // sizeof(*ctrl);
    rdma->raddr    = htonl64(wr_rdma_remote_addr);
    rdma->rkey     = htonl(wr_rdma_rkey);
    rdma->reserved = 0;

    struct mlx5_wqe_data_seg *data = (struct mlx5_wqe_data_seg *) (seg + 32);
    data->byte_count = htonl(wr_sg_length); // htonl(wr_sg_list->length);
    data->lkey       = htonl(wr_sg_lkey); // htonl(wr_sg_list->lkey);
    data->addr       = htonl64(wr_sg_addr); // htonl64(wr_sg_list->addr);
    
    // cur_post++;
    // qp_sq->cur_post += 1;
    // qp_sq->head += 1;
    // if(cur_post == 0)
    // qp_db[0] = (uint16_t) (cur_post + 1) ; // htonl(cur_post & 0xffff);
    __threadfence_system();
    
    *(volatile uint64_t *)bf_reg = *(uint64_t *) ctrl ;// 
	return 0;
}

__device__ int post(uint64_t wr_rdma_remote_addr, uint32_t wr_rdma_rkey,
                    uint32_t wr_sg_length, uint32_t wr_sg_lkey, uint64_t wr_sg_addr, 
                    int wr_opcode, uint32_t qp_num, int cur_post, void *qp_buf, void *bf_reg, unsigned int *qp_db)
{
    // printf("inside post\n");
    // int wr_opcode = wr.wr_opcode;
    // uint32_t qp_num = wr.qp_num;
    // uint64_t wr_rdma_remote_addr = wr.wr_rdma_remote_addr;
    // uint32_t wr_rdma_rkey = wr.wr_rdma_rkey;
    // uint64_t wr_sg_addr = wr.wr_sg_addr;
    // uint32_t wr_sg_length = wr.wr_sg_length;
    // uint32_t wr_sg_lkey = wr.wr_sg_lkey;

	void *seg;
	struct mlx5_wqe_ctrl_seg *ctrl = NULL;
	int err = 0;
	unsigned idx = cur_post & 63;
	uint32_t mlx5_opcode;

    seg = (qp_buf + 256 + (idx * 64)); // mlx5_get_send_wqe(qp, idx);
    ctrl = (struct mlx5_wqe_ctrl_seg *) seg;

    mlx5_opcode = wr_opcode*2 + 8 - 2*(wr_opcode == 2); // mlx5_ib_opcode[wr->opcode];
    ctrl->opmod_idx_opcode = htonl(((cur_post & 0xffff) << 8) | mlx5_opcode);
    ctrl->qpn_ds = htonl(3 | (qp_num << 8));
    ctrl->signature = 0;
    ctrl->fm_ce_se = 8; // MLX5_WQE_CTRL_CQ_UPDATE;
    ctrl->imm = 0; // 

    struct mlx5_wqe_raddr_seg *rdma = (struct mlx5_wqe_raddr_seg *)(seg + 16); // seg + 16; // sizeof(*ctrl);
    rdma->raddr    = htonl64(wr_rdma_remote_addr);
    rdma->rkey     = htonl(wr_rdma_rkey);
    rdma->reserved = 0;

    struct mlx5_wqe_data_seg *data = (struct mlx5_wqe_data_seg *) (seg + 32);
    data->byte_count = htonl(wr_sg_length); // htonl(wr_sg_list->length);
    data->lkey       = htonl(wr_sg_lkey); // htonl(wr_sg_list->lkey);
    data->addr       = htonl64(wr_sg_addr); // htonl64(wr_sg_list->addr);
    printf("bf_reg: %p\n", bf_reg);
    printf("*bf_reg: %lld, *bf_reg+4: %lld\n", *(uint32_t *)bf_reg, *(uint32_t *)(bf_reg+4));
    cur_post++;
    qp_db[0] = htonl(cur_post & 0xffff);
    uint64_t val;
    val = *(uint64_t *) ctrl;
    *(volatile uint32_t *)bf_reg = (volatile uint32_t) htonl(htonl64(val) >> 32);
    *(volatile uint32_t *)(bf_reg+4) = (volatile uint32_t) htonl(htonl64(val));
    printf("after *bf_reg: %lld, *bf_reg+4: %lld\n", *(volatile uint32_t *)bf_reg, *(volatile uint32_t *)(bf_reg+4));
    printf("htonl(htonl64(val) >> 32: %lld, htonl(htonl64(val)): %lld\n", htonl(htonl64(val) >> 32), htonl(htonl64(val)));    
    // __threadfence_system()
    
	return 0;
}

__device__ int poll(void *cq_buf, struct ibv_wc *wc, uint32_t *cons_index,
                    int ibv_cqe, uint32_t cqe_sz, int n, void *cq_dbrec) 
{
    uint32_t *gpu_dbrec = (uint32_t *) cq_dbrec;
    void *cqe;
    struct mlx5_cqe64 *cqe64;
    uint32_t cons_index_dev = *cons_index;
    cqe = cq_buf + (cons_index_dev & ibv_cqe) * cqe_sz;
    cqe64 = (struct mlx5_cqe64 *)((cqe_sz == 64) ? cqe : cqe + 64);
    // printf("cqe64->op_own: %d\n", cqe64->op_own);
        // if(((cqe64->op_own != 240) && !((cqe64->op_own & 1) ^ !!(cons_index_dev & (ibv_cqe + 1))))==0){
        //     // gpu_dbrec[0] = htonl(*cons_index & 0xffffff);
        //     return 0;
        // }
    
    // while(((cqe64->op_own != 240) && !((cqe64->op_own & 1) ^ !!(*cons_index & (ibv_cqe + 1))))==0);
    while(cqe64->op_own == 240);
    // if((cqe64->op_own >> 4) != 0) //Check opcode
    // {
    //     struct mlx5_err_cqe *err_cqe = (struct mlx5_err_cqe *)cqe64;
    //     printf("Got completion with error, opcode = %d , syndrome = %d\n",(cqe64->op_own >> 4), err_cqe->syndrome);
    //     wc->status = IBV_WC_GENERAL_ERR;
    // }
    // printf("cqe64->op_own: %d\n", cqe64->op_own);
    // __threadfence_system();
    *cons_index = cons_index_dev + 1;
    wc->qp_num =  htonl(cqe64->sop_drop_qpn) & 0xffffff;
    wc->status = (ibv_wc_status) 0; // (ibv_wc_status) IBV_WC_SUCCESS;
    // __threadfence_system();
    *gpu_dbrec = (uint32_t) htonl((cons_index_dev + 1) & 0xffffff);
    return 1;// err; 
}

/*initialize data structures for */
int prepare_post_poll_content(struct context *s_ctx, struct post_content *post_cont, struct poll_content *poll_cont, struct post_content2 *post_cont2, \
                              struct post_content *host_post, struct poll_content *host_poll, struct host_keys *host_post2){

    cudaError_t success;
    printf("Function name: %s, line number: %d\n", __func__, __LINE__);
    struct mlx5_qp *qp_0 = to_mqp(s_ctx->gpu_qp[0]);
    printf("Function name: %s, line number: %d\n", __func__, __LINE__);
    post_cont->wr_rdma_remote_addr = (uintptr_t)s_ctx->server_memory.addresses[0];
    remote_address = s_ctx->server_memory.addresses[0];
    post_cont->wr_rdma_rkey = s_ctx->gpu_mr->rkey;
    post_cont->wr_sg_length = 4096; // fixed for now by default
    post_cont->wr_sg_lkey = s_ctx->gpu_mr->lkey;

    post_cont->wr_sg_addr = (uintptr_t) s_ctx->gpu_buffer;
    post_cont->wr_opcode = IBV_WR_RDMA_READ; // for read request by default
    post_cont->qp_num = s_ctx->gpu_qp[0]->qp_num;
    post_cont->qp_buf = qp_0->buf.buf;

    post_cont->qpbf_bufsize = qp_0->bf->buf_size;
    post_cont->wr_rdma_rkey = 1;
    printf("Function name: %s, line number: %d\n", __func__, __LINE__);

    struct mlx5_qp *main_qp = to_mqp(s_ctx->main_qp);
    host_post->wr_rdma_remote_addr = (uintptr_t)s_ctx->server_memory.addresses[0];
    host_post->wr_rdma_rkey = s_ctx->gpu_mr->rkey;
    host_post->wr_sg_length = 1024; // fixed for now by default
    host_post->wr_sg_lkey = s_ctx->gpu_mr->lkey;
    // printf("Function name: %s, line number: %d\n", __func__, __LINE__);
    host_post->wr_sg_addr = (uintptr_t) s_ctx->gpu_buffer;
    host_post->wr_opcode = IBV_WR_RDMA_READ; // for read request by default
    host_post->qp_num = s_ctx->main_qp->qp_num;
    host_post->qp_buf = main_qp->buf.buf;
    host_post->qpbf_bufsize = main_qp->bf->buf_size;
    // printf("Function name: %s, line number: %d\n", __func__, __LINE__);
    host_post->wr_rdma_rkey = 1;
    host_post->bf_reg[0] = /*(long long int)*/ main_qp->bf->reg; // device_db;
    host_post->qp_db[0] = (unsigned int *) main_qp->db; // dev_qp_db;
    host_post->dev_qp_sq[0] = &main_qp->sq; // dev_qp_sq;
    host_post->n_post[0] = main_qp->sq.cur_post;
    host_post->cq_lock[0] = 0;
    printf("Function name: %s, line number: %d\n", __func__, __LINE__);
    for(int i = 0; i < N_8GB_Region; i++){
        post_cont2->wr_rdma_rkey[i] = s_ctx->server_memory.rkeys[i];
        post_cont2->wr_rdma_lkey[i] = s_ctx->server_memory.lkeys[i];
        post_cont2->addrs[i] = s_ctx->server_memory.addresses[i];
        // post_cont2->wr_rdma_lkey[i] = s_ctx->server_memory.addresses[i];
        host_post2->rkeys[i] = s_ctx->server_memory.rkeys[i];
        host_post2->lkeys[i] = s_ctx->server_memory.lkeys[i];
        host_post2->addrs[i] = s_ctx->server_memory.addresses[i];
        printf("server rkey: %d server memory address: %p\n", \
        s_ctx->server_memory.rkeys[i], s_ctx->server_memory.addresses[i]);
    }

    printf("gpu mr lkey: %d\n", \
        s_ctx->gpu_mr->lkey);
    printf("gpu mr rkey: %d\n", \
        s_ctx->gpu_mr->rkey);

    printf("Function name: %s, line number: %d\n", __func__, __LINE__);
    printf("qp->sq.wqe_cnt: %d\n", qp_0->sq.wqe_cnt);
    printf("[]s_ctx->n_bufs: %d\n\n\n\n", s_ctx->n_bufs);
    for(int i = 0; i < s_ctx->n_bufs; i++){
        // printf("qp_num[%d]: %d\n", i, s_ctx->gpu_qp[i]->qp_num);
        struct mlx5_qp *qp = to_mqp(s_ctx->gpu_qp[i]);
        // printf("qp->bf->reg[%d]: %p\n", i, qp->bf->reg);
        // printf("qp->buf.buf[%d]: %p \n", i, qp->buf.buf);

        void *device_db;
        // printf("qp->bf->reg[%d]: 0x%llx\n", i, qp->bf->reg);
        success = cudaHostRegister(qp->bf->reg,  8, cudaHostRegisterIoMemory);
        if (success != cudaSuccess && success != cudaErrorHostMemoryAlreadyRegistered) return -1;
        success = cudaHostGetDevicePointer(&device_db, qp->bf->reg, 0);
        if (success != cudaSuccess) return -1;
        // printf("device_db[%d]: 0x%llx\n", i, device_db);
        // printf("device_db[%d]: 0x%llx\n", i, qp->cqe);
        post_cont->bf_reg[i] = /*(long long int)*/ device_db;

        void *dev_qp_db;
        success = cudaHostRegister(qp->db, sizeof(qp->db), cudaHostRegisterMapped);
        if(success != cudaSuccess && success != cudaErrorHostMemoryAlreadyRegistered)
            return -1;
        if(cudaHostGetDevicePointer(&dev_qp_db, qp->db, 0) != cudaSuccess)
            return -1;
        // printf("dev_qp_db[%d]: %p \n", i, dev_qp_db);
        post_cont->qp_db[i] = (unsigned int *) dev_qp_db;

        void *dev_qp_sq;
        success = cudaHostRegister(&qp->sq, sizeof(qp->sq), cudaHostRegisterMapped);
        if(success != cudaErrorHostMemoryAlreadyRegistered && success !=  cudaSuccess)
                return -1;
        // get GPU pointer for qp->sq
        if(cudaHostGetDevicePointer(&dev_qp_sq, &qp->sq, 0) != cudaSuccess)
            return -1;
        post_cont->dev_qp_sq[i] = dev_qp_sq;
        post_cont->n_post[i] = 0;
        post_cont->queue_count[i] = 0;
        post_cont->queue_lock[i] = 0;
        for(size_t k = 0; k < 64; k++)
            post_cont->cq_lock[i*64+k] = 0;

        if(i == 0){
            printf("qp_buf[%d]: %p \n", i, post_cont->qp_buf);
            printf("dev_qp_db[%d]: %p \n", i, dev_qp_db);
            printf("bf_reg[%d]: 0x%llx\n", i, device_db);
        }

    }

    printf("Function name: %s, line number: %d\n", __func__, __LINE__);
    struct mlx5_cq *cq1 = to_mcq(s_ctx->gpu_cq[0]);
    poll_cont->cq_buf = cq1->active_buf->buf;
    
    poll_cont->ibv_cqe = cq1->verbs_cq.cq.cqe;
    poll_cont->cqe_sz = cq1->cqe_sz;
    printf("poll_cont->ibv_cqe: %d\n",poll_cont->ibv_cqe);
    printf("poll_cont->cqe_sz: %d\n",poll_cont->cqe_sz);
    poll_cont->n = 1; // subject to change

    struct mlx5_cq *main_cq = to_mcq(s_ctx->main_cq);
    host_poll->cq_buf = main_cq->active_buf->buf;
    host_poll->ibv_cqe = main_cq->verbs_cq.cq.cqe;
    host_poll->cqe_sz = main_cq->cqe_sz;
    host_poll->n = 1; // subject to change
    host_poll->cons_index[0] = (long long int) &main_cq->cons_index;
    host_poll->cq_dbrec[0] = (long long int) main_cq->dbrec;
    
    for(int i = 0; i < s_ctx->n_bufs; i++){
        struct mlx5_cq *cq = to_mcq(s_ctx->gpu_cq[i]);
        struct mlx5_context *mctx = container_of(cq->verbs_cq.cq.context, struct mlx5_context, ibv_ctx.context);
        struct mlx5_resource *rsc = mctx->uidx_table[0].table[0];
        struct mlx5_qp *qp = (struct mlx5_qp *)(rsc);

        void *dev_cons_index;
        success = cudaHostRegister(&cq->cons_index, sizeof(cq->cons_index), cudaHostRegisterMapped);
        if(success != cudaSuccess && success != cudaErrorHostMemoryAlreadyRegistered) return -1;
        if(cudaHostGetDevicePointer(&dev_cons_index, &cq->cons_index, 0) != cudaSuccess) return -1;
        poll_cont->cons_index[i] = (long long int) dev_cons_index;

        
        void *dev_cq_dbrec;
        success = cudaHostRegister(cq->dbrec, sizeof(cq->dbrec), cudaHostRegisterMapped);
        if(success != cudaErrorHostMemoryAlreadyRegistered && success !=  cudaSuccess) return -1;
        if(cudaHostGetDevicePointer(&dev_cq_dbrec, cq->dbrec, 0) != cudaSuccess)
            return -1;
        poll_cont->cq_dbrec[i] = (long long int) dev_cq_dbrec;

        if(i == 0){
            printf("dev_cq_dbrec[%d]: %p \n", i, dev_cq_dbrec);
            printf("dev_cons_index[%d]: %p \n", i, dev_cons_index);
            printf("poll_cont->cq_buf[%d]: 0x%llx\n", i, poll_cont->cq_buf);
        }   
    }

    main_content.cq = s_ctx->main_cq;
    main_content.qp = s_ctx->main_qp;
    main_content.pd = s_ctx->pd;
    
    printf("Function name: %s, line number: %d\n", __func__, __LINE__);
    return 0;
}

int prepare_post_poll_content_2gpu(struct context_2gpu *s_ctx, struct post_content *post_cont, struct poll_content *poll_cont, struct post_content2 *post_cont2, \
                              struct post_content *host_post, struct poll_content *host_poll, struct host_keys *host_post2, struct gpu_memory_info *gpu_infos){
    cudaError_t success;
    printf("Function name: %s, line number: %d\n", __func__, __LINE__);
    struct mlx5_qp *qp_0 = to_mqp(s_ctx->gpu_qp[0]);
    struct mlx5_qp *qp_128 = to_mqp(s_ctx->gpu_qp[s_ctx->n_bufs]);
    printf("Function name: %s, line number: %d\n", __func__, __LINE__);
    post_cont->wr_rdma_remote_addr = (uintptr_t)s_ctx->server_memory.addresses[0];
    remote_address = s_ctx->server_memory.addresses[0];
    // post_cont->wr_rdma_rkey = s_ctx->gpu_mr->rkey;
    post_cont->wr_sg_length = 4096; // fixed for now by default
    // post_cont->wr_sg_lkey = s_ctx->gpu_mr->lkey;

    gpu_infos->addrs[0] = (uint64_t) s_ctx->gpu_buffer1;
    gpu_infos->addrs[1] = (uint64_t) s_ctx->gpu_buffer2;
    gpu_infos->wr_rdma_lkey[0] = (uint64_t) s_ctx->gpu1_mr->lkey;
    gpu_infos->wr_rdma_lkey[1] = (uint64_t) s_ctx->gpu2_mr->lkey;
    gpu_infos->wr_rdma_rkey[0] = (uint64_t) s_ctx->gpu1_mr->rkey;
    gpu_infos->wr_rdma_rkey[1] = (uint64_t) s_ctx->gpu2_mr->rkey;
    gpu_infos->qp_buf_gpu[0] = (uint64_t) qp_0->buf.buf;
    gpu_infos->qp_buf_gpu[1] = (uint64_t) qp_128->buf.buf;

    // post_cont->wr_sg_addr = (uintptr_t) s_ctx->gpu_buffer;
    post_cont->wr_opcode = IBV_WR_RDMA_READ; // for read request by default
    post_cont->qp_num = s_ctx->gpu_qp[0]->qp_num;
    post_cont->qp_buf = qp_0->buf.buf;

    post_cont->qpbf_bufsize = qp_0->bf->buf_size;
    post_cont->wr_rdma_rkey = 1;
    printf("Function name: %s, line number: %d\n", __func__, __LINE__);

    struct mlx5_qp *main_qp = to_mqp(s_ctx->main_qp);
    host_post->wr_rdma_remote_addr = (uintptr_t)s_ctx->server_memory.addresses[0];
    // host_post->wr_rdma_rkey = s_ctx->gpu_mr->rkey;
    host_post->wr_sg_length = 1024; // fixed for now by default
    // host_post->wr_sg_lkey = s_ctx->gpu_mr->lkey;
    // printf("Function name: %s, line number: %d\n", __func__, __LINE__);
    // host_post->wr_sg_addr = (uintptr_t) s_ctx->gpu_buffer;
    host_post->wr_opcode = IBV_WR_RDMA_READ; // for read request by default
    host_post->qp_num = s_ctx->main_qp->qp_num;
    host_post->qp_buf = main_qp->buf.buf;
    host_post->qpbf_bufsize = main_qp->bf->buf_size;
    // printf("Function name: %s, line number: %d\n", __func__, __LINE__);
    host_post->wr_rdma_rkey = 1;
    host_post->bf_reg[0] = /*(long long int)*/ main_qp->bf->reg; // device_db;
    host_post->qp_db[0] = (unsigned int *) main_qp->db; // dev_qp_db;
    host_post->dev_qp_sq[0] = &main_qp->sq; // dev_qp_sq;
    host_post->n_post[0] = main_qp->sq.cur_post;
    host_post->cq_lock[0] = 0;
    printf("Function name: %s, line number: %d\n", __func__, __LINE__);
    for(int i = 0; i < N_8GB_Region; i++){
        post_cont2->wr_rdma_rkey[i] = s_ctx->server_memory.rkeys[i];
        post_cont2->wr_rdma_lkey[i] = s_ctx->server_memory.lkeys[i];
        post_cont2->addrs[i] = s_ctx->server_memory.addresses[i];
        // post_cont2->wr_rdma_lkey[i] = s_ctx->server_memory.addresses[i];
        host_post2->rkeys[i] = s_ctx->server_memory.rkeys[i];
        host_post2->lkeys[i] = s_ctx->server_memory.lkeys[i];
        host_post2->addrs[i] = s_ctx->server_memory.addresses[i];
        printf("server rkey: %d server memory address: %p\n", \
        s_ctx->server_memory.rkeys[i], s_ctx->server_memory.addresses[i]);
    }

    printf("gpu0 mr lkey: %d\n", \
        s_ctx->gpu1_mr->lkey);
    printf("gpu0 mr rkey: %d\n", \
        s_ctx->gpu1_mr->rkey);

    printf("Function name: %s, line number: %d\n", __func__, __LINE__);
    printf("qp->sq.wqe_cnt: %d\n", qp_0->sq.wqe_cnt);
    printf("[]s_ctx->n_bufs: %d\n\n\n\n", s_ctx->n_bufs);
    for(int i = 0; i < s_ctx->n_bufs*2; i++){
        // printf("qp_num[%d]: %d\n", i, s_ctx->gpu_qp[i]->qp_num);
        struct mlx5_qp *qp = to_mqp(s_ctx->gpu_qp[i]);
        // printf("qp->bf->reg[%d]: %p\n", i, qp->bf->reg);
        // printf("qp->buf.buf[%d]: %p \n", i, qp->buf.buf);

        void *device_db;
        // printf("qp->bf->reg[%d]: 0x%llx\n", i, qp->bf->reg);
        success = cudaHostRegister(qp->bf->reg,  8, cudaHostRegisterIoMemory);
        if (success != cudaSuccess && success != cudaErrorHostMemoryAlreadyRegistered) return -1;
        success = cudaHostGetDevicePointer(&device_db, qp->bf->reg, 0);
        if (success != cudaSuccess) return -1;
        // printf("device_db[%d]: 0x%llx\n", i, device_db);
        // printf("device_db[%d]: 0x%llx\n", i, qp->cqe);
        post_cont->bf_reg[i] = /*(long long int)*/ device_db;

        void *dev_qp_db;
        success = cudaHostRegister(qp->db, sizeof(qp->db), cudaHostRegisterMapped);
        if(success != cudaSuccess && success != cudaErrorHostMemoryAlreadyRegistered)
            return -1;
        if(cudaHostGetDevicePointer(&dev_qp_db, qp->db, 0) != cudaSuccess)
            return -1;
        // printf("dev_qp_db[%d]: %p \n", i, dev_qp_db);
        post_cont->qp_db[i] = (unsigned int *) dev_qp_db;

        void *dev_qp_sq;
        success = cudaHostRegister(&qp->sq, sizeof(qp->sq), cudaHostRegisterMapped);
        if(success != cudaErrorHostMemoryAlreadyRegistered && success !=  cudaSuccess)
                return -1;
        // get GPU pointer for qp->sq
        if(cudaHostGetDevicePointer(&dev_qp_sq, &qp->sq, 0) != cudaSuccess)
            return -1;
        post_cont->dev_qp_sq[i] = dev_qp_sq;
        post_cont->n_post[i] = 0;
        post_cont->queue_count[i] = 0;
        post_cont->queue_lock[i] = 0;
        for(size_t k = 0; k < 64; k++)
            post_cont->cq_lock[i*64+k] = 0;

        if(i == 0){
            printf("qp_buf[%d]: %p \n", i, post_cont->qp_buf);
            printf("dev_qp_db[%d]: %p \n", i, dev_qp_db);
            printf("bf_reg[%d]: 0x%llx\n", i, device_db);
        }

    }

    printf("Function name: %s, line number: %d\n", __func__, __LINE__);
    struct mlx5_cq *cq1 = to_mcq(s_ctx->gpu_cq[0]);
    struct mlx5_cq *cq128 = to_mcq(s_ctx->gpu_cq[s_ctx->n_bufs]);
    poll_cont->cq_buf = cq1->active_buf->buf;
    
    gpu_infos->cq_buf_gpu[0] = (uint64_t) cq1->active_buf->buf; 
    gpu_infos->cq_buf_gpu[1] = (uint64_t) cq128->active_buf->buf;

    poll_cont->ibv_cqe = cq1->verbs_cq.cq.cqe;
    poll_cont->cqe_sz = cq1->cqe_sz;
    printf("poll_cont->ibv_cqe: %d\n",poll_cont->ibv_cqe);
    printf("poll_cont->cqe_sz: %d\n",poll_cont->cqe_sz);
    poll_cont->n = 1; // subject to change

    struct mlx5_cq *main_cq = to_mcq(s_ctx->main_cq);
    host_poll->cq_buf = main_cq->active_buf->buf;
    host_poll->ibv_cqe = main_cq->verbs_cq.cq.cqe;
    host_poll->cqe_sz = main_cq->cqe_sz;
    host_poll->n = 1; // subject to change
    host_poll->cons_index[0] = (long long int) &main_cq->cons_index;
    host_poll->cq_dbrec[0] = (long long int) main_cq->dbrec;
    
    for(int i = 0; i < s_ctx->n_bufs*2; i++){
        struct mlx5_cq *cq = to_mcq(s_ctx->gpu_cq[i]);
        struct mlx5_context *mctx = container_of(cq->verbs_cq.cq.context, struct mlx5_context, ibv_ctx.context);
        struct mlx5_resource *rsc = mctx->uidx_table[0].table[0];
        struct mlx5_qp *qp = (struct mlx5_qp *)(rsc);

        void *dev_cons_index;
        success = cudaHostRegister(&cq->cons_index, sizeof(cq->cons_index), cudaHostRegisterMapped);
        if(success != cudaSuccess && success != cudaErrorHostMemoryAlreadyRegistered) return -1;
        if(cudaHostGetDevicePointer(&dev_cons_index, &cq->cons_index, 0) != cudaSuccess) return -1;
        poll_cont->cons_index[i] = (long long int) dev_cons_index;

        
        void *dev_cq_dbrec;
        success = cudaHostRegister(cq->dbrec, sizeof(cq->dbrec), cudaHostRegisterMapped);
        if(success != cudaErrorHostMemoryAlreadyRegistered && success !=  cudaSuccess) return -1;
        if(cudaHostGetDevicePointer(&dev_cq_dbrec, cq->dbrec, 0) != cudaSuccess)
            return -1;
        poll_cont->cq_dbrec[i] = (long long int) dev_cq_dbrec;

        if(i == 0){
            printf("dev_cq_dbrec[%d]: %p \n", i, dev_cq_dbrec);
            printf("dev_cons_index[%d]: %p \n", i, dev_cons_index);
            printf("poll_cont->cq_buf[%d]: 0x%llx\n", i, poll_cont->cq_buf);
        }   
    }

    main_content.cq = s_ctx->main_cq;
    main_content.qp = s_ctx->main_qp;
    main_content.pd = s_ctx->pd;
    
    printf("Function name: %s, line number: %d\n", __func__, __LINE__);
    return 0;
}

int prepare_post_poll_content_2gpu_2nic(struct context_2gpu_2nic *s_ctx, struct post_content *post_cont, struct poll_content *poll_cont, struct server_content_2nic *post_cont2, \
                              struct post_content *host_post, struct poll_content *host_poll, struct host_keys *host_post2, struct gpu_memory_info *gpu_infos){
    cudaError_t success;
    printf("Function name: %s, line number: %d\n", __func__, __LINE__);
    struct mlx5_qp *qp_0 = to_mqp(s_ctx->gpu_qp[0]);
    struct mlx5_qp *qp_128 = to_mqp(s_ctx->gpu_qp[s_ctx->n_bufs]);
    printf("Function name: %s, line number: %d\n", __func__, __LINE__);
    post_cont->wr_rdma_remote_addr = (uintptr_t)s_ctx->server_memory[0].addresses[0];
    remote_address_2nic[0] = s_ctx->server_memory[0].addresses[0];
    remote_address_2nic[1] = s_ctx->server_memory[1].addresses[0];
    // post_cont->wr_rdma_rkey = s_ctx->gpu_mr->rkey;
    post_cont->wr_sg_length = 4096; // fixed for now by default
    // post_cont->wr_sg_lkey = s_ctx->gpu_mr->lkey;

    gpu_infos->addrs[0] = (uint64_t) s_ctx->gpu_buffer[0];
    gpu_infos->addrs[1] = (uint64_t) s_ctx->gpu_buffer[1];
    gpu_infos->wr_rdma_lkey[0] = (uint64_t) s_ctx->gpu_mr[0]->lkey;
    gpu_infos->wr_rdma_lkey[1] = (uint64_t) s_ctx->gpu_mr[1]->lkey;
    gpu_infos->wr_rdma_rkey[0] = (uint64_t) s_ctx->gpu_mr[0]->rkey;
    gpu_infos->wr_rdma_rkey[1] = (uint64_t) s_ctx->gpu_mr[1]->rkey;
    gpu_infos->qp_buf_gpu[0] = (uint64_t) qp_0->buf.buf;
    gpu_infos->qp_buf_gpu[1] = (uint64_t) qp_128->buf.buf;

    gpu_infos->server_address[0] = (uint64_t) s_ctx->server_memory[0].addresses[0];
    gpu_infos->server_address[1] = (uint64_t) s_ctx->server_memory[1].addresses[0];

    gpu_infos->qp_num_gpu[0] = s_ctx->gpu_qp[0]->qp_num;
    gpu_infos->qp_num_gpu[1] = s_ctx->gpu_qp[s_ctx->n_bufs]->qp_num;

    // post_cont->wr_sg_addr = (uintptr_t) s_ctx->gpu_buffer;
    post_cont->wr_opcode = IBV_WR_RDMA_READ; // for read request by default
    post_cont->qp_num = s_ctx->gpu_qp[0]->qp_num;
    post_cont->qp_buf = qp_0->buf.buf;

    post_cont->qpbf_bufsize = qp_0->bf->buf_size;
    post_cont->wr_rdma_rkey = 1;
    printf("Function name: %s, line number: %d\n", __func__, __LINE__);

    // host_post is not used but I leave it here in case needed
    struct mlx5_qp *main_qp = to_mqp(s_ctx->main_qp[0]);
    host_post->wr_rdma_remote_addr = (uintptr_t)s_ctx->server_memory[0].addresses[0];
    // host_post->wr_rdma_rkey = s_ctx->gpu_mr->rkey;
    host_post->wr_sg_length = 1024; // fixed for now by default
    // host_post->wr_sg_lkey = s_ctx->gpu_mr->lkey;
    // printf("Function name: %s, line number: %d\n", __func__, __LINE__);
    // host_post->wr_sg_addr = (uintptr_t) s_ctx->gpu_buffer;
    host_post->wr_opcode = IBV_WR_RDMA_READ; // for read request by default
    host_post->qp_num = s_ctx->main_qp[0]->qp_num;
    host_post->qp_buf = main_qp->buf.buf;
    host_post->qpbf_bufsize = main_qp->bf->buf_size;
    // printf("Function name: %s, line number: %d\n", __func__, __LINE__);
    host_post->wr_rdma_rkey = 1;
    host_post->bf_reg[0] = /*(long long int)*/ main_qp->bf->reg; // device_db;
    host_post->qp_db[0] = (unsigned int *) main_qp->db; // dev_qp_db;
    host_post->dev_qp_sq[0] = &main_qp->sq; // dev_qp_sq;
    host_post->n_post[0] = main_qp->sq.cur_post;
    host_post->cq_lock[0] = 0;
    printf("Function name: %s, line number: %d\n", __func__, __LINE__);
    for (size_t server = 0; server < 2; server++)
    {
        for(int i = 0; i < N_8GB_Region; i++){
            post_cont2->wr_rdma_rkey[i + N_8GB_Region*server] = s_ctx->server_memory[server].rkeys[i];
            post_cont2->wr_rdma_lkey[i + N_8GB_Region*server] = s_ctx->server_memory[server].lkeys[i];
            post_cont2->addrs[i + N_8GB_Region*server] = s_ctx->server_memory[server].addresses[i];

            printf("post_cont2->wr_rdma_rkey[%d + N_8GB_Region*%d]: %d\n", i, server, post_cont2->wr_rdma_rkey[i + N_8GB_Region*server]);


            // post_cont2->wr_rdma_lkey[i] = s_ctx->server_memory.lkeys[i];
            // post_cont2->addrs[i] = s_ctx->server_memory.addresses[i];
            // post_cont2->wr_rdma_lkey[i] = s_ctx->server_memory.addresses[i];
            host_post2->rkeys[i] = s_ctx->server_memory[0].rkeys[i];
            host_post2->lkeys[i] = s_ctx->server_memory[0].lkeys[i];
            host_post2->addrs[i] = s_ctx->server_memory[0].addresses[i];
            printf("server rkey: %d server memory address: %p\n", \
            s_ctx->server_memory[server].rkeys[i], s_ctx->server_memory[server].addresses[i]);
        }
    }

    printf("gpu0 mr lkey: %d\n", \
        s_ctx->gpu_mr[0]->lkey);
    printf("gpu0 mr rkey: %d\n", \
        s_ctx->gpu_mr[0]->rkey);

    printf("gpu1 mr lkey: %d\n", \
        s_ctx->gpu_mr[1]->lkey);
    printf("gpu1 mr rkey: %d\n", \
        s_ctx->gpu_mr[1]->rkey);

    printf("Function name: %s, line number: %d\n", __func__, __LINE__);
    printf("qp->sq.wqe_cnt: %d\n", qp_0->sq.wqe_cnt);
    printf("[]s_ctx->n_bufs: %d\n\n\n\n", s_ctx->n_bufs);
    for(int i = 0; i < s_ctx->n_bufs*2; i++){
        // printf("qp_num[%d]: %d\n", i, s_ctx->gpu_qp[i]->qp_num);
        struct mlx5_qp *qp = to_mqp(s_ctx->gpu_qp[i]);
        // printf("qp->bf->reg[%d]: %p\n", i, qp->bf->reg);
        // printf("qp->buf.buf[%d]: %p \n", i, qp->buf.buf);

        void *device_db;
        // printf("qp->bf->reg[%d]: 0x%llx\n", i, qp->bf->reg);
        success = cudaHostRegister(qp->bf->reg,  8, cudaHostRegisterIoMemory);
        if (success != cudaSuccess && success != cudaErrorHostMemoryAlreadyRegistered) return -1;
        success = cudaHostGetDevicePointer(&device_db, qp->bf->reg, 0);
        if (success != cudaSuccess) return -1;
        // printf("device_db[%d]: 0x%llx\n", i, device_db);
        // printf("device_db[%d]: 0x%llx\n", i, qp->cqe);
        post_cont->bf_reg[i] = /*(long long int)*/ device_db;

        void *dev_qp_db;
        success = cudaHostRegister(qp->db, sizeof(qp->db), cudaHostRegisterMapped);
        if(success != cudaSuccess && success != cudaErrorHostMemoryAlreadyRegistered)
            return -1;
        if(cudaHostGetDevicePointer(&dev_qp_db, qp->db, 0) != cudaSuccess)
            return -1;
        // printf("dev_qp_db[%d]: %p \n", i, dev_qp_db);
        post_cont->qp_db[i] = (unsigned int *) dev_qp_db;

        void *dev_qp_sq;
        success = cudaHostRegister(&qp->sq, sizeof(qp->sq), cudaHostRegisterMapped);
        if(success != cudaErrorHostMemoryAlreadyRegistered && success !=  cudaSuccess)
                return -1;
        // get GPU pointer for qp->sq
        if(cudaHostGetDevicePointer(&dev_qp_sq, &qp->sq, 0) != cudaSuccess)
            return -1;
        post_cont->dev_qp_sq[i] = dev_qp_sq;
        post_cont->n_post[i] = 0;
        post_cont->queue_count[i] = 0;
        post_cont->queue_lock[i] = 0;
        for(size_t k = 0; k < 64; k++)
            post_cont->cq_lock[i*64+k] = 0;

        if(i == 0){
            printf("qp_buf[%d]: %p \n", i, post_cont->qp_buf);
            printf("dev_qp_db[%d]: %p \n", i, dev_qp_db);
            printf("bf_reg[%d]: 0x%llx\n", i, device_db);
        }

    }

    printf("Function name: %s, line number: %d\n", __func__, __LINE__);
    struct mlx5_cq *cq1 = to_mcq(s_ctx->gpu_cq[0]);
    struct mlx5_cq *cq2 = to_mcq(s_ctx->gpu_cq[1]);
    struct mlx5_cq *cq128 = to_mcq(s_ctx->gpu_cq[s_ctx->n_bufs]);
    poll_cont->cq_buf = cq1->active_buf->buf;
    
    gpu_infos->cq_buf_gpu[0] = (uint64_t) cq1->active_buf->buf; 
    gpu_infos->cq_buf_gpu[1] = (uint64_t) cq128->active_buf->buf;
    printf("Function name: %s, line number: %d gpu_infos->cq_buf_gpu[0]: %p\n gpu_infos->cq_buf_gpu[1]: %p gpu_infos->cq_buf_gpu[2]: %p\n", 
            __func__, __LINE__, gpu_infos->cq_buf_gpu[0], gpu_infos->cq_buf_gpu[1], cq2->active_buf->buf);

    poll_cont->ibv_cqe = cq1->verbs_cq.cq.cqe;
    poll_cont->cqe_sz = cq1->cqe_sz;
    printf("poll_cont->ibv_cqe: %d\n",poll_cont->ibv_cqe);
    printf("poll_cont->cqe_sz: %d\n",poll_cont->cqe_sz);
    poll_cont->n = 1; // subject to change

    struct mlx5_cq *main_cq = to_mcq(s_ctx->main_cq[0]);
    host_poll->cq_buf = main_cq->active_buf->buf;
    host_poll->ibv_cqe = main_cq->verbs_cq.cq.cqe;
    host_poll->cqe_sz = main_cq->cqe_sz;
    host_poll->n = 1; // subject to change
    host_poll->cons_index[0] = (long long int) &main_cq->cons_index;
    host_poll->cq_dbrec[0] = (long long int) main_cq->dbrec;
    
    for(int i = 0; i < s_ctx->n_bufs*2; i++){
        struct mlx5_cq *cq = to_mcq(s_ctx->gpu_cq[i]);
        struct mlx5_context *mctx = container_of(cq->verbs_cq.cq.context, struct mlx5_context, ibv_ctx.context);
        struct mlx5_resource *rsc = mctx->uidx_table[0].table[0];
        struct mlx5_qp *qp = (struct mlx5_qp *)(rsc);

        void *dev_cons_index;
        success = cudaHostRegister(&cq->cons_index, sizeof(cq->cons_index), cudaHostRegisterMapped);
        if(success != cudaSuccess && success != cudaErrorHostMemoryAlreadyRegistered) return -1;
        if(cudaHostGetDevicePointer(&dev_cons_index, &cq->cons_index, 0) != cudaSuccess) return -1;
        poll_cont->cons_index[i] = (long long int) dev_cons_index;

        
        void *dev_cq_dbrec;
        success = cudaHostRegister(cq->dbrec, sizeof(cq->dbrec), cudaHostRegisterMapped);
        if(success != cudaErrorHostMemoryAlreadyRegistered && success !=  cudaSuccess) return -1;
        if(cudaHostGetDevicePointer(&dev_cq_dbrec, cq->dbrec, 0) != cudaSuccess)
            return -1;
        poll_cont->cq_dbrec[i] = (long long int) dev_cq_dbrec;

        if(i == 0){
            printf("dev_cq_dbrec[%d]: %p \n", i, dev_cq_dbrec);
            printf("dev_cons_index[%d]: %p \n", i, dev_cons_index);
            printf("poll_cont->cq_buf[%d]: 0x%llx\n", i, poll_cont->cq_buf);
        }   
    }

    main_content.cq = s_ctx->main_cq[0];
    main_content.qp = s_ctx->main_qp[0];
    main_content.pd = s_ctx->pd[0];
    
    printf("Function name: %s, line number: %d\n", __func__, __LINE__);
    return 0;
}

int prepare_post_poll_content_2nic(struct context_2nic *s_ctx, struct post_content *post_cont, struct poll_content *poll_cont, struct server_content_2nic *post_cont2, \
                              struct post_content *host_post, struct poll_content *host_poll, struct host_keys *host_post2, struct gpu_memory_info *gpu_infos){
    cudaError_t success;
    printf("Function name: %s, line number: %d\n", __func__, __LINE__);
    struct mlx5_qp *qp_0 = to_mqp(s_ctx->gpu_qp[0]);
    struct mlx5_qp *qp_128 = to_mqp(s_ctx->gpu_qp[s_ctx->n_bufs]);
    printf("Function name: %s, line number: %d\n", __func__, __LINE__);
    post_cont->wr_rdma_remote_addr = (uintptr_t)s_ctx->server_memory[0].addresses[0];
    remote_address_2nic[0] = s_ctx->server_memory[0].addresses[0];
    remote_address_2nic[1] = s_ctx->server_memory[1].addresses[0];
    // post_cont->wr_rdma_rkey = s_ctx->gpu_mr->rkey;
    post_cont->wr_sg_length = 4096; // fixed for now by default
    // post_cont->wr_sg_lkey = s_ctx->gpu_mr->lkey;

    gpu_infos->addrs[0] = (uint64_t) s_ctx->gpu_buffer;
    gpu_infos->addrs[1] = (uint64_t) s_ctx->gpu_buffer;
    gpu_infos->wr_rdma_lkey[0] = (uint64_t) s_ctx->gpu_mr[0]->lkey;
    gpu_infos->wr_rdma_lkey[1] = (uint64_t) s_ctx->gpu_mr[1]->lkey;
    gpu_infos->wr_rdma_rkey[0] = (uint64_t) s_ctx->gpu_mr[0]->rkey;
    gpu_infos->wr_rdma_rkey[1] = (uint64_t) s_ctx->gpu_mr[1]->rkey;
    gpu_infos->qp_buf_gpu[0] = (uint64_t) qp_0->buf.buf;
    gpu_infos->qp_buf_gpu[1] = (uint64_t) qp_128->buf.buf;

    gpu_infos->server_address[0] = (uint64_t) s_ctx->server_memory[0].addresses[0];
    gpu_infos->server_address[1] = (uint64_t) s_ctx->server_memory[1].addresses[0];

    gpu_infos->qp_num_gpu[0] = s_ctx->gpu_qp[0]->qp_num;
    gpu_infos->qp_num_gpu[1] = s_ctx->gpu_qp[s_ctx->n_bufs]->qp_num;

    printf("s_ctx->gpu_qp[0]->qp_num: %d s_ctx->gpu_qp[%d]->qp_num: %d\n", \
            s_ctx->gpu_qp[0]->qp_num, s_ctx->n_bufs, s_ctx->gpu_qp[s_ctx->n_bufs]->qp_num);

    // post_cont->wr_sg_addr = (uintptr_t) s_ctx->gpu_buffer;
    post_cont->wr_opcode = IBV_WR_RDMA_READ; // for read request by default
    post_cont->qp_num = s_ctx->gpu_qp[0]->qp_num;
    post_cont->qp_buf = qp_0->buf.buf;

    post_cont->qpbf_bufsize = qp_0->bf->buf_size;
    post_cont->wr_rdma_rkey = 1;
    printf("Function name: %s, line number: %d\n", __func__, __LINE__);

    // host_post is not used but I leave it here in case needed
    struct mlx5_qp *main_qp = to_mqp(s_ctx->main_qp[0]);
    host_post->wr_rdma_remote_addr = (uintptr_t)s_ctx->server_memory[0].addresses[0];
    // host_post->wr_rdma_rkey = s_ctx->gpu_mr->rkey;
    host_post->wr_sg_length = 1024; // fixed for now by default
    // host_post->wr_sg_lkey = s_ctx->gpu_mr->lkey;
    // printf("Function name: %s, line number: %d\n", __func__, __LINE__);
    // host_post->wr_sg_addr = (uintptr_t) s_ctx->gpu_buffer;
    host_post->wr_opcode = IBV_WR_RDMA_READ; // for read request by default
    host_post->qp_num = s_ctx->main_qp[0]->qp_num;
    host_post->qp_buf = main_qp->buf.buf;
    host_post->qpbf_bufsize = main_qp->bf->buf_size;
    // printf("Function name: %s, line number: %d\n", __func__, __LINE__);
    host_post->wr_rdma_rkey = 1;
    host_post->bf_reg[0] = /*(long long int)*/ main_qp->bf->reg; // device_db;
    host_post->qp_db[0] = (unsigned int *) main_qp->db; // dev_qp_db;
    host_post->dev_qp_sq[0] = &main_qp->sq; // dev_qp_sq;
    host_post->n_post[0] = main_qp->sq.cur_post;
    host_post->cq_lock[0] = 0;
    printf("Function name: %s, line number: %d\n", __func__, __LINE__);
    for (size_t server = 0; server < 2; server++)
    {
        for(int i = 0; i < N_8GB_Region; i++){
            post_cont2->wr_rdma_rkey[i + N_8GB_Region*server] = s_ctx->server_memory[server].rkeys[i];
            post_cont2->wr_rdma_lkey[i + N_8GB_Region*server] = s_ctx->server_memory[server].lkeys[i];
            post_cont2->addrs[i + N_8GB_Region*server] = s_ctx->server_memory[server].addresses[i];

            printf("post_cont2->wr_rdma_rkey[%d + N_8GB_Region*%d]: %d\n", i, server, post_cont2->wr_rdma_rkey[i + N_8GB_Region*server]);


            // post_cont2->wr_rdma_lkey[i] = s_ctx->server_memory.lkeys[i];
            // post_cont2->addrs[i] = s_ctx->server_memory.addresses[i];
            // post_cont2->wr_rdma_lkey[i] = s_ctx->server_memory.addresses[i];
            host_post2->rkeys[i] = s_ctx->server_memory[0].rkeys[i];
            host_post2->lkeys[i] = s_ctx->server_memory[0].lkeys[i];
            host_post2->addrs[i] = s_ctx->server_memory[0].addresses[i];
            printf("server rkey: %d server memory address: %p\n", \
            s_ctx->server_memory[server].rkeys[i], s_ctx->server_memory[server].addresses[i]);
        }
    }

    printf("gpu mr[0] lkey: %d\n", s_ctx->gpu_mr[0]->lkey);
    printf("gpu mr[0] rkey: %d\n", s_ctx->gpu_mr[0]->rkey);

    printf("gpu mr[1] lkey: %d\n", s_ctx->gpu_mr[1]->lkey);
    printf("gpu mr[1] rkey: %d\n", s_ctx->gpu_mr[1]->rkey);

    printf("Function name: %s, line number: %d\n", __func__, __LINE__);
    printf("qp->sq.wqe_cnt: %d\n", qp_0->sq.wqe_cnt);
    printf("[]s_ctx->n_bufs: %d\n\n\n\n", s_ctx->n_bufs);
    for(int i = 0; i < s_ctx->n_bufs*2; i++){
        // printf("qp_num[%d]: %d\n", i, s_ctx->gpu_qp[i]->qp_num);
        struct mlx5_qp *qp = to_mqp(s_ctx->gpu_qp[i]);
        // printf("qp->bf->reg[%d]: %p\n", i, qp->bf->reg);
        // printf("qp->buf.buf[%d]: %p \n", i, qp->buf.buf);

        void *device_db;
        // printf("qp->bf->reg[%d]: 0x%llx\n", i, qp->bf->reg);
        success = cudaHostRegister(qp->bf->reg,  8, cudaHostRegisterIoMemory);
        if (success != cudaSuccess && success != cudaErrorHostMemoryAlreadyRegistered) return -1;
        success = cudaHostGetDevicePointer(&device_db, qp->bf->reg, 0);
        if (success != cudaSuccess) return -1;
        // printf("device_db[%d]: 0x%llx\n", i, device_db);
        // printf("device_db[%d]: 0x%llx\n", i, qp->cqe);
        post_cont->bf_reg[i] = /*(long long int)*/ device_db;

        void *dev_qp_db;
        success = cudaHostRegister(qp->db, sizeof(qp->db), cudaHostRegisterMapped);
        if(success != cudaSuccess && success != cudaErrorHostMemoryAlreadyRegistered)
            return -1;
        if(cudaHostGetDevicePointer(&dev_qp_db, qp->db, 0) != cudaSuccess)
            return -1;
        // printf("dev_qp_db[%d]: %p \n", i, dev_qp_db);
        post_cont->qp_db[i] = (unsigned int *) dev_qp_db;

        void *dev_qp_sq;
        success = cudaHostRegister(&qp->sq, sizeof(qp->sq), cudaHostRegisterMapped);
        if(success != cudaErrorHostMemoryAlreadyRegistered && success !=  cudaSuccess)
                return -1;
        // get GPU pointer for qp->sq
        if(cudaHostGetDevicePointer(&dev_qp_sq, &qp->sq, 0) != cudaSuccess)
            return -1;
        post_cont->dev_qp_sq[i] = dev_qp_sq;
        post_cont->n_post[i] = 0;
        post_cont->queue_count[i] = 0;
        post_cont->queue_lock[i] = 0;
        for(size_t k = 0; k < 64; k++)
            post_cont->cq_lock[i*64+k] = 0;

        if(i == 0){
            printf("qp_buf[%d]: %p \n", i, post_cont->qp_buf);
            printf("dev_qp_db[%d]: %p \n", i, dev_qp_db);
            printf("bf_reg[%d]: 0x%llx\n", i, device_db);
        }

    }

    printf("Function name: %s, line number: %d\n", __func__, __LINE__);
    struct mlx5_cq *cq1 = to_mcq(s_ctx->gpu_cq[0]);
    struct mlx5_cq *cq2 = to_mcq(s_ctx->gpu_cq[1]);
    struct mlx5_cq *cq128 = to_mcq(s_ctx->gpu_cq[s_ctx->n_bufs]);
    poll_cont->cq_buf = cq1->active_buf->buf;
    
    gpu_infos->cq_buf_gpu[0] = (uint64_t) cq1->active_buf->buf; 
    gpu_infos->cq_buf_gpu[1] = (uint64_t) cq128->active_buf->buf;
    printf("Function name: %s, line number: %d gpu_infos->cq_buf_gpu[0]: %p\n gpu_infos->cq_buf_gpu[1]: %p gpu_infos->cq_buf_gpu[2]: %p\n", 
            __func__, __LINE__, gpu_infos->cq_buf_gpu[0], gpu_infos->cq_buf_gpu[1], cq2->active_buf->buf);

    poll_cont->ibv_cqe = cq1->verbs_cq.cq.cqe;
    poll_cont->cqe_sz = cq1->cqe_sz;
    printf("poll_cont->ibv_cqe: %d\n",poll_cont->ibv_cqe);
    printf("poll_cont->cqe_sz: %d\n",poll_cont->cqe_sz);
    poll_cont->n = 1; // subject to change

    struct mlx5_cq *main_cq = to_mcq(s_ctx->main_cq[0]);
    host_poll->cq_buf = main_cq->active_buf->buf;
    host_poll->ibv_cqe = main_cq->verbs_cq.cq.cqe;
    host_poll->cqe_sz = main_cq->cqe_sz;
    host_poll->n = 1; // subject to change
    host_poll->cons_index[0] = (long long int) &main_cq->cons_index;
    host_poll->cq_dbrec[0] = (long long int) main_cq->dbrec;
    
    for(int i = 0; i < s_ctx->n_bufs*2; i++){
        struct mlx5_cq *cq = to_mcq(s_ctx->gpu_cq[i]);
        struct mlx5_context *mctx = container_of(cq->verbs_cq.cq.context, struct mlx5_context, ibv_ctx.context);
        struct mlx5_resource *rsc = mctx->uidx_table[0].table[0];
        struct mlx5_qp *qp = (struct mlx5_qp *)(rsc);

        void *dev_cons_index;
        success = cudaHostRegister(&cq->cons_index, sizeof(cq->cons_index), cudaHostRegisterMapped);
        if(success != cudaSuccess && success != cudaErrorHostMemoryAlreadyRegistered) return -1;
        if(cudaHostGetDevicePointer(&dev_cons_index, &cq->cons_index, 0) != cudaSuccess) return -1;
        poll_cont->cons_index[i] = (long long int) dev_cons_index;

        
        void *dev_cq_dbrec;
        success = cudaHostRegister(cq->dbrec, sizeof(cq->dbrec), cudaHostRegisterMapped);
        if(success != cudaErrorHostMemoryAlreadyRegistered && success !=  cudaSuccess) return -1;
        if(cudaHostGetDevicePointer(&dev_cq_dbrec, cq->dbrec, 0) != cudaSuccess)
            return -1;
        poll_cont->cq_dbrec[i] = (long long int) dev_cq_dbrec;

        if(i == 0){
            printf("dev_cq_dbrec[%d]: %p \n", i, dev_cq_dbrec);
            printf("dev_cons_index[%d]: %p \n", i, dev_cons_index);
            printf("poll_cont->cq_buf[%d]: 0x%llx\n", i, poll_cont->cq_buf);
        }   
    }

    main_content.cq = s_ctx->main_cq[0];
    main_content.qp = s_ctx->main_qp[0];
    main_content.pd = s_ctx->pd[0];
    
    printf("Function name: %s, line number: %d\n", __func__, __LINE__);
    return 0;
}

int benchmark(struct context *s_ctx, int num_msg, int mesg_size, float *bandwidth){
    struct ibv_wc wc;
    struct ibv_send_wr wr, *bad_wr = NULL;
    struct ibv_sge sge;
    // conn = (struct connection *)(uintptr_t)wc.wr_id;
    bad_wr = NULL;
    memset(&wr, 0, sizeof(wr));
    // wr.wr_id = (uintptr_t) 100;
    wr.opcode = IBV_WR_RDMA_READ;
    wr.sg_list = &sge;
    wr.num_sge = 1;
    wr.send_flags = IBV_SEND_SIGNALED;
    wr.wr.rdma.remote_addr = (uintptr_t)s_ctx->server_memory.addresses[0];
    wr.wr.rdma.rkey = s_ctx->server_memory.rkeys[0];
    sge.addr = (uintptr_t)s_ctx->gpu_mr->addr; // conn->rdma_local_region;
    sge.length = 4096;
    sge.lkey = s_ctx->gpu_mr->lkey;
    printf("s_ctx->server_mr.addr: 0x%llx\n", s_ctx->server_memory.addresses[0]);
    printf("s_ctx->server_mr.rkey: %d\n", s_ctx->server_memory.rkeys[0]);
    printf("s_ctx->gpu_mr->addr: 0x%llx\n", s_ctx->gpu_mr->addr);
    printf("s_ctx->gpu_mr->lkey: %d\n", s_ctx->gpu_mr->lkey);
    printf("Function name: %s, line number: %d\n", __func__, __LINE__);
    cpu_benchmark_whole(s_ctx->gpu_cq[0], 1, &wc, s_ctx->gpu_qp[0], &wr, &bad_wr, num_msg, mesg_size, bandwidth);
    return 0;
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
    printf("Function name: %s, line number: %d\n", __func__, __LINE__);
    if (0){
        success = cudaHostRegister(qp->buf.buf, qp->buf.length, cudaHostRegisterMapped);
        if(success !=  cudaSuccess && success != cudaErrorHostMemoryAlreadyRegistered) exit(0);
        if(cudaHostGetDevicePointer(&qp_buf, qp->buf.buf, 0) != cudaSuccess) exit(0);
    }
    else qp_buf = qp->buf.buf;
    printf("Function name: %s, line number: %d\n", __func__, __LINE__);
    success = cudaHostRegister(qp->sq.wqe_head, sizeof(qp->sq.wqe_head), cudaHostRegisterMapped);
    printf("success: %s, %d\n", cudaGetErrorString(success), success);
    if(success != cudaErrorHostMemoryAlreadyRegistered && success !=  cudaSuccess)
            exit(0);
    printf("Function name: %s, line number: %d\n", __func__, __LINE__);
    // get GPU pointer for qp->sq.wqe_head
    if(cudaHostGetDevicePointer(&dev_qpsq_wqe_head, qp->sq.wqe_head, 0) != cudaSuccess)
        exit(0);
    printf("Function name: %s, line number: %d\n", __func__, __LINE__);
    // printf("Function name: %s, line number: %d\n", __func__, __LINE__);
    //   cur_post = qp->sq.cur_post;
    // printf("Function name: %s, line number: %d cur_post: %d\n", __func__, __LINE__, cur_post);
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
    printf("Function name: %s, line number: %d\n", __func__, __LINE__);
    cudaStatus1 = cudaHostGetDevicePointer(&device_db, bf->reg, 0);
    if (cudaStatus1 != cudaSuccess) exit(0);


    cudaError_t cudaState;
    cudaState = cudaHostRegister(qp->db, sizeof(qp->db), cudaHostRegisterMapped);

    if(cudaState != cudaSuccess && cudaState != cudaErrorHostMemoryAlreadyRegistered)
            exit(0);
    if(cudaHostGetDevicePointer(&dev_qp_db, qp->db, 0) != cudaSuccess)
        exit(0);
    cudaError_t cudaStatus;
    
    printf("Function name: %s, line number: %d\n", __func__, __LINE__);
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
    printf("Function name: %s, line number: %d\n", __func__, __LINE__);
    cudaError_t crc = cudaSuccess;
    // cudaError_t cudaStatus;
    
    if (0){
        cudaStatus = cudaHostRegister(cq->active_buf->buf /*cqbuf*/, cq->active_buf->length /*cqbuf_size*/, cudaHostRegisterMapped);
        if(cudaStatus != cudaErrorHostMemoryAlreadyRegistered && cudaStatus != cudaSuccess) exit(0);
        if(cudaHostGetDevicePointer(&dev_cq_ptr, /*cqbuf*/ cq->active_buf->buf, 0) !=  cudaSuccess) exit(0);
    }
    else dev_cq_ptr = cq->active_buf->buf;
    printf("Function name: %s, line number: %d\n", __func__, __LINE__);
    
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
        // printf("timer[i]: %d\n", timer[i]);
        // printf("timer[i+1]: %d\n", timer[i+1]);
        // printf("timer[i+2]: %d\n", timer[i+2]);
        // printf("timer[i+3]: %d\n", timer[i+3]);
    
        poll += timer[4*i+3] - timer[4*i+2];
        post += timer[4*i+1] - timer[4*i];
    
    }
    printf("/*************************************************************/\n");
    printf("Test results for %d packets each with %d bytes\n", num_of_packets, mesg_size*4);

    printf("post: %d, poll: %d\n", post, poll );

    float freq_post = (float)1/((float)devProp.clockRate*1000);
        float g_usec_post = (float)((float)1/(devProp.clockRate*1000))*((post)) * 1000000;
        printf("POST - INTERNAL MEASUREMENT: %f useconds to execute \n", g_usec_post / num_of_packets);
    
    float freq_poll = (float)1/(devProp.clockRate*1000);
    float g_usec__poll = (float)((float)1/(devProp.clockRate*1000))*((poll)) * 1000000;
    printf("POLLING - INTERNAL MEASUREMENT: %f useconds to execute \n", g_usec__poll / num_of_packets);

    float total_usec = /*g_usec_post +*/ g_usec__poll;
    printf("Total time: %f useconds for %d bytes data\n", total_usec, num_of_packets*mesg_size*4);

    float throughtput = (float)(num_of_packets*mesg_size*4)/(total_usec*1e-6*1e9);
    *bandwidth = throughtput;
    printf("Throughput: %f GBps\n", throughtput);
    printf("/*************************************************************/\n");
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
    sge.length =  1500; // (uint32_t) mesg_size*sizeof(int); // RDMA_BUFFER_SIZE*sizeof(int); //RDMA_BUFFER_SIZE;
    printf("mesg_size: %d\n", mesg_size*sizeof(int));
    printf("wr_sg_addr: 0x%llx\n", wr_sg_addr);
    // printf("read_after_write_buffer: 0x%llx\n", read_after_write_buffer);
    sge.lkey = wr_sg_lkey; // conn->rdma_local_mr->lkey;
    int *addr = (int *) wr_sg_addr;
    int cur_post = 0;

    printf("wr1.wr.rdma.remote_addr: 0x%llx\n", wr1.wr.rdma.remote_addr);
    printf("wr1.wr.rdma.rkey: %d\n", wr1.wr.rdma.rkey);

    printf("wr_sg_addr: 0x%llx\n", wr_sg_addr);
    printf("wr_sg_lkey: %d\n", wr_sg_lkey);
    printf("bf_reg: 0x%llx \n", bf_reg);
    for(int p_num=0;  p_num < num_of_packets ; p_num++)
    {

        for(int i = 0; i < mesg_size; i++)
        addr[i] = 0;
        int cons_index_dev = *(int *)cons_index;
        uint32_t *gpu_dbrec = (uint32_t *) dev_cq_dbrec;
        void *cqe = cq_buf + (cons_index_dev & ibv_cqe) * cqe_sz;
        struct mlx5_cqe64 *cqe64 = (struct mlx5_cqe64 *)((cqe_sz == 64) ? cqe : cqe + 64);
        int cond1 = (cqe64->op_own != 240) &&
        !((cqe64->op_own & 1) ^ !!(1 & (ibv_cqe + 1)));
        timer[4*p_num] = clock64();
        int ret = post(wr.wr.rdma.remote_addr + sge.length*p_num, wr.wr.rdma.rkey,
                sge.length, wr_sg_lkey, wr_sg_addr + sge.length*p_num, wr_opcode, 
                qp_num, cur_post, qp_buf, bf_reg, (unsigned int *) dev_qp_db);
        timer[4*p_num+1] = clock64();
    
        cur_post += 1;
        // printf("cqe64->op_own: %d\n", cqe64->op_own);
        
        // printf("cons_index_dev: %d\n", cons_index_dev);
        // printf("ibv_cqe: %d\n", ibv_cqe);
        // printf("cqe_sz: %d\n", cqe_sz);
    
        timer[4*p_num+2] = clock64();
        // while(poll(cq_buf /* the CQ, we got notification for */, 
        //     &wc1, // twc/* where to store */,
        //     (uint32_t *) cons_index,
        //     ibv_cqe,
        //     cqe_sz,
        //     max_wc /* number of remaining WC elements*/,
        //     (uint32_t *) dev_cq_dbrec) == 0);
            // printf("gpu polling\n");
        // while (addr[mesg_size-1] == 0);
        while(!((cqe64->op_own != 240) &&
        !((cqe64->op_own & 1) ^ !!(1 & (ibv_cqe + 1))))); 
        timer[4*p_num+3] = clock64();
    (*(int *)cons_index)++;
        // wc1.qp_num =  htonl(cqe64->sop_drop_qpn) & 0xffffff;
        // wc1.status = (ibv_wc_status) IBV_WC_SUCCESS;
        // gpu_dbrec[0] = htonl((*(int *)cons_index) & 0xffffff);
        printf("addr[0+ sge.length*p_num]: %d\n", addr[0]);
        printf("addr[mesg_size-1]: %d\n", addr[mesg_size-1]);
        printf("addr[%d]: %d\n", 0+ sge.length/4*p_num, addr[0+ sge.length/4*p_num]);
        printf("addr[0+ sge.length*p_num]: %d\n", addr[0+ sge.length/4*p_num]);
        printf("addr[0+ sge.length*p_num]: %d\n", addr[0+ sge.length/4*p_num]);
        printf("addr[1+ sge.length*p_num]: %d\n", addr[1+ sge.length/4*p_num]);
        // for(int i = 0; i < mesg_size; i++)
        //   addr[i] = 0;
        if(wc1.status != IBV_WC_SUCCESS){
        printf("WC1 status is not success: %d\n", wc1.status);
        return;
        }
    }
}


__device__ int get_qp(int id){
    printf("qp_num: %d\n", gpost_cont1->qp_num);
}

__global__ void alloc_content(struct post_content *post_cont, struct poll_content *poll_cont){
    // copy poll and post content to global 
    gpost_cont1 = post_cont;
    gpoll_cont1 = poll_cont;
    printf("qp_num: %d\n", gpost_cont1->qp_num);
}

__global__ void add_vectors_rdma_64MB_512KB(int *a, int *b, int *c, int size, \
                                uint8_t *tlb_A, uint8_t *tlb_B, uint8_t *tlb_C, clock_t *timer,
                                /*struct post_content *post_cont1, struct poll_content *poll_cont1,*/ int data_size1, int num_iter)
{   

    
    // get_qp(0);
	int id = blockDim.x * blockIdx.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
        int single_thread = threadIdx.x;
      
        struct ibv_wc wc;
    // printf("id: %d\n", id);
        int cur_post = 0;
            int data_size = 512*1024; // data_size1;
            int index = blockIdx.x;
            int index_poll = blockIdx.x;
            int tlb_id;;
            index = id/131072;
            /********************************************************************************************/
            uint64_t remote_addr = gpost_cont1->wr_rdma_remote_addr; // + data_size*4*index;
            uint64_t local_addr = gpost_cont1->wr_sg_addr; // + data_size*4*index;
            void *qp_buf = gpost_cont1->qp_buf + 8192*index;
            uint32_t qp_num = gpost_cont1->qp_num + index;
            void *dev_qp_sq = gpost_cont1->dev_qp_sq[index];
            void *bf_reg = (void *) gpost_cont1->bf_reg[index];
            unsigned int *qp_db = gpost_cont1->qp_db[index];
            void *cq_buf = gpoll_cont1->cq_buf + 8192*index;
            uint32_t *cons_index = (uint32_t *) gpoll_cont1->cons_index[index];
            void *cq_dbrec = (void *) gpoll_cont1->cq_dbrec[index];
            uint32_t length = 4*data_size;
            int wr_opcode = gpost_cont1->wr_opcode;
            uint64_t wr_rdma_remote_addr = remote_addr;
            uint32_t wr_rdma_rkey; // = gpost_cont1->wr_rdma_rkey[0];
            int wr_sg_length = data_size;
            uint32_t wr_sg_lkey = gpost_cont1->wr_sg_lkey;
            uint64_t wr_sg_addr = local_addr;
            /********************************************************************************************/
        
        for(int i = id; i < 16777216; i += 524288)
        {
            tlb_id = (int)i/131072;
            remote_addr = wr_rdma_remote_addr + data_size*tlb_id;
            local_addr = wr_sg_addr + data_size*tlb_id;

            if(tlb_A[tlb_id] == 0){
                if(id % 131072 == 0){
                    post_m(remote_addr, wr_rdma_rkey, 
                            wr_sg_length, wr_sg_lkey, local_addr, wr_opcode, 
                            qp_num, cur_post, NULL, qp_buf, bf_reg, qp_db, dev_qp_sq, id);
                    cur_post++;
                    tlb_A[tlb_id] = 1;
                }
                while(tlb_A[tlb_id] == 0);
            }

            if(tlb_B[tlb_id] == 0){
                if(id % 131072 == 0){
                    void *cqe = cq_buf + (cur_post & 63) * 64;
                    struct mlx5_cqe64 *cqe64 = (struct mlx5_cqe64 *) cqe;
                    post_m(remote_addr+64*1024*1024, wr_rdma_rkey, 
                            wr_sg_length, wr_sg_lkey, local_addr+64*1024*1024, wr_opcode, 
                            qp_num, cur_post, NULL, qp_buf, bf_reg, qp_db, dev_qp_sq, id);
                    cur_post++;
                    while(cqe64->op_own == 240);
                    *(uint32_t *) cq_dbrec = (uint32_t) htonl((cur_post ) & 0xffffff);
                    cqe64->op_own = 240;
                    
                    tlb_B[tlb_id] = 1;
                }
                while(tlb_B[tlb_id] == 0);
            }
            
            c[i] = a[i] + b[i];
            
        }
        
}

__global__ void add_vectors_rdma_64MB_64KB(int *a, int *b, int *c, int size, \
                                uint8_t *tlb_A, uint8_t *tlb_B, uint8_t *tlb_C, clock_t *timer,
                                /*struct post_content *post_cont1, struct poll_content *poll_cont1,*/ int data_size1, int num_iter)
{   

    
    // get_qp(0);
	int id = blockDim.x * blockIdx.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
        int single_thread = threadIdx.x;
      
        struct ibv_wc wc;
    // printf("id: %d\n", id);
        int cur_post = 0;
            int data_size = 64*1024; // data_size1;
            int index = blockIdx.x;
            int index_poll = blockIdx.x;
            int tlb_id;;
            index = id/16384;
            /********************************************************************************************/
            uint64_t remote_addr = gpost_cont1->wr_rdma_remote_addr; // + data_size*4*index;
            uint64_t local_addr = gpost_cont1->wr_sg_addr; // + data_size*4*index;
            void *qp_buf = gpost_cont1->qp_buf + 8192*index;
            uint32_t qp_num = gpost_cont1->qp_num + index;
            void *dev_qp_sq = gpost_cont1->dev_qp_sq[index];
            void *bf_reg = (void *) gpost_cont1->bf_reg[index];
            unsigned int *qp_db = gpost_cont1->qp_db[index];
            void *cq_buf = gpoll_cont1->cq_buf + 8192*index;
            uint32_t *cons_index = (uint32_t *) gpoll_cont1->cons_index[index];
            void *cq_dbrec = (void *) gpoll_cont1->cq_dbrec[index];
            uint32_t length = 4*data_size;
            int wr_opcode = gpost_cont1->wr_opcode;
            uint64_t wr_rdma_remote_addr = remote_addr;
            uint32_t wr_rdma_rkey; // = gpost_cont1->wr_rdma_rkey[0];
            int wr_sg_length = data_size;
            uint32_t wr_sg_lkey = gpost_cont1->wr_sg_lkey;
            uint64_t wr_sg_addr = local_addr;
            /********************************************************************************************/
        
        for(int i = id; i < 16777216; i += 524288)
        {
            tlb_id = (int)i/16384;
            remote_addr = wr_rdma_remote_addr + data_size*tlb_id;
            local_addr = wr_sg_addr + data_size*tlb_id;
           
            if(tlb_A[tlb_id] == 0){
                if(id % 16384 == 0){
                    // void *cqe = cq_buf + (cur_post & 15) * 64;
                    // struct mlx5_cqe64 *cqe64 = (struct mlx5_cqe64 *) cqe;
                    
                    post_m(remote_addr, wr_rdma_rkey, 
                            wr_sg_length, wr_sg_lkey, local_addr, wr_opcode, 
                            qp_num, cur_post, NULL, qp_buf, bf_reg, qp_db, dev_qp_sq, id);
                    

                    cur_post++;
                    // while(cqe64->op_own == 240);
                    // *(uint32_t *) cq_dbrec = (uint32_t) htonl((cur_post ) & 0xffffff);
                    // poll(cq_buf, &wc, cons_index, gpoll_cont1->ibv_cqe, gpoll_cont1->cqe_sz, 1, cq_dbrec);
                   
            
                    tlb_A[tlb_id] = 1;
                    
                }
                while(tlb_A[tlb_id] == 0);
                // __syncthreads();
            }
           
            if(tlb_B[tlb_id] == 0){
                if(id % 16384 == 0){
                    // printf(" B: id: %d, index: %d\n", id, data_size*index);
                    // for(int i = 0; i < num_iter; i++){
                    // __syncthreads();
                    // if (/*(threadIdx.x | threadIdx.y | threadIdx.z)*/ threadIdx.x == 0) {
                        
                        void *cqe = cq_buf + (cur_post & 63) * 64;
                        struct mlx5_cqe64 *cqe64 = (struct mlx5_cqe64 *) cqe;
                        // uint32_t cons_index_dev = cur_post; // = *cons_index;
                        // cqe = cq_buf + (cur_post & 15/*gpoll_cont1->ibv_cqe*/) * 64 /*gpoll_cont1->cqe_sz*/;
                        // printf("gpoll_cont1->ibv_cqe: %d gpoll_cont1->cqe_sz: %d\n", gpoll_cont1->ibv_cqe, gpoll_cont1->cqe_sz);
                        // cqe64 = (struct mlx5_cqe64 *) ((gpoll_cont1->cqe_sz == 64) ? cqe : cqe + 64);

                    // timer[4*blockIdx.x] = clock();
                    post_m(remote_addr+64*1024*1024, wr_rdma_rkey, 
                            wr_sg_length, wr_sg_lkey, local_addr+64*1024*1024, wr_opcode, 
                            qp_num, cur_post, NULL, qp_buf, bf_reg, qp_db, dev_qp_sq, id);
                    // timer[4*blockIdx.x+1] = clock();
                    
                    // ((cqe64->op_own != 240) && !((cqe64->op_own & 1) ^ !!(1 & (poll_cont.ibv_cqe + 1))))==0

                        cur_post++;
                        // // // timer[4*blockIdx.x] = clock();
                        // cons_index_dev = cur_post; // *cons_index;
                        // uint32_t *gpu_dbrec = (uint32_t *) cq_dbrec;

                    // timer[4*blockIdx.x+2] = clock();
                        while(cqe64->op_own == 240);
                        // poll(cq_buf, &wc, cons_index, gpoll_cont1->ibv_cqe, gpoll_cont1->cqe_sz, 1, cq_dbrec);
                    // timer[4*blockIdx.x+3] = clock();
                    
                    
                        // *cons_index = cons_index_dev + 1;
                        *(uint32_t *) cq_dbrec = (uint32_t) htonl((cur_post ) & 0xffffff);
                        cqe64->op_own = 240;
                        // gpu_dbrec[0] = htonl((cons_index_dev ) & 0xffffff);
                    // while(poll(cq_buf, &wc, cons_index, poll_cont.ibv_cqe, poll_cont.cqe_sz, 1, cq_dbrec)==0);

                    tlb_B[tlb_id] = 1;
                    // printf(" B: id: %d, index: %d tlb_B[%d]: %d\n", id, data_size*index, tlb_id, tlb_B[tlb_id]);
                }
                while(tlb_B[tlb_id] == 0);
                // __syncthreads();
            }
            // if(i == 4194336){
            //     printf("i: %d, id: %d, a[i]: %d, b[i]: %d\n", i, id, a[i], b[i]);
            //     printf("i: %d, id: %d, tlb_B[%d]: %d\n", i, id, tlb_id, tlb_B[tlb_id]);
            // }
            c[i] = a[i] + b[i];
            
        }
        
}


__global__ void add_vectors_rdma(int *a, int *b, int *c, int size, \
                                uint8_t *tlb_A, uint8_t *tlb_B, uint8_t *tlb_C, clock_t *timer,
                                /*struct post_content *post_cont1, struct poll_content *poll_cont1,*/ int data_size1, int num_iter)
{   

    
    // get_qp(0);
	int id = blockDim.x * blockIdx.x + threadIdx.x;
    
    
        int single_thread = threadIdx.x;
        // if (tlb_A[blockIdx.x] == 0){
            
            
        // struct post_content post_cont = *gpost_cont1; 
        // struct poll_content poll_cont = *gpoll_cont1;
        struct ibv_wc wc;
        // configure work request here
        // default works fpr this request
        // post and poll using CQ
        struct post_wr wr;
        // wr.qp_num = post_cont.qp_num;
        // wr.wr_opcode = IBV_WR_RDMA_READ; // post_cont.wr_opcode;
        // wr.wr_rdma_remote_addr = post_cont.wr_rdma_remote_addr;
        // wr.wr_rdma_rkey = post_cont.wr_rdma_rkey;
        // wr.wr_sg_addr = post_cont.wr_sg_addr;
        // wr.wr_sg_length = 128*4; // post_cont.wr_sg_length;
        // wr.wr_sg_lkey = post_cont.wr_sg_lkey;
        int cur_post = 0;
        
        
        // if(threadIdx.x == 0){
            cur_post = 0;
            int data_size = data_size1;
            int index = blockIdx.x;
            int index_poll = blockIdx.x;
            /*********************************************************************************************/
            // uint64_t remote_addr = post_cont.wr_rdma_remote_addr + data_size*4*index;
            // uint64_t local_addr = post_cont.wr_sg_addr + data_size*4*index;
            // void *qp_buf = post_cont.qp_buf + 8192*index;
            // uint32_t qp_num = post_cont.qp_num + index;
            // void *dev_qp_sq = post_cont.dev_qp_sq[index];
            // void *bf_reg = (void *) post_cont.bf_reg[index];
            // unsigned int *qp_db = post_cont.qp_db[index];
            // void *cq_buf = poll_cont.cq_buf + 4096*index;
            // uint32_t *cons_index = (uint32_t *) poll_cont.cons_index[index];
            // void *cq_dbrec = (void *) poll_cont.cq_dbrec[index];
            // uint32_t length = 4*data_size;
            // int wr_opcode = post_cont.wr_opcode;
            // uint64_t wr_rdma_remote_addr = remote_addr;
            // uint32_t wr_rdma_rkey = post_cont.wr_rdma_rkey;
            // int wr_sg_length = data_size*4;
            // uint32_t wr_sg_lkey = post_cont.wr_sg_lkey;
            // uint64_t wr_sg_addr = local_addr;
            /********************************************************************************************/
            uint64_t remote_addr = gpost_cont1->wr_rdma_remote_addr + data_size*4*index*1*num_iter;
            uint64_t local_addr = gpost_cont1->wr_sg_addr + data_size*4*index*1*num_iter;
            void *qp_buf = gpost_cont1->qp_buf + 8192*index;
            uint32_t qp_num = gpost_cont1->qp_num + index;
            void *dev_qp_sq = gpost_cont1->dev_qp_sq[index];
            void *bf_reg = (void *) gpost_cont1->bf_reg[index];
            unsigned int *qp_db = gpost_cont1->qp_db[index];
            void *cq_buf = gpoll_cont1->cq_buf + 4096*index_poll;
            uint32_t *cons_index = (uint32_t *) gpoll_cont1->cons_index[index_poll];
            void *cq_dbrec = (void *) gpoll_cont1->cq_dbrec[index_poll];
            uint32_t length = 4*data_size;
            int wr_opcode = gpost_cont1->wr_opcode;
            uint64_t wr_rdma_remote_addr = remote_addr;
            uint32_t wr_rdma_rkey; // = gpost_cont1->wr_rdma_rkey[0];
            int wr_sg_length = data_size*4;
            uint32_t wr_sg_lkey = gpost_cont1->wr_sg_lkey;
            uint64_t wr_sg_addr = local_addr;
            /********************************************************************************************/
            
            // __syncthreads();
            // if(blockIdx.x >= 0){
                // timer[4*blockIdx.x+1] = clock();
                // if(id == 0){
                    // printf("id: %d, blockIdx.x: %d, qp_num: %d: post_cont.wr_rdma_rkey: %d\n",id, index, qp_num, gpost_cont1->wr_rdma_rkey);
                    // printf("id: %d, blockIdx.x: %d, remote_addr: %p: local_addr: %p\n",id, index, remote_addr, local_addr);
                    
                // if(tlb_A[blockIdx.x]==0){
                    if(threadIdx.x == 0){
                        for (int i = 0; i < num_iter; i++){
                        // for(int i = 0; i < 1; i++){
                        // __syncthreads();
                        // if (/*(threadIdx.x | threadIdx.y | threadIdx.z)*/ threadIdx.x == 0) {
                            
                                
                            // printf("wr_sg_length: %d local_addr: %p\n", wr_sg_length, local_addr + data_size*4*0);
                            timer[4*blockIdx.x] = clock();
                            // request qp atomically
                            post_m(remote_addr + wr_sg_length*i, wr_rdma_rkey, 
                                    wr_sg_length, wr_sg_lkey, local_addr  + wr_sg_length*i, wr_opcode, 
                                    qp_num, cur_post, NULL, qp_buf, bf_reg, qp_db, dev_qp_sq, id);
                            // timer[4*blockIdx.x+1] = clock();
                            
                           
                            // printf("wr_sg_length: %d local_addr: %p\n", wr_sg_length, local_addr + data_size*4*0);
                            // ((cqe64->op_own != 240) && !((cqe64->op_own & 1) ^ !!(1 & (poll_cont.ibv_cqe + 1))))==0

                                cur_post++;
                            
                            
                            // post_m(remote_addr + wr_sg_length*0.5, wr_rdma_rkey, 
                            //         wr_sg_length*0.5, wr_sg_lkey, local_addr  + wr_sg_length*0.5, wr_opcode, 
                            //         qp_num, cur_post, qp_buf, bf_reg, qp_db, dev_qp_sq, id);
                            //     timer[4*blockIdx.x] = clock();
                                // post_m(remote_addr + wr_sg_length*i, wr_rdma_rkey, 
                                //         wr_sg_length, wr_sg_lkey, local_addr + wr_sg_length*i, wr_opcode, 
                                //         qp_num, cur_post, qp_buf, bf_reg, qp_db, dev_qp_sq, id);
                                timer[4*blockIdx.x+1] = clock();

                            

                                // cqe = cq_buf + (cur_post & 15) * 64;
                                // cqe64 = (struct mlx5_cqe64 *) cqe;

                                
                                // // // timer[4*blockIdx.x] = clock();
                                // cons_index_dev = *cons_index;
                                // uint32_t *gpu_dbrec = (uint32_t *) cq_dbrec;

                            // 
                                // while(cqe64->op_own == 240){
                                //     // gpu_dbrec[0] = htonl(*cons_index & 0xffffff);
                                // }

                               

                                // poll(cq_buf, &wc, cons_index, gpoll_cont1->ibv_cqe, gpoll_cont1->cqe_sz, 1, cq_dbrec);
                            
                            
                            
                                // *cons_index = cons_index_dev + 1;
                                // gpu_dbrec[0] = htonl((cons_index_dev + 1) & 0xffffff);
                            // while(poll(cq_buf, &wc, cons_index, poll_cont.ibv_cqe, poll_cont.cqe_sz, 1, cq_dbrec)==0);
                    
                            // tlb_A[blockIdx.x] = 1;
                        // }
                        }
                        cur_post--;
                        
                        timer[4*blockIdx.x+2] = clock();                        

                        int icqe = cur_post;
                        void *cqe = cq_buf + (icqe & 15) * 64;
                        struct mlx5_cqe64 *cqe64 = (struct mlx5_cqe64 *) cqe;
                        timer[4*blockIdx.x+2] = clock();
                        // while (icqe>=0){
                            while(cqe64->op_own == 240){
                            }
                            // *(uint32_t *) cq_dsbrec = (uint32_t) htonl((icqe +1) & 0xffffff);
                        //     icqe--;
                        //     cqe = cq_buf + (icqe & 15) * 64;
                        //     cqe64 = (struct mlx5_cqe64 *) cqe;
                        // }
                        timer[4*blockIdx.x+3] = clock();
                        //  *(uint32_t *) cq_dbrec = (uint32_t) htonl((cur_post ) & 0xffffff);
                        // cqe64->op_own = 240;
                        // tlb_A[blockIdx.x] = 1;
                    }
                    // while (tlb_A[blockIdx.x]==0)
                    // {
                    //     /* code */
                    // }
                    
               
        __syncthreads();    
       
        // if (a[id] == 2)
        // // //     // printf("a[%d]: %d a[%d]: %d\n", id, a[id], id+blockDim.x, a[id+blockDim.x]);
        //     printf("a[%d]: %d b[%d]: %d\n", id, a[id], id, b[id]);
        // c[id] = a[id] + b[id];
        // __syncthreads();
	// }

}


// __global__ void add_vectors_rdma(int *a, int *b, int *c, int size, \
//                                 uint8_t *tlb_A, uint8_t *tlb_B, uint8_t *tlb_C, clock_t *timer,
//                                 struct post_content *post_cont1, struct poll_content *poll_cont1, int data_size1)
// {   

    

// 	int id = blockDim.x * blockIdx.x + threadIdx.x;
    
    
//         int single_thread = threadIdx.x;
//         // if (tlb_A[blockIdx.x] == 0){
            
            
//         struct post_content post_cont = *post_cont1; 
//         struct poll_content poll_cont = *poll_cont1;
//         struct ibv_wc wc;
//         // configure work request here
//         // default works fpr this request
//         // post and poll using CQ
//         struct post_wr wr;
//         // wr.qp_num = post_cont.qp_num;
//         // wr.wr_opcode = IBV_WR_RDMA_READ; // post_cont.wr_opcode;
//         // wr.wr_rdma_remote_addr = post_cont.wr_rdma_remote_addr;
//         // wr.wr_rdma_rkey = post_cont.wr_rdma_rkey;
//         // wr.wr_sg_addr = post_cont.wr_sg_addr;
//         // wr.wr_sg_length = 128*4; // post_cont.wr_sg_length;
//         // wr.wr_sg_lkey = post_cont.wr_sg_lkey;
//         int cur_post = 0;
        
        
//         // if(threadIdx.x == 0){
//             cur_post = 0;
//             int data_size = data_size1;
//             int index = id; // threadIdx.x;
//             uint64_t remote_addr = post_cont.wr_rdma_remote_addr + 1*4*index;
//             uint64_t local_addr = post_cont.wr_sg_addr + 1*4*index;
//             void *qp_buf = post_cont.qp_buf + 8192*index;
//             uint32_t qp_num = post_cont.qp_num + index;
//             void *dev_qp_sq = post_cont.dev_qp_sq[index];
//             void *bf_reg = (void *) post_cont.bf_reg[index];
//             unsigned int *qp_db = post_cont.qp_db[index];
//             void *cq_buf = poll_cont.cq_buf + 4096*index;
//             uint32_t *cons_index = (uint32_t *) poll_cont.cons_index[index];
//             void *cq_dbrec = (void *) poll_cont.cq_dbrec[index];
//             uint32_t length = 4*1;
//             int wr_opcode = post_cont.wr_opcode;
//             uint64_t wr_rdma_remote_addr = remote_addr;
//             uint32_t wr_rdma_rkey = post_cont.wr_rdma_rkey;
//             int wr_sg_length = 1*4;
//             uint32_t wr_sg_lkey = post_cont.wr_sg_lkey;
//             uint64_t wr_sg_addr = local_addr;
//             printf("id: %d, blockIdx.x: %d, qp_num: %d: remote_addr: %p qp_buf: %p\n", id, index, qp_num, remote_addr, qp_buf);
//             // __syncthreads();
//             // if(blockIdx.x >= 0){
                
//                 // if(tlb_A[threadIdx.x]==0){
//                 //     if(threadIdx.x == 0){
//                         // __syncthreads();
//                         // if (/*(threadIdx.x | threadIdx.y | threadIdx.z)*/ threadIdx.x == 0) {
//                         // timer[4*threadIdx.x] = clock(); 
//                         void *cqe;
//                         struct mlx5_cqe64 *cqe64;
//                         uint32_t cons_index_dev = *cons_index;
//                         cqe = cq_buf + (cons_index_dev & poll_cont.ibv_cqe) * poll_cont.cqe_sz;
                        
//                         cqe64 = (struct mlx5_cqe64 *)((poll_cont.cqe_sz == 64) ? cqe : cqe + 64);
                        
//                         post_m(remote_addr, post_cont.wr_rdma_rkey, 
//                                 wr_sg_length, post_cont.wr_sg_lkey, local_addr, wr_opcode, 
//                                 qp_num, 0, qp_buf, bf_reg, qp_db, dev_qp_sq, id);
                        
                        
//                         // ((cqe64->op_own != 240) && !((cqe64->op_own & 1) ^ !!(1 & (poll_cont.ibv_cqe + 1))))==0

//                         // cons_index_dev = *cons_index;
//                         uint32_t *gpu_dbrec = (uint32_t *) cq_dbrec;
//                         // while(cqe64->op_own == 240){
//                         //     // gpu_dbrec[0] = htonl(*cons_index & 0xffffff);
//                         // }

//                          if((cqe64->op_own >> 4) != 0) //Check opcode
//                         {
//                             struct mlx5_err_cqe *err_cqe = (struct mlx5_err_cqe *)cqe64;
//                             printf("Got completion with error, opcode = %d , syndrome = %d\n",(cqe64->op_own >> 4), err_cqe->syndrome);
//                             // wc->status = IBV_WC_GENERAL_ERR;
//                         }

//                         // poll(cq_buf, &wc, cons_index, poll_cont.ibv_cqe, poll_cont.cqe_sz, 1, cq_dbrec);
                        
                        
//                         *cons_index = cons_index_dev + 1;
//                         gpu_dbrec[0] = htonl((cons_index_dev + 1) & 0xffffff);
//                         // timer[4*threadIdx.x+1] = clock();
//                         // while(poll(cq_buf, &wc, cons_index, poll_cont.ibv_cqe, poll_cont.cqe_sz, 1, cq_dbrec)==0);
//                         // 
                        
//                         // uint64_t local_addr1 = post_cont.wr_rdma_rkey + wr_sg_length;
//                         // uint64_t remote_addr = post_cont.wr_rdma_remote_addr + wr_sg_length;
//                         // post_m(remote_addr + wr_sg_length, post_cont.wr_rdma_rkey, 
//                         //         wr_sg_length, post_cont.wr_sg_lkey, local_addr + wr_sg_length, wr_opcode, 
//                         //         qp_num, 1, qp_buf, bf_reg, qp_db, dev_qp_sq, id); 

                        
                            
//                         // while(poll(cq_buf, &wc, cons_index, poll_cont.ibv_cqe, poll_cont.cqe_sz, 1, cq_dbrec)==0);
//                 //         tlb_A[blockIdx.x] = 1;
//                 //     }
//                 //     while(tlb_A[blockIdx.x]==0);
//                 //     // __syncthreads();
//                 // }
            
//             // }
            
//             // if(tlb_B[blockIdx.x]==0){
//             //     if(threadIdx.x == 0){
//                     // void *cqe;
//                     // struct mlx5_cqe64 *cqe64;
//                     // uint32_t cons_index_dev = *cons_index;
//                     // uint32_t *gpu_dbrec = (uint32_t *) cq_dbrec;
//                     // cqe = cq_buf + (*cons_index & poll_cont.ibv_cqe) * poll_cont.cqe_sz;
                    
//                     // cqe64 = (struct mlx5_cqe64 *)((poll_cont.cqe_sz == 64) ? cqe : cqe + 64);
                    
//                     // post_m(remote_addr + 1024*1024, post_cont.wr_rdma_rkey, 
//                     //     wr_sg_length, post_cont.wr_sg_lkey, local_addr + 1024*1024, wr_opcode, 
//                     //     qp_num, 1, qp_buf, bf_reg, qp_db, dev_qp_sq, id);
                     
                    
                    
//                     // // while(cqe64->op_own == 240){
//                     // //     // gpu_dbrec[0] = htonl(*cons_index & 0xffffff);
//                     // // }
//                     // // while(poll(cq_buf, &wc, cons_index, poll_cont.ibv_cqe, poll_cont.cqe_sz, 1, cq_dbrec)==0);
//                     // timer[4*blockIdx.x+2] = clock();
//                     // poll(cq_buf, &wc, cons_index, poll_cont.ibv_cqe, poll_cont.cqe_sz, 1, cq_dbrec);
//                     // timer[4*blockIdx.x+3] = clock();  
//                     // *cons_index++;
//                     // gpu_dbrec[0] = htonl((*cons_index) & 0xffffff);
//             //         tlb_B[blockIdx.x] = 1;    
//             //     }
//             //     while(tlb_B[blockIdx.x]==0);
                
//             // }
            
//             __syncthreads();
            
//         // __syncthreads();
//         for(int del = 0; del < 20000000; del++);
//         __syncthreads();    
       
//         // if (a[id] != 2)
//         //     // printf("a[%d]: %d a[%d]: %d\n", id, a[id], id+blockDim.x, a[id+blockDim.x]);
//         // for(int i = data_size*index; i < data_size*2*index; i++)
//             printf("a[%d]: %d b[%d]: %d\n", id, a[id], id, b[id]);

//         c[id] = a[id] + b[id];

//         // __syncthreads();
       
// 	// }
    
// }

int destroy(struct context *s_ctx){
    struct connection *conn = (struct connection *)s_ctx;
    rdma_destroy_qp(s_ctx->id);
    printf("line number %d\n", __LINE__);
    for(int i = 0; i < s_ctx->n_bufs; i++){
        if( 1 != 0) {
            printf("line number ibv_destroy_qp(s_ctx->gpu_qp[i]): %d, %d\n", ibv_destroy_qp(s_ctx->gpu_qp[i]), __LINE__);
            printf("Destroy failed on qp: %d\n", i);
            return -1;
        }
    }
}

// __device__ struct post_content *post_cont;
// __device__ struct poll_content *poll_cont;
__device__ int qps[256];

__device__ int get_qp(void){
    return 1;
}