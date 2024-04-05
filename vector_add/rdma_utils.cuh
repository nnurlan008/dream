#ifndef RDMA_UTILS_CUH
#define RDMA_UTILS_CUH

#include <sys/types.h>
#include <infiniband/verbs.h>

// static int N_BUFs = 256;

struct context {
  struct ibv_context *ctx;
  struct ibv_pd *pd;
  struct ibv_cq *main_cq;
  struct ibv_comp_channel *comp_channel;
  struct ibv_qp *main_qp;

   struct ibv_qp **gpu_qp;
  struct ibv_cq **gpu_cq;
  void **cqbuf;
  int cqbuf_size = 4096;
  void **wqbuf;
  int wqbuf_size = 8192;
  int n_bufs;

  void *gpu_buffer;
  int gpu_buf_size = 3*1024*1024; // 3 MB

  struct ibv_mr *gpu_mr;
  struct ibv_mr server_mr;

  pthread_t cq_poller_thread;
};

struct remote_qp_info{
    uint32_t target_qp_num[256];
    uint16_t target_lid;
    union ibv_gid target_gid;
};

struct __attribute__((__packed__)) post_wr{
  uint64_t wr_rdma_remote_addr; 
  uint32_t wr_rdma_rkey;
  uint32_t wr_sg_length; 
  uint32_t wr_sg_lkey; 
  uint64_t wr_sg_addr; 
  int wr_opcode; 
  uint32_t qp_num; 
};

struct __attribute__((__packed__)) post_content{
  unsigned int qpbf_bufsize; // explain all these variables
  uint64_t wr_rdma_remote_addr; 
  uint32_t wr_rdma_rkey;
  uint32_t wr_sg_length; 
  uint32_t wr_sg_lkey; 
  uint64_t wr_sg_addr; 
  int wr_opcode; 
  uint32_t qp_num; 
  void *qp_buf;  
  long long int bf_reg[20];
};

struct __attribute__((__packed__)) poll_content{
  void *cq_buf; // explain all these variables
  long long int cons_index[20];
  int ibv_cqe; 
  uint32_t cqe_sz; 
  int n; 
  long long int cq_dbrec[20];
};

struct benchmark_content{
  struct ibv_cq *cq_ptr; 
  int num_entries; 
  struct ibv_wc *wc;
  struct ibv_qp *ibqp; 
  struct ibv_send_wr *wr;
  struct ibv_send_wr **bad_wr; 
  int num_packets; 
  int mesg_size; 
  float *bandwidth;
};

int init_gpu(int gpu);
int connect(const char *ip, struct context *s_ctx);
int prepare_post_poll_content(struct context *s_ctx, struct post_content *post_cont, struct poll_content *poll_cont);
void host_poll_fake(struct ibv_cq *cq1, struct ibv_wc *wc);

__device__ int poll(void *cq_buf, struct ibv_wc *wc, uint32_t *cons_index,
                    int ibv_cqe, uint32_t cqe_sz, int n, void *cq_dbrec);

__device__ int post_s(struct post_wr wr, int cur_post, void *qp_buf, void *bf_reg);


__device__ int post(uint64_t wr_rdma_remote_addr, uint32_t wr_rdma_rkey,
                    uint32_t wr_sg_length, uint32_t wr_sg_lkey, uint64_t wr_sg_addr, 
                    int wr_opcode, uint32_t qp_num, int cur_post, void *qp_buf, void *bf_reg);

int benchmark(struct context *s_ctx, int num_msg, int mesg_size, float *bandwidth);

int cpu_benchmark_whole(struct ibv_cq *cq_ptr, int num_entries, struct ibv_wc *wc,
                   struct ibv_qp *ibqp, struct ibv_send_wr *wr,
                   struct ibv_send_wr **bad_wr, int num_packets, int mesg_size, float *bandwidth);

          
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
);

void *benchmark(void *param);

#endif