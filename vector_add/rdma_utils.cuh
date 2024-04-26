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
  // void* volatile cqbuf = NULL;
  // void* volatile wqbuf = NULL;
  void** volatile cqbuf;
  int cqbuf_size = 4096;
  void** volatile wqbuf;
  int wqbuf_size = 8192;
  int n_bufs;

  void *gpu_buffer;
  int gpu_buf_size; // 3 MB

  struct ibv_mr *gpu_mr;
  struct ibv_mr server_mr;

  struct rdma_cm_id	*id;

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
  unsigned int cq_lock[256];
  size_t n_post[256];
  void *qp_buf;  
  void *bf_reg[256];
  unsigned int *qp_db[256];
  void *dev_qp_sq[256];

  __forceinline__
  __host__ __device__ 
  post_content& operator=(const post_content& obj) {
        
    this->qpbf_bufsize = obj.qpbf_bufsize;
    this->wr_rdma_remote_addr = obj.wr_rdma_remote_addr;
    this->wr_rdma_rkey = obj.wr_rdma_rkey;
    this->wr_sg_length = obj.wr_sg_length;
    this->wr_sg_lkey = obj.wr_sg_lkey;
    this->wr_opcode = obj.wr_opcode;
    this->wr_sg_addr = obj.wr_sg_addr;
    this->qp_num = obj.qp_num;
    this->qp_buf = obj.qp_buf;

    for(int i = 0; i < 256; i++)
    {
      this->n_post[i] = obj.n_post[i];
      this->bf_reg[i] = obj.bf_reg[i];
      this->qp_db[i] = obj.qp_db[i];
      this->dev_qp_sq[i] = obj.dev_qp_sq[i];
      this->cq_lock[i] = obj.cq_lock[i];
    }
            
    return *this;
  }
};

struct __attribute__((__packed__)) poll_content{
  void *cq_buf; // explain all these variables
  long long int cons_index[256];
  int ibv_cqe; 
  uint32_t cqe_sz; 
  int n; 
  long long int cq_dbrec[256];

  __forceinline__
  __host__ __device__ 
  poll_content& operator=(const poll_content& obj) {
        
    this->cq_buf = obj.cq_buf;
    this->ibv_cqe = obj.ibv_cqe;
    this->cqe_sz = obj.cqe_sz;
    this->n = obj.n;
    for(int i = 0; i < 256; i++)
    {
      this->cons_index[i] = obj.cons_index[i];
      this->cq_dbrec[i] = obj.cq_dbrec[i];
    }
    return *this;
  }
};

struct __attribute__((__packed__)) wqe_segment_ctrl {
    uint8_t     opmod;
    uint16_t    wqe_index;
    uint8_t     opcode;
    uint32_t    qpn_ds;
    uint8_t     signature;
    uint16_t    rsvd;
    uint8_t     fcs;
    uint32_t    imm;
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

//  __device__ struct post_content *gpost_cont;
//  __device__ struct poll_content *gpoll_cont;

int init_gpu(int gpu);
int connect(const char *ip, struct context *s_ctx);
int prepare_post_poll_content(struct context *s_ctx, struct post_content *post_cont, struct poll_content *poll_cont);
void host_poll_fake(struct ibv_cq *cq1, struct ibv_wc *wc);

__device__ int poll(void *cq_buf, struct ibv_wc *wc, uint32_t *cons_index,
                    int ibv_cqe, uint32_t cqe_sz, int n, void *cq_dbrec);

__device__ int post_s(struct post_wr wr, int cur_post, void *qp_buf, void *bf_reg);

__device__ int post(uint64_t wr_rdma_remote_addr, uint32_t wr_rdma_rkey,
                    uint32_t wr_sg_length, uint32_t wr_sg_lkey, uint64_t wr_sg_addr, 
                    int wr_opcode, uint32_t qp_num, int cur_post, void *qp_buf, void *bf_reg, unsigned int *qp_db);

__device__ int post_m(uint64_t wr_rdma_remote_addr, uint32_t wr_rdma_rkey,
                    uint32_t wr_sg_length, uint32_t wr_sg_lkey, uint64_t wr_sg_addr, 
                    int wr_opcode, uint32_t qp_num, int cur_post, void *qp_buf, void *bf_reg, unsigned int *qp_db, void *dev_qp_sq, int id);

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

__global__ void add_vectors_rdma(int *a, int *b, int *c, int size, \
                                uint8_t *tlb_A, uint8_t *tlb_B, uint8_t *tlb_C, clock_t *timer,/* struct post_content *post_cont1, struct poll_content *poll_cont1,*/ int data_size, int num_iter);

__global__ void add_vectors_rdma_64MB_512KB(int *a, int *b, int *c, int size, \
                                uint8_t *tlb_A, uint8_t *tlb_B, uint8_t *tlb_C, clock_t *timer,
                                /*struct post_content *post_cont1, struct poll_content *poll_cont1,*/ int data_size1, int num_iter);

__global__ void alloc_content(struct post_content *post_cont, struct poll_content *poll_cont);

void *benchmark(void *param);

int destroy(struct context *s_ctx);


// template<typename T>
// struct rdma_buf {
//     uint64_t address;
//     // uint64_t address;
//     size_t size;

//     // constructor
//     rdma_buf(uint64_t user_address, size_t user_size){
//         address = user_address;
//         size = user_address;
//     }

//     // destructor ~

//     // assignment operator
    
//     __device__
//     T& operator[](size_t index) {
//         T *add = (int *) address;
//         return add[index];
//     }



//     // tlb entry update
// };

#endif