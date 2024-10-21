#ifndef RDMA_UTILS_CUH
#define RDMA_UTILS_CUH

#include <sys/types.h>
#include <infiniband/verbs.h>



// static int N_BUFs = 256;

enum { 
<<<<<<< HEAD
  N_8GB_Region = 8,
=======
  N_8GB_Region = 5,
>>>>>>> origin/cloudlab
  Region_Size = 8*1024*1024*1024llu
 };
  
struct MemPool{
  uint64_t addresses[N_8GB_Region];
  uint32_t rkeys[N_8GB_Region];
  uint32_t lkeys[N_8GB_Region];
};

extern struct MemPool MemoryPool;

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
  unsigned long long int gpu_buf_size; // 3 MB

  struct ibv_mr *pool_mr;

  struct ibv_mr *gpu_mr;
  // struct ibv_mr server_mr;
  struct MemPool server_memory;

  struct rdma_cm_id	*id;

  pthread_t cq_poller_thread;
};

struct context_2gpu {
  struct ibv_context *ctx;
  struct ibv_pd *pd;
  struct ibv_cq *main_cq;
  struct ibv_comp_channel *comp_channel;
  struct ibv_qp *main_qp;

  // struct ibv_qp **gpu1_qp;
  // struct ibv_cq **gpu1_cq;
  // the foolowing double pointers contain qp and cq pointers
  // for both gpus (first 128 for GPU0 and the next 128 for GPU1)
  struct ibv_qp **gpu_qp;
  struct ibv_cq **gpu_cq;
  // void* volatile cqbuf = NULL;
  // void* volatile wqbuf = NULL;
  void** volatile cqbuf;
  int cqbuf_size;
  void** volatile wqbuf;
  int wqbuf_size;
  int n_bufs;

  void *gpu_buffer1;
  unsigned long long int gpu_buf1_size; // 3 MB
  void *gpu_buffer2;
  unsigned long long int gpu_buf2_size; // 3 MB

  struct ibv_mr *pool_mr;

  struct ibv_mr *gpu1_mr;
  struct ibv_mr *gpu2_mr;
  // struct ibv_mr server_mr;
  struct MemPool server_memory;

  struct rdma_cm_id	*id;
};

struct context_2gpu_2nic {
  struct ibv_context *ctx[2];
  struct ibv_pd *pd[2];
  struct ibv_cq *main_cq[2];
  struct ibv_comp_channel *comp_channel[2];
  struct ibv_qp *main_qp[2];

  // struct ibv_qp **gpu1_qp;
  // struct ibv_cq **gpu1_cq;
  // the foolowing double pointers contain qp and cq pointers
  // for both gpus (first 128 for GPU0 and the next 128 for GPU1)
  struct ibv_qp **gpu_qp;
  struct ibv_cq **gpu_cq;
  // void* volatile cqbuf = NULL;
  // void* volatile wqbuf = NULL;
  void** volatile cqbuf;
  int cqbuf_size;
  void** volatile wqbuf;
  int wqbuf_size;
  int n_bufs;

  void *gpu_buffer[2];
  unsigned long long int gpu_buf_size[2]; // 3 MB

  struct ibv_mr *pool_mr[2];
  struct ibv_mr *gpu_mr[2];
  
  // struct ibv_mr server_mr;
  struct MemPool server_memory[2];

  struct rdma_cm_id	*id[2];
};

struct context_2nic {
  struct ibv_context *ctx[2];
  struct ibv_pd *pd[2];
  struct ibv_cq *main_cq[2];
  struct ibv_comp_channel *comp_channel[2];
  struct ibv_qp *main_qp[2];

  // struct ibv_qp **gpu1_qp;
  // struct ibv_cq **gpu1_cq;
  // the foolowing double pointers contain qp and cq pointers
  // for both gpus (first 128 for GPU0 and the next 128 for GPU1)
  struct ibv_qp **gpu_qp;
  struct ibv_cq **gpu_cq;
  // void* volatile cqbuf = NULL;
  // void* volatile wqbuf = NULL;
  void** volatile cqbuf;
  int cqbuf_size;
  void** volatile wqbuf;
  int wqbuf_size;
  int n_bufs;

  void *gpu_buffer;
  unsigned long long int gpu_buf_size; // 3 MB

  struct ibv_mr *pool_mr;
  struct ibv_mr *gpu_mr[2];
  
  // struct ibv_mr server_mr;
  struct MemPool server_memory[2];

  struct rdma_cm_id	*id[2];
};

struct remote_qp_info{
    uint32_t target_qp_num[256];
    uint16_t target_lid;
    union ibv_gid target_gid;
};

struct rdma_content{
    struct ibv_cq *cq;
    struct ibv_qp *qp;
    struct ibv_pd *pd;
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

struct __attribute__((__packed__)) host_keys{
  uint32_t rkeys[N_8GB_Region];
  uint32_t lkeys[N_8GB_Region];
  uint64_t addrs[N_8GB_Region];
  

  host_keys& operator=(const host_keys& obj) {
  
    for(int i = 0; i < N_8GB_Region; i++){
      this->rkeys[i] = obj.rkeys[i];
      this->lkeys[i] = obj.lkeys[i];
      this->addrs[i] = obj.addrs[i];
    }
   
    return *this;
  }
};

struct __attribute__((__packed__)) gpu_memory_info{
  uint32_t wr_rdma_rkey[2];
  uint32_t wr_rdma_lkey[2];
  uint64_t addrs[2];
  uint64_t qp_buf_gpu[2];
  uint64_t cq_buf_gpu[2];
  uint64_t server_address[2];
  uint32_t qp_num_gpu[2];
  

  __forceinline__
  __host__ __device__ 
  gpu_memory_info& operator=(const gpu_memory_info& obj) {
  
    for(int i = 0; i < 2; i++){
      this->wr_rdma_rkey[i] = obj.wr_rdma_rkey[i];
      this->wr_rdma_lkey[i] = obj.wr_rdma_lkey[i];
      this->addrs[i] = obj.addrs[i];
      this->qp_buf_gpu[i] = obj.qp_buf_gpu[i];
      this->cq_buf_gpu[i] = obj.cq_buf_gpu[i];
      this->server_address[i] = obj.server_address[i];
    }
    return *this;
  }
};

struct __attribute__((__packed__)) post_content2{
  uint32_t wr_rdma_rkey[N_8GB_Region];
  uint32_t wr_rdma_lkey[N_8GB_Region];
  uint64_t addrs[N_8GB_Region];

  __forceinline__
  __host__ __device__ 
  post_content2& operator=(const post_content2& obj) {
  
    for(int i = 0; i < N_8GB_Region; i++){
      this->wr_rdma_rkey[i] = obj.wr_rdma_rkey[i];
      this->wr_rdma_lkey[i] = obj.wr_rdma_lkey[i];
      this->addrs[i] = obj.addrs[i];
    }
    return *this;
  }
}; 

struct __attribute__((__packed__)) server_content_2nic{
  
  // struct post_content2 servers[2];

  // __forceinline__
  // __host__ __device__ 
  // server_content_2nic& operator=(const server_content_2nic& obj) {
  
  //   for(int i = 0; i < 2; i++){
  //     this->servers[i] = obj.servers[i];
  //   }
  //   return *this;
  // }

  uint32_t wr_rdma_rkey[N_8GB_Region*2];
  uint32_t wr_rdma_lkey[N_8GB_Region*2];
  uint64_t addrs[N_8GB_Region*2];

  __forceinline__
  __host__ __device__ 
  server_content_2nic& operator=(const server_content_2nic& obj) {
  
    for(int i = 0; i < N_8GB_Region*2; i++){
      this->wr_rdma_rkey[i] = obj.wr_rdma_rkey[i];
      this->wr_rdma_lkey[i] = obj.wr_rdma_lkey[i];
      this->addrs[i] = obj.addrs[i];
    }
    return *this;
  }

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
  unsigned int cq_lock[256*64];
  size_t queue_count[256];
  unsigned int queue_lock[256];
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
    this->wr_sg_length = obj.wr_sg_length;
    this->wr_sg_lkey = obj.wr_sg_lkey;
    this->wr_opcode = obj.wr_opcode;
    this->wr_sg_addr = obj.wr_sg_addr;
    this->qp_num = obj.qp_num;
    this->qp_buf = obj.qp_buf;
    this->wr_rdma_rkey = obj.wr_rdma_rkey;

    // for(int i = 0; i < N_8GB_Region; i++)
    //   this->wr_rdma_rkey[i] = obj.wr_rdma_rkey[i];

    for(int i = 0; i < 256; i++)
    {
      this->n_post[i] = obj.n_post[i];
      this->bf_reg[i] = obj.bf_reg[i];
      this->qp_db[i] = obj.qp_db[i];
      this->dev_qp_sq[i] = obj.dev_qp_sq[i];
      // this->cq_lock[i] = obj.cq_lock[i];
      this->queue_count[i] = obj.queue_count[i];
      this->queue_lock[i] = obj.queue_lock[i];
      for(size_t k = 0; k < 64; k++)
        this->cq_lock[i*64+k] = obj.cq_lock[i*64+k];
            // post_cont->cq_lock[i*64+k] = 0;
    }
            
    return *this;
  }
};

struct __attribute__((__packed__)) batch{
  
  unsigned int wait_queue[256*64];
  size_t queue_lock[256];
  size_t global_post_number[256]; // per queue
  
  __forceinline__
  __host__ __device__ 
  batch& operator=(const batch& obj) {
        
    for(int i = 0; i < 256; i++)
    {
      this->queue_lock[i] = obj.queue_lock[i];
      for(size_t k = 0; k < 64; k++)
        this->wait_queue[i*64+k] = obj.wait_queue[i*64+k];
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

struct cpu_benchmark_content{
  struct ibv_cq *cpu_cq; 
  int num_entries; 
  // struct ibv_wc *wc;
  struct ibv_qp *cpu_qp; 
  // struct ibv_send_wr *wr;
  void *rem_addr;
  // struct ibv_send_wr **bad_wr; 
  void *gpu_addr;
  uint32_t rkey;
  uint32_t lkey;
  int num_packets; 
  int mesg_size; 
  float *bandwidth;
  int thread_num;

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
int local_connect(const char *mlx_name, struct context *s_ctx);
int local_connect_2gpu(const char *mlx_name, struct context_2gpu *s_ctx);
int local_connect_2gpu_2nic(const char *mlx_name, struct context_2gpu_2nic *s_ctx, int gpu);
int local_connect_2nic(const char *mlx_name, struct context_2nic *s_ctx, int nic, int gpu);

int prepare_post_poll_content(struct context *s_ctx, struct post_content *post_cont, struct poll_content *poll_cont, struct post_content2 *post_cont2, \
                              struct post_content *host_post, struct poll_content *host_poll, struct host_keys *host_post2);

int prepare_post_poll_content_2gpu(struct context_2gpu *s_ctx, struct post_content *post_cont, struct poll_content *poll_cont, struct post_content2 *post_cont2, \
                              struct post_content *host_post, struct poll_content *host_poll, struct host_keys *host_post2, struct gpu_memory_info *gpu_infos);

int prepare_post_poll_content_2gpu_2nic(struct context_2gpu_2nic *s_ctx, struct post_content *post_cont, struct poll_content *poll_cont, struct server_content_2nic *post_cont2, \
                              struct post_content *host_post, struct poll_content *host_poll, struct host_keys *host_post2, struct gpu_memory_info *gpu_infos);

int prepare_post_poll_content_2nic(struct context_2nic *s_ctx, struct post_content *post_cont, struct poll_content *poll_cont, struct server_content_2nic *post_cont2, \
                              struct post_content *host_post, struct poll_content *host_poll, struct host_keys *host_post2, struct gpu_memory_info *gpu_infos);

void host_poll_fake(struct ibv_cq *cq1, struct ibv_wc *wc);

__device__ int poll(void *cq_buf, struct ibv_wc *wc, uint32_t *cons_index,
                    int ibv_cqe, uint32_t cqe_sz, int n, void *cq_dbrec);

__device__ int post_s(struct post_wr wr, int cur_post, void *qp_buf, void *bf_reg);

__device__ int post(uint64_t wr_rdma_remote_addr, uint32_t wr_rdma_rkey,
                    uint32_t wr_sg_length, uint32_t wr_sg_lkey, uint64_t wr_sg_addr, 
                    int wr_opcode, uint32_t qp_num, int cur_post, void *qp_buf, void *bf_reg, unsigned int *qp_db);

__device__ int update_db(uint64_t *ctrl, void *bf_reg);

__device__ int update_db_spec(void *bf_reg, void *qp_buf, unsigned int cur_post);

__device__ int update_db_index_opt(unsigned int cur_post, int qp_index);

__device__ int post_m(uint64_t wr_rdma_remote_addr, uint32_t wr_rdma_rkey,
                    uint32_t wr_sg_length, uint32_t wr_sg_lkey, uint64_t wr_sg_addr, 
                    int wr_opcode, uint32_t qp_num, int cur_post, uint64_t *value_ctrl, 
                    void *qp_buf, void *bf_reg, unsigned int *qp_db, void *dev_qp_sq, int id);

__device__ void post_opt(uint64_t wr_rdma_remote_addr, uint32_t wr_rdma_rkey,            
                        uint64_t wr_sg_addr,
                        int cur_post, int qp_index);

__device__ 
void post_opt_write(uint64_t wr_rdma_remote_addr, uint32_t wr_rdma_rkey,            
                        uint64_t wr_sg_addr,
                        int cur_post,
                        int qp_index);

__device__ 
void post_opt_2nic(uint64_t wr_rdma_remote_addr, uint32_t wr_rdma_rkey,            
                        uint64_t wr_sg_addr,
                        int cur_post,
                        int qp_index, int warpId);

__device__ int post_write(uint64_t wr_rdma_remote_addr, uint32_t wr_rdma_rkey,            
                      uint32_t wr_sg_length, uint32_t wr_sg_lkey, uint64_t wr_sg_addr,
                      int wr_opcode, uint32_t qp_num, int cur_post, void *qp_buf, void *bf_reg, unsigned int *qp_db, void *dev_qp_sq, int id);

int benchmark(struct context *s_ctx, int num_msg, int mesg_size, float *bandwidth);
void *cpu_benchmark(void *param1);

int local_connect_cpu_benchmark(const char *mlx_name, struct context *s_ctx, int mesg_size, int u_iter);

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
                          
__global__ void add_vectors_rdma_64MB_64KB(int *a, int *b, int *c, int size, \
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