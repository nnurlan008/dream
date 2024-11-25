#include <infiniband/verbs.h>
#include "rdma_utils.h"

// int cpu_poll_cq(struct ibv_cq *ibcq, int n, struct ibv_wc *wc){

// }

int post(uint64_t wr_rdma_remote_addr, uint32_t wr_rdma_rkey,            
                      uint32_t wr_sg_length, uint32_t wr_sg_lkey, uint64_t wr_sg_addr,
                      int wr_opcode, uint32_t qp_num, int cur_post, void *qp_buf, void *bf_reg, unsigned int *qp_db, void *dev_qp_sq)
{
    
    
    struct mlx5_wq *qp_sq = (struct mlx5_wq *) dev_qp_sq; // &qp->sq;
	void *seg;
	
	unsigned int idx = cur_post & 63;
	uint32_t mlx5_opcode;

    seg = (qp_buf + 256 + (idx * 64)); // mlx5_get_send_wqe(qp, idx);
    struct mlx5_wqe_ctrl_seg *ctrl = (struct mlx5_wqe_ctrl_seg *) seg;
    
    ctrl->opmod_idx_opcode = htonl(((uint16_t) cur_post * 256) | 16);
    
    ctrl->qpn_ds = htonl(3 | (qp_num *256));
    ctrl->signature = 0;
    ctrl->fm_ce_se = 8; // MLX5_WQE_CTRL_CQ_UPDATE;
    ctrl->imm = 0; // 
    

    struct mlx5_wqe_raddr_seg *rdma = (struct mlx5_wqe_raddr_seg *)(seg + 16); // seg + 16; // sizeof(*ctrl);
    rdma->raddr    = htonl64(wr_rdma_remote_addr);
    rdma->rkey     = htonl(wr_rdma_rkey);
    rdma->reserved = 0;

    struct mlx5_wqe_data_seg *data = (struct mlx5_wqe_data_seg *) (seg + 32);
    // *(unsigned long long *) (seg + 32) = ((unsigned long long)htonl(wr_sg_length) | (unsigned long long) htonl(wr_sg_lkey) << 32); 
    // *(unsigned long long *) (seg + 96) = (unsigned long long) htonl64(wr_sg_addr) << 64;
    // *(uint64_t *) (seg + 32) = (uint64_t) (htonl(wr_sg_length) | (uint64_t) htonl(wr_sg_lkey) << 32);
    data->byte_count = htonl(wr_sg_length); // htonl(wr_sg_list->length);
    data->lkey       = htonl(wr_sg_lkey); // htonl(wr_sg_list->lkey);
    data->addr       = htonl64(wr_sg_addr); // htonl64(wr_sg_list->addr);
   
    // cur_post++;
    qp_sq->cur_post += 1;
    qp_sq->head += 1;
    // if(cur_post == 0)
    qp_db[1] = (uint16_t) (cur_post + 1) ; // htonl(cur_post & 0xffff);
    
        *(volatile uint64_t *)bf_reg = *(uint64_t *) ctrl ;// 

	return 0;
}