#include <infiniband/mlx5dv.h>
#include <linux/kernel.h>
#include <valgrind/memcheck.h>
#include <rdma/mlx5-abi.h>

#include <linux/types.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stddef.h>
#include <endian.h>

#include "gpu-utils.h"

int cpu_poll_cq(struct ibv_cq *ibcq, int n, struct ibv_wc *wc) 
{
    struct mlx5_cq *cq = to_mcq(ibcq);
	// ((struct mlx5_cq *)(ibcq - offsetof(struct mlx5_cq, verbs_cq.cq)));
    int npolled=0;
	int err = CQ_OK;
	struct mlx5_resource *rsc = NULL;
	struct mlx5_srq *srq = NULL;
	printf("cq stall enable: %d\n", cq->stall_enable);
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

    for (npolled = 0 ; npolled < n; ++npolled) {
		struct mlx5_cqe64 *cqe64;
		void *cqe;
		int err;
		cqe = cq->active_buf->buf + (cq->cons_index & cq->verbs_cq.cq.cqe) * cq->cqe_sz;
		cqe64 = (cq->cqe_sz == 64) ? cqe : cqe + 64;
		if ((cqe64->op_own >> 4 != MLX5_CQE_INVALID) &&
			!((cqe64->op_own & MLX5_CQE_OWNER_MASK) ^ !!(n & (cq->verbs_cq.cq.cqe + 1)))) {
			++cq->cons_index;
		} else {
			err = CQ_EMPTY;
			break;
		}
		
		is_srq = 0;
		err = 0; 
		//((struct mlx5_context *)(cq->verbs_cq.cq.context - offsetof(struct mlx5_context, ibv_ctx.context)));//
		mctx = to_mctx(cq->verbs_cq.cq.context);
		qpn = be32toh(cqe64->sop_drop_qpn) & 0xffffff;
		printf("Function: %s line number: %d qpn: %d \n\n",__func__, __LINE__, qpn);
			(wc)->wc_flags = 0;
			(wc)->qp_num = qpn;
		opcode = cqe64->op_own >> 4;
		printf("opcode: %d\n\n\n",opcode);
		switch (opcode) {
		case MLX5_CQE_REQ:
		{
			uint32_t rsn = (cqe_ver ? (be32toh(cqe64->srqn_uidx) & 0xffffff) : qpn);
			if (!rsc || (rsn != rsc->rsn)){
				
				if(cqe_ver) {
					int tind = rsn >> MLX5_UIDX_TABLE_SHIFT;
					printf("Function: %s line number: %d rsn & MLX5_UIDX_TABLE_MASK: %d tind: %d\n",__func__, __LINE__, rsn & MLX5_UIDX_TABLE_MASK, tind);
                    printf("mctx->uidx_table[tind].refcnt: %d\n", mctx->uidx_table[tind].refcnt);
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
				(wc+npolled)->opcode    = IBV_WC_RDMA_READ;
				(wc+npolled)->byte_len  = htonl(cqe64->byte_cnt);
				(wc+npolled)->wr_id = wq->wrid[idx];
				printf("Function: %s line number: %d idx: %d \n\n",__func__, __LINE__, idx);
				printf("Function: %s line number: %d wq->wrid[idx]: %d \n\n",__func__, __LINE__, wq->wrid[idx]);
				(wc+npolled)->status = err;
			wq->tail = wq->wqe_head[idx] + 1;
			printf("Function: %s line number: %d wq->wqe_head[idx]: %d \n\n",__func__, __LINE__, wq->wqe_head[idx]);
			break;
		}
		case MLX5_CQE_RESP_SEND:
		{
			srqn_uidx = be32toh(cqe64->srqn_uidx) & 0xffffff;
			printf("Function: %s line number: %d srqn_uidx: %d \n\n",__func__, __LINE__, srqn_uidx);
			struct mlx5_qp *mqp;

			if (!rsc || (srqn_uidx != rsc->rsn)) {
				int tind = srqn_uidx >> MLX5_UIDX_TABLE_SHIFT;
				printf("Function: %s line number: %d rsn & MLX5_UIDX_TABLE_MASK: %d tind: %d\n",__func__, __LINE__, srqn_uidx & MLX5_UIDX_TABLE_MASK, tind);
                printf("mctx->uidx_table[tind].refcnt: %d\n", mctx->uidx_table[tind].refcnt);
				if ((mctx->uidx_table[tind].refcnt))
					rsc = mctx->uidx_table[tind].table[srqn_uidx & MLX5_UIDX_TABLE_MASK];
				if ((!rsc)){
					err = CQ_POLL_ERR;
					break;
				}
			}
			
			mqp = (struct mlx5_qp *) rsc;
			printf("Function: %s line number: %d mqp->verbs_qp.qp.srq: %d\n",__func__, __LINE__, mqp->verbs_qp.qp.srq);
			if (mqp->verbs_qp.qp.srq) {
				printf("Function: %s line number: %d \n",__func__, __LINE__);
				srq = to_msrq(mqp->verbs_qp.qp.srq);
				is_srq = 1;
			}
			err = CQ_OK;
			printf("Function: %s line number: %d \n",__func__, __LINE__);
			
			uint16_t	wqe_ctr;
			struct mlx5_wq *wq;
			struct mlx5_qp *qp = rsc_to_mqp(rsc);
			uint8_t g;
			int err = 0;

			wc->byte_len = be32toh(cqe64->byte_cnt);
			
			if ((rsc->type == MLX5_RSC_TYPE_QP)) {
				printf("Function: %s line number: %d wc->byte_len: %d \n",__func__, __LINE__);
				wq = &qp->rq;
			} 
			
			wqe_ctr = wq->tail & (wq->wqe_cnt - 1);
			wc->wr_id = wq->wrid[wqe_ctr];
			printf("Function: %s line number: %d wq->tail: %d\n",__func__, __LINE__, wq->tail);
			printf("Function: %s line number: %d wqe_ctr: %d\n",__func__, __LINE__, wqe_ctr);
			printf("Function: %s line number: %d wc->wr_id: %d\n",__func__, __LINE__, wc->wr_id);
			++wq->tail;
			if (cqe64->op_own & MLX5_INLINE_SCATTER_32){
				printf("Function: %s line number: %d wqe_ctr: %d\n",__func__, __LINE__, wqe_ctr);
				int size = wc->byte_len;
				struct mlx5_context *ctx = to_mctx(qp->ibv_qp->pd->context);
				struct mlx5_wqe_data_seg *scat;
				int max = 1 << (qp->rq.wqe_shift - 4);
				printf("Function: %s line number: %d max:%d qp->rq.offset: %d wqe_ctr: %d qp->rq.wqe_shift: %d\n",
                __func__, __LINE__, max, qp->rq.offset, wqe_ctr, qp->rq.wqe_shift);
				scat = qp->buf.buf + qp->rq.offset + (wqe_ctr << qp->rq.wqe_shift);
				if ((qp->wq_sig))
					++scat;
				int copy;
				int i, success = 0;
				if ((!size)){
					err = IBV_WC_SUCCESS;
					printf("Function: %s line number: %d \n",__func__, __LINE__);
				}
				else{
					printf("Function: %s line number: %d \n",__func__, __LINE__);
					for (i = 0; i < max; ++i) {
						printf("Function: %s line number: %d \n",__func__, __LINE__);
						copy = size <= be32toh(scat->byte_count) ? size : be32toh(scat->byte_count);
						printf("Function: %s line number: %d be32toh(scat->byte_count):%d\n",__func__, __LINE__, (scat->byte_count));
						printf("Function: %s line number: %d copy:%d\n",__func__, __LINE__, copy);
						// min_t(long, size, be32toh(scat->byte_count));
						/* When NULL MR is used can't copy to target,
						* expected to be NULL.
						*/if (likely(rsc->type == MLX5_RSC_TYPE_QP)) {
							wq = &qp->rq;
							printf("Function: %s line number: %d \n",__func__, __LINE__);
						} 
						
						wqe_ctr = wq->tail & (wq->wqe_cnt - 1);
						printf("Function: %s line number: %d wq->wqe_cnt: %d\n",__func__, __LINE__, wq->wqe_cnt);
						printf("Function: %s line number: %d wq->tail: %d\n",__func__, __LINE__, wq->tail);
						wc->wr_id = wq->wrid[wqe_ctr];
						printf("Function: %s line number: %d wqe_ctr: %d\n",__func__, __LINE__, wqe_ctr);
            			printf("Function: %s line number: %d wc->wr_id: %d\n",__func__, __LINE__, wc->wr_id);
						++wq->tail;
						if (cqe64->op_own & MLX5_INLINE_SCATTER_32){
							int size = wc->byte_len;
							printf("Function: %s line number: %d \n",__func__, __LINE__);
							// struct mlx5_context *ctx = to_mctx(qp->ibv_qp->pd->context);
							struct mlx5_wqe_data_seg *scat;
							int max = 1 << (qp->rq.wqe_shift - 4);
							printf("Function: %s line number: %d max:%d qp->rq.offset: %d wqe_ctr: %d qp->rq.wqe_shift: %d\n",
                			__func__, __LINE__, max, qp->rq.offset, wqe_ctr, qp->rq.wqe_shift);
							scat = qp->buf.buf + qp->rq.offset + (wqe_ctr << qp->rq.wqe_shift);
							if ((qp->wq_sig))
								++scat;

							int copy;
							int i, success = 0;

							if ((!size)){
								err = IBV_WC_SUCCESS;
								printf("Function: %s line number: %d \n",__func__, __LINE__);
							}
							else{
								
								for (i = 0; i < max; ++i) {
									printf("Function: %s line number: %d be32toh(scat->byte_count):%d\n",__func__, __LINE__, be32toh(scat->byte_count));
									copy = size <= be32toh(scat->byte_count) ? size : be32toh(scat->byte_count);
									printf("Function: %s line number: %d copy:%d\n",__func__, __LINE__, copy);
									
									// copy = min_t(long, size, be32toh(scat->byte_count));

									/* When NULL MR is used can't copy to target,
									* expected to be NULL.
									*/
									printf("Function: %s line number: %d ctx->dump_fill_mkey_be:%d\n",__func__, __LINE__, ctx->dump_fill_mkey_be);
                                    printf("Function: %s line number: %d scat->lkey:%d\n",__func__, __LINE__, scat->lkey);
									if ((scat->lkey != ctx->dump_fill_mkey_be)){
										memcpy((void *)(unsigned long)htonl64(scat->addr),
											cqe64, copy);
										printf("Function: %s line number: %d ctx->dump_fill_mkey_be:%d\n",__func__, __LINE__, ctx->dump_fill_mkey_be);
									}
									printf("Function: %s line number: %d copy:%d\n",__func__, __LINE__, copy);
									size -= copy;
									if (size == 0){
										printf("Function: %s line number: %d \n",__func__, __LINE__);
										err = IBV_WC_SUCCESS;
										success = 1;
										break;
									}

									cqe64 += copy;
									++scat;
									printf("Function: %s line number: %d copy:%d\n",__func__, __LINE__, copy);
								}
								if(!success){
									printf("Function: %s line number: %d \n",__func__, __LINE__);
									err = IBV_WC_LOC_LEN_ERR;
									// break;
								}
							}
								
						}
						printf("Function: %s line number: %d copy :%d, size: %d\n",__func__, __LINE__, copy, size);
						if ((scat->lkey != ctx->dump_fill_mkey_be))
							memcpy((void *)(unsigned long)be64toh(scat->addr),
								cqe64, copy);
						printf("Function: %s line number: %d copy :%d, size: %d\n",__func__, __LINE__, copy, size);
						size -= copy;
						if (size == 0){
							printf("Function: %s line number: %d \n",__func__, __LINE__);
							err = IBV_WC_SUCCESS;
							success = 1;
							break;
						}

						cqe64 += copy;
						++scat;
					}
					if(!success)
						err = IBV_WC_LOC_LEN_ERR;
						// break;
				}
				
			}
				
			if (err){
				(wc+npolled)->status = err;
				printf("Function: %s line number: %d (wc+npolled)->status: %d\n",__func__, __LINE__, (wc+npolled)->status);
				break;
			}
			wc->opcode   = IBV_WC_RECV;
			wc->slid	   = htons(cqe64->slid);
			wc->sl		   = (htonl(cqe64->flags_rqpn) >> 24) & 0xf;
			wc->src_qp	   = htonl(cqe64->flags_rqpn) & 0xffffff;
			wc->dlid_path_bits = cqe64->ml_path & 0x7f;
			g = (htonl(cqe64->flags_rqpn) >> 28) & 3;
			wc->wc_flags |= g ? IBV_WC_GRH : 0;
			wc->pkey_index     = htonl(cqe64->imm_inval_pkey) & 0xffff;
			(wc+npolled)->status = IBV_WC_SUCCESS;
			printf("Function: %s line number: %d wc->opcode: %d\n",__func__, __LINE__, wc->opcode);
			printf("Function: %s line number: %d wc->slid: %d\n",__func__, __LINE__, wc->slid);
			printf("Function: %s line number: %d wc->sl: %d\n",__func__, __LINE__, wc->sl);
			printf("Function: %s line number: %d wc->src_qp: %d\n",__func__, __LINE__, wc->src_qp);
			printf("Function: %s line number: %d wc->dlid_path_bits: %d\n",__func__, __LINE__, wc->dlid_path_bits);
			printf("Function: %s line number: %d g: %d\n",__func__, __LINE__, g);
			printf("Function: %s line number: %d wc->pkey_index: %d\n",__func__, __LINE__, wc->pkey_index);
			break;	
		}
		}
		if (err != CQ_OK){
			break;
		}
		
	}
	/* Update cons index */
	cq->dbrec[0] = htonl(cq->cons_index & 0xffffff);
    return err == CQ_POLL_ERR ? err : npolled;
}
