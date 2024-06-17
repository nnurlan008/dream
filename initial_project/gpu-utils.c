#include <infiniband/mlx5dv.h>
#include <linux/kernel.h>
#include <valgrind/memcheck.h>
#include <rdma/mlx5-abi.h>

#include <linux/types.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stddef.h>
#include <endian.h>

#include <pthread.h>
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
    printf("Function name: %s, line number: %d\n", __func__, __LINE__);
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
    printf("Function name: %s, line number: %d\n", __func__, __LINE__);
    for (npolled = 0 ; npolled < n; ++npolled) {
		struct mlx5_cqe64 *cqe64;
		void *cqe;
		int err;
        printf("Function name: %s, line number: %d\n", __func__, __LINE__);
		cqe = cq->active_buf->buf + (cq->cons_index & cq->verbs_cq.cq.cqe) * cq->cqe_sz;
        printf("Function name: %s, line number: %d\n", __func__, __LINE__);
		cqe64 = (cq->cqe_sz == 64) ? cqe : cqe + 64;
		printf("Function name: %s, line number: %d cqe64->op_own >> 4: %d\n", __func__, __LINE__, cqe64->op_own >> 4);
		if ((cqe64->op_own >> 4 != MLX5_CQE_INVALID) &&
			!((cqe64->op_own & MLX5_CQE_OWNER_MASK) ^ !!(n & (cq->verbs_cq.cq.cqe + 1)))) {
			++cq->cons_index;
		} else {
			err = CQ_EMPTY;
			break;
		}
		printf("Function name: %s, line number: %d\n", __func__, __LINE__);
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
                    printf("mctx->uidx_table[tind=%d].refcnt: %d\n", tind, mctx->uidx_table[tind].refcnt);
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
            if(be32toh(cqe64->sop_drop_qpn) >> 24 == MLX5_OPCODE_RDMA_WRITE){
				(wc+npolled)->opcode    = IBV_WC_RDMA_WRITE;
            }
            else {
                (wc+npolled)->opcode    = IBV_WC_RDMA_READ;
                (wc+npolled)->byte_len  = htonl(cqe64->byte_cnt);
            }
				
				(wc+npolled)->wr_id = wq->wrid[idx];
				printf("Function: %s line number: %d idx: %d \n\n", __func__, __LINE__, idx);
				printf("Function: %s line number: %d wq->wrid[idx]: %d \n\n", __func__, __LINE__, wq->wrid[idx]);
				(wc+npolled)->status = err;
			wq->tail = wq->wqe_head[idx] + 1;
			printf("Function: %s line number: %d wq->wqe_head[idx]: %d \n\n",__func__, __LINE__, wq->wqe_head[idx]);
			printf("idx: %d, wc->wr_id: %d, wc->status: %d, wq->tail: %d\n", idx, wc->wr_id, wc->status, wq->tail);
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
			
			// mqp = (struct mlx5_qp *) rsc;
			// // printf("Function: %s line number: %d mqp->verbs_qp.qp.srq: %d\n",__func__, __LINE__, mqp->verbs_qp.qp.srq);
			// if (mqp->verbs_qp.qp.srq) {
			// 	// printf("Function: %s line number: %d \n",__func__, __LINE__);
			// 	srq = to_msrq(mqp->verbs_qp.qp.srq);
			// 	is_srq = 1;
			// }
			// err = CQ_OK;
			// printf("Function: %s line number: %d \n",__func__, __LINE__);
			
			uint16_t	wqe_ctr;
			struct mlx5_wq *wq;
			struct mlx5_qp *qp = rsc_to_mqp(rsc);
			uint8_t g;
			int err = 0;

			wc->byte_len = be32toh(cqe64->byte_cnt);
			printf("Function: %s line number: %d\n",__func__, __LINE__);
			if ((rsc->type == MLX5_RSC_TYPE_QP)) {
				printf("Function: %s line number: %d wc->byte_len: %d \n",__func__, __LINE__);
				wq = &qp->rq;
			} 
			
			wqe_ctr = wq->tail & (wq->wqe_cnt - 1);
            printf("Function: %s line number: %d wc->byte_len: %d wqe_ctr: %d\n",__func__, __LINE__, wqe_ctr);
			wc->wr_id = wq->wrid[wqe_ctr];
			++wq->tail;
				
			if (err){
				(wc)->status = err;
				break;
			}
			wc->opcode   = IBV_WC_RECV;
			(wc+npolled)->status = IBV_WC_SUCCESS;
			
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


const uint32_t mlx5_ib_opcode[] = {
	[IBV_WR_SEND]			= MLX5_OPCODE_SEND,
	[IBV_WR_SEND_WITH_INV]		= MLX5_OPCODE_SEND_INVAL,
	[IBV_WR_SEND_WITH_IMM]		= MLX5_OPCODE_SEND_IMM,
	[IBV_WR_RDMA_WRITE]		= MLX5_OPCODE_RDMA_WRITE,
	[IBV_WR_RDMA_WRITE_WITH_IMM]	= MLX5_OPCODE_RDMA_WRITE_IMM,
	[IBV_WR_RDMA_READ]		= MLX5_OPCODE_RDMA_READ,
	[IBV_WR_ATOMIC_CMP_AND_SWP]	= MLX5_OPCODE_ATOMIC_CS,
	[IBV_WR_ATOMIC_FETCH_AND_ADD]	= MLX5_OPCODE_ATOMIC_FA,
	[IBV_WR_BIND_MW]		= MLX5_OPCODE_UMR,
	[IBV_WR_LOCAL_INV]		= MLX5_OPCODE_UMR,
	[IBV_WR_TSO]			= MLX5_OPCODE_TSO,
	[IBV_WR_DRIVER1]		= MLX5_OPCODE_UMR,
};

int mlx5_post_recv(struct ibv_qp *ibqp, struct ibv_recv_wr *wr,
		   struct ibv_recv_wr **bad_wr)
{
	struct mlx5_qp *qp = to_mqp(ibqp);
	struct mlx5_wqe_data_seg *scat;
	int err = 0;
	int nreq;
	int ind;
	int i, j;
	struct mlx5_rwqe_sig *sig;
	int head = qp->rq.head;
	

	ind = head & (qp->rq.wqe_cnt - 1);

	for (nreq = 0; wr; ++nreq, wr = wr->next) {
		printf("nreq: %d\n\n\n\n\n\n\n", nreq);
		scat = qp->buf.buf + qp->rq.offset + (ind << qp->rq.wqe_shift); 
        printf("qp->rq.offset: %d, ind: %d, qp->rq.wqe_shift: %d\n", qp->rq.offset, ind, qp->rq.wqe_shift);
		sig = (struct mlx5_rwqe_sig *)scat;

		for (i = 0, j = 0; i < wr->num_sge; ++i) {
			if ((!wr->sg_list[i].length))
				continue;
			struct mlx5_wqe_data_seg *dseg = scat ;
			struct ibv_sge *sg = wr->sg_list;
			int offset = 0;
			dseg->byte_count = htonl(sg->length - offset);
			dseg->lkey       = htonl(sg->lkey);
			dseg->addr       = htonl64(sg->addr + offset);
		}

		qp->rq.wrid[ind] = wr->wr_id;

		ind = (ind + 1) & (qp->rq.wqe_cnt - 1);
	}
	 printf("after for - nreq: %d\n\n\n\n\n\n\n", nreq);
out:
	if ((nreq)) {
		head += nreq;

        printf("out\n ");
		if ((!((ibqp->qp_type == 8 ||
			      qp->flags & 1) &&
			     ibqp->state < 2))){
			qp->db[0] = htonl(head & 0xffff);
            printf("Function: %s line number: %d\n",__func__, __LINE__);
        }
	}

	return err;
}

int HEYmlx5_post_send(struct ibv_qp *ibqp, struct ibv_send_wr *wr,
				  struct ibv_send_wr **bad_wr)
{
	struct mlx5_qp *qp =  to_mqp(ibqp);///(struct mlx5_qp *) (ibqp - offsetof(struct mlx5_qp, verbs_qp));
	 
	void *seg;
	struct mlx5_wqe_eth_seg *eseg;
	struct mlx5_wqe_ctrl_seg *ctrl = NULL;
	struct mlx5_wqe_data_seg *dpseg;
	struct mlx5_sg_copy_ptr sg_copy_ptr = {.index = 0, .offset = 0};
	int nreq;
	int inl = 0;
	int err = 0;
	int size = 0;
	int i;
	unsigned idx;
	uint8_t opmod = 0;
	struct mlx5_bf *bf = qp->bf;
	void *qend = qp->sq.qend;
	uint32_t mlx5_opcode;
	struct mlx5_wqe_xrc_seg *xrc;
	uint8_t fence;
	uint8_t next_fence;
	uint32_t max_tso = 0;

	next_fence = qp->fm_cache;
	for (nreq = 0; wr; ++nreq, wr = wr->next) {
		if (wr->send_flags & IBV_SEND_FENCE)
			fence = MLX5_WQE_CTRL_FENCE;
		else
			fence = next_fence;
		next_fence = 0;
		idx = qp->sq.cur_post & (qp->sq.wqe_cnt - 1);
		ctrl = seg = qp->sq_start + (idx << MLX5_SEND_WQE_SHIFT);
		// mlx5_get_send_wqe(qp, idx);
		*(uint32_t *)(seg + 8) = 0;
		
		if((wr->opcode == IBV_WR_SEND_WITH_IMM) | (wr->opcode == IBV_WR_RDMA_WRITE_WITH_IMM))
			ctrl->imm = wr->imm_data;
		else if (wr->opcode == IBV_WR_SEND_WITH_INV)
			ctrl->imm = htonl(wr->invalidate_rkey);
		else ctrl->imm = 0;
		// ctrl->imm = send_ieth(wr);
		ctrl->fm_ce_se = qp->sq_signal_bits | fence |
			(wr->send_flags & IBV_SEND_SIGNALED ?
			 MLX5_WQE_CTRL_CQ_UPDATE : 0) |
			(wr->send_flags & IBV_SEND_SOLICITED ?
			 MLX5_WQE_CTRL_SOLICITED : 0);

		seg += sizeof (struct mlx5_wqe_ctrl_seg);// *ctrl;
		size = sizeof (struct mlx5_wqe_ctrl_seg) / 16;
		qp->sq.wr_data[idx] = 0;
		// printf("ibqp->qp_type: %d\n\n\n", ibqp->qp_type);
		 
		dpseg = seg;
		for (i = sg_copy_ptr.index; i < wr->num_sge; ++i) {
			if (unlikely(dpseg == qend)) {
				seg = qp->sq_start + (0 << MLX5_SEND_WQE_SHIFT);
				dpseg = seg;
			}
			if ((wr->sg_list[i].length)) {
				if ((wr->opcode == IBV_WR_ATOMIC_FETCH_AND_ADD)){
					dpseg->byte_count = htonl(MLX5_ATOMIC_SIZE);
					dpseg->lkey       = htonl((wr->sg_list + i)->lkey);
					dpseg->addr       = htonl64((wr->sg_list + i)->addr);
					}
				else {
					if ((wr->opcode == IBV_WR_TSO)) {
						if (max_tso < wr->sg_list[i].length) {
							err = EINVAL;
							*bad_wr = wr;
							goto out;
						}
						max_tso -= wr->sg_list[i].length;
					}
					dpseg->byte_count = htonl((wr->sg_list + i)->length - sg_copy_ptr.offset);
					dpseg->lkey       = htonl((wr->sg_list + i)->lkey);
					dpseg->addr       = htonl64((wr->sg_list + i)->addr + sg_copy_ptr.offset);
				}
				sg_copy_ptr.offset = 0;
				++dpseg;
				size += sizeof(struct mlx5_wqe_data_seg) / 16;
			}
		}
		
		
		mlx5_opcode = mlx5_ib_opcode[wr->opcode];
		ctrl->opmod_idx_opcode = htonl(((qp->sq.cur_post & 0xffff) << 8) |
					       mlx5_opcode			 |
					       (opmod << 24));
		ctrl->qpn_ds = htonl(size | (ibqp->qp_num << 8));
		
		if (unlikely(qp->wq_sig)){
            uint8_t *p = (uint8_t *) ctrl;
	        uint8_t res = 0;
            int size = (htonl(ctrl->qpn_ds) & 0x3f) << 4;
            for (int i = 0; i < size; ++i)
		        res ^= p[i];
		    ctrl->signature = ~res;
		
        }
		
		qp->sq.wrid[idx] = wr->wr_id;
		qp->sq.wqe_head[idx] = qp->sq.head + nreq;
		qp->sq.cur_post += (unsigned long) (size * 16 + MLX5_SEND_WQE_BB - 1)/MLX5_SEND_WQE_BB;
	}

out:
	qp->fm_cache = next_fence;

	struct mlx5_context *ctx;
	if (unlikely(!nreq))
		return;

	qp->sq.head += nreq;
	qp->db[MLX5_SND_DBR] = htonl(qp->sq.cur_post & 0xffff);
	
	ctx = container_of(qp->ibv_qp->context, struct mlx5_context, ibv_ctx.context);
    // to_mctx(qp->ibv_qp->context);

	// printf("Function: %s line number: %d\n",__func__, __LINE__);
	if (!ctx->shut_up_bf && nreq == 1 && bf->uuarn &&
	    (inl || ctx->prefer_bf) && size > 1 &&
	    size <= bf->buf_size / 16){
		// printf("Function: %s line number: %d\n",__func__, __LINE__);
		
		uint64_t *dst = bf->reg + bf->offset;
		const uint64_t *src = ctrl;
		unsigned bytecnt = (size * 16 + 64 - 1) & ~(64 - 1);
		// align(size * 16, 64);
		do {
		
		size_t bytecnt_temp = 64;
		uintptr_t *dst_p = bf->reg + bf->offset;

		if (sizeof(*dst_p) == 8) {
			const __be64 *src_p = ctrl;
			void *addr;
			__be64 val;
			do {
				/* Do 64 bytes at a time */
				addr = dst_p++;
				val = *src_p++;
				__be32 first_dword = htonl(htonl64(val) >> 32);
				__be32 second_dword = htonl(htonl64(val));
				
				*(volatile uint32_t *)addr = (uint32_t)first_dword;
				*(volatile uint32_t *)(addr+4) = (uint32_t)second_dword;
				bytecnt_temp -= sizeof(*dst_p);
			} while (bytecnt_temp > 0);
		} else if (sizeof(*dst_p) == 4) {
			const __be32 *src_p = src;
			do {
				*(volatile uint32_t *)dst_p++ = ( uint32_t)*src_p++;
				*(volatile uint32_t *)dst_p++ = ( uint32_t)*src_p++;
				bytecnt_temp -= 2 * sizeof(*dst_p);
			} while (bytecnt_temp > 0);
		}

		bytecnt -= 64;
		dst += 8;
		src += 8;
		if (unlikely(src == qp->sq.qend))
			src = qp->sq_start;
		} while (bytecnt > 0);
	}
	bf->offset ^= bf->buf_size;

	return err;
}

static inline int mlx5_post_send_underlay(struct mlx5_qp *qp, struct ibv_send_wr *wr,
					  void **pseg, int *total_size,
					  struct mlx5_sg_copy_ptr *sg_copy_ptr)
{
	struct mlx5_wqe_eth_seg *eseg;
	int inl_hdr_copy_size;
	void *seg = *pseg;
	int size = 0;

	if (unlikely(wr->opcode == IBV_WR_SEND_WITH_IMM))
		return EINVAL;

	memset(seg, 0, sizeof(struct mlx5_wqe_eth_pad));
	size += sizeof(struct mlx5_wqe_eth_pad);
	seg += sizeof(struct mlx5_wqe_eth_pad);
	eseg = seg;
	*((uint64_t *)eseg) = 0;
	eseg->rsvd2 = 0;

	if (wr->send_flags & IBV_SEND_IP_CSUM) {
		if (!(qp->qp_cap_cache & MLX5_CSUM_SUPPORT_UNDERLAY_UD))
			return EINVAL;

		eseg->cs_flags |= MLX5_ETH_WQE_L3_CSUM | MLX5_ETH_WQE_L4_CSUM;
	}

	if (likely(wr->sg_list[0].length >= MLX5_SOURCE_QPN_INLINE_MAX_HEADER_SIZE))
		/* Copying the minimum required data unless inline mode is set */
		inl_hdr_copy_size = (wr->send_flags & IBV_SEND_INLINE) ?
				MLX5_SOURCE_QPN_INLINE_MAX_HEADER_SIZE :
				MLX5_IPOIB_INLINE_MIN_HEADER_SIZE;
	else {
		inl_hdr_copy_size = MLX5_IPOIB_INLINE_MIN_HEADER_SIZE;
		/* We expect at least 4 bytes as part of first entry to hold the IPoIB header */
		if (unlikely(wr->sg_list[0].length < inl_hdr_copy_size))
			return EINVAL;
	}

	memcpy(eseg->inline_hdr_start, (void *)(uintptr_t)wr->sg_list[0].addr,
	       inl_hdr_copy_size);
	eseg->inline_hdr_sz = htobe16(inl_hdr_copy_size);
	size += sizeof(struct mlx5_wqe_eth_seg);
	seg += sizeof(struct mlx5_wqe_eth_seg);

	/* If we copied all the sge into the inline-headers, then we need to
	 * start copying from the next sge into the data-segment.
	 */
	if (unlikely(wr->sg_list[0].length == inl_hdr_copy_size))
		sg_copy_ptr->index++;
	else
		sg_copy_ptr->offset = inl_hdr_copy_size;

	*pseg = seg;
	*total_size += (size / 16);
	return 0;
}

static inline uint8_t calc_sig(void *wqe, int size)
{
	int i;
	uint8_t *p = wqe;
	uint8_t res = 0;

	for (i = 0; i < size; ++i)
		res ^= p[i];

	return ~res;
}

static int mlx5_wq_overflow(struct mlx5_wq *wq, int nreq, struct mlx5_cq *cq)
{
	unsigned cur;

	cur = wq->head - wq->tail;
	if (cur + nreq < wq->max_post)
		return 0;

	// mlx5_spin_lock(&cq->lock);
	cur = wq->head - wq->tail;
	// mlx5_spin_unlock(&cq->lock);

	return cur + nreq >= wq->max_post;
}

void *mlx5_get_send_wqe(struct mlx5_qp *qp, int n)
{
	return qp->sq_start + (n << MLX5_SEND_WQE_SHIFT);
}

static __be32 send_ieth(struct ibv_send_wr *wr)
{
	switch (wr->opcode) {
	case IBV_WR_SEND_WITH_IMM:
	case IBV_WR_RDMA_WRITE_WITH_IMM:
		return wr->imm_data;
	case IBV_WR_SEND_WITH_INV:
		return htobe32(wr->invalidate_rkey);
	default:
		return 0;
	}
}

static inline void set_raddr_seg(struct mlx5_wqe_raddr_seg *rseg,
				 uint64_t remote_addr, uint32_t rkey)
{
	rseg->raddr    = htobe64(remote_addr);
	rseg->rkey     = htobe32(rkey);
	rseg->reserved = 0;
}

static inline struct mlx5_ah *to_mah(struct ibv_ah *ibah)
{
	return to_mxxx(ah, ah);
}

static inline void _set_datagram_seg(struct mlx5_wqe_datagram_seg *dseg,
				     struct mlx5_wqe_av *av,
				     uint32_t remote_qpn,
				     uint32_t remote_qkey)
{
	memcpy(&dseg->av, av, sizeof(dseg->av));
	dseg->av.dqp_dct = htobe32(remote_qpn | MLX5_EXTENDED_UD_AV);
	dseg->av.key.qkey.qkey = htobe32(remote_qkey);
}

static void set_datagram_seg(struct mlx5_wqe_datagram_seg *dseg,
			     struct ibv_send_wr *wr)
{
	_set_datagram_seg(dseg, &to_mah(wr->wr.ud.ah)->av, wr->wr.ud.remote_qpn,
			  wr->wr.ud.remote_qkey);
}

static inline void _set_atomic_seg(struct mlx5_wqe_atomic_seg *aseg,
				   enum ibv_wr_opcode opcode,
				   uint64_t swap,
				   uint64_t compare_add)
{
	if (opcode == IBV_WR_ATOMIC_CMP_AND_SWP) {
		aseg->swap_add = htobe64(swap);
		aseg->compare = htobe64(compare_add);
	} else {
		aseg->swap_add = htobe64(compare_add);
	}
}

static void set_atomic_seg(struct mlx5_wqe_atomic_seg *aseg,
			   enum ibv_wr_opcode opcode,
			   uint64_t swap,
			   uint64_t compare_add)
{
	_set_atomic_seg(aseg, opcode, swap, compare_add);
}

static inline unsigned long align(unsigned long val, unsigned long align)
{
	return (val + align - 1) & ~(align - 1);
}

static int set_data_inl_seg(struct mlx5_qp *qp, struct ibv_send_wr *wr,
			    void *wqe, int *sz,
			    struct mlx5_sg_copy_ptr *sg_copy_ptr)
{
	struct mlx5_wqe_inline_seg *seg;
	void *addr;
	int len;
	int i;
	int inl = 0;
	void *qend = qp->sq.qend;
	int copy;
	int offset = sg_copy_ptr->offset;

	seg = wqe;
	wqe += sizeof *seg;
	for (i = sg_copy_ptr->index; i < wr->num_sge; ++i) {
		addr = (void *) (unsigned long)(wr->sg_list[i].addr + offset);
		len  = wr->sg_list[i].length - offset;
		inl += len;
		offset = 0;

		if (unlikely(inl > qp->max_inline_data))
			return ENOMEM;

		if (unlikely(wqe + len > qend)) {
			copy = qend - wqe;
			memcpy(wqe, addr, copy);
			addr += copy;
			len -= copy;
			wqe = mlx5_get_send_wqe(qp, 0);
		}
		memcpy(wqe, addr, len);
		wqe += len;
	}

	if (likely(inl)) {
		seg->byte_count = htobe32(inl | MLX5_INLINE_SEG);
		*sz = align(inl + sizeof seg->byte_count, 16) / 16;
	} else
		*sz = 0;

	return 0;
}

static void set_data_ptr_seg(struct mlx5_wqe_data_seg *dseg, struct ibv_sge *sg,
			     int offset)
{
	dseg->byte_count = htobe32(sg->length - offset);
	dseg->lkey       = htobe32(sg->lkey);
	dseg->addr       = htobe64(sg->addr + offset);
}

static void set_data_ptr_seg_atomic(struct mlx5_wqe_data_seg *dseg,
				    struct ibv_sge *sg)
{
	dseg->byte_count = htobe32(MLX5_ATOMIC_SIZE);
	dseg->lkey       = htobe32(sg->lkey);
	dseg->addr       = htobe64(sg->addr);
}

static void set_data_ptr_seg_end(struct mlx5_wqe_data_seg *dseg)
{
	dseg->byte_count = 0;
	dseg->lkey       = htobe32(MLX5_INVALID_LKEY);
	dseg->addr       = 0;
}

static uint8_t wq_sig(struct mlx5_wqe_ctrl_seg *ctrl)
{
	return calc_sig(ctrl, (be32toh(ctrl->qpn_ds) & 0x3f) << 4);
}

// static inline void post_send_db(struct mlx5_qp *qp, struct mlx5_bf *bf,
// 				int nreq, int inl, int size, void *ctrl)
// {
// 	struct mlx5_context *ctx;

// 	if ((!nreq))
// 		return;

// 	qp->sq.head += nreq;

// 	/*
// 	 * Make sure that descriptors are written before
// 	 * updating doorbell record and ringing the doorbell
// 	 */
// 	udma_to_device_barrier();
// 	qp->db[MLX5_SND_DBR] = htobe32(qp->sq.cur_post & 0xffff);

// 	/* Make sure that the doorbell write happens before the memcpy
// 	 * to WC memory below
// 	 */
// 	ctx = to_mctx(qp->ibv_qp->context);
// 	if (bf->need_lock)
// 		mmio_wc_spinlock(&bf->lock.lock);
// 	else
// 		mmio_wc_start();

// 	if (!ctx->shut_up_bf && nreq == 1 && bf->uuarn &&
// 	    (inl || ctx->prefer_bf) && size > 1 &&
// 	    size <= bf->buf_size / 16)
// 		mlx5_bf_copy(bf->reg + bf->offset, ctrl,
// 			     align(size * 16, 64), qp);
// 	else
// 		mmio_write64_be(bf->reg + bf->offset, *(__be64 *)ctrl);

// 	/*
// 	 * use mmio_flush_writes() to ensure write combining buffers are
// 	 * flushed out of the running CPU. This must be carried inside
// 	 * the spinlock. Otherwise, there is a potential race. In the
// 	 * race, CPU A writes doorbell 1, which is waiting in the WC
// 	 * buffer. CPU B writes doorbell 2, and it's write is flushed
// 	 * earlier. Since the mmio_flush_writes is CPU local, this will
// 	 * result in the HCA seeing doorbell 2, followed by doorbell 1.
// 	 * Flush before toggling bf_offset to be latency oriented.
// 	 */
// 	mmio_flush_writes();
// 	bf->offset ^= bf->buf_size;
// 	if (bf->need_lock)
// 		mlx5_spin_unlock(&bf->lock);
// }

static inline unsigned long DIV_ROUND_UP(unsigned long n, unsigned long d)
{
	return (n + d - 1) / d;
}

int cpu_post_send(struct ibv_qp *ibqp, struct ibv_send_wr *wr,
				  struct ibv_send_wr **bad_wr)
{
	struct mlx5_qp *qp = to_mqp(ibqp);
	void *seg;
	struct mlx5_wqe_eth_seg *eseg;
	struct mlx5_wqe_ctrl_seg *ctrl = NULL;
	struct mlx5_wqe_data_seg *dpseg;
	struct mlx5_sg_copy_ptr sg_copy_ptr = {.index = 0, .offset = 0};
	int nreq;
	int inl = 0;
	int err = 0;
	int size = 0;
	int i;
	unsigned idx;
	uint8_t opmod = 0;
	struct mlx5_bf *bf = qp->bf;
	void *qend = qp->sq.qend;
	uint32_t mlx5_opcode;
	struct mlx5_wqe_xrc_seg *xrc;
	uint8_t fence;
	uint8_t next_fence;

	next_fence = qp->fm_cache;

	for (nreq = 0; nreq < 1; ++nreq) {
        printf("Function: %s line number: %d nreq: %d\n",__func__, __LINE__, nreq);
		fence = next_fence;
		next_fence = 0;
		idx = qp->sq.cur_post & (qp->sq.wqe_cnt - 1);
		ctrl = seg = qp->sq_start + (idx << MLX5_SEND_WQE_SHIFT); // mlx5_get_send_wqe(qp, idx);
		*(uint32_t *)(seg + 8) = 0;
		ctrl->imm = 0; // send_ieth(wr);
		ctrl->fm_ce_se = qp->sq_signal_bits | fence |
			(wr->send_flags & IBV_SEND_SIGNALED ?
			 MLX5_WQE_CTRL_CQ_UPDATE : 0) |
			(wr->send_flags & IBV_SEND_SOLICITED ?
			 MLX5_WQE_CTRL_SOLICITED : 0);

		seg += sizeof *ctrl;
		size = sizeof *ctrl / 16;
		qp->sq.wr_data[idx] = 0;
        printf("ibqp->qp_type: %d \n", ibqp->qp_type);
        if(ibqp->qp_type == IBV_QPT_RC){
            if(wr->opcode == IBV_WR_RDMA_READ || wr->opcode == IBV_WR_RDMA_WRITE){
                ((struct mlx5_wqe_raddr_seg *) seg)->raddr    = htobe64(wr->wr.rdma.remote_addr);
                ((struct mlx5_wqe_raddr_seg *) seg)->rkey     = htobe32(wr->wr.rdma.rkey);
                ((struct mlx5_wqe_raddr_seg *) seg)->reserved = 0;
				seg  += sizeof(struct mlx5_wqe_raddr_seg);
				size += sizeof(struct mlx5_wqe_raddr_seg) / 16;
            }
        }

        printf("Function: %s line number: %d\n",__func__, __LINE__);
        dpseg = seg;
        for (i = sg_copy_ptr.index; i < wr->num_sge; ++i) {
            
            if ((wr->sg_list[i].length)) {
                
                printf("Function: %s line number: %d\n",__func__, __LINE__);
                
                // set_data_ptr_seg(dpseg, wr->sg_list + i, sg_copy_ptr.offset);

                dpseg->byte_count = htonl((wr->sg_list + i)->length - sg_copy_ptr.offset);
                dpseg->lkey       = htonl((wr->sg_list + i)->lkey);
                dpseg->addr       = htonl64((wr->sg_list + i)->addr + sg_copy_ptr.offset);
                size += sizeof(struct mlx5_wqe_data_seg) / 16;
            }
        }
		
        printf("Function: %s line number: %d\n",__func__, __LINE__);
		mlx5_opcode = mlx5_ib_opcode[wr->opcode];
        printf("mlx5_opcode: %d, wr->opcode: %d\n\n\n",mlx5_opcode, wr->opcode);
		ctrl->opmod_idx_opcode = htonl(((qp->sq.cur_post & 0xffff) << 8) | mlx5_opcode | (opmod << 24));
		ctrl->qpn_ds = htonl(size | (ibqp->qp_num << 8));

		qp->sq.wrid[idx] = wr->wr_id;
		qp->sq.wqe_head[idx] = qp->sq.head + nreq;
		qp->sq.cur_post += (size * 16 + 64 - 1) / 64;
	}

out:
	qp->fm_cache = next_fence;
	// post_send_db(qp, bf, nreq, inl, size, ctrl);
    printf("Function: %s line number: %d\n",__func__, __LINE__);
    struct mlx5_context *ctx;
	if ((!nreq))
		return;

	qp->sq.head += nreq;
	qp->db[MLX5_SND_DBR] = htonl(qp->sq.cur_post & 0xffff);
	
	ctx = container_of(qp->ibv_qp->context, struct mlx5_context, ibv_ctx.context);

	printf("Function: %s line number: %d\n",__func__, __LINE__);
	if (!ctx->shut_up_bf && nreq == 1 && bf->uuarn &&
	    (inl || ctx->prefer_bf) && size > 1 &&
	    size <= bf->buf_size / 16){
		printf("Function: %s line number: %d\n",__func__, __LINE__);
	
		uintptr_t *dst_p = bf->reg + bf->offset;

		if (sizeof(*dst_p) == 8) {
			const __be64 *src_p = ctrl;
			void *addr;
			__be64 val;
            printf("Func name: %s, line number: %d\n", __func__, __LINE__);
            /* Do 64 bytes at a time */
            addr = bf->reg + bf->offset;
            val = *src_p;
            __be32 first_dword = htonl(htonl64(val) >> 32);
            __be32 second_dword = htonl(htonl64(val));
            *(volatile uint32_t *)addr = (uint32_t)first_dword;
            *(volatile uint32_t *)(addr+4) = (uint32_t)second_dword;
		} 
        // printf("Func name: %s, line number: %d bytecnt: %d\n", __func__, __LINE__, bytecnt);
	
		// if ((src == qp->sq.qend))
		// 	src = qp->sq_start;
	}
	bf->offset ^= bf->buf_size;


	return err;
}


int mlx5_post_send(struct ibv_qp *ibqp, struct ibv_send_wr *wr,
				  struct ibv_send_wr **bad_wr)
{

	struct mlx5_qp *qp = to_mqp(ibqp);
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
	struct mlx5_bf *bf = qp->bf;
	void *qend = qp->sq.qend;
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
    // unsigned int qpsq_cur_post = qp->sq.cur_post;

    // pointers
    // void* qp_buf = qp->buf.buf;
    // uint32_t *qpsq_wr_data = (uint32_t *) dev_qpsq_wr_data;//qp->sq.wr_data;
    // unsigned int *qpsq_wqe_head = (unsigned int *) dev_qpsq_wqe_head;// qp->sq.wqe_head;
    // struct mlx5_wq *qp_sq = (struct mlx5_wq *) dev_qp_sq; // &qp->sq;
    // void *bf_reg = bf->reg;
    

    idx = qp->sq.cur_post & (qp->sq.wqe_cnt - 1);
    printf("Function: %s line number: %d qp->buf.buf: 0x%llx \n",__func__, __LINE__, qp->buf.buf);
    printf("Function: %s line number: %d qp->bf->buf_size: %d \n",__func__, __LINE__, qp->bf->buf_size);
    printf("Function: %s line number: %d bf->reg: 0x%llx \n",__func__, __LINE__, bf->reg);
    printf("Function: %s line number: %d qp->buf.buf: 0x%llx \n",__func__, __LINE__, qp->sq_start);
    seg = (qp->buf.buf + qp->bf->buf_size + (idx * 64)); // mlx5_get_send_wqe(qp, idx);
    ctrl = (struct mlx5_wqe_ctrl_seg *) seg;
    printf("Function: %s line number: %d qp->buf.buf: 0x%llx \n",__func__, __LINE__, ctrl);
    *(uint32_t *)(seg + 8) = 0;
    ctrl->imm = 0; // send_ieth(wr);
    ctrl->fm_ce_se = MLX5_WQE_CTRL_CQ_UPDATE;

    seg = seg + sizeof(*ctrl);
    size = sizeof(*ctrl) / 16;
    qp->sq.wr_data[idx] = 0;
    printf("ibqp->qp_type: %d \n", ibqp->qp_type);

    // we generally have qp_type: IBV_QPT_RC and RDMA READ or WRITE request
    if(wr->opcode == IBV_WR_RDMA_READ || wr->opcode == IBV_WR_RDMA_WRITE){
        ((struct mlx5_wqe_raddr_seg *) seg)->raddr    = htonl64(wr->wr.rdma.remote_addr);
        ((struct mlx5_wqe_raddr_seg *) seg)->rkey     = htonl(wr->wr.rdma.rkey);
        ((struct mlx5_wqe_raddr_seg *) seg)->reserved = 0;
        seg = seg + sizeof(struct mlx5_wqe_raddr_seg);
        size += sizeof(struct mlx5_wqe_raddr_seg) / 16;
    }

    dpseg = (struct mlx5_wqe_data_seg *) seg;
    // for simplicity, we'll have one sge per request/post
    i = 0;// sg_copy_ptr.index;
    dpseg->byte_count = htonl(wr->sg_list->length);
    dpseg->lkey       = htonl(wr->sg_list->lkey);
    dpseg->addr       = htonl64(wr->sg_list->addr);
    size += sizeof(struct mlx5_wqe_data_seg) / 16;
    // 4 -> 16, 0 -> 8
    mlx5_opcode = wr->opcode*2 + 8 - 2*(wr->opcode == 2); // mlx5_ib_opcode[wr->opcode];
    // printf("mlx5_opcode: %d, wr->opcode: %d wr->opcode*2 + 8 - 2*(mlx5_opcode == 2): %d\n\n\n",mlx5_opcode, wr->opcode, wr->opcode*2 + 8 - 2*(wr->opcode == 2));
    ctrl->opmod_idx_opcode = htonl(((qp->sq.cur_post & 0xffff) << 8) | mlx5_opcode);
    ctrl->qpn_ds = htonl(size | (ibqp->qp_num << 8));

    qp->sq.wrid[idx] = wr->wr_id;
    qp->sq.wqe_head[idx] = qp->sq.head;
    qp->sq.cur_post += (size * 16 + 64 - 1) / 64;
        
    // db update:
    qp->sq.head += 1;
    qp->db[MLX5_SND_DBR] = htonl(qp->sq.cur_post & 0xffff);

    void *addr;
    uint64_t val;
    /* Do 64 bytes at a time */
    addr = bf->reg + bf->offset;
    val = *(uint64_t *)ctrl;
    uint32_t first_dword = htonl(htonl64(val) >> 32);
    uint32_t second_dword = htonl(htonl64(val));
    *(volatile uint32_t *)addr = (uint32_t)first_dword;
    *(volatile uint32_t *)(addr+4) = (uint32_t)second_dword;
    bf->offset ^= bf->buf_size;

	return err;
}

static inline int mlx5_spin_lock(struct mlx5_spinlock *lock)
{
	if (lock->need_lock)
		return pthread_spin_lock(&lock->lock);

	if (unlikely(lock->in_use)) {
		fprintf(stderr, "*** ERROR: multithreading violation ***\n"
			"You are running a multithreaded application but\n"
			"you set MLX5_SINGLE_THREADED=1. Please unset it.\n");
		abort();
	} else {
		lock->in_use = 1;
		/*
		 * This fence is not at all correct, but it increases the
		 * chance that in_use is detected by another thread without
		 * much runtime cost. */
		atomic_thread_fence(memory_order_acq_rel);
	}

	return 0;
}

static inline int mlx5_spin_unlock(struct mlx5_spinlock *lock)
{
	if (lock->need_lock)
		return pthread_spin_unlock(&lock->lock);

	lock->in_use = 0;

	return 0;
}

void mmio_wc_spinlock(pthread_spinlock_t *lock)
{
	pthread_spin_lock(lock);
#if !defined(__i386__) && !defined(__x86_64__)
	/* For x86 the serialization within the spin lock is enough to
	 * strongly order WC and other memory types. */
	mmio_wc_start();
#endif
}

void mlx5_bf_copy(uint64_t *dst, const uint64_t *src, unsigned bytecnt,
			 struct mlx5_qp *qp)
{
	do {
		// mmio_memcpy_x64(dst, src, 64);


		uintptr_t *dst_p = dst;

		/* Caller must guarantee:
			assert(bytecnt != 0);
			assert((bytecnt % 64) == 0);
			assert(((uintptr_t)dest) % __alignof__(*dst) == 0);
			assert(((uintptr_t)src) % __alignof__(*dst) == 0);
		*/

		/* Use the native word size for the copy */
		// if (sizeof(*dst_p) == 8) {
		// 	const __be64 *src_p = src;
		// 	void *addr;
		// 	__be64 val;

		// 	do {
		// 		/* Do 64 bytes at a time */
		// 		addr = dst_p++;
		// 		val = *src_p++;
		// 		__be32 first_dword = htonl(htonl64(val) >> 32);
		// 		__be32 second_dword = htonl(htonl64(val));

		// 		*(volatile uint32_t *)addr = (uint32_t)first_dword;
		// 		*(volatile uint32_t *)(addr+4) = (uint32_t)second_dword;

		// 		bytecnt -= sizeof(*dst_p);
		// 	} while (bytecnt > 0);
		// } 


		bytecnt -= 64;
		dst += 8;
		src += 8;
		if (unlikely(src == qp->sq.qend))
			src = qp->sq_start;
	} while (bytecnt > 0);
}

void mmio_memcpy_x64(void *dest, const void *src, size_t bytecnt)
{
	uintptr_t *dst_p = dest;

	/* Caller must guarantee:
	    assert(bytecnt != 0);
	    assert((bytecnt % 64) == 0);
	    assert(((uintptr_t)dest) % __alignof__(*dst) == 0);
	    assert(((uintptr_t)src) % __alignof__(*dst) == 0);
	*/

	/* Use the native word size for the copy */
	if (sizeof(*dst_p) == 8) {
		const __be64 *src_p = src;
		void *addr;
		__be64 val;

		do {
			/* Do 64 bytes at a time */
			addr = dst_p++;
			val = *src_p++;
			__be32 first_dword = htonl(htonl64(val) >> 32);
			__be32 second_dword = htonl(htonl64(val));

			*(volatile uint32_t *)addr = (uint32_t)first_dword;
			*(volatile uint32_t *)(addr+4) = (uint32_t)second_dword;

			bytecnt -= sizeof(*dst_p);
		} while (bytecnt > 0);
	} else if (sizeof(*dst_p) == 4) {
		const __be32 *src_p = src;

		do {
			*(volatile uint32_t *)dst_p++ = ( uint32_t)*src_p++;
			*(volatile uint32_t *)dst_p++ = ( uint32_t)*src_p++;
			bytecnt -= 2 * sizeof(*dst_p);
		} while (bytecnt > 0);
	}
}

void post_send_db(struct mlx5_qp *qp, struct mlx5_bf *bf,
				int nreq, int inl, int size, void *ctrl)
{

	if (unlikely(!nreq))
		return;
	qp->sq.head += nreq;
	qp->db[MLX5_SND_DBR] = htobe32(qp->sq.cur_post & 0xffff);	
	void *addr2;
	__be64 val;
	addr2 = bf->reg + bf->offset;
	val = *(__be64 *)ctrl;
	__be32 first_dword = htonl(htonl64(val) >> 32);
	__be32 second_dword = htonl(htonl64(val));

	*(volatile uint32_t *)addr2 = (uint32_t)first_dword;
	*(volatile uint32_t *)(addr2+4) = (uint32_t)second_dword;

	bf->offset ^= bf->buf_size;

}

int my_post_send(struct ibv_qp *ibqp, struct ibv_send_wr *wr,
				  struct ibv_send_wr **bad_wr)
{
	struct mlx5_qp *qp = to_mqp(ibqp);
	void *seg;
	struct mlx5_wqe_eth_seg *eseg;
	struct mlx5_wqe_ctrl_seg *ctrl = NULL;
	struct mlx5_wqe_data_seg *dpseg;
	struct mlx5_sg_copy_ptr sg_copy_ptr = {.index = 0, .offset = 0};
	int nreq;
	int inl = 0;
	int err = 0;
	int size = 0;
	int i;
	unsigned idx;
	uint8_t opmod = 0;
	struct mlx5_bf *bf = qp->bf;
	void *qend = qp->sq.qend;
	uint32_t mlx5_opcode;
	struct mlx5_wqe_xrc_seg *xrc;
	uint8_t fence;
	uint8_t next_fence;
	uint32_t max_tso = 0;
	FILE *fp = to_mctx(ibqp->context)->dbg_fp; /* The compiler ignores in non-debug mode */

	// next_fence = qp->fm_cache;

	// for (nreq = 0; wr; ++nreq, wr = wr->next) {
		

		// if (wr->send_flags & IBV_SEND_FENCE)
		// 	fence = MLX5_WQE_CTRL_FENCE;
		// else
		// 	fence = next_fence;
		// next_fence = 0;
		idx = qp->sq.cur_post & (qp->sq.wqe_cnt - 1);
		printf("qp->buf.buf: 0x%llx, qp->sq_start: 0x%llx\n", qp->buf.buf, qp->sq_start);
		printf("qp->bf->buf_size: 0x%llx\n", qp->bf->buf_size);
		ctrl = seg = qp->sq_start + (idx << MLX5_SEND_WQE_SHIFT); // mlx5_get_send_wqe(qp, idx);
		*(uint32_t *)(seg + 8) = 0;
		ctrl->imm = 0; // send_ieth(wr);
		ctrl->fm_ce_se = MLX5_WQE_CTRL_CQ_UPDATE;
		// qp->sq_signal_bits | fence |
		// 	(wr->send_flags & IBV_SEND_SIGNALED ?
		// 	 MLX5_WQE_CTRL_CQ_UPDATE : 0) |
		// 	(wr->send_flags & IBV_SEND_SOLICITED ?
		// 	 MLX5_WQE_CTRL_SOLICITED : 0);

		seg += sizeof(*ctrl);
		size = sizeof(*ctrl)/ 16;
		qp->sq.wr_data[idx] = 0;

		
		if(wr->opcode == IBV_WR_RDMA_READ || wr->opcode == IBV_WR_RDMA_WRITE){
			((struct mlx5_wqe_raddr_seg *) seg)->raddr    = htonl64(wr->wr.rdma.remote_addr);
			((struct mlx5_wqe_raddr_seg *) seg)->rkey     = htonl(wr->wr.rdma.rkey);
			((struct mlx5_wqe_raddr_seg *) seg)->reserved = 0;
			seg = seg + sizeof(struct mlx5_wqe_raddr_seg);
			size += sizeof(struct mlx5_wqe_raddr_seg) / 16;
    	}

		
		dpseg = seg;

		dpseg->byte_count = htonl(wr->sg_list->length);
		dpseg->lkey       = htonl(wr->sg_list->lkey);
		dpseg->addr       = htonl64(wr->sg_list->addr);

		size += sizeof(struct mlx5_wqe_data_seg) / 16;
		
		mlx5_opcode = wr->opcode*2 + 8 - 2*(wr->opcode == 2); // mlx5_ib_opcode[wr->opcode];
		ctrl->opmod_idx_opcode = htonl(((qp->sq.cur_post & 0xffff) << 8) | mlx5_opcode | (opmod << 24));
		ctrl->qpn_ds = htonl(size | (ibqp->qp_num << 8));

		qp->sq.wrid[idx] = wr->wr_id;
		qp->sq.wqe_head[idx] = qp->sq.head;
		qp->sq.cur_post += (size * 16 + 64 - 1) / 64;; //DIV_ROUND_UP(size * 16, MLX5_SEND_WQE_BB);


	nreq = 1;

out:
	// qp->fm_cache = next_fence;
	// post_send_db(qp, bf, nreq, inl, size, ctrl);

	if (unlikely(!nreq))
		return;
	qp->sq.head += nreq;
	qp->db[MLX5_SND_DBR] = htobe32(qp->sq.cur_post & 0xffff);	
	void *addr2;
	__be64 val;
	addr2 = bf->reg + bf->offset;
	val = *(__be64 *)ctrl;
	__be32 first_dword = htonl(htonl64(val) >> 32);
	__be32 second_dword = htonl(htonl64(val));

	*(volatile uint32_t *)addr2 = (uint32_t)first_dword;
	*(volatile uint32_t *)(addr2+4) = (uint32_t)second_dword;

	bf->offset ^= bf->buf_size;


	return err;
}




// __device__ 
// int device_gpu_post_send(struct ibv_qp *ibqp, struct ibv_send_wr *wr,
// 				  struct ibv_send_wr **bad_wr, unsigned int qpbf_bufsize, unsigned int qpsq_cur_post,
//           unsigned int qpsq_wqe_cnt, uint64_t wr_rdma_remote_addr, uint32_t wr_rdma_rkey,
//           uint32_t wr_sg_length, uint32_t wr_sg_lkey, uint64_t wr_sg_addr, int wr_opcode, 
//           unsigned int bf_offset, uint32_t qp_num, uint64_t wr_id, 
//           void *qp_buf, void *dev_qpsq_wr_data, void *dev_qpsq_wqe_head, 
//           void *dev_qp_sq, void *bf_reg, void *dev_qp_db)
// {

// 	// struct mlx5_qp *qp = to_mqp(ibqp);
// 	void *seg;
// 	struct mlx5_wqe_eth_seg *eseg;
// 	struct mlx5_wqe_ctrl_seg *ctrl = NULL;
// 	struct mlx5_wqe_data_seg *dpseg;
// 	struct mlx5_sg_copy_ptr sg_copy_ptr = {.index = 0, .offset = 0};
// 	int nreq = 0;
// 	int inl = 0;
// 	int err = 0;
// 	int size = 0;
// 	int i;
// 	unsigned idx;
// 	uint8_t opmod = 0;
// 	// struct mlx5_bf *bf = qp->bf;

// 	uint32_t mlx5_opcode;
// 	struct mlx5_wqe_xrc_seg *xrc;

//     // input variables:
//     // unsigned int qpbf_bufsize = qp->bf->buf_size; // 256
//     // unsigned int qpsq_cur_post = qp->sq.cur_post;
//     // unsigned int qpsq_wqe_cnt = qp->sq.wqe_cnt;
//     // uint64_t wr_rdma_remote_addr = wr->wr.rdma.remote_addr;
//     // uint32_t wr_rdma_rkey = wr->wr.rdma.rkey;
//     // uint32_t wr_sg_length = wr->sg_list->length;
//     // uint32_t wr_sg_lkey = wr->sg_list->lkey;
//     // uint64_t wr_sg_addr = wr->sg_list->addr;
//     // int wr_opcode = wr->opcode;
//     // uint32_t qp_num = ibqp->qp_num;
//     // unsigned int bf_offset = bf->offset; 
//     // uint64_t wr_id = wr->wr_id;

//     // pointers
//     // void* qp_buf = qp->buf.buf;
//     uint32_t *qpsq_wr_data = (uint32_t *) dev_qpsq_wr_data;//qp->sq.wr_data;
//     unsigned int *qpsq_wqe_head = (unsigned int *) dev_qpsq_wqe_head;// qp->sq.wqe_head;
//     struct mlx5_wq *qp_sq = (struct mlx5_wq *) dev_qp_sq; // &qp->sq;
//     // void *bf_reg = bf->reg;
//     unsigned int *qp_db = (unsigned int *) dev_qp_db; // qp->db
    

//     idx = qpsq_cur_post & (qpsq_wqe_cnt - 1);
//     printf("Function: %s line number: %d qp->buf.buf: 0x%llx \n",__func__, __LINE__, qp_buf);
//     printf("Function: %s line number: %d qp->bf->buf_size: %d \n",__func__, __LINE__, qpbf_bufsize);
//     printf("Function: %s line number: %d bf->reg: 0x%llx \n",__func__, __LINE__, bf_reg);
//     // printf("Function: %s line number: %d qp->buf.buf: 0x%llx \n",__func__, __LINE__, qp->sq_start);
//     seg = (qp_buf + qpbf_bufsize + (idx * 64)); // mlx5_get_send_wqe(qp, idx);
//     ctrl = (struct mlx5_wqe_ctrl_seg *) seg;
//     printf("Function: %s line number: %d qp->buf.buf: 0x%llx \n",__func__, __LINE__, ctrl);
//     *(uint32_t *)(seg + 8) = 0;
//     ctrl->imm = 0; // send_ieth(wr);
//     ctrl->fm_ce_se = MLX5_WQE_CTRL_CQ_UPDATE;

//     seg = seg + sizeof(*ctrl);
//     size = sizeof(*ctrl) / 16;
//     qp_sq->wr_data[idx] = 0;
//     // printf("ibqp->qp_type: %d \n", ibqp->qp_type);

//     // we generally have qp_type: IBV_QPT_RC and RDMA READ or WRITE request
//     ((struct mlx5_wqe_raddr_seg *) seg)->raddr    = htonl64(wr_rdma_remote_addr);
//     ((struct mlx5_wqe_raddr_seg *) seg)->rkey     = htonl(wr_rdma_rkey);
//     ((struct mlx5_wqe_raddr_seg *) seg)->reserved = 0;
//     seg = seg + sizeof(struct mlx5_wqe_raddr_seg);
//     size += sizeof(struct mlx5_wqe_raddr_seg) / 16;

//     dpseg = (struct mlx5_wqe_data_seg *) seg;
//     // for simplicity, we'll have one sge per request/post
//     i = 0;// sg_copy_ptr.index;
//     dpseg->byte_count = htonl(wr_sg_length);
//     dpseg->lkey       = htonl(wr_sg_lkey);
//     dpseg->addr       = htonl64(wr_sg_addr);
//     size += sizeof(struct mlx5_wqe_data_seg) / 16;
//     // 4 -> 16, 0 -> 8
//     mlx5_opcode = wr_opcode*2 + 8; // mlx5_ib_opcode[wr->opcode];
//     ctrl->opmod_idx_opcode = htonl(((qpsq_cur_post & 0xffff) << 8) | mlx5_opcode);
//     ctrl->qpn_ds = htonl(size | (qp_num << 8));

//     qp_sq->wrid[idx] = wr_id;
//     qp_sq->wqe_head[idx] = qp_sq->head;
//     qp_sq->cur_post += (size * 16 + 64 - 1) / 64;
        
//     // db update:
//     qp_sq->head += 1;
//     qp_db[MLX5_SND_DBR] = htonl(qpsq_cur_post & 0xffff);

//     void *addr;
//     uint64_t val;
//     /* Do 64 bytes at a time */
//     addr = bf_reg + bf_offset;
//     val = *(uint64_t *) ctrl;
//     uint32_t first_dword = htonl(htonl64(val) >> 32);
//     uint32_t second_dword = htonl(htonl64(val));
//     *(volatile uint32_t *)addr = (uint32_t)first_dword;
//     *(volatile uint32_t *)(addr+4) = (uint32_t)second_dword;
//     // bf->offset ^= bf->buf_size;

// 	return err;
// }

// __global__
// void global_gpu_post_send(struct ibv_qp *ibqp, struct ibv_send_wr *wr,
// 				  struct ibv_send_wr **bad_wr, unsigned int qpbf_bufsize, unsigned int qpsq_cur_post,
//           unsigned int qpsq_wqe_cnt, uint64_t wr_rdma_remote_addr, uint32_t wr_rdma_rkey,
//           uint32_t wr_sg_length, uint32_t wr_sg_lkey, uint64_t wr_sg_addr, int wr_opcode, 
//           unsigned int bf_offset, uint32_t qp_num, uint64_t wr_id,
//           void *qp_buf, void *dev_qpsq_wr_data, void *dev_qpsq_wqe_head, 
//           void *dev_qp_sq, void *bf_reg, void *dev_qp_db, int *ret){

//   *ret = device_gpu_post_send(ibqp, wr,
// 				  bad_wr, qpbf_bufsize, qpsq_cur_post,
//           qpsq_wqe_cnt, wr_rdma_remote_addr, wr_rdma_rkey,
//           wr_sg_length, wr_sg_lkey, wr_sg_addr, wr_opcode, 
//           bf_offset, qp_num, wr_id, qp_buf, dev_qpsq_wr_data, 
//           dev_qpsq_wqe_head, dev_qp_sq, bf_reg, dev_qp_db);

// }

// int gpu_post_send(struct ibv_qp *ibqp, struct ibv_send_wr *wr, struct ibv_send_wr **bad_wr){

//   int ret;
//   struct mlx5_qp *qp = to_mqp(ibqp);
//   struct mlx5_bf *bf = qp->bf;

//   // input variables:
//   unsigned int qpbf_bufsize = qp->bf->buf_size; // 256
//   unsigned int qpsq_cur_post = qp->sq.cur_post;
//   unsigned int qpsq_wqe_cnt = qp->sq.wqe_cnt;
//   uint64_t wr_rdma_remote_addr = wr->wr.rdma.remote_addr;
//   uint32_t wr_rdma_rkey = wr->wr.rdma.rkey;
//   uint32_t wr_sg_length = wr->sg_list->length;
//   uint32_t wr_sg_lkey = wr->sg_list->lkey;
//   uint64_t wr_sg_addr = wr->sg_list->addr;
//   int wr_opcode = wr->opcode;
//   uint32_t qp_num = ibqp->qp_num;
//   unsigned int bf_offset = bf->offset; 
//   uint64_t wr_id = wr->wr_id;

//   // pointers
//   void *qp_buf = qp->buf.buf;
//   void *dev_qpsq_wr_data; //qp->sq.wr_data;
//   void *dev_qpsq_wqe_head;// qp->sq.wqe_head;
//   void *dev_qp_sq; // &qp->sq;
//   void *bf_reg; // bf->reg;
//   void *dev_qp_db; // qp->db
//   int *dev_ret; // &ret

//   // pin host memory and get device pointer:

//   if(cudaHostRegister(qp->buf.buf, sizeof(qp->buf.buf), cudaHostRegisterMapped) !=  cudaSuccess)
//         exit(0);
//   // get GPU pointer for qp_buf
//   if(cudaHostGetDevicePointer(&qp_buf, qp->buf.buf, 0) != cudaSuccess)
//       exit(0);

//   if(cudaHostRegister(qp->sq.wr_data, sizeof(qp->sq.wr_data), cudaHostRegisterMapped) !=  cudaSuccess)
//         exit(0);
//   // get GPU pointer for qp->sq.wr_data
//   if(cudaHostGetDevicePointer(&dev_qpsq_wr_data, qp->sq.wr_data, 0) != cudaSuccess)
//       exit(0);

//   if(cudaHostRegister(qp->sq.wqe_head, sizeof(qp->sq.wqe_head), cudaHostRegisterMapped) !=  cudaSuccess)
//         exit(0);
//   // get GPU pointer for qp->sq.wqe_head
//   if(cudaHostGetDevicePointer(&dev_qpsq_wqe_head, qp->sq.wqe_head, 0) != cudaSuccess)
//       exit(0);

//   if(cudaHostRegister(&qp->sq, sizeof(qp->sq), cudaHostRegisterMapped) !=  cudaSuccess)
//         exit(0);
//   // get GPU pointer for qp->sq
//   if(cudaHostGetDevicePointer(&dev_qp_sq, &qp->sq, 0) != cudaSuccess)
//       exit(0);

//   if(cudaHostRegister(bf->reg, sizeof(bf->reg), cudaHostRegisterMapped) !=  cudaSuccess)
//         exit(0);
//   // get GPU pointer for bf->reg
//   if(cudaHostGetDevicePointer(&bf_reg, bf->reg, 0) != cudaSuccess)
//       exit(0);

//   if(cudaHostRegister(qp->db, sizeof(qp->db), cudaHostRegisterMapped) !=  cudaSuccess)
//         exit(0);
//   // get GPU pointer for qp->db
//   if(cudaHostGetDevicePointer(&dev_qp_db, qp->db, 0) != cudaSuccess)
//       exit(0);

//   if(cudaHostRegister(&ret, sizeof(ret), cudaHostRegisterMapped) !=  cudaSuccess)
//         exit(0);
//   // get GPU pointer for qp->db
//   if(cudaHostGetDevicePointer(&dev_ret, &ret, 0) != cudaSuccess)
//       exit(0);

//   if (cudaDeviceSynchronize() != 0) exit(0);

//   global_gpu_post_send<<<1,1>>>(ibqp, wr,
// 				  bad_wr, qpbf_bufsize, qpsq_cur_post, qpsq_wqe_cnt, 
//           wr_rdma_remote_addr, wr_rdma_rkey,
//           wr_sg_length,
//           wr_sg_lkey, wr_sg_addr, wr_opcode, 
//           bf_offset, qp_num, wr_id, qp_buf, dev_qpsq_wr_data, 
//           dev_qpsq_wqe_head, dev_qp_sq, bf_reg, dev_qp_db, dev_ret);

//   if (cudaDeviceSynchronize() != 0) exit(0);

//   return ret;

// }