#ifndef RDMA_UTILS_H
#define RDMA_UTILS_H

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <errno.h>
#include <getopt.h>

#include <netdb.h>
#include <netinet/in.h>	
#include <arpa/inet.h>
#include <sys/socket.h>

#include <rdma/rdma_cma.h>
#include <infiniband/verbs.h>
#include <infiniband/mlx5dv.h>
#include <linux/kernel.h>
#include <valgrind/memcheck.h>
#include <rdma/mlx5-abi.h>
#include <stddef.h>


static const int RDMA_BUFFER_SIZE = 1024*1024*50+1;

struct message {
  enum {
    MSG_MR,
    MSG_DONE
  } type;

  union {
    struct ibv_mr mr;
  } data;
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
  char *rdma_remote_region;

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

enum mode {
  M_WRITE,
  M_READ
};

enum {
	CQ_OK					=  0,
	CQ_EMPTY				= -1,
	CQ_POLL_ERR				= -2,
	CQ_POLL_NODATA				= ENOENT
};


/*MLX structs, defs and enums*/

#define htons(x)  ((((uint16_t)(x) & 0xff00) >> 8) |\
                   (((uint16_t)(x) & 0x00ff) << 8))
#define htonl(x)  ((((uint32_t)(x) & 0xff000000) >> 24) |\
                   (((uint32_t)(x) & 0x00ff0000) >>  8) |\
                   (((uint32_t)(x) & 0x0000ff00) <<  8) |\
                   (((uint32_t)(x) & 0x000000ff) << 24))
#define htonl64(x)((((uint64_t)(x) & 0xff00000000000000ull) >> 56)	\
				|  (((uint64_t)(x) & 0x00ff000000000000ull) >> 40)	\
				|  (((uint64_t)(x) & 0x0000ff0000000000ull) >> 24)	\
				|  (((uint64_t)(x) & 0x000000ff00000000ull) >> 8)	\
				|  (((uint64_t)(x) & 0x00000000ff000000ull) << 8)	\
				|  (((uint64_t)(x) & 0x0000000000ff0000ull) << 24)	\
				|  (((uint64_t)(x) & 0x000000000000ff00ull) << 40)	\
				|  (((uint64_t)(x) & 0x00000000000000ffull) << 56))
#define ntohs htons
#define ntohl htonl

#define unlikely(x)      __builtin_expect(!!(x), 0)
#define likely(x)       __builtin_expect(!!(x), 1)

#define VALGRIND_MAKE_MEM_DEFINED(_qzz_addr,_qzz_len)            \
    VALGRIND_DO_CLIENT_REQUEST_EXPR(0 /* default return */,      \
                            VG_USERREQ__MAKE_MEM_DEFINED,        \
                            (_qzz_addr), (_qzz_len), 0, 0, 0)

#define udma_from_device_barrier() asm volatile("lfence" ::: "memory")

#define PFX		"mlx5: "

#define BITS_PER_LONG	   (8 * sizeof(long))

#define BUILD_ASSERT(cond) \
	do { (void) sizeof(char [1 - 2*!(cond)]); } while(0)

#define MINMAX_ASSERT_COMPATIBLE(a, b) \
	BUILD_ASSERT(__builtin_types_compatible_p(a, b))

#define min(a, b) \
	({ \
		typeof(a) _a = (a); \
		typeof(b) _b = (b); \
		MINMAX_ASSERT_COMPATIBLE(typeof(_a), typeof(_b)); \
		_a < _b ? _a : _b; \
	})

#define min_t(t, a, b) \
	({ \
		t _ta = (a); \
		t _tb = (b); \
		min(_ta, _tb); \
	})

#define SWITCH_FALLTHROUGH __attribute__ ((fallthrough))

#define udma_to_device_barrier() asm volatile("" ::: "memory")


#define check_types_match(expr1, expr2)		\
	((typeof(expr1) *)0 != (typeof(expr2) *)0)

#define container_off(containing_type, member)	\
	offsetof(containing_type, member)

#ifndef container_of
#define container_of(member_ptr, containing_type, member)		\
	 ((containing_type *)						\
	  ((char *)(member_ptr)						\
	   - container_off(containing_type, member))			\
	  + check_types_match(*(member_ptr), ((containing_type *)0)->member))
#endif

#define ALIGN(x, log_a) ((((x) + (1 << (log_a)) - 1)) & ~((1 << (log_a)) - 1))

#if (__GNUC__ >= 6 && __GNUC__ < 12 && !defined(__powerpc__)) || defined(__clang__)
#define uninitialized_var(x) x
#else
#define uninitialized_var(x) x = x
#endif

#define mmio_wc_start() mmio_flush_writes()

#define mmio_flush_writes() asm volatile("sfence" ::: "memory")

#define MAKE_WRITE(_NAME_, _SZ_)                                               \
	static inline void _NAME_(void *addr, uint##_SZ_##_t value)            \
	{                                                                      \
		_NAME_##_le(addr, htole##_SZ_(value));                         \
	}

#define MLX5_ATOMIC_SIZE 8

#define to_mxxx(xxx, type) container_of(ib##xxx, struct mlx5_##type, ibv_##xxx)


enum {
	MLX5_TM_MAX_SYNC_DIFF = 0x3fff
};

enum {
	MLX5_CQE_APP_OP_TM_CONSUMED = 0x1,
	MLX5_CQE_APP_OP_TM_EXPECTED = 0x2,
	MLX5_CQE_APP_OP_TM_UNEXPECTED = 0x3,
	MLX5_CQE_APP_OP_TM_NO_TAG = 0x4,
	MLX5_CQE_APP_OP_TM_APPEND = 0x5,
	MLX5_CQE_APP_OP_TM_REMOVE = 0x6,
	MLX5_CQE_APP_OP_TM_NOOP = 0x7,
	MLX5_CQE_APP_OP_TM_CONSUMED_SW_RDNV = 0x9,
	MLX5_CQE_APP_OP_TM_CONSUMED_MSG = 0xA,
	MLX5_CQE_APP_OP_TM_CONSUMED_MSG_SW_RDNV = 0xB,
	MLX5_CQE_APP_OP_TM_MSG_COMPLETION_CANCELED = 0xC,
};

enum {
	MLX5_CQ_FLAGS_RX_CSUM_VALID = 1 << 0,
	MLX5_CQ_FLAGS_EMPTY_DURING_POLL = 1 << 1,
	MLX5_CQ_FLAGS_FOUND_CQES = 1 << 2,
	MLX5_CQ_FLAGS_EXTENDED = 1 << 3,
	MLX5_CQ_FLAGS_SINGLE_THREADED = 1 << 4,
	MLX5_CQ_FLAGS_DV_OWNED = 1 << 5,
	MLX5_CQ_FLAGS_TM_SYNC_REQ = 1 << 6,
	MLX5_CQ_FLAGS_RAW_WQE = 1 << 7,
};

enum {
	MLX5_CQ_LAZY_FLAGS =
		MLX5_CQ_FLAGS_RX_CSUM_VALID |
		MLX5_CQ_FLAGS_TM_SYNC_REQ |
		MLX5_CQ_FLAGS_RAW_WQE
};

enum {
	MLX5_CQE_APP_TAG_MATCHING = 1,
};

enum {
	MLX5_CSUM_SUPPORT_RAW_OVER_ETH  = (1 << 0),
	MLX5_CSUM_SUPPORT_UNDERLAY_UD   = (1 << 1),
	/*
	 * Only report rx checksum when the validation
	 * is valid.
	 */
	MLX5_RX_CSUM_VALID              = (1 << 16),
};

enum {
	MLX5_QP_TABLE_SHIFT		= 12,
	MLX5_QP_TABLE_MASK		= (1 << MLX5_QP_TABLE_SHIFT) - 1,
	MLX5_QP_TABLE_SIZE		= 1 << (24 - MLX5_QP_TABLE_SHIFT),
};

enum {
	MLX5_SRQ_TABLE_SHIFT		= 12,
	MLX5_SRQ_TABLE_MASK		= (1 << MLX5_SRQ_TABLE_SHIFT) - 1,
	MLX5_SRQ_TABLE_SIZE		= 1 << (24 - MLX5_SRQ_TABLE_SHIFT),
};

enum {
	MLX5_UIDX_TABLE_SHIFT		= 12,
	MLX5_UIDX_TABLE_MASK		= (1 << MLX5_UIDX_TABLE_SHIFT) - 1,
	MLX5_UIDX_TABLE_SIZE		= 1 << (24 - MLX5_UIDX_TABLE_SHIFT),
};

enum {
	MLX5_MKEY_TABLE_SHIFT		= 12,
	MLX5_MKEY_TABLE_MASK		= (1 << MLX5_MKEY_TABLE_SHIFT) - 1,
	MLX5_MKEY_TABLE_SIZE		= 1 << (24 - MLX5_MKEY_TABLE_SHIFT),
};

enum {
	MLX5_NUM_NON_FP_BFREGS_PER_UAR	= 2,
	NUM_BFREGS_PER_UAR		= 4,
	MLX5_MAX_UARS			= 1 << 8,
	MLX5_MAX_BFREGS			= MLX5_MAX_UARS * MLX5_NUM_NON_FP_BFREGS_PER_UAR,
	MLX5_DEF_TOT_UUARS		= 8 * MLX5_NUM_NON_FP_BFREGS_PER_UAR,
	MLX5_MED_BFREGS_TSHOLD		= 12,
};

typedef enum _cl_map_color {
	CL_MAP_RED,
	CL_MAP_BLACK
} cl_map_color_t;

enum mlx5_uar_type {
	MLX5_UAR_TYPE_REGULAR,
	MLX5_UAR_TYPE_NC,
	MLX5_UAR_TYPE_REGULAR_DYN,
};

enum {
	MLX5_MAX_PORTS_NUM = 2,
};

enum {
	MLX5_DBG_QP		= 1 << 0,
	MLX5_DBG_CQ		= 1 << 1,
	MLX5_DBG_QP_SEND	= 1 << 2,
	MLX5_DBG_QP_SEND_ERR	= 1 << 3,
	MLX5_DBG_CQ_CQE		= 1 << 4,
	MLX5_DBG_CONTIG		= 1 << 5,
	MLX5_DBG_DR		= 1 << 6,
};

enum mlx5_devx_obj_type {
	MLX5_DEVX_FLOW_TABLE		= 1,
	MLX5_DEVX_FLOW_COUNTER		= 2,
	MLX5_DEVX_FLOW_METER		= 3,
	MLX5_DEVX_QP			= 4,
	MLX5_DEVX_PKT_REFORMAT_CTX	= 5,
	MLX5_DEVX_TIR			= 6,
	MLX5_DEVX_FLOW_GROUP		= 7,
	MLX5_DEVX_FLOW_TABLE_ENTRY	= 8,
	MLX5_DEVX_FLOW_SAMPLER		= 9,
	MLX5_DEVX_ASO_FIRST_HIT		= 10,
	MLX5_DEVX_ASO_FLOW_METER	= 11,
	MLX5_DEVX_ASO_CT		= 12,
};

enum mlx5_sig_type {
	MLX5_SIG_TYPE_NONE = 0,
	MLX5_SIG_TYPE_CRC,
	MLX5_SIG_TYPE_T10DIF,
};

enum mlx5_mkey_bsf_state {
	MLX5_MKEY_BSF_STATE_INIT,
	MLX5_MKEY_BSF_STATE_RESET,
	MLX5_MKEY_BSF_STATE_SET,
	MLX5_MKEY_BSF_STATE_UPDATED,
};

enum {
	MLX5_IPOIB_INLINE_MIN_HEADER_SIZE	= 4,
	MLX5_SOURCE_QPN_INLINE_MAX_HEADER_SIZE	= 18,
	MLX5_ETH_L2_INLINE_HEADER_SIZE	= 18,
	MLX5_ETH_L2_MIN_HEADER_SIZE	= 14,
};

struct mlx5_sig_err {
	uint16_t syndrome;
	uint64_t expected;
	uint64_t actual;
	uint64_t offset;
	uint8_t sig_type;
	uint8_t domain;
};

enum mlx5_qp_flags {
	MLX5_QP_FLAGS_USE_UNDERLAY = 0x01,
	MLX5_QP_FLAGS_DRAIN_SIGERR = 0x02,
};

enum {
	MLX5_CQ_SET_CI	= 0,
	MLX5_CQ_ARM_DB	= 1,
};


struct mlx5_psv {
	uint32_t index;
	struct mlx5dv_devx_obj *devx_obj;
};

struct mlx5_sig_block_domain {
	enum mlx5_sig_type sig_type;
	union {
		struct mlx5dv_sig_t10dif dif;
		struct mlx5dv_sig_crc crc;
	} sig;
	enum mlx5dv_block_size block_size;
};

struct mlx5_sig_block_attr {
	struct mlx5_sig_block_domain mem;
	struct mlx5_sig_block_domain wire;
	uint32_t flags;
	uint8_t check_mask;
	uint8_t copy_mask;
};

struct mlx5dv_devx_obj {
	struct ibv_context *context;
	uint32_t handle;
	enum mlx5_devx_obj_type type;
	uint32_t object_id;
	uint64_t rx_icm_addr;
	uint8_t log_obj_range;
	void *priv;
};

struct mlx5_sig_block {
	struct mlx5_psv *mem_psv;
	struct mlx5_psv *wire_psv;
	struct mlx5_sig_block_attr attr;
	enum mlx5_mkey_bsf_state state;
};

struct mlx5_crypto_attr {
	enum mlx5dv_crypto_standard crypto_standard;
	bool encrypt_on_tx;
	enum mlx5dv_signature_crypto_order signature_crypto_order;
	enum mlx5dv_block_size data_unit_size;
	char initial_tweak[16];
	struct mlx5dv_dek *dek;
	char keytag[8];
	enum mlx5_mkey_bsf_state state;
};

struct mlx5_sig_ctx {
	struct mlx5_sig_block block;
	struct mlx5_sig_err err_info;
	uint32_t err_count;
	bool err_exists;
	bool err_count_updated;
};

struct mlx5_mkey {
	struct mlx5dv_mkey dv_mkey;
	struct mlx5dv_devx_obj *devx_obj;
	uint16_t num_desc;
	uint64_t length;
	struct mlx5_sig_ctx *sig;
	struct mlx5_crypto_attr *crypto;
};

struct mlx5_sigerr_cqe {
	uint8_t rsvd0[16];
	__be32 expected_trans_sig;
	__be32 actual_trans_sig;
	__be32 expected_ref_tag;
	__be32 actual_ref_tag;
	__be16 syndrome;
	uint8_t sig_type;
	uint8_t domain;
	__be32 mkey;
	__be64 sig_err_offset;
	uint8_t rsvd30[14];
	uint8_t signature;
	uint8_t op_own;
};

struct mlx5_wq {
	uint64_t		       *wrid;
	unsigned		       *wqe_head;
	struct mlx5_spinlock		lock;
	unsigned			wqe_cnt;
	unsigned			max_post;
	unsigned			head;
	unsigned			tail;
	unsigned			cur_post;
	int				max_gs;
	/*
	 * Equal to max_gs when qp is in RTS state for sq, or in INIT state for
	 * rq, equal to -1 otherwise, used to verify qp_state in data path.
	 */
	int				qp_state_max_gs;
	int				wqe_shift;
	int				offset;
	void			       *qend;
	uint32_t			*wr_data;
};

struct verbs_qp {
	union {
		struct ibv_qp qp;
		struct ibv_qp_ex qp_ex;
	};
	uint32_t		comp_mask;
	struct verbs_xrcd       *xrcd;
};

struct mlx5_qp {
	struct mlx5_resource            rsc; /* This struct must be first */
	struct verbs_qp			verbs_qp;
	struct mlx5dv_qp_ex		dv_qp;
	struct ibv_qp		       *ibv_qp;
	struct mlx5_buf                 buf;
	int                             max_inline_data;
	int                             buf_size;
	/* For Raw Packet QP, use different buffers for the SQ and RQ */
	struct mlx5_buf                 sq_buf;
	int				sq_buf_size;
	struct mlx5_bf		       *bf;

	/* Start of new post send API specific fields */
	bool				inl_wqe;
	uint8_t				cur_setters_cnt;
	uint8_t				num_mkey_setters;
	uint8_t				fm_cache_rb;
	int				err;
	int				nreq;
	uint32_t			cur_size;
	uint32_t			cur_post_rb;
	void				*cur_eth;
	void				*cur_data;
	struct mlx5_wqe_ctrl_seg	*cur_ctrl;
	struct mlx5_mkey		*cur_mkey;
	/* End of new post send API specific fields */

	uint8_t				fm_cache;
	uint8_t	                        sq_signal_bits;
	void				*sq_start;
	struct mlx5_wq                  sq;

	__be32                         *db;
	bool				custom_db;
	struct mlx5_wq                  rq;
	int                             wq_sig;
	uint32_t			qp_cap_cache;
	int				atomics_enabled;
	uint32_t			max_tso;
	uint16_t			max_tso_header;
	int                             rss_qp;
	uint32_t			flags; /* Use enum mlx5_qp_flags */
	enum mlx5dv_dc_type		dc_type;
	uint32_t			tirn;
	uint32_t			tisn;
	uint32_t			rqn;
	uint32_t			sqn;
	uint64_t			tir_icm_addr;
	/*
	 * ECE configuration is done in create/modify QP stages,
	 * so this value is cached version of the requested ECE prior
	 * to its execution. This field will be cleared after successful
	 * call to relevant "executor".
	 */
	uint32_t			set_ece;
	/*
	 * This field indicates returned ECE options from the device
	 * as were received from the HW in previous stage. Every
	 * write to the set_ece will clear this field.
	 */
	uint32_t			get_ece;

	uint8_t				need_mmo_enable:1;
};

struct mlx5_rwq {
	struct mlx5_resource rsc;
	struct ibv_wq wq;
	struct mlx5_buf buf;
	int buf_size;
	struct mlx5_wq rq;
	__be32  *db;
	bool	custom_db;
	void	*pbuff;
	__be32	*recv_db;
	int wq_sig;
};

typedef struct _cl_list_item {
        struct _cl_list_item *p_next;
        struct _cl_list_item *p_prev;
} cl_list_item_t;

typedef struct _cl_pool_item {
        cl_list_item_t list_item;
} cl_pool_item_t;

typedef struct _cl_map_item {
	/* Must be first to allow casting. */
	cl_pool_item_t pool_item;
	struct _cl_map_item *p_left;
	struct _cl_map_item *p_right;
	struct _cl_map_item *p_up;
	cl_map_color_t color;
	uint64_t key;
#ifdef _DEBUG_
	struct _cl_qmap *p_map;
#endif
} cl_map_item_t;

typedef struct _cl_qmap {
	cl_map_item_t root;
	cl_map_item_t nil;
	size_t count;
} cl_qmap_t;

struct list_node
{
	struct list_node *next, *prev;
};

struct list_head
{
	struct list_node n;
};

struct mlx5_uar_info {
	void				*reg;
	enum mlx5_uar_type		type;
};

struct mlx5_entropy_caps {
	uint8_t num_lag_ports;
	uint8_t lag_tx_port_affinity:1;
	uint8_t rts2rts_qp_udp_sport:1;
	uint8_t rts2rts_lag_tx_port_affinity:1;
};

struct mlx5_qos_caps {
	uint8_t qos:1;

	uint8_t nic_sq_scheduling:1;
	uint8_t nic_bw_share:1;
	uint8_t nic_rate_limit:1;
	uint8_t nic_qp_scheduling:1;

	uint32_t nic_element_type;
	uint32_t nic_tsar_type;
};

struct mlx5_hca_cap_2_caps {
	uint32_t log_reserved_qpns_per_obj;
};

struct mlx5_dma_mmo_caps {
	uint8_t dma_mmo_sq:1; /* Indicates that RC and DCI support DMA MMO */
	uint8_t dma_mmo_qp:1;
	uint64_t dma_max_size;
};

struct mlx5_reserved_qpns {
	struct list_head blk_list;
	pthread_mutex_t mutex;
};

struct mlx5_context {
	struct verbs_context		ibv_ctx;
	int				max_num_qps;
	int				bf_reg_size;
	int				tot_uuars;
	int				low_lat_uuars;
	int				num_uars_per_page;
	int				bf_regs_per_page;
	int				num_bf_regs;
	int				prefer_bf;
	int				shut_up_bf;
	struct {
		struct mlx5_qp        **table;
		int			refcnt;
	}				qp_table[MLX5_QP_TABLE_SIZE];
	pthread_mutex_t			qp_table_mutex;

	struct {
		struct mlx5_srq	      **table;
		int			refcnt;
	}				srq_table[MLX5_SRQ_TABLE_SIZE];
	pthread_mutex_t			srq_table_mutex;

	struct {
		struct mlx5_resource  **table;
		int                     refcnt;
	}				uidx_table[MLX5_UIDX_TABLE_SIZE];
	pthread_mutex_t                 uidx_table_mutex;

	struct {
		struct mlx5_mkey      **table;
		int			refcnt;
	}				mkey_table[MLX5_MKEY_TABLE_SIZE];
	pthread_mutex_t			mkey_table_mutex;

	struct mlx5_uar_info		uar[MLX5_MAX_UARS];
	struct list_head		dbr_available_pages;
	cl_qmap_t		        dbr_map;
	pthread_mutex_t			dbr_map_mutex;
	int				cache_line_size;
	int				max_sq_desc_sz;
	int				max_rq_desc_sz;
	int				max_send_wqebb;
	int				max_recv_wr;
	unsigned			max_srq_recv_wr;
	int				num_ports;
	int				stall_enable;
	int				stall_adaptive_enable;
	int				stall_cycles;
	struct mlx5_bf		       *bfs;
	FILE			       *dbg_fp;
	char				hostname[40];
	struct mlx5_spinlock            hugetlb_lock;
	struct list_head                hugetlb_list;
	int				cqe_version;
	uint8_t				cached_link_layer[MLX5_MAX_PORTS_NUM];
	uint8_t				cached_port_flags[MLX5_MAX_PORTS_NUM];
	unsigned int			cached_device_cap_flags;
	enum ibv_atomic_cap		atomic_cap;
	struct {
		uint64_t                offset;
		uint64_t                mask;
	} core_clock;
	void			       *hca_core_clock;
	const struct mlx5_ib_clock_info *clock_info_page;
	struct mlx5_ib_tso_caps		cached_tso_caps;
	int				cmds_supp_uhw;
	uint32_t			uar_size;
	uint64_t			vendor_cap_flags; /* Use enum mlx5_vendor_cap_flags */
	struct mlx5dv_cqe_comp_caps	cqe_comp_caps;
	struct mlx5dv_ctx_allocators	extern_alloc;
	struct mlx5dv_sw_parsing_caps	sw_parsing_caps;
	struct mlx5dv_striding_rq_caps	striding_rq_caps;
	struct mlx5dv_dci_streams_caps  dci_streams_caps;
	uint32_t			tunnel_offloads_caps;
	struct mlx5_packet_pacing_caps	packet_pacing_caps;
	struct mlx5_entropy_caps	entropy_caps;
	struct mlx5_qos_caps		qos_caps;
	struct mlx5_hca_cap_2_caps	hca_cap_2_caps;
	uint64_t			general_obj_types_caps;
	uint8_t				qpc_extension_cap:1;
	struct mlx5dv_sig_caps		sig_caps;
	struct mlx5_dma_mmo_caps	dma_mmo_caps;
	struct mlx5dv_crypto_caps	crypto_caps;
	pthread_mutex_t			dyn_bfregs_mutex; /* protects the dynamic bfregs allocation */
	uint32_t			num_dyn_bfregs;
	uint32_t			max_num_legacy_dyn_uar_sys_page;
	uint32_t			curr_legacy_dyn_sys_uar_page;
	uint16_t			flow_action_flags;
	uint64_t			max_dm_size;
	uint32_t                        eth_min_inline_size;
	uint32_t                        dump_fill_mkey;
	__be32                          dump_fill_mkey_be;
	uint32_t			flags;
	struct list_head		dyn_uar_bf_list;
	struct list_head		dyn_uar_db_list;
	struct list_head		dyn_uar_qp_shared_list;
	struct list_head		dyn_uar_qp_dedicated_list;
	uint16_t			qp_max_dedicated_uuars;
	uint16_t			qp_alloc_dedicated_uuars;
	uint16_t			qp_max_shared_uuars;
	uint16_t			qp_alloc_shared_uuars;
	struct mlx5_bf			*nc_uar;
	void				*cq_uar_reg;
	struct mlx5_reserved_qpns	reserved_qpns;
	uint8_t				qp_data_in_order_cap:1;
	struct mlx5_dv_context_ops	*dv_ctx_ops;
	struct mlx5dv_devx_obj		*crypto_login;
	pthread_mutex_t			crypto_login_mutex;
	uint64_t			max_dc_rd_atom;
	uint64_t			max_dc_init_rd_atom;
};

struct mlx5_sg_copy_ptr {
	int	index;
	int	offset;
};

struct mlx5_wqe_xrc_seg {
	__be32		xrc_srqn;
	uint8_t		rsvd[12];
};

struct mlx5_wqe_eth_pad {
	uint8_t rsvd0[16];
};

struct mlx5_wqe_inline_seg {
	__be32		byte_count;
};

enum ibv_mr_type {
	IBV_MR_TYPE_MR,
	IBV_MR_TYPE_NULL_MR,
	IBV_MR_TYPE_IMPORTED_MR,
	IBV_MR_TYPE_DMABUF_MR,
};

struct verbs_mr {
	struct ibv_mr		ibv_mr;
	enum ibv_mr_type        mr_type;
	int access;
};

struct mlx5_ah {
	struct ibv_ah			ibv_ah;
	struct mlx5_wqe_av		av;
	bool				kern_ah;
	pthread_mutex_t			mutex;
	uint8_t				is_global;
	struct mlx5dv_devx_obj		*ah_qp_mapping;
};

struct mlx5_mr {
	struct verbs_mr                 vmr;
	uint32_t			alloc_flags;
};

struct mlx5_rwqe_sig {
	uint8_t		rsvd0[4];
	uint8_t		signature;
	uint8_t		rsvd1[11];
};

static inline struct mlx5_context *to_mctx(struct ibv_context *ibctx)
{
	return container_of(ibctx, struct mlx5_context, ibv_ctx.context);
}

static inline struct mlx5_qp *rsc_to_mqp(struct mlx5_resource *rsc)
{
	return (struct mlx5_qp *)rsc;
}

static inline struct mlx5_qp *to_mqp(struct ibv_qp *ibqp)
{
	struct verbs_qp *vqp = (struct verbs_qp *)ibqp;

	return container_of(vqp, struct mlx5_qp, verbs_qp);
}



static inline struct mlx5_cq *to_mcq(struct ibv_cq *ibcq)
{
	return container_of(ibcq, struct mlx5_cq, verbs_cq.cq);
}

static inline struct mlx5_srq *to_msrq(struct ibv_srq *ibsrq)
{
	struct verbs_srq *vsrq = (struct verbs_srq *)ibsrq;

	return container_of(vsrq, struct mlx5_srq, vsrq);
}

static inline struct mlx5_rwq *rsc_to_mrwq(struct mlx5_resource *rsc)
{
	return (struct mlx5_rwq *)rsc;
}

static inline struct mlx5_srq *rsc_to_msrq(struct mlx5_resource *rsc)
{
	return (struct mlx5_srq *)rsc;
}

int cpu_poll_cq(struct ibv_cq *ibcq, int n, struct ibv_wc *wc);

int mlx5_post_send(struct ibv_qp *ibqp, struct ibv_send_wr *wr,
				  struct ibv_send_wr **bad_wr);

int mlx5_post_recv(struct ibv_qp *ibqp, struct ibv_recv_wr *wr,
		   struct ibv_recv_wr **bad_wr);
		


#endif