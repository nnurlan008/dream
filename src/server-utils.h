#ifndef SERVER_UTILS_H
#define SERVER_UTILS_H

#include <netdb.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <rdma/rdma_cma.h>

#define TEST_NZ(x) do { if ( (x)) die("error: " #x " failed (returned non-zero)." ); } while (0)
#define TEST_Z(x)  do { if (!(x)) die("error: " #x " failed (returned zero/null)."); } while (0)

#define PORT 9700

static unsigned long long int RDMA_BUFFER_SIZE = 80*1024*1024*1024llu;
static int N_QPs = 256;
// const static int N_8GB_Region = 10;
enum { 
  N_8GB_Region = 5,
  Region_Size = 8*1024*1024*1024llu
 };
  
struct MemPool{
  uint64_t addresses[N_8GB_Region];
  uint32_t rkeys[N_8GB_Region];
  uint32_t lkeys[N_8GB_Region];
};

enum mode {
  M_WRITE,
  M_READ
};



void die(const char *reason);

void build_connection(struct rdma_cm_id *id);
void build_params(struct rdma_conn_param *params);
void destroy_connection(void *context);
void * get_local_message_region(void *context);
void on_connect(void *context);
void send_mr(void *context);
void set_mode(enum mode m);

#endif