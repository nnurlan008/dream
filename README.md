# gpu_rdma_read_write
## TODO:
1. change the structure of the code to have the following:
   1. server has the buffer
   2. client makes a read and write requests
2. add gpu as a client:
   1. ibv_poll_cq -> gpu_poll_cq
   2. ibv_post_send/recv -> gpu_poll_send/recv
3. Compare the read and write time for CPU and GPU client
