.PHONY: clean

CFLAGS  := -Wall  -g
LD      := gcc
LDLIBS  := ${LDLIBS} -lrdmacm -libverbs -lpthread

mlx5_path := /home/nurlan/Desktop/rdma-core-installed_use_this_one
nvidia_driver := /home/nurlan/Downloads/NVIDIA-Linux-x86_64-470.223.02
cuda_memalloc = /usr/lib/x86_64-linux-gnu
add_flags := -I $(mlx5_path)/providers/mlx5 -I $(mlx5_path)/build/include/ -I $(mlx5_path)/buildlib/sparse-include/ -I$(cuda_memalloc)

CUDA_ROOT:=/usr/local/cuda-11.4
CUDA_LIBS= -I $(CUDA_ROOT)/lib64 #-I $(nvidia_driver)/kernel/nvidia
CUDA_INCLUDE=-I $(CUDA_ROOT)/include -I$(CUDA_SDK_ROOT)/C/common/inc 
NVCC=$(CUDA_ROOT)/bin/nvcc -g -G --generate-code code=sm_35,arch=compute_35 $(CUDA_INCLUDE)

APPS    := rdma-client rdma-server cpu-client gpu-client

all: ${APPS}
gpu-client:  gpu-client.o
	$(NVCC) $^ -libverbs -lrdmacm -lcuda -o $@ $(add_flags)

# gpu-client.o: gpu-utils.h

# gpu-utils.o: gpu-utils.c
# 	$(CC) $(CFLAGS) $(LIBS) -g -c gpu-utils.c

rdma-client: client-utils.o rdma-client.o
	${LD} -o $@ $^ ${LDLIBS}

rdma-server: server-utils.o rdma-server.o
	${LD} -o $@ $^ ${LDLIBS}

cpu-client: cpu-client.o gpu-utils.o
	${LD} -o $@ $^ ${LDLIBS} $(add_flags)

gpu-client.o: gpu-client.cu
	$(NVCC) -dc gpu-client.cu -o gpu-client.o -libverbs -lrdmacm -lcuda -lcudart $(CUDA_LIBS) $(add_flags)
	
# $(add_flags) $(CUDA_LIBS) 

clean:
	rm -f *.o ${APPS}

