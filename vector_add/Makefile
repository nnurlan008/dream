# NVCCCFLAGS =  -std=c++11 -lcuda -O0
# NVCC = /usr/local/cuda-11.4/bin/nvcc



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

CXXARGS = -w -std=c++14 -g -arch=sm_35

APPS    := main server

all: ${APPS}

main:  main.o rdma_utils.o
	$(NVCC) $^ -libverbs -lrdmacm -lcuda -o $@ $(add_flags)

rdma_utils:  rdma_utils.o
	$(NVCC) $^ -libverbs -lrdmacm -lcuda -o $@ $(add_flags)

# server-utils.o: server-utils.c
# 	${LD}  server-utils.c -o server-utils.o ${LDLIBS}

main.o: rdma_utils.cuh

server: server-utils.o server.o
	${LD} -o $@ $^ ${LDLIBS}

main.o: main.cu 
	$(NVCC) -dc main.cu -o main.o -libverbs -lrdmacm -lcuda -lcudart $(CUDA_LIBS) $(add_flags)

rdma_utils.o: rdma_utils.cu 
	$(NVCC) -dc rdma_utils.cu -o rdma_utils.o -libverbs -lrdmacm -lcuda -lcudart $(CUDA_LIBS) $(add_flags)



clean:
	rm -f *.o ${APPS}

