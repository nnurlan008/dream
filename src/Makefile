.PHONY: clean

CFLAGS  := -Wall  -g
LD      := gcc
LDLIBS  := ${LDLIBS} -lrdmacm -libverbs -lpthread

# folder_path := /users/Nurlan/gpu_rdma_read_write
# freest := -I $(folder_path)/freestanding/include 
mlx5_path := /users/Nurlan/rdma-core-installed_use_this_one
# nvidia_driver := /home/nurlan/Downloads/NVIDIA-Linux-x86_64-470.223.02
cuda_memalloc = /usr/lib/x86_64-linux-gnu
add_flags := -I $(mlx5_path)/providers/mlx5 -I $(mlx5_path)/build/include/ -I $(mlx5_path)/buildlib/sparse-include/ -I$(cuda_memalloc) 

CUDA_ROOT:=/usr/local/cuda-12.2
CUDA_LIBS= -I $(CUDA_ROOT)/lib64 #-I $(nvidia_driver)/kernel/nvidia
CUDA_INCLUDE=-I $(CUDA_ROOT)/include -I$(CUDA_SDK_ROOT)/C/common/inc 
NVCC2=$(CUDA_ROOT)/bin/nvcc -g -G --generate-code code=sm_35,arch=compute_35 $(CUDA_INCLUDE)
NVCC=$(CUDA_ROOT)/bin/nvcc -g -G --generate-code code=sm_70,arch=compute_70 $(CUDA_INCLUDE)

CXXARGS = -w -std=c++14 -g -arch=sm_70

APPS    := server rdma_utils.o

all: ${APPS}

rdma_utils:  rdma_utils.o
	$(NVCC) $^ -libverbs -lrdmacm -lcuda -o $@ $(add_flags)

# server-utils.o: server-utils.c
# 	${LD}  server-utils.c -o server-utils.o ${LDLIBS}

rdma_utils.o: rdma_utils.cuh

server: server-utils.o server.o
	${LD} -o $@ $^ ${LDLIBS}

# temp_cuda.o: temp_cuda.cu 
# 	$(NVCC) -dc temp_cuda.cu -o temp_cuda.o -libverbs -lrdmacm -lcuda -lcudart $(CUDA_LIBS)

rdma_utils.o: rdma_utils.cu 
	$(NVCC) -dc rdma_utils.cu -o rdma_utils.o -libverbs -lrdmacm -lcuda -lcudart $(CUDA_LIBS) $(add_flags)

clean:
	rm -f *.o ${APPS}