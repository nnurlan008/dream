.PHONY: clean

CFLAGS  := -Wall -g
LD      := gcc
LDLIBS  := ${LDLIBS} -lrdmacm -libverbs -lpthread

# folder_path := /users/Nurlan/gpu_rdma_read_write
# freest := -I $(folder_path)/freestanding/include 
mlx5_path := #/users/Nurlan/rdma-core-installed_use_this_one
# nvidia_driver := /home/nurlan/Downloads/NVIDIA-Linux-x86_64-470.223.02
cuda_memalloc = /usr/lib/x86_64-linux-gnu
add_flags := -I $(mlx5_path)/providers/mlx5 -I $(mlx5_path)/build/include/ -I $(mlx5_path)/buildlib/sparse-include/ -I$(cuda_memalloc) 

CXX := /usr/bin/g++
CC := /usr/bin/gcc

CUDA_ROOT := /usr/local/cuda-12.2
CUDA_LIBS := -I $(CUDA_ROOT)/lib64 -libverbs -lrdmacm -lcuda -lcudart #-I $(nvidia_driver)/kernel/nvidia
CUDA_INCLUDE := -I$(CUDA_ROOT)/include -I$(CUDA_SDK_ROOT)/C/common/inc 
NVCC2 := $(CUDA_ROOT)/bin/nvcc -g -G --generate-code code=sm_35,arch=compute_35 $(CUDA_INCLUDE)
NVCC := $(CUDA_ROOT)/bin/nvcc -g -G --generate-code code=sm_70,arch=compute_70 $(CUDA_INCLUDE)

free_standing1 := /users/Nurlan/gpu_rdma_read_write/include/freestanding/include

INCLUDE_DIR := /users/Nurlan/gpu_rdma_read_write/include
SRC_DIR := /users/Nurlan/gpu_rdma_read_write/src

CXXFLAGS = -w -std=c++14 -O3 -Wno-deprecated-gpu-targets
# -arch=sm_70

PWD := $(shell pwd)

APPS := main

all: ${APPS}

# bfsCPU.o: bfsCPU.cpp bfsCPU.h
# 	$(CXX) -c bfsCPU.cpp -o bfsCPU.o $(add_flags) $(CXXFLAGS)  

readGraph.o: readGraph.cpp graph.h
	$(CXX) -c readGraph.cpp -o readGraph.o $(add_flags) $(CXXFLAGS)

main: main.o   $(SRC_DIR)/rdma_utils.o readGraph.o
	$(NVCC) $^ -libverbs -lrdmacm -lcuda -lcudart -o $@ $(add_flags)  $(CXXFLAGS)  -I $(INCLUDE_DIR) -I $(SRC_DIR) -I $(free_standing1)

main.o: main.cu  graph.h 
<<<<<<< HEAD
	$(NVCC) -dc -Xptxas -v main.cu -o main.o $(CUDA_LIBS) $(add_flags) $(CXXFLAGS)  -I $(INCLUDE_DIR) -I $(SRC_DIR) -I $(free_standing1)
=======
	$(NVCC) -dc main.cu -o main.o $(CUDA_LIBS) $(add_flags) $(CXXFLAGS)  -I $(INCLUDE_DIR) -I $(SRC_DIR) -I $(free_standing1)
>>>>>>> origin/cloudlab

# bfs.o: bfs.cu bfs.cuh
# 	$(NVCC) -dc bfs.cu -o bfs.o -libverbs -lrdmacm -lcuda -lcudart $(add_flags) $(CUDA_LIBS) 

  

clean:
	rm -f *.o ${APPS}