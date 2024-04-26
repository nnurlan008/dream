#include <cuda.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <stdbool.h>
#include <cuda_profiler_api.h>
#include <vector>
#include <map>
#include <bits/stdc++.h> 
#include <stdio.h>

//#include "helper.h"
//#include "definitions.h"
//#include "EvictionSetGenerator.cuh"

#define stride 32
#define EvictionBufferSize (1*1024*1024)
#define EvictionNumbers numHashedAddr
#define dataType uint32_t
#define dummyType float
#define timeType float

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
        if (code != cudaSuccess)
        {
                fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
                if (abort) exit(code);
        }
}






__global__ void pointerChaseRead(dataType* p,
                                long long int* times,
                                 dummyType* dumy)
{
    long long int start, end;
    
    //printf("Hello\n");
    dataType tempDummy = 0;
    dummyType dummysum=0;
    
    int i=0;
    //printf("tempDummy: %lld\n",tempDummy);
    //printf("p[tempDummy]: %d\n",p[tempDummy]);
    while(tempDummy != (dataType)-1)
    {
        start = clock();
        
        tempDummy = p[tempDummy];
        dummysum = dummysum + tempDummy + start;
        end = clock();
        times[i] = end - start;
        printf("end-start: %lld\n",end -start);
        i++;
    }

    *dumy = (dummyType) dummysum;
    
    
}



int main()
{
    long long int* times;
    dummyType* dummy;
    long long int* times_host;
    dummyType* dummy_host;

    FILE* fptr;
    char line[100];
    dataType* p;

    

    int cuda_device_id = 0;
	int cuda_pci_bus_id;
	int cuda_pci_device_id;
	int index;
	CUdevice cu_device;
	CUdevice cuDevice_selected;

	// printf("initializing CUDA\n");
	CUresult error = cuInit(0);
	if (error != CUDA_SUCCESS) {
		// printf("cuInit(0) returned %d\n", error);
		return -1;
	}
  // printf("Function name: %s, line number: %d cqe64->op_own >> 4: %d\n", __func__, __LINE__, cqe64->op_own >> 4);
	int deviceCount = 0;
	error = cuDeviceGetCount(&deviceCount);
	if (error != CUDA_SUCCESS) {
		printf("cuDeviceGetCount() returned %d\n", error);
		return -1;
	}
	/* This function call returns 0 if there are no CUDA capable devices. */
	if (deviceCount == 0) {
		printf("There are no available device(s) that support CUDA\n");
		return -1;
	}
  // printf("Function name: %s, line number: %d cqe64->op_own >> 4: %d\n", __func__, __LINE__, cqe64->op_own >> 4);
	printf("Listing all CUDA devices in system:\n");
	for (int index = 0; index < deviceCount; index++) {
		if(cuDeviceGet(&cu_device, index) != CUDA_SUCCESS) exit(0);
		cuDeviceGetAttribute(&cuda_pci_bus_id, CU_DEVICE_ATTRIBUTE_PCI_BUS_ID , cu_device);
		cuDeviceGetAttribute(&cuda_pci_device_id, CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID , cu_device);
		printf("CUDA device %d: PCIe address is %02X:%02X\n", index, (unsigned int)cuda_pci_bus_id, (unsigned int)cuda_pci_device_id);
	}

	if(cuDeviceGet(&cuDevice_selected, cuda_device_id) != cudaSuccess){
		exit(0);
	}
	char name[128];
	if(cuDeviceGetName(name, sizeof(name), cuda_device_id) != cudaSuccess){
		exit(0);
	}
    printf("GPU: %s\n", name);


    gpuErrchk(cudaMalloc((void **)&times,((EvictionBufferSize)/(stride))*sizeof(long long int)));
    gpuErrchk(cudaMalloc((void **)&dummy,sizeof(dummyType)));

    times_host = (long long int*) malloc((EvictionBufferSize)/(stride)*sizeof(long long int));
    dummy_host = (dummyType *) malloc(sizeof(dummyType));

    __clock_t start, end;
    start = clock();
    cudaMallocManaged((void **)&p,EvictionBufferSize);
    end = clock();
    
    
    for(int i=0; i<(EvictionBufferSize)/(stride);i++)
    {
        p[i*stride/sizeof(dataType)] = (i+1)*stride/sizeof(dataType);
        if(i == (EvictionBufferSize)/(stride)-1)
        {
            p[i*stride/sizeof(dataType)] = (dataType)-1;
        }

    }
    dataType temp=0;
    for(int i=0; i< 16; i++)
    {
        temp = p[temp];
        printf("temp: %d\n",temp);
    }
    //gpuErrchk(cudaMemPrefetchAsync(p, EvictionBufferSize, 0 , 0));
    pointerChaseRead<<<1,1>>>(p,times,dummy);
    cudaDeviceSynchronize();
    gpuErrchk(cudaMemcpy(times_host,times,(EvictionBufferSize)/(stride)*sizeof(long long int),cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(dummy_host,dummy,sizeof(dummyType),cudaMemcpyDeviceToHost));
    fptr = fopen("uvm_exp_access_latency.txt","w");
    for(int i=0; i<(EvictionBufferSize)/(stride) ;i++)
    {
        sprintf(line,"%lld\n",times_host[i]);
        fputs(line, fptr);
    }
    fclose(fptr);
    
    return 0;
}
