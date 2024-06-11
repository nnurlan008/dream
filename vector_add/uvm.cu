#include <stdio.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <unistd.h>

int init_gpu(int gpu){
    int cuda_device_id = 0;
	int cuda_pci_bus_id;
	int cuda_pci_device_id;
	int index;
	CUdevice cu_device;
	CUdevice cuDevice_selected;

	printf("initializing CUDA...\n");
	CUresult error = cuInit(gpu);
	if (error != CUDA_SUCCESS) {
		printf("cuInit(0) returned %d\n", error);
		return -1;
	}
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
  
	for (int index = 0; index < deviceCount; index++) {
		if(cuDeviceGet(&cu_device, index) != CUDA_SUCCESS) return -1;
		cuDeviceGetAttribute(&cuda_pci_bus_id, CU_DEVICE_ATTRIBUTE_PCI_BUS_ID , cu_device);
		cuDeviceGetAttribute(&cuda_pci_device_id, CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID , cu_device);
		printf("CUDA device %d: PCIe address is %02X:%02X\n", index, (unsigned int)cuda_pci_bus_id, (unsigned int)cuda_pci_device_id);
	}


	if(cuDeviceGet(&cuDevice_selected, cuda_device_id) != cudaSuccess){
		printf("error on cuDeviceGet\n");
		return -1;
	}
	char name[128];
	if(cuDeviceGetName(name, sizeof(name), cuda_device_id) != cudaSuccess){
		printf("error on cuDeviceGetName\n");
		return -1;
	}
	printf("[pid = %d, dev = %d] device name = [%s]\n", getpid(), cuDevice_selected, name);

    size_t free_memory, total_memory;
    if(cudaSuccess != cudaMemGetInfo(&free_memory, &total_memory)){
        printf("error on cudaMemGetInfo\n");
        return -1;
    }
    printf("free memory: %zu, total_memory: %zu\n", free_memory/(1024 * 1024), total_memory/(1024 * 1024));
    return 0;
}


__global__
void initWith(float num, float *a, int N)
{

    int index = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    
        
    for(int i = index; i < N; i += stride)
    {
        a[i] = num;
    }
}

__global__
void add_vectors_UVM(float *result, float *a, float *b, int N)
{
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    
    for(int i = index; i < N; i += stride)
    {
        result[i] = a[i] + b[i];
    }
}

void checkElementsAre(float target, float *vector, int N)
{
    for(int i = 0; i < N; i++)
    {
        if(vector[i] != target)
        {
            printf("FAIL: vector[%d] - %0.0f does not equal %0.0f\n", i, vector[i], target);
            exit(1);
        }
    }
    printf("Success! All values calculated correctly.\n");
}

int main()
{
    int deviceId;
    int numberOfSMs;
    init_gpu(0);
    cudaGetDevice(&deviceId);
    cudaDeviceGetAttribute(&numberOfSMs, cudaDevAttrMultiProcessorCount, deviceId);

    printf("sizeof(float): %d\n", sizeof(float));
    const int N = 256*1024*1024/sizeof(float); // (2<<24)/8;
    size_t size = N * sizeof(float);
    printf("size: %d MB\n", size/1024/1024);
    float *a;
    float *b;
    float *c;

    cudaMallocManaged(&a, size);
    cudaMallocManaged(&b, size);
    cudaMallocManaged(&c, size);

    // cudaMemPrefetchAsync(a, size, deviceId);
    // cudaMemPrefetchAsync(b, size, deviceId);
    // cudaMemPrefetchAsync(c, size, deviceId);

    size_t threadsPerBlock;
    size_t numberOfBlocks;

    threadsPerBlock = 256;
    numberOfBlocks = 2048; // 32 * numberOfSMs;

    cudaError_t addVectorsErr;
    cudaError_t asyncErr;

    for (int i = 0; i < N; i++){
        a[i] = 2;
        b[i] = 2;
    }

    // initWith<<<numberOfBlocks, threadsPerBlock>>>(3, a, N);
    // initWith<<<numberOfBlocks, threadsPerBlock>>>(4, b, N);
    // initWith<<<numberOfBlocks, threadsPerBlock>>>(0, c, N);
    cudaEvent_t event1, event2;
    cudaEventCreate(&event1);
    cudaEventCreate(&event2);


    // clock_gettime(CLOCK_REALTIME, &start);
    cudaEventRecord(event1, (cudaStream_t)0); //where 0 is the default stream

    add_vectors_UVM<<<numberOfBlocks, threadsPerBlock>>>(c, a, b, N);
    cudaEventRecord(event2, (cudaStream_t) 0);
    
    addVectorsErr = cudaGetLastError();
    if(addVectorsErr != cudaSuccess) printf("Error: %s\n", cudaGetErrorString(addVectorsErr));

    asyncErr = cudaDeviceSynchronize();
    if(asyncErr != cudaSuccess) printf("Error: %s\n", cudaGetErrorString(asyncErr));

    cudaEventSynchronize(event1); //optional
    cudaEventSynchronize(event2); //wait for the event to be executed!

    //calculate time
    float dt_ms;
    cudaEventElapsedTime(&dt_ms, event1, event2);
    printf("dt_ms: %f\n", dt_ms);
    // cudaMemPrefetchAsync(c, size, cudaCpuDeviceId);

    checkElementsAre(4, c, N);

    cudaFree(a);
    cudaFree(b);
    cudaFree(c);
}