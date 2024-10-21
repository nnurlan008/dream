#include <cstdio>
#include <cuda.h>
#include <cuda_runtime.h>
#include <string>
#include <cstring>
#include <math.h>
#include <chrono>
#include <stdlib.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

using namespace std;
// using namespace std;


// extern "C"{
//   #include "rdma_utils.h"
// } 

// #include "../../src/rdma_utils.cuh"
#include <time.h>
// #include "../../include/runtime_prefetching.h"
// #include "../../include/runtime_eviction.h"
#include "../../include/runtime_prefetching_2nic.h"

// #include "../../include/runtime_micro.h"

#define MAX_TRIPS 1000000000llu  // Set a maximum number of trips
#define LINE_LENGTH 256   // Set the maximum line length

//define the error threshold for the results "not matching"
#define PERCENT_DIFF_ERROR_THRESHOLD 0.05

//define a small float value
#define SMALL_FLOAT_VAL 0.00000001f

/* Problem size. */
#define NX 4096*16llu
#define NY 4096*16llu

#define BLOCK_NUM 1024ULL
#define MYINFINITY 2147483647llu

#define BLOCK_SIZE 1024
#define WARP_SHIFT 5
#define WARP_SIZE 32

#define THRESHOLD_SECONDS 9000

typedef float DATA_TYPE;

/* Thread block dimensions */
#define DIM_THREAD_BLOCK_X 1024
#define DIM_THREAD_BLOCK_Y 1

#define GPU 0

#ifndef M_PI
#define M_PI 3.14159
#endif

__device__ rdma_buf<unsigned int> D_adjacencyList;

__global__ void test(rdma_buf<unsigned int> *a/*, rdma_buf<int> *b, rdma_buf<int> *c*/);


// Kernel
__global__ void add_vectors_uvm(int *a, int *b, int *c, int size)
{
	int id = blockDim.x * blockIdx.x + threadIdx.x;
	// if(id < size) {
		c[id] = a[id] + b[id];
		// printf("c[%d]: %d\n", id, c[id]);
	// }
}

#define htonl(x)  ((((uint32_t)(x) & 0xff000000) >> 24) |\
                   (((uint32_t)(x) & 0x00ff0000) >>  8) |\
                   (((uint32_t)(x) & 0x0000ff00) <<  8) |\
                   (((uint32_t)(x) & 0x000000ff) << 24))

#define WARP_SIZE 32

void delay(int number_of_seconds)
{
    // Converting time into milli_seconds
    int milli_seconds = 1000000 * number_of_seconds;
 
    // Storing start time
    clock_t start_time = clock();
 
    // looping till required time is not achieved
    while (clock() < start_time + milli_seconds)
        ;
}

enum { NS_PER_SECOND = 1000000000 };

void sub_timespec(struct timespec t1, struct timespec t2, struct timespec *td)
{
    td->tv_nsec = t2.tv_nsec - t1.tv_nsec;
    td->tv_sec  = t2.tv_sec - t1.tv_sec;
    if (td->tv_sec > 0 && td->tv_nsec < 0)
    {
        td->tv_nsec += NS_PER_SECOND;
        td->tv_sec--;
    }
    else if (td->tv_sec < 0 && td->tv_nsec > 0)
    {
        td->tv_nsec -= NS_PER_SECOND;
        td->tv_sec++;
    }
}

__device__ float AtomicAdd(float *address, float value) {
    // Convert address to integer representation (since atomicCAS works on integers)
    uint32_t *address_as_int = (uint32_t *)address;
    uint32_t old = *address_as_int, assumed;

    // Loop to perform atomic addition
    do {
        assumed = old;
        // Convert the integer bits back to a float and perform the addition
        float old_f = __int_as_float(assumed);
        float new_f = old_f + value;

        // Use atomicCAS to try and set the new value (converted to int)
        old = atomicCAS(address_as_int, assumed, __float_as_int(new_f));
    } while (assumed != old);  // Retry if the value changed during the process

    // Return the old value (before the addition)
    return __int_as_float(old);
}


void usage(const char *argv0)
{
  fprintf(stderr, "usage: %s <mode> <server-address> <server-port>\n  mode = \"read\", \"write\"\n", argv0);
  exit(1);
}


#define check_cuda_error(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }

}

__global__ void transfer(size_t size, rdma_buf<DATA_TYPE> *d_adjacencyList)
{
    size_t id = blockDim.x * blockIdx.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;
    
        for (size_t i = id; i < size ; i += stride)
        {
            DATA_TYPE y = (*d_adjacencyList)[i];
        }
}

__global__ void check(size_t size, rdma_buf<DATA_TYPE> *d_adjacencyList, DATA_TYPE *a)
{
    size_t id = blockDim.x * blockIdx.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;
    
        for (size_t i = id; i < size ; i += stride)
        {
            DATA_TYPE y = (*d_adjacencyList)[i];
            if(a[i] != y){
                printf("y: %f %f ", y, a[i]);
            }
        }
}

__global__ void assign_array(rdma_buf<unsigned int> *adjacencyList){
    D_adjacencyList = *adjacencyList;
    printf("D_adjacencyList.d_TLB[0].state: %d\n", D_adjacencyList.d_TLB[0].state);
    printf("D_adjacencyList.d_TLB[0].device_address: %p\n", D_adjacencyList.d_TLB[0].device_address);
}

int alloc_global_cont(struct post_content *post_cont, struct poll_content *poll_cont, struct server_content_2nic *post_cont2, 
                      struct gpu_memory_info gpu_mem){
    struct post_content *d_post;
    struct poll_content *d_poll;
    struct server_content_2nic *d_post2;

    cudaError_t ret0 = cudaMalloc((void **)&d_post, sizeof(struct post_content));
    if(ret0 != cudaSuccess){
        printf("Error on allocation post content!\n");
        return -1;
    }
    ret0 = cudaMalloc((void **)&d_poll, sizeof(struct poll_content));
    if(ret0 != cudaSuccess){
        printf("Error on allocation poll content!\n");
        return -1;
    }
    printf("sizeof(struct post_content): %d, sizeof(struct poll_content): %d\n", sizeof(struct post_content), sizeof(struct poll_content));
    ret0 = cudaMemcpy(d_post, post_cont, sizeof(struct post_content), cudaMemcpyHostToDevice);
    if(ret0 != cudaSuccess){
        printf("Error on post copy!\n");
        return -1;
    }
    ret0 = cudaMemcpy(d_poll, poll_cont, sizeof(struct poll_content), cudaMemcpyHostToDevice);
    if(ret0 != cudaSuccess){
        printf("Error on poll copy!\n");
        return -1;
    }

    ret0 = cudaMalloc((void **)&d_post2, sizeof(struct server_content_2nic));
    if(ret0 != cudaSuccess){
        printf("Error on allocation post content!\n");
        return -1;
    }
    ret0 = cudaMemcpy(d_post2, post_cont2, sizeof(struct server_content_2nic), cudaMemcpyHostToDevice);
    if(ret0 != cudaSuccess){
        printf("Error on poll copy!\n");
        return -1;
    }

    // cudaSetDevice(0);
    alloc_content<<<1,1>>>(d_post, d_poll);
    alloc_global_content<<<1,1>>>(d_post, d_poll, d_post2, gpu_mem);
    ret0 = cudaDeviceSynchronize();
    if(ret0 != cudaSuccess){
        printf("Error on alloc_content!\n");
        return -1;
    }
    return 0;
}


__device__ size_t sum_page_faults = 0;

__global__ void
print_retires(void){
    // size_t max = cq_wait[0];
    // for (size_t i = 0; i < 128; i++)
    // {
    //     if(max < cq_wait[i]) max = cq_wait[i];
    // }
    sum_page_faults += g_qp_index;
    printf("g_qp_index: %llu sum page fault: %llu\n", g_qp_index, sum_page_faults);
    g_qp_index = 0;
    // for (size_t i = 0; i < 128; i++)
    // {
    //     max = 0;
    // }
}

__global__ // __launch_bounds__(1024,2) 
void
calculate_opt(size_t n, size_t size, rdma_buf<unsigned int> *rdma_array, unsigned int *array) {

    // Page size in elements (64KB / 4 bytes per unsigned int)
    const size_t pageSize = 8*1024 / sizeof(unsigned int);
    // Elements per warp
    const size_t elementsPerWarp = pageSize / warpSize;

    // Global thread ID
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    // if(tid == 0) printf("warpSize: %d\n", warpSize);
    // Warp ID within the block
    size_t warpId = tid / warpSize;

    // Thread lane within the warp
    size_t lane = threadIdx.x % warpSize;

    // Determine which page this warp will process
    size_t pageStart = warpId * pageSize;

    // Ensure we don't process out-of-bounds pages
    if (pageStart < n * pageSize) {
        
        // Process elements within the page
        // for (size_t i = 0; i < elementsPerWarp; ++i) {
        //     size_t elementIdx = pageStart + lane + i * warpSize;
            uint end = (warpId + 1)*pageSize > size ? size : (warpId + 1)*pageSize;
            for(size_t j = warpId*pageSize + lane; j < end; j += warpSize) {
                uint end_edge = (*rdma_array)[j]; // shared_data[j - pageStart];
                array[j] = end_edge;
            }
        // }
    }
}


__global__ void transfer_opt(size_t n, rdma_buf<unsigned int> *rdma_array) {


    // Page size in elements (64KB / 4 bytes per unsigned int)
    const size_t pageSize = 4*1024 / sizeof(unsigned int);
    // Elements per warp
    const size_t elementsPerWarp = pageSize / warpSize;

    // Global thread ID
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    // if(tid == 0) printf("warpSize: %d\n", warpSize);
    // Warp ID within the block
    size_t warpId = tid / warpSize;

    // Thread lane within the warp
    size_t lane = threadIdx.x % warpSize;

    // Determine which page this warp will process
    size_t pageStart = warpId * pageSize;

    // Ensure we don't process out-of-bounds pages
    if (pageStart < n * pageSize) {
        
        // Process elements within the page
        // for (size_t i = 0; i < elementsPerWarp; ++i) {
        //     size_t elementIdx = pageStart + lane + i * warpSize;
                
            for(size_t j = warpId*pageSize + lane; j < (warpId + 1)*pageSize; j += warpSize) {
                uint end_edge = (*rdma_array)[j]; // shared_data[j - pageStart];
            }
        // }
    }
}

void compute_benchmark(){

    cudaError_t ret;
    unsigned int *cuda_array, *h_cuda_array;
    uint64_t numblocks_update, numthreads, numblocks_kernel;
    double avg_milliseconds;
    float milliseconds;
    size_t num_elements, size = 12*1024*1024*1024llu; 
    num_elements = size/sizeof(uint);
    cudaEvent_t start, end;

    rdma_buf<unsigned int> *rdma_array;
    check_cuda_error(cudaMallocManaged((void **) &rdma_array, sizeof(rdma_buf<unsigned int>)));
    rdma_array->start(num_elements *sizeof(unsigned int), GPU, NULL);
    for(size_t i = 0; i < num_elements; i++){
        rdma_array->local_buffer[i] = 14;
    }

    check_cuda_error(cudaEventCreate(&start));
    check_cuda_error(cudaEventCreate(&end));

    h_cuda_array = new uint[num_elements];
    check_cuda_error(cudaMalloc((void **) &cuda_array, size));

    // numblocks_update = ((numVertex + numthreads) / numthreads);
    dim3 blockDim_kernel(numthreads, (numblocks_kernel+numthreads)/numthreads);
    // dim3 blockDim_kernel(BLOCK_SIZE, (numblocks_kernel+BLOCK_SIZE)/BLOCK_SIZE);
    numthreads = BLOCK_SIZE;
    // numblocks_update = ((numVertex + numthreads) / numthreads);
    dim3 blockDim_update(BLOCK_SIZE, (numblocks_update+BLOCK_SIZE)/BLOCK_SIZE);

    avg_milliseconds = 0.0f;

    ret = cudaDeviceSynchronize();
    printf("ret: %d cudaGetLastError: %d\n", ret, cudaGetLastError());
    auto start_chrono = std::chrono::steady_clock::now();
    printf("starting kernel\n");
    size_t n_pages = size/(8*1024);

    check_cuda_error(cudaEventRecord(start, (cudaStream_t) 1));
    numthreads = 1024;
    calculate_opt<<<(n_pages*32)/numthreads+1, numthreads>>>(num_elements, num_elements, rdma_array, cuda_array);
    check_cuda_error(cudaEventRecord(end, (cudaStream_t) 1));

    ret = cudaDeviceSynchronize();               
    auto end_chrono = std::chrono::steady_clock::now();
    long duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_chrono - start_chrono).count();
    print_retires<<<1,1>>>();
    printf("ret: %d cudaGetLastError: %d\n", ret, cudaGetLastError());
    printf("Elapsed time in milliseconds : %li ms\n\n", duration);
    // printf("ret: %d cudaGetLastError: %d\n", ret, cudaGetLastError());
    
    check_cuda_error(cudaMemcpy(h_cuda_array, cuda_array, size, cudaMemcpyDeviceToHost));
    size_t num_errs = 0;
    for (size_t i = 0; i < num_elements; i++)
    {
        if(h_cuda_array[i] != 14){
            num_errs++;
            // printf("i: %d h_cuda_array[i]: %d\n", i, h_cuda_array[i]);
        }
    }
    

    
    check_cuda_error(cudaEventSynchronize(start));
    check_cuda_error(cudaEventSynchronize(end));
    check_cuda_error(cudaEventElapsedTime(&milliseconds, start, end));
    printf("CUDA elapsed time in milliseconds : %0.3f ms num_errs: %llu bw: %.2f REQUEST_SIZE: %d\n\n", milliseconds, num_errs, (float) size/(milliseconds*0.001*1024*1024*1024), REQUEST_SIZE/1024);

}

void transfer_benchmark(){
    cudaError_t ret;
    
    uint64_t numblocks_update, numthreads, numblocks_kernel;
    double avg_milliseconds;
    float milliseconds;
    size_t num_elements, size = 1*1024*1024*1024llu; 
    num_elements = size/sizeof(uint);
    
    cudaEvent_t start, end;

    rdma_buf<unsigned int> *rdma_array;
    check_cuda_error(cudaMallocManaged((void **) &rdma_array, sizeof(rdma_buf<unsigned int>)));

    rdma_array->start(num_elements *sizeof(unsigned int), GPU, NULL);

    for(size_t i = 0; i < num_elements; i++){
        rdma_array->local_buffer[i] = 14;
    }

   
    check_cuda_error(cudaEventCreate(&start));
    check_cuda_error(cudaEventCreate(&end));
   

    // numblocks_update = ((numVertex + numthreads) / numthreads);

    dim3 blockDim_kernel(numthreads, (numblocks_kernel+numthreads)/numthreads);
    // dim3 blockDim_kernel(BLOCK_SIZE, (numblocks_kernel+BLOCK_SIZE)/BLOCK_SIZE);
    numthreads = BLOCK_SIZE;
    // numblocks_update = ((numVertex + numthreads) / numthreads);
    dim3 blockDim_update(BLOCK_SIZE, (numblocks_update+BLOCK_SIZE)/BLOCK_SIZE);

    avg_milliseconds = 0.0f;

    printf("Initialization done\n");
    fflush(stdout);

    
    ret = cudaDeviceSynchronize();
    printf("ret: %d cudaGetLastError: %d\n", ret, cudaGetLastError());
    

    auto start_chrono = std::chrono::steady_clock::now();
    
    printf("starting kernel\n");
    size_t n_pages = size/(4*1024);

    check_cuda_error(cudaEventRecord(start, (cudaStream_t) 1));
    transfer_opt<<<(n_pages*32)/512+1, 512>>>(num_elements, rdma_array);
    check_cuda_error(cudaEventRecord(end, (cudaStream_t) 1));

    ret = cudaDeviceSynchronize();               
    auto end_chrono = std::chrono::steady_clock::now();
    long duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_chrono - start_chrono).count();
    print_retires<<<1,1>>>();
    printf("ret: %d cudaGetLastError: %d\n", ret, cudaGetLastError());
    printf("Elapsed time in milliseconds : %li ms\n\n", duration);
    // printf("ret: %d cudaGetLastError: %d\n", ret, cudaGetLastError());
    

    
    check_cuda_error(cudaEventSynchronize(start));
    check_cuda_error(cudaEventSynchronize(end));
    check_cuda_error(cudaEventElapsedTime(&milliseconds, start, end));
    printf("CUDA elapsed time in milliseconds : %li ms\n\n", milliseconds);

}

// Custom function to convert string to integer
__device__ int my_atoi(const char *str) {
    int result = 0;
    int sign = 1;
    int i = 0;

    // Handle negative numbers
    if (str[0] == '-') {
        sign = -1;
        i++;
    }

    for (; str[i] != '\0' && str[i] != '\n'; i++) {
        result = result * 10 + (str[i] - '0');
    }
    return result * sign;
}

// Custom function to convert string to float
__forceinline__
__device__ float my_atof(const char *str) {
    float result = 0.0f;
    float divisor = 1.0f;
    int i = 0;
    int sign = 1;

    // Handle negative numbers
    if (str[0] == '-') {
        sign = -1;
        i++;
    }

    // Convert integer part
    for (; str[i] != '\0' && str[i] != '.' && str[i] != '\n'; i++) {
        result = result * 10.0f + (str[i] - '0');
    }

    // Convert decimal part
    if (str[i] == '.') {
        i++;
        for (; str[i] != '\0' && str[i] != '\n'; i++) {
            result = result * 10.0f + (str[i] - '0');
            divisor *= 10.0f;  // Increment divisor for each decimal place
        }
    }

    return sign * (result / divisor);
}

__forceinline__
__device__ void parse_trip(const char *line, float &trip_seconds, float &trip_miles, float &fare_amount, float &tip_amount, float &tolls, float &extra) {
    // Manually parse the line
    char buffer[LINE_LENGTH];
    int index = 0;
    int field = 0;
    
    for (int i = 0; line[i] != '\0' && i < LINE_LENGTH; i++) {
        if (line[i] == ',' || line[i] == '\n') {
            buffer[index] = '\0'; // Null-terminate the string
            switch (field) {
                // case 0: // VendorID
                //     *vendorID = my_atoi(buffer);
                //     break;
                // case 6: // Pickup latitude
                //     pickup_latitude = my_atof(buffer);
                //     break;
                // case 5: // Pickup longitude
                //     *pickup_longitude = my_atof(buffer);
                //     break;
                case 4: // Trip Distance
                    trip_seconds = my_atof(buffer);
                    break;
                case 5: // Trip Distance
                    trip_miles = my_atof(buffer);
                    break;
                case 10: // fare_amount
                    fare_amount = my_atof(buffer);
                    break;
                case 11: // tip_amount
                    tip_amount = my_atof(buffer);
                    break;
                case 12: // mta_tax
                    tolls = my_atof(buffer);
                    break;
                case 13: // extra
                    extra = my_atof(buffer);
                    break;
            }
            field++;
            index = 0; // Reset index for next field
        } else {
            buffer[index++] = line[i]; // Collect character into buffer
        }
    }
}

__global__ __launch_bounds__(1024,2)
void process_trips(const char *buffer, int count, float *total_amount, float *total_miles) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        
        // Declare an array to hold the line
        char line[LINE_LENGTH];  // Create a character array for one line

        // Copy the line from the buffer into the line array
        for (int i = 0; i < LINE_LENGTH; ++i) {
            line[i] = buffer[idx * LINE_LENGTH + i];
        }

        int vendorID;
        float trip_seconds;
        float trip_miles;
        float fare_amount;
        float tip_amount;
        float tolls;
        float extra;
        
        // Parse the trip details
        if(idx == 0 || idx == 1 || idx == 2) printf("first line: %s\n", line);
        parse_trip(line, trip_seconds, trip_miles, fare_amount, tip_amount, tolls, extra);

        if(trip_seconds > 1000 && trip_miles < 1050) {
            float local_total = fare_amount - extra - tolls + tip_amount;
            // size_t value = (size_t) local_total;
            // printf("local_total : %llu\n", value);
            // *total_amount += 1;
            // if(trip_distance < 1000){
                // printf("trip_seconds: %f trip_miles: %f, fare_amount: %f, extra: %f tolls: %f tip_amount: %f\nline: %s\n", 
                //         trip_seconds, trip_miles, fare_amount, extra, tolls, tip_amount, line);
            //     // printf("line: %s\n", line);
                AtomicAdd(total_amount, local_total);
                AtomicAdd(total_miles, trip_miles);
            // }
        }

        // Example operation: Store or print the parsed values (printing from device is not recommended)
        // printf("Vendor ID: %d, Pickup Latitude: %f, Pickup Longitude: %f\n", vendorID, pickup_latitude, pickup_longitude);
    }
}

__global__ __launch_bounds__(1024,2)
void process_trips_rdma(rdma_buf<char> *buffer, int count, float *total_amount, float *total_miles) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        
        // Declare an array to hold the line
        char line[LINE_LENGTH];  // Create a character array for one line

        // Copy the line from the buffer into the line array
        for (int i = 0; i < LINE_LENGTH; ++i) {
            line[i] = (*buffer)[idx * LINE_LENGTH + i];
        }

        int vendorID;
        float trip_seconds;
        float trip_miles;
        float fare_amount;
        float tip_amount;
        float tolls;
        float extra;
        
        // Parse the trip details
        // if(idx == 0 || idx == 1 || idx == 2) printf("first line: %s\n", line);
        parse_trip(line, trip_seconds, trip_miles, fare_amount, tip_amount, tolls, extra);

        if(trip_seconds > 1000 && trip_miles < 1050) {
            float local_total = fare_amount - extra - tolls + tip_amount;
            // size_t value = (size_t) local_total;
            // printf("local_total : %llu\n", value);
            // *total_amount += 1;
            // if(trip_distance < 1000){
                // printf("trip_seconds: %f trip_miles: %f, fare_amount: %f, extra: %f tolls: %f tip_amount: %f\nline: %s\n", 
                //         trip_seconds, trip_miles, fare_amount, extra, tolls, tip_amount, line);
            //     // printf("line: %s\n", line);
                AtomicAdd(total_amount, local_total);
                AtomicAdd(total_miles, trip_miles);
            // }
        }

        // Example operation: Store or print the parsed values (printing from device is not recommended)
        // printf("Vendor ID: %d, Pickup Latitude: %f, Pickup Longitude: %f\n", vendorID, pickup_latitude, pickup_longitude);
    }
}

void trips_seconds(char *file){

}

// Function to read CSV data into a single buffer
int read_csv(const char *filename, char *&buffer, size_t *count) {
    FILE *file = fopen(filename, "r");
    if (!file) {
        perror("Failed to open file read_csv");
        return -1;
    }

    fseek(file, 0L, SEEK_END);
    size_t file_size = ftell(file);
    fseek(file, 0L, SEEK_SET);  // Reset to beginning of the file

    // Adjust size based on file size if needed
    size_t max_size = (file_size > (MAX_TRIPS * LINE_LENGTH)) ? file_size : (MAX_TRIPS * LINE_LENGTH);
    printf("max_size: %llu\n", max_size);


    size_t size = MAX_TRIPS * LINE_LENGTH * sizeof(char);
    printf("size: %llu\n", size);
    // buffer = (char *)malloc(size);
    // check_cuda_error(cudaMallocHost(&buffer, size));
    if (!buffer) {
        perror("Failed to allocate memory");
        fclose(file);
        return -1;
    }
    printf("size1: %llu\n", size);
    *count = 0;
    while (fgets(buffer + (*count * LINE_LENGTH), LINE_LENGTH, file) && *count < MAX_TRIPS) {
        (*count)++;
    }
    printf("size2: %llu\n", size);
    fclose(file);
    return 0;
}

int rapids_CUDA(char *file){
    cudaEvent_t event1, event2;
    cudaEventCreate(&event1);
    cudaEventCreate(&event2);
    cudaError_t ret;
    char *buffer, *d_buffer;
    size_t count;
    float *total_amount, *h_total, h_miles, *d_miles;
    size_t size = MAX_TRIPS * LINE_LENGTH * sizeof(char);
    h_total = (float *) malloc(sizeof(float));
    buffer = (char *)malloc(size);
    // cudaError_t err = cudaMallocHost((void**)&buffer, size);
    printf("line: %d\n", __LINE__);
    // buffer[0] = 12;
    // if (err != cudaSuccess) {
    //     printf("cudaMallocHost failed: %s\n", cudaGetErrorString(err));
    //     return -1;
    // }
    // Step 1: Read CSV into a single buffer
    if (read_csv(file, buffer, &count) != 0) {
        return -1;
    }

    printf("count: %llu\n", count);

    cudaDeviceSynchronize();
    auto start = std::chrono::steady_clock::now();                
    cudaEventRecord(event1, (cudaStream_t)1);

    // Step 2: Allocate device memory
    check_cuda_error(cudaMallocManaged((void **)&d_buffer, count * LINE_LENGTH * sizeof(char)));
    memcpy(d_buffer, buffer, count * LINE_LENGTH * sizeof(char));
    check_cuda_error(cudaMalloc((void **)&total_amount, sizeof(float)));
    check_cuda_error(cudaMalloc((void **)&d_miles, sizeof(float)));
    check_cuda_error(cudaMemset(total_amount, 0, sizeof(float)));

    // Step 3: Copy data to device
    // check_cuda_error(cudaMemcpy(d_buffer, buffer, count * LINE_LENGTH * sizeof(char), cudaMemcpyHostToDevice));

    // Step 4: Launch kernel
    int threadsPerBlock = 1024;
    int blocksPerGrid = (count + threadsPerBlock - 1) / threadsPerBlock;
    ret = cudaDeviceSynchronize();
    printf("ret: %d cudaGetLastError(): %d\n", ret, cudaGetLastError());
    process_trips<<<blocksPerGrid, threadsPerBlock>>>(d_buffer, count, total_amount, d_miles);
    ret = cudaDeviceSynchronize();
    printf("ret: %d cudaGetLastError(): %d\n", ret, cudaGetLastError());

    check_cuda_error(cudaMemcpy(h_total, total_amount, sizeof(float), cudaMemcpyDeviceToHost));
    check_cuda_error(cudaMemcpy(&h_miles, d_miles, sizeof(float), cudaMemcpyDeviceToHost));
    cudaDeviceSynchronize();

    cudaEventRecord(event2, (cudaStream_t) 1);
            
    cudaEventSynchronize(event1); //optional
    cudaEventSynchronize(event2); //wait for the event to be executed!

    auto end = std::chrono::steady_clock::now();
    long duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    printf("Elapsed time for direct transfer  ms : %li ms.\n\n", duration);
    
    float dt_ms;
    cudaEventElapsedTime(&dt_ms, event1, event2);
    printf("The execution time with direct transfer on GPU: %f ms\n", dt_ms);

    printf("h_total: %f\n", *h_total);
    printf("h_miles: %f\n", h_miles);

    // Step 5: Free memory
    free(buffer);
    free(h_total);
    cudaFree(d_buffer);
    cudaFree(total_amount);

    return 0;
}

int rapids_RDMA(char *filename){

    rdma_buf<char> *rdma_buffer;
    size_t count;
    check_cuda_error(cudaMallocManaged((void **)&rdma_buffer, sizeof(rdma_buf<char>)));
    
    // char *buffer;
    FILE *file = fopen(filename, "r");
    if (!file) {
        perror("Failed to open file rapids_RDMA");
        return -1;
    }
    size_t size = MAX_TRIPS * LINE_LENGTH * sizeof(char);
    printf("size: %llu\n", size);
    rdma_buffer->start(size, GPU, NULL);
    printf("size0: %llu\n", size);
    // buffer = (char *) malloc(size);
    // if (!buffer) {
    //     perror("Failed to allocate memory");
    //     fclose(file);
    //     return -1;
    // }
    
    printf("size1: %llu\n", size);

    count = 0;
    while(fgets(rdma_buffer->local_buffer + (count * LINE_LENGTH), LINE_LENGTH, file) && count < MAX_TRIPS) {
        (count)++;
    }
    printf("size2: %llu\n", size);
    fclose(file);
    // return 0;




    cudaEvent_t event1, event2;
    cudaEventCreate(&event1);
    cudaEventCreate(&event2);
    cudaError_t ret;
    
    
    float *total_amount, *h_total, *d_miles, h_miles;
    h_total = (float *) malloc(sizeof(float));

    // Step 1: Read CSV into a single buffer
    // if (read_csv(file, buffer, &count) != 0) {
    //     return -1;
    // }

    printf("count: %llu\n", count);

    auto start = std::chrono::steady_clock::now();                
    cudaEventRecord(event1, (cudaStream_t)1);

    // Step 2: Allocate device memory
    // check_cuda_error(cudaMalloc((void **)&d_buffer, count * LINE_LENGTH * sizeof(char)));
    check_cuda_error(cudaMalloc((void **)&d_miles, sizeof(float)));
    check_cuda_error(cudaMalloc((void **)&total_amount, sizeof(float)));
    check_cuda_error(cudaMemset(total_amount, 0, sizeof(float)));

    // Step 3: Copy data to device
    // check_cuda_error(cudaMemcpy(d_buffer, buffer, count * LINE_LENGTH * sizeof(char), cudaMemcpyHostToDevice));

    // Step 4: Launch kernel
    int threadsPerBlock = 1024;
    int blocksPerGrid = (count + threadsPerBlock - 1) / threadsPerBlock;
    ret = cudaDeviceSynchronize();
    printf("ret: %d cudaGetLastError(): %d\n", ret, cudaGetLastError());
    process_trips_rdma<<<blocksPerGrid, threadsPerBlock>>>(rdma_buffer, count, total_amount, d_miles);
    ret = cudaDeviceSynchronize();
    printf("ret: %d cudaGetLastError(): %d\n", ret, cudaGetLastError());

    check_cuda_error(cudaMemcpy(h_total, total_amount, sizeof(float), cudaMemcpyDeviceToHost));
    check_cuda_error(cudaMemcpy(&h_miles, d_miles, sizeof(float), cudaMemcpyDeviceToHost));
    cudaDeviceSynchronize();

    cudaEventRecord(event2, (cudaStream_t) 1);
            
    cudaEventSynchronize(event1); //optional
    cudaEventSynchronize(event2); //wait for the event to be executed!

    auto end = std::chrono::steady_clock::now();
    long duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    printf("Elapsed time for rdma transfer  ms : %li ms.\n\n", duration);
    
    float dt_ms;
    cudaEventElapsedTime(&dt_ms, event1, event2);
    printf("The cudaEvent execution time with rdma on GPU: %f ms\n", dt_ms);

    printf("h_total: %f\n", *h_total);
    printf("h_miles: %f\n", h_miles);

    // Step 5: Free memory
    // free(buffer);
    // cudaFree(d_buffer);

    return 0;
}

// __global__ __launch_bounds__(1024,2)
// void process_trips_uvm_direct(float *uvm_file, int count, float *total_amount, float *total_miles) {
//     size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
//     if (idx < count) {
        
//         // Declare an array to hold the line
//         // char line[LINE_LENGTH];  // Create a character array for one line

//         // int vendorID;
//         float trip_seconds = uvm_file[idx];
        
//         // parse_trip(line, trip_seconds, trip_miles, fare_amount, tip_amount, tolls, extra);
//         // trip_miles < 1050

//         if(trip_seconds > 1000) {

//             // printf("trip_seconds: %f\n", trip_seconds);

//             float trip_miles  = uvm_file[5*idx  + count + 0];
//             float fare_amount = uvm_file[5*idx + count + 1];
//             float tip_amount  = uvm_file[5*idx + count + 2];
//             float tolls       = uvm_file[5*idx + count + 3];
//             float extra       = uvm_file[5*idx + count + 4];

//             float local_total = fare_amount; // - extra - tolls + tip_amount;
            
//                 AtomicAdd(total_amount, local_total);
//                 AtomicAdd(total_miles, trip_miles);
//         }

//         // Example operation: Store or print the parsed values (printing from device is not recommended)
//         // printf("Vendor ID: %d, Pickup Latitude: %f, Pickup Longitude: %f\n", vendorID, pickup_latitude, pickup_longitude);
//     }
// }

__global__ __launch_bounds__(1024,2)
void process_trips_uvm_direct_trip_miles(float *uvm_file, size_t n, size_t count, float *total_amount, float *total_miles, int *array, float *sum) {


    // // Page size in elements (64KB / 4 bytes per unsigned int)
    // const size_t pageSize = REQUEST_SIZE / sizeof(float);
    // // Elements per warp
    // const size_t elementsPerWarp = pageSize / WARP_SIZE;

    // // Global thread ID
    // size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    // // if(tid == 0) printf("warpSize: %d\n", warpSize);
    // // Warp ID within the block
    // size_t warpId = tid / WARP_SIZE;

    // // Thread lane within the warp
    // size_t lane = threadIdx.x % WARP_SIZE; // warpSize;

    // // Determine which page this warp will process
    // size_t pageStart = warpId * pageSize;

    // // Ensure we don't process out-of-bounds pages
    // if (pageStart < n * pageSize) {
        
    //     // Process elements within the page
    //     // for (size_t i = 0; i < elementsPerWarp; ++i) {
    //     //     size_t elementIdx = pageStart + lane + i * warpSize;
    //         size_t end = (warpId + 1)*pageSize > count ? count : (warpId + 1)*pageSize;
    //         for(size_t j = warpId*pageSize + lane; j < end; j += WARP_SIZE) {
    //             float trip_seconds = uvm_file[j];
    //             if(trip_seconds > THRESHOLD_SECONDS) {
    //                 // 
    //                 // if(trip_miles < 3460.0){
    //                     // size_t index = 5*j + count;
    //                     // float trip_miles  = uvm_file[index ];
    //                     // float fare_amount = uvm_file[index + 1];     
    //                     // float extra       = uvm_file[index + 4];
    //                     // float tip_amount  = uvm_file[index + 2];
    //                     // float tolls       = uvm_file[index + 3];

    //                     float trip_miles  = uvm_file[j + count];
    //                     // float fare_amount = uvm_file[j + count*2];
                        
    //                     // float extra       = uvm_file[j + count*5];
    //                     // float tip_amount  = uvm_file[j + count*3];
    //                     // float tolls       = uvm_file[j + count*4];
                        
                        

    //                     // float local_total = fare_amount; // - extra; // + tip_amount; // - tolls;
    //                     // sum[j] += local_total;
    //                     total_miles[j] += trip_miles;
                        
    //                     // atomicAdd(total_amount, local_total);
    //                     // atomicAdd(total_miles, trip_miles);
    //                     // array[j] = 1;
    //                     // atomicAdd(ones, 1);
    //                 // }
    //             }
    //             // else{
    //             //     printf("trip_seconds: %f\n", trip_seconds);
    //             // }
    //         }
    //     // }
    // }


    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        float trip_seconds = uvm_file[idx];
        if(trip_seconds > THRESHOLD_SECONDS) {
            float trip_miles  = uvm_file[idx  + count];
            float fare_amount = uvm_file[idx + count*2];
            float extra       = uvm_file[idx + count*5];
            float tip_amount  = uvm_file[idx + count*3];
            float tolls       = uvm_file[idx + count*4];

            float local_total = fare_amount - extra + tip_amount - tolls;
            sum[idx] += local_total;
            total_miles[idx] += trip_miles;
        }
    }
}

__global__ // __launch_bounds__(1024,2)
void process_trips_uvm_direct_trip_miles2(float *uvm_seconds, float *uvm_miles, float *uvm_fare, float *uvm_extra, float *uvm_tips, float *uvm_tolls, 
                    size_t n, size_t count, float *total_amount, float *total_miles, int *array, float *sum) {

    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        float trip_seconds = uvm_seconds[idx];
        if(trip_seconds > THRESHOLD_SECONDS) {
            float trip_miles  = uvm_miles[idx];
            float fare_amount = uvm_fare[idx];
            float extra       = uvm_extra[idx];
            float tip_amount  = uvm_tips[idx];
            float tolls       = uvm_tolls[idx];

            float local_total = fare_amount - extra + tip_amount - tolls;
            sum[idx] += local_total;
            total_miles[idx] += trip_miles;
        }
    }
}


int read_bin_uvm(char *filename, size_t &max1, float *&seconds, float *&miles, float *&fare, float *&extra, float *&tips, float *&tolls){
    size_t max = 211670894llu;
    max1 = max;
    float *trip_seconds; // [MAX_TRIPS];
    float *trip_miles;
    float *fare_amount;
    float *tip_amount;
    float *tolls_amount;
    float *extra_amount;
    trip_seconds = (float *) malloc(sizeof(float)*max);
    trip_miles = (float *) malloc(sizeof(float)*max);
    fare_amount = (float *) malloc(sizeof(float)*max);
    tip_amount = (float *) malloc(sizeof(float)*max);
    tolls_amount = (float *) malloc(sizeof(float)*max);
    extra_amount = (float *) malloc(sizeof(float)*max);
    

    size_t res;
    size_t size = max*sizeof(float);
    check_cuda_error(cudaMallocManaged((void **)&seconds, size));
    check_cuda_error(cudaMallocManaged((void **)&miles, size));
    check_cuda_error(cudaMallocManaged((void **)&fare, size));
    check_cuda_error(cudaMallocManaged((void **)&extra, size));
    check_cuda_error(cudaMallocManaged((void **)&tips, size));
    check_cuda_error(cudaMallocManaged((void **)&tolls, size));

    // check_cuda_error(cudaMemAdvise(seconds, size, cudaMemAdviseSetReadMostly, 0));
    // check_cuda_error(cudaMemAdvise(miles, size, cudaMemAdviseSetReadMostly, 0));
    // check_cuda_error(cudaMemAdvise(fare, size, cudaMemAdviseSetReadMostly, 0));
    // check_cuda_error(cudaMemAdvise(extra, size, cudaMemAdviseSetReadMostly, 0));
    // check_cuda_error(cudaMemAdvise(tips, size, cudaMemAdviseSetReadMostly, 0));
    // check_cuda_error(cudaMemAdvise(tolls, size, cudaMemAdviseSetReadMostly, 0));

    auto start = std::chrono::steady_clock::now();
    
    FILE *file = fopen(filename, "rb");
    if (file == NULL) {
        perror("Failed to open file for reading");
        return EXIT_FAILURE;
    }

    res = fread(trip_seconds, sizeof(float), max, file);
    if (res != max) {
        perror("Failed to read all data in trip_seconds");
    }

    res = fread(trip_miles, sizeof(float), max, file);
    if (res != max) {
        perror("Failed to read all data in trip_miles");
    }
    
    res = fread(fare_amount, sizeof(float), max, file);
    if (res != max) {
        perror("Failed to read all data in fare_amount");
    }
    
    res = fread(tip_amount, sizeof(float), max, file);
    if (res != max) {
        perror("Failed to read all data in tip_amount");
    }
    
    res = fread(tolls_amount, sizeof(float), max, file);
    if (res != max) {
        perror("Failed to read all data in tolls");
    }
    
    res = fread(extra_amount, sizeof(float), max, file);
    if (res != max) {
        perror("Failed to read all data in extra");
    }

    auto end = std::chrono::steady_clock::now();
    long duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    printf("Elapsed time for copying file  ms : %li ms.\n\n", duration);

    fclose(file);
    
    for(size_t i = 0; i < 10; i++){
        printf("trip_seconds[%d]: %f, trip_miles: %f fare_amount: %f tip_amount: %f tolls: %f extra: %f\n", 
                i, trip_seconds[i], trip_miles[i], fare_amount[i], 
                   tip_amount[i], tolls[i], extra[i]);
    }

    printf("1\n");

    size_t number = 0;
    for(size_t i = 0; i < max; i++){

        if(trip_seconds[i] > THRESHOLD_SECONDS){
            number++;
        }
    }

    printf("number: %llu\n", number);

    printf("size1: %llu\n", size);

    for(size_t i = 0; i < max; i++){
        seconds[i] = trip_seconds[i];
        miles[i] = trip_miles[i];
        fare[i] = fare_amount[i];
        extra[i] = tip_amount[i];
        tips[i] = tolls_amount[i];
        tolls[i] = extra_amount[i];
    }

    free(trip_seconds);
    free(trip_miles);
    free(fare_amount);
    free(tip_amount);
    free(tolls_amount);
    free(extra_amount);

    printf("line: %d\n", __LINE__);

    return 0;

}

int rapids_uvm_direct(char *filename){

    printf("Started processing 0.\n");
    size_t count = 0;
    
    // file = fopen(filename, "r");
    
    float *seconds, *miles, *fare, *extra, *tips, *tolls;
    read_bin_uvm(filename, count, seconds, miles, fare, extra, tips, tolls);

    size_t size = count*sizeof(float)*6; // MAX_TRIPS * LINE_LENGTH * sizeof(char);
    printf("size: %llu\n", size);
    
    
    printf("size0: %llu\n", size);
    printf("size1: %llu\n", size);

    printf("size2: %llu\n", size);
    // fclose(file);
    // return 0;

    cudaEvent_t event1, event2;
    cudaEventCreate(&event1);
    cudaEventCreate(&event2);
    cudaError_t ret;
    
    float *total_amount, *h_total, *d_miles, *h_miles, *h_sum, *d_sum;
    h_miles = (float *) malloc(sizeof(float)*count);
    h_sum = (float *) malloc(sizeof(float)*count);
    check_cuda_error(cudaMalloc((void **)&d_sum, sizeof(float)*count));
    for(size_t i = 0; i < count; i++) {
        h_sum[i] = 0;
        h_miles[i] = 0;
    }
    check_cuda_error(cudaMemcpy(d_sum, h_sum, sizeof(float)*count, cudaMemcpyHostToDevice));

    check_cuda_error(cudaMalloc((void **)&d_miles, sizeof(float)*count));
    check_cuda_error(cudaMemcpy(d_miles, h_miles, sizeof(float)*count, cudaMemcpyHostToDevice));
    // check_cuda_error(cudaMemset(d_miles, 0, sizeof(float)));
    
    h_total = (float *) malloc(sizeof(float));

    // Step 1: Read CSV into a single buffer
    // if (read_csv(file, buffer, &count) != 0) {
    //     return -1;
    // }

    printf("count: %llu\n", count);

    int *h_array, *d_array, h_ones, *d_ones; 
    h_array = (int *) malloc(sizeof(int)*count);

    for(size_t i = 0; i < count; i++) h_array[i] = 0;

    check_cuda_error(cudaMalloc((void **)&d_ones, sizeof(int)));
    check_cuda_error(cudaMemset(d_ones, 0, sizeof(int)));

    check_cuda_error(cudaMalloc((void **)&d_array, sizeof(int)*count));
    check_cuda_error(cudaMemcpy(d_array, h_array, sizeof(int)*count, cudaMemcpyHostToDevice));

    
    
    
    check_cuda_error(cudaMalloc((void **)&total_amount, sizeof(float)));
    check_cuda_error(cudaMemset(total_amount, 0, sizeof(float)));
    

    ret = cudaDeviceSynchronize();
    cudaEventRecord(event1, (cudaStream_t)1);

    // Step 3: Copy data to device
    // check_cuda_error(cudaMemcpy(d_buffer, buffer, count * LINE_LENGTH * sizeof(char), cudaMemcpyHostToDevice));

    // Step 4: Launch kernel
    size_t n_pages = size/(REQUEST_SIZE*6);
    int threadsPerBlock = 1024;
    int blocksPerGrid = (count + threadsPerBlock - 1) / threadsPerBlock;
    ret = cudaDeviceSynchronize();
    printf("ret: %d cudaGetLastError(): %d\n", ret, cudaGetLastError());
    auto start = std::chrono::steady_clock::now();
    process_trips_uvm_direct_trip_miles2<<</*(n_pages*WARP_SIZE)/threadsPerBlock+1, threadsPerBlock */blocksPerGrid, threadsPerBlock >>>
                    (seconds, miles, fare, extra, tips, tolls, n_pages, count, total_amount, d_miles, d_array, d_sum);
    ret = cudaDeviceSynchronize();
    printf("ret: %d cudaGetLastError(): %d\n", ret, cudaGetLastError());
    auto end = std::chrono::steady_clock::now();
    long duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    printf("Elapsed time for trip_miles uvm transfer  ms : %li ms.\n\n", duration);

    // start = std::chrono::steady_clock::now();
    // process_trips_rdma_direct_fare_amount<<< (n_pages*32)/threadsPerBlock+1, threadsPerBlock /*blocksPerGrid, threadsPerBlock*/ >>>
    //                 (rdma_buffer_global, n_pages, count, total_amount, d_miles, d_array, d_ones);
    // ret = cudaDeviceSynchronize();
    // printf("ret: %d cudaGetLastError(): %d\n", ret, cudaGetLastError());
    // end = std::chrono::steady_clock::now();
    // duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    // printf("Elapsed time for fare_amount rdma transfer  ms : %li ms.\n\n", duration);

    // start = std::chrono::steady_clock::now();
    // process_trips_rdma_direct_tip_amount<<< blocksPerGrid, threadsPerBlock >>>(rdma_buffer_global, count, total_amount, d_miles);
    // ret = cudaDeviceSynchronize();
    // printf("ret: %d cudaGetLastError(): %d\n", ret, cudaGetLastError());
    // end = std::chrono::steady_clock::now();
    // duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    // printf("Elapsed time for tip_amount rdma transfer  ms : %li ms.\n\n", duration);

    // start = std::chrono::steady_clock::now();
    // process_trips_rdma_direct_tolls<<< blocksPerGrid, threadsPerBlock >>>(rdma_buffer_global, count, total_amount, d_miles);
    // ret = cudaDeviceSynchronize();
    // printf("ret: %d cudaGetLastError(): %d\n", ret, cudaGetLastError());
    // end = std::chrono::steady_clock::now();
    // duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    // printf("Elapsed time for tolls rdma transfer  ms : %li ms.\n\n", duration);

    // start = std::chrono::steady_clock::now();
    // process_trips_rdma_direct_extra<<< blocksPerGrid, threadsPerBlock >>>(rdma_buffer_global, count, total_amount, d_miles);
    // ret = cudaDeviceSynchronize();
    // printf("ret: %d cudaGetLastError(): %d\n", ret, cudaGetLastError());
    // end = std::chrono::steady_clock::now();
    // duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    // printf("Elapsed time for extra rdma transfer  ms : %li ms.\n\n", duration);

    
    cudaDeviceSynchronize();

    cudaEventRecord(event2, (cudaStream_t) 1);
            
    cudaEventSynchronize(event1); //optional
    cudaEventSynchronize(event2); //wait for the event to be executed!

    ret = cudaDeviceSynchronize();

    print_retires<<<1,1>>>();

    check_cuda_error(cudaMemcpy(h_array, d_array, sizeof(int)*count, cudaMemcpyDeviceToHost));
    check_cuda_error(cudaMemcpy(h_total, total_amount, sizeof(float), cudaMemcpyDeviceToHost));
    check_cuda_error(cudaMemcpy(h_miles, d_miles, sizeof(float)*count, cudaMemcpyDeviceToHost));
    check_cuda_error(cudaMemcpy(&h_ones, d_ones, sizeof(int), cudaMemcpyDeviceToHost));
    check_cuda_error(cudaMemcpy(h_sum, d_sum, sizeof(float)*count, cudaMemcpyDeviceToHost));
    
    float dt_ms;
    cudaEventElapsedTime(&dt_ms, event1, event2);
    printf("The cudaEvent execution time with rdma on GPU: %f ms\n", dt_ms);

    size_t ones = 0;
    float sum = 0;
    float sum_miles = 0;
    for(size_t i = 0; i < count; i++) {
        if(h_array[i] == 1)
            ones++;
        sum += h_sum[i];
        sum_miles += h_miles[i];

    }

    printf("sum: %f\n", sum);
    printf("atomic ones: %d\n", h_ones);
    printf("ones: %d\n", ones);
    printf("h_total: %f\n", *h_total);
    printf("sum_miles: %f\n", sum_miles);
    printf("Avg. $/mile: %f\n", sum/sum_miles);

    // Step 5: Free memory
    // free(buffer);
    // cudaFree(d_buffer);

    return 0;
}

int filter_bin(char *filename){
    size_t max = 211670894llu;
    // max1 = max;
    float *trip_seconds; // [MAX_TRIPS];
    float *trip_miles;
    float *fare_amount;
    float *tip_amount;
    float *tolls;
    float *extra;
    trip_seconds = (float *) malloc(sizeof(float)*max);
    trip_miles = (float *) malloc(sizeof(float)*max);
    fare_amount = (float *) malloc(sizeof(float)*max);
    tip_amount = (float *) malloc(sizeof(float)*max);
    tolls = (float *) malloc(sizeof(float)*max);
    extra = (float *) malloc(sizeof(float)*max);
    
    

    size_t res;
    size_t size = max*sizeof(float)*6;
    // check_cuda_error(cudaMallocManaged((void **)&rdma_buffer_global, sizeof(rdma_buf<float>)));
    auto start = std::chrono::steady_clock::now();
    
    FILE *file = fopen(filename, "rb");
    if (file == NULL) {
        perror("Failed to open file for reading");
        return EXIT_FAILURE;
    }

    res = fread(trip_seconds, sizeof(float), max, file);
    if (res != max) {
        perror("Failed to read all data in trip_seconds");
    }

    res = fread(trip_miles, sizeof(float), max, file);
    if (res != max) {
        perror("Failed to read all data in trip_miles");
    }
    
    res = fread(fare_amount, sizeof(float), max, file);
    if (res != max) {
        perror("Failed to read all data in fare_amount");
    }
    
    res = fread(tip_amount, sizeof(float), max, file);
    if (res != max) {
        perror("Failed to read all data in tip_amount");
    }
    
    res = fread(tolls, sizeof(float), max, file);
    if (res != max) {
        perror("Failed to read all data in tolls");
    }
    
    res = fread(extra, sizeof(float), max, file);
    if (res != max) {
        perror("Failed to read all data in extra");
    }

    auto end = std::chrono::steady_clock::now();
    long duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    printf("Elapsed time for copying file  ms : %li ms.\n\n", duration);

    fclose(file);
    
    for(size_t i = 0; i < 10; i++){
        printf("trip_seconds[%d]: %f, trip_miles: %f fare_amount: %f tip_amount: %f tolls: %f extra: %f\n", 
                i, trip_seconds[i], trip_miles[i], fare_amount[i], 
                   tip_amount[i], tolls[i], extra[i]);
    }

    printf("1\n");

    size_t number = 0;
    float sum_fare = 0;
    for(size_t i = 0; i < max; i++){

        if(trip_seconds[i] > THRESHOLD_SECONDS){
            number++;
            sum_fare += fare_amount[i];
        }

        // sum_fare += fare_amount[i];


    }

    printf("sum_fare: %f\n", sum_fare);

    free(trip_seconds);
    free(trip_miles);
    free(fare_amount);
    free(tip_amount);
    free(tolls);
    free(extra);

    printf("number: %llu\n", number);

    printf("size1: %llu\n", size);

    return 0;

}

__global__ __launch_bounds__(1024,2)
void process_trips_rdma_direct(rdma_buf<float> *rdma_file, int count, float *total_amount, float *total_miles) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        
        // Declare an array to hold the line
        // char line[LINE_LENGTH];  // Create a character array for one line

        // int vendorID;
        float trip_seconds = (*rdma_file)[idx];
        
        // parse_trip(line, trip_seconds, trip_miles, fare_amount, tip_amount, tolls, extra);
        // trip_miles < 1050

        if(trip_seconds > 1000) {

            // printf("trip_seconds: %f\n", trip_seconds);

            float trip_miles  = (*rdma_file)[5*idx  + count + 0];
            float fare_amount = (*rdma_file)[5*idx + count + 1];
            float tip_amount  = (*rdma_file)[5*idx + count + 2];
            float tolls       = (*rdma_file)[5*idx + count + 3];
            float extra       = (*rdma_file)[5*idx + count + 4];

            float local_total = fare_amount; // - extra - tolls + tip_amount;
            
                AtomicAdd(total_amount, local_total);
                AtomicAdd(total_miles, trip_miles);
        }

        // Example operation: Store or print the parsed values (printing from device is not recommended)
        // printf("Vendor ID: %d, Pickup Latitude: %f, Pickup Longitude: %f\n", vendorID, pickup_latitude, pickup_longitude);
    }
}

__global__ // __launch_bounds__(1024,2)
void process_trips_rdma_direct_trip_miles(rdma_buf<float> *rdma_file, size_t n, size_t count, float *total_amount, float *total_miles, int *array, float *sum) {


    // Page size in elements (64KB / 4 bytes per unsigned int)
    const size_t pageSize = REQUEST_SIZE / sizeof(float);
    // Elements per warp
    const size_t elementsPerWarp = pageSize / WARP_SIZE;

    // Global thread ID
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    // if(tid == 0) printf("warpSize: %d\n", warpSize);
    // Warp ID within the block
    size_t warpId = tid / WARP_SIZE;

    // Thread lane within the warp
    size_t lane = threadIdx.x % WARP_SIZE; // warpSize;

    // Determine which page this warp will process
    size_t pageStart = warpId * pageSize;

    // Ensure we don't process out-of-bounds pages
    if (pageStart < n * pageSize) {
        
        // Process elements within the page
        // for (size_t i = 0; i < elementsPerWarp; ++i) {
        //     size_t elementIdx = pageStart + lane + i * warpSize;
            size_t end = (warpId + 1)*pageSize > count ? count : (warpId + 1)*pageSize;
            for(size_t j = warpId*pageSize + lane; j < end; j += WARP_SIZE) {
                float trip_seconds = (*rdma_file)[j];
                if(trip_seconds > THRESHOLD_SECONDS) {
                    // 
                    // if(trip_miles < 3460.0){
                        // size_t index = 5*j + count;
                        // float trip_miles  = (*rdma_file)[index ];
                        // float fare_amount = (*rdma_file)[index + 1];     
                        // float extra       = (*rdma_file)[index + 4];
                        // float tip_amount  = (*rdma_file)[index + 2];
                        // float tolls       = (*rdma_file)[index + 3];

                        float trip_miles  = (*rdma_file)[j + count];
                        float fare_amount = (*rdma_file)[j + count*2];
                        
                        float extra       = (*rdma_file)[j + count*5];
                        float tip_amount  = (*rdma_file)[j + count*3];
                        float tolls       = (*rdma_file)[j + count*4];
                        
                        

                        float local_total = fare_amount - extra + tip_amount - tolls;
                        sum[j] += local_total;
                        total_miles[j] += trip_miles;
                        
                        atomicAdd(total_amount, local_total);
                        atomicAdd(total_miles, trip_miles);
                        // array[j] = 1;
                        // atomicAdd(ones, 1);
                    // }
                }
                // else{
                //     printf("trip_seconds: %f\n", trip_seconds);
                // }
            }
        // }
    }


    // size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    // if (idx < count) {
    //     float trip_seconds = (*rdma_file)[idx];
    //     if(trip_seconds > 1000) {
    //         float trip_miles  = (*rdma_file)[5*idx  + count + 0];
    //         // float fare_amount = (*rdma_file)[5*idx + count + 1];
    //         // float tip_amount  = (*rdma_file)[5*idx + count + 2];
    //         // float tolls       = (*rdma_file)[5*idx + count + 3];
    //         // float extra       = (*rdma_file)[5*idx + count + 4];

    //         // float local_total = fare_amount - extra - tolls + tip_amount;
            
    //         //     AtomicAdd(total_amount, local_total);
    //             AtomicAdd(total_miles, trip_miles);
    //     }
    // }
}


__global__ __launch_bounds__(1024,2)
void process_trips_rdma_direct_fare_amount(rdma_buf<float> *rdma_file, size_t n, size_t count, float *total_amount, float *total_miles, int *array, int *ones) {

    // Page size in elements (64KB / 4 bytes per unsigned int)
    const size_t pageSize = REQUEST_SIZE / sizeof(float);
    // Elements per warp
    const size_t elementsPerWarp = pageSize / warpSize;

    // Global thread ID
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    // if(tid == 0) printf("warpSize: %d\n", warpSize);
    // Warp ID within the block
    size_t warpId = tid / warpSize;

    // Thread lane within the warp
    size_t lane = threadIdx.x % warpSize; // warpSize;

    // Determine which page this warp will process
    size_t pageStart = warpId * pageSize;

    // Ensure we don't process out-of-bounds pages
    if (pageStart < n * pageSize) {
        
        // Process elements within the page
        // for (size_t i = 0; i < elementsPerWarp; ++i) {
        //     size_t elementIdx = pageStart + lane + i * warpSize;
            size_t end = (warpId + 1)*pageSize > count ? count : (warpId + 1)*pageSize;
            for(size_t j = warpId*pageSize + lane; j < end; j += warpSize) {
                
                if(array[j] == 1) {
                    float fare_amount = (*rdma_file)[j + count*2];
                    // atomicAdd(total_amount, fare_amount);
                    atomicAdd(ones, 1);
                }
                
            }
        // }
    }

    // size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    // if (idx < count) {
    //     // float trip_seconds = (*rdma_file)[idx];
    //     if(array[idx]) {
    //         // float fare_amount  = (*rdma_file)[5*idx  + count + 1];
    //         float fare_amount = (*rdma_file)[idx + count*2];
            
    //             
    //             // AtomicAdd(total_miles, trip_miles);
    //     }
    // }
}

__global__ __launch_bounds__(1024,2)
void process_trips_rdma_direct_tip_amount(rdma_buf<float> *rdma_file, size_t count, float *total_amount, float *total_miles) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        float trip_seconds = (*rdma_file)[idx];
        if(trip_seconds > THRESHOLD_SECONDS) {
            // float trip_miles  = (*rdma_file)[5*idx  + count + 0];
            // float fare_amount = (*rdma_file)[5*idx + count + 1];
            float tip_amount  = (*rdma_file)[idx + count*3];
            // float tolls       = (*rdma_file)[5*idx + count + 3];
            // float extra       = (*rdma_file)[5*idx + count + 4];

            // float local_total = fare_amount - extra - tolls + tip_amount;
            
                atomicAdd(total_amount, tip_amount);
                // AtomicAdd(total_miles, trip_miles);
        }
    }
}

__global__ __launch_bounds__(1024,2)
void process_trips_rdma_direct_tolls(rdma_buf<float> *rdma_file, size_t count, float *total_amount, float *total_miles) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        float trip_seconds = (*rdma_file)[idx];
        if(trip_seconds > THRESHOLD_SECONDS) {
            // float trip_miles  = (*rdma_file)[5*idx  + count + 0];
            // float fare_amount = (*rdma_file)[5*idx + count + 1];
            // float tip_amount  = (*rdma_file)[5*idx + count + 2];
            float tolls       = (*rdma_file)[idx + count*4] * (-1);
            // float extra       = (*rdma_file)[5*idx + count + 4];

            // float local_total = fare_amount - extra - tolls + tip_amount;
            
                atomicAdd(total_amount, tolls);
                // AtomicAdd(total_miles, trip_miles);
        }
    }
}

__global__ __launch_bounds__(1024,2)
void process_trips_rdma_direct_extra(rdma_buf<float> *rdma_file, size_t count, float *total_amount, float *total_miles) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        float trip_seconds = (*rdma_file)[idx];
        if(trip_seconds > THRESHOLD_SECONDS) {
            // float trip_miles  = (*rdma_file)[5*idx  + count + 0];
            // float fare_amount = (*rdma_file)[5*idx + count + 1];
            // float tip_amount  = (*rdma_file)[5*idx + count + 2];
            // float tolls       = (*rdma_file)[5*idx + count + 3] * (-1);
            float extra       = (*rdma_file)[idx + count*5] * (-1);

            // float local_total = fare_amount - extra - tolls + tip_amount;
            
                atomicAdd(total_amount, extra);
                // AtomicAdd(total_miles, trip_miles);
        }
    }
}

int write_bin(char *filename, char *bin_file){
    size_t max = 211670894*2;
    // char line1[LINE_LENGTH];
    // FILE *file1 = fopen(filename, "r");
    // if (!file1) {
    //     perror("Failed to open file rapids_RDMA_direct");
    //     return -1;
    // }

    // while (fgets(line1, LINE_LENGTH, file1)){
    //     max++;
    // }
    // printf("max: %llu\n", max);

    // fclose(file1);


    float *trip_seconds; // [MAX_TRIPS];
    float *trip_miles;
    float *fare_amount;
    float *tip_amount;
    float *tolls;
    float *extra;
    trip_seconds = (float *) malloc(sizeof(float)*max);
    trip_miles = (float *) malloc(sizeof(float)*max);
    fare_amount = (float *) malloc(sizeof(float)*max);
    tip_amount = (float *) malloc(sizeof(float)*max);
    tolls = (float *) malloc(sizeof(float)*max);
    extra = (float *) malloc(sizeof(float)*max);
    if(trip_seconds == NULL || trip_miles == NULL || fare_amount == NULL ||
       tip_amount == NULL || tolls == NULL || extra == NULL ){
        printf("error on allocation\n");
        return -1;
    }
    char *line; //[LINE_LENGTH];
    size_t i_seconds = 0, i_miles = 0, i_fare = 0, i_tip = 0, i_tolls = 0, i_extra = 0;
    printf("Started processing 1.\n");
    
    // char *buffer;
    FILE *file = fopen(filename, "r");
    if (!file) {
        perror("Failed to open file rapids_RDMA_direct");
        return -1;
    }
    printf("Started processing.\n");
    size_t counter_line = 0;
    size_t count = 0, len = 0;
    while (getline(&line, &len, file) != -1/* && count < MAX_ROWS*/) {
        if (counter_line == 0) {
            counter_line++;
            continue;  // Skip the header line
        }

        char small_buffer[50]; // Temporary buffer to store individual field data
        size_t field = 0; // Field index
        size_t index = 0; // Index for small_buffer
        
        // Iterate through the line character by character
        for (size_t i = 0; i < len; i++) {
            if (line[i] == ',' || line[i] == '\n') { // Check for comma or newline
                small_buffer[index] = '\0'; // Null-terminate the string

                // Check for empty fields (NA) or invalid values
                if (index == 0 || strcmp(small_buffer, "") == 0) {
                    strcpy(small_buffer, "0"); // Replace with "0" if empty
                }

                // Store the value in the appropriate buffer
                if (field == 4) {
                    
                    trip_seconds[i_seconds] = atof(small_buffer); // Convert to float
                    if(i_seconds < 10) printf("i_seconds: %d small_buffer: %s trip_seconds[i_seconds]; %f\n", i_seconds, small_buffer, trip_seconds[i_seconds]);
                    // printf(" small_buffer: %s trip_seconds[%llu]: %f ", small_buffer, i_seconds, trip_seconds[i_seconds]);
                    i_seconds++;
                } else if (field == 5) {
                    trip_miles[i_miles] = atof(small_buffer);
                    i_miles++;
                } else if (field == 10) {
                    fare_amount[i_fare] = atof(small_buffer);
                    i_fare++;
                } else if (field == 11) {
                    tip_amount[i_tip] = atof(small_buffer);
                    i_tip++;
                } else if (field == 12) {
                    tolls[i_tolls] = atof(small_buffer);
                    i_tolls++;
                } else if (field == 13) {
                    extra[i_extra] = atof(small_buffer);
                    i_extra++; // Increment count after the last field
                }
                
                field++; // Move to the next field
                index = 0; // Reset index for the next field
            } else {
                small_buffer[index++] = line[i]; // Collect character into buffer
            }
        }
        
        counter_line++;
    
        // if(count < 10){
        //     printf("trip_seconds[%d]: %f ", count, trip_seconds[count]);
        //     printf("trip_miles[%d]: %f ", count, trip_miles[count]);
        //     printf("fare_amount[%d]: %f ", count, fare_amount[count]);
        //     printf("tip_amount[%d]: %f ", count, tip_amount[count]);
        //     printf("tolls[%d]: %f ", count, tolls[count]);
        //     printf("extra[%d]: %f\n", count, extra[count]);
        // }

    }

    free(line); // Free the line buffer
    fclose(file);

    printf("i_seconds: %d, i_miles: %d i_fare: %d i_tip: %d i_tolls: %d i_extra: %d max; %d\n", 
                i_seconds, i_miles, i_fare, i_tip, i_tolls, i_extra, max);

    for(size_t i = 0; i < 10; i++){
        printf("trip_seconds[%d]: %f, trip_miles: %f fare_amount: %f tip_amount: %f tolls: %f extra: %f\n", 
                i, trip_seconds[i], trip_miles[i], fare_amount[i], tip_amount[i], tolls[i], extra[i]);
    }

    // Assume we have two columns of data: column1 and column2
    FILE *f = fopen(bin_file, "wb");

    printf("count: %llu\n", count);

    // Write column1 data (assume it's an array of floats)
    size_t res = fwrite(trip_seconds, sizeof(float), i_seconds, f);
    if (res != i_seconds) {
        perror("Failed to write all data in trip_seconds");
    }
    printf("Trip Seconds written\n");
    res = fwrite(trip_miles, sizeof(float), i_miles, f);
    if (res != i_miles) {
        perror("Failed to write all data in trip_miles");
    }
    printf("Trip Miles written\n");
    res = fwrite(fare_amount, sizeof(float), i_fare, f);
    if (res != i_fare) {
        perror("Failed to write all data in fare_amount");
    }
    printf("Fare written\n");
    res = fwrite(tip_amount, sizeof(float), i_tip, f);
    if (res != i_tip) {
        perror("Failed to write all data in tip_amount");
    }
    printf("Tip amounts written\n");
    res = fwrite(tolls, sizeof(float), i_tolls, f);
    if (res != i_tolls) {
        perror("Failed to write all data in tolls");
    }
    printf("Tolls written\n");
    res = fwrite(extra, sizeof(float), i_extra, f);
    if (res != i_extra) {
        perror("Failed to write all data in extra");
    }
    printf("Extra written\n");

    fclose(f);

    printf("File is closed: %s\n", bin_file);

    free(trip_seconds);
    free(trip_miles);
    free(fare_amount);
    free(tip_amount);
    free(tolls);
    free(extra);

    return 0;
}

rdma_buf<float> *rdma_buffer_global = NULL;

int read_bin(char *filename, size_t &max1){
    size_t max = 211670894llu;
    max1 = max;
    float *trip_seconds; // [MAX_TRIPS];
    float *trip_miles;
    float *fare_amount;
    float *tip_amount;
    float *tolls;
    float *extra;
    trip_seconds = (float *) malloc(sizeof(float)*max);
    trip_miles = (float *) malloc(sizeof(float)*max);
    fare_amount = (float *) malloc(sizeof(float)*max);
    tip_amount = (float *) malloc(sizeof(float)*max);
    tolls = (float *) malloc(sizeof(float)*max);
    extra = (float *) malloc(sizeof(float)*max);
    

    size_t res;
    size_t size = max*sizeof(float)*6;
    check_cuda_error(cudaMallocManaged((void **)&rdma_buffer_global, sizeof(rdma_buf<float>)));

    rdma_buffer_global->start(size, GPU, NULL);

    auto start = std::chrono::steady_clock::now();
    
    FILE *file = fopen(filename, "rb");
    if (file == NULL) {
        perror("Failed to open file for reading");
        return EXIT_FAILURE;
    }

    res = fread(trip_seconds, sizeof(float), max, file);
    if (res != max) {
        perror("Failed to read all data in trip_seconds");
    }

    res = fread(trip_miles, sizeof(float), max, file);
    if (res != max) {
        perror("Failed to read all data in trip_miles");
    }
    
    res = fread(fare_amount, sizeof(float), max, file);
    if (res != max) {
        perror("Failed to read all data in fare_amount");
    }
    
    res = fread(tip_amount, sizeof(float), max, file);
    if (res != max) {
        perror("Failed to read all data in tip_amount");
    }
    
    res = fread(tolls, sizeof(float), max, file);
    if (res != max) {
        perror("Failed to read all data in tolls");
    }
    
    res = fread(extra, sizeof(float), max, file);
    if (res != max) {
        perror("Failed to read all data in extra");
    }

    auto end = std::chrono::steady_clock::now();
    long duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    printf("Elapsed time for copying file  ms : %li ms.\n\n", duration);

    fclose(file);
    
    for(size_t i = 0; i < 10; i++){
        printf("trip_seconds[%d]: %f, trip_miles: %f fare_amount: %f tip_amount: %f tolls: %f extra: %f\n", 
                i, trip_seconds[i], trip_miles[i], fare_amount[i], 
                   tip_amount[i], tolls[i], extra[i]);
    }

    printf("1\n");

    size_t number = 0;
    for(size_t i = 0; i < max; i++){

        if(trip_seconds[i] > THRESHOLD_SECONDS){
            number++;
        }
    }

    printf("number: %llu\n", number);

    printf("size1: %llu\n", size);

    for(size_t i = 0; i < max; i++){
        rdma_buffer_global->local_buffer[i] = trip_seconds[i];

        // rdma_buffer_global->local_buffer[5*i + max + 0] = trip_miles[i];
        // rdma_buffer_global->local_buffer[5*i + max + 1] = fare_amount[i];
        // rdma_buffer_global->local_buffer[5*i + max + 2] = tip_amount[i];
        // rdma_buffer_global->local_buffer[5*i + max + 3] = tolls[i];
        // rdma_buffer_global->local_buffer[5*i + max + 4] = extra[i];

        rdma_buffer_global->local_buffer[i + max] = trip_miles[i];
        rdma_buffer_global->local_buffer[i + max*2] = fare_amount[i];
        rdma_buffer_global->local_buffer[i + max*3] = tip_amount[i];
        rdma_buffer_global->local_buffer[i + max*4] = tolls[i];
        rdma_buffer_global->local_buffer[i + max*5] = extra[i];
    }

    free(trip_seconds);
    free(trip_miles);
    free(fare_amount);
    free(tip_amount);
    free(tolls);
    free(extra);

    printf("line: %d\n", __LINE__);

    return 0;

}

int rapids_RDMA_direct(char *filename){

    printf("Started processing 0.\n");
    size_t count = 0;
    
    // file = fopen(filename, "r");
    
    read_bin(filename, count);

    size_t size = count*sizeof(float)*6; // MAX_TRIPS * LINE_LENGTH * sizeof(char);
    printf("size: %llu\n", size);
    
    
    printf("size0: %llu\n", size);
    printf("size1: %llu\n", size);

    printf("size2: %llu\n", size);
    // fclose(file);
    // return 0;

    cudaEvent_t event1, event2;
    cudaEventCreate(&event1);
    cudaEventCreate(&event2);
    cudaError_t ret;
    
    float *total_amount, *h_total, *d_miles, *h_miles, *h_sum, *d_sum;
    h_miles = (float *) malloc(sizeof(float)*count);
    h_sum = (float *) malloc(sizeof(float)*count);
    check_cuda_error(cudaMalloc((void **)&d_sum, sizeof(float)*count));
    for(size_t i = 0; i < count; i++) {
        h_sum[i] = 0;
        h_miles[i] = 0;
    }
    check_cuda_error(cudaMemcpy(d_sum, h_sum, sizeof(float)*count, cudaMemcpyHostToDevice));

    check_cuda_error(cudaMalloc((void **)&d_miles, sizeof(float)*count));
    check_cuda_error(cudaMemcpy(d_miles, h_miles, sizeof(float)*count, cudaMemcpyHostToDevice));
    // check_cuda_error(cudaMemset(d_miles, 0, sizeof(float)));
    
    h_total = (float *) malloc(sizeof(float));

    // Step 1: Read CSV into a single buffer
    // if (read_csv(file, buffer, &count) != 0) {
    //     return -1;
    // }

    printf("count: %llu\n", count);

    int *h_array, *d_array, h_ones, *d_ones; 
    h_array = (int *) malloc(sizeof(int)*count);

    for(size_t i = 0; i < count; i++) h_array[i] = 0;

    check_cuda_error(cudaMalloc((void **)&d_ones, sizeof(int)));
    check_cuda_error(cudaMemset(d_ones, 0, sizeof(int)));

    check_cuda_error(cudaMalloc((void **)&d_array, sizeof(int)*count));
    check_cuda_error(cudaMemcpy(d_array, h_array, sizeof(int)*count, cudaMemcpyHostToDevice));

    
    
    
    check_cuda_error(cudaMalloc((void **)&total_amount, sizeof(float)));
    check_cuda_error(cudaMemset(total_amount, 0, sizeof(float)));
    

    ret = cudaDeviceSynchronize();
    cudaEventRecord(event1, (cudaStream_t)1);

    // Step 3: Copy data to device
    // check_cuda_error(cudaMemcpy(d_buffer, buffer, count * LINE_LENGTH * sizeof(char), cudaMemcpyHostToDevice));

    // Step 4: Launch kernel
    size_t n_pages = size/(REQUEST_SIZE*6);
    int threadsPerBlock = 1024;
    int blocksPerGrid = (count + threadsPerBlock - 1) / threadsPerBlock;
    ret = cudaDeviceSynchronize();
    printf("ret: %d cudaGetLastError(): %d\n", ret, cudaGetLastError());
    auto start = std::chrono::steady_clock::now();
    process_trips_rdma_direct_trip_miles<<< (n_pages*WARP_SIZE)/threadsPerBlock+1, threadsPerBlock /*blocksPerGrid, threadsPerBlock*/ >>>
                    (rdma_buffer_global, n_pages, count, total_amount, d_miles, d_array, d_sum);
    ret = cudaDeviceSynchronize();
    printf("ret: %d cudaGetLastError(): %d\n", ret, cudaGetLastError());
    auto end = std::chrono::steady_clock::now();
    long duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    printf("Elapsed time for trip_miles rdma transfer  ms : %li ms.\n\n", duration);

    // start = std::chrono::steady_clock::now();
    // process_trips_rdma_direct_fare_amount<<< (n_pages*32)/threadsPerBlock+1, threadsPerBlock /*blocksPerGrid, threadsPerBlock*/ >>>
    //                 (rdma_buffer_global, n_pages, count, total_amount, d_miles, d_array, d_ones);
    // ret = cudaDeviceSynchronize();
    // printf("ret: %d cudaGetLastError(): %d\n", ret, cudaGetLastError());
    // end = std::chrono::steady_clock::now();
    // duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    // printf("Elapsed time for fare_amount rdma transfer  ms : %li ms.\n\n", duration);

    // start = std::chrono::steady_clock::now();
    // process_trips_rdma_direct_tip_amount<<< blocksPerGrid, threadsPerBlock >>>(rdma_buffer_global, count, total_amount, d_miles);
    // ret = cudaDeviceSynchronize();
    // printf("ret: %d cudaGetLastError(): %d\n", ret, cudaGetLastError());
    // end = std::chrono::steady_clock::now();
    // duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    // printf("Elapsed time for tip_amount rdma transfer  ms : %li ms.\n\n", duration);

    // start = std::chrono::steady_clock::now();
    // process_trips_rdma_direct_tolls<<< blocksPerGrid, threadsPerBlock >>>(rdma_buffer_global, count, total_amount, d_miles);
    // ret = cudaDeviceSynchronize();
    // printf("ret: %d cudaGetLastError(): %d\n", ret, cudaGetLastError());
    // end = std::chrono::steady_clock::now();
    // duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    // printf("Elapsed time for tolls rdma transfer  ms : %li ms.\n\n", duration);

    // start = std::chrono::steady_clock::now();
    // process_trips_rdma_direct_extra<<< blocksPerGrid, threadsPerBlock >>>(rdma_buffer_global, count, total_amount, d_miles);
    // ret = cudaDeviceSynchronize();
    // printf("ret: %d cudaGetLastError(): %d\n", ret, cudaGetLastError());
    // end = std::chrono::steady_clock::now();
    // duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    // printf("Elapsed time for extra rdma transfer  ms : %li ms.\n\n", duration);

    
    cudaDeviceSynchronize();

    cudaEventRecord(event2, (cudaStream_t) 1);
            
    cudaEventSynchronize(event1); //optional
    cudaEventSynchronize(event2); //wait for the event to be executed!

    ret = cudaDeviceSynchronize();

    print_retires<<<1,1>>>();

    check_cuda_error(cudaMemcpy(h_array, d_array, sizeof(int)*count, cudaMemcpyDeviceToHost));
    check_cuda_error(cudaMemcpy(h_total, total_amount, sizeof(float), cudaMemcpyDeviceToHost));
    check_cuda_error(cudaMemcpy(h_miles, d_miles, sizeof(float)*count, cudaMemcpyDeviceToHost));
    check_cuda_error(cudaMemcpy(&h_ones, d_ones, sizeof(int), cudaMemcpyDeviceToHost));
    check_cuda_error(cudaMemcpy(h_sum, d_sum, sizeof(float)*count, cudaMemcpyDeviceToHost));
    
    float dt_ms;
    cudaEventElapsedTime(&dt_ms, event1, event2);
    printf("The cudaEvent execution time with rdma on GPU: %f ms\n", dt_ms);

    size_t ones = 0;
    float sum = 0;
    float sum_miles = 0;
    for(size_t i = 0; i < count; i++) {
        if(h_array[i] == 1)
            ones++;
        sum += h_sum[i];
        sum_miles += h_miles[i];

    }

    printf("sum: %f\n", sum);
    printf("atomic ones: %d\n", h_ones);
    printf("ones: %d\n", ones);
    printf("h_total: %f\n", *h_total);
    printf("sum_miles: %f\n", sum_miles);
    printf("Avg. $/mile: %f\n", sum/sum_miles);

    // Step 5: Free memory
    // free(buffer);
    // cudaFree(d_buffer);

    return 0;
}


// Main program
int main(int argc, char **argv)
{   
    init_gpu(0);
    cudaSetDevice(0);
    printf("hello from rapid\n");
    char *file = argv[7];
    printf("hello from rapid file : %s\n", file);
    // rapids_CUDA(file);
    
    
    // char *bin_file = "/mydata/chicago_2b_trial.bin";
    // write_bin(file, bin_file);

    // printf("Binary file is written\n");

    bool rdma_flag = false;
    cudaError_t ret1;
    struct context_2nic *s_ctx = (struct context_2nic *)malloc(sizeof(struct context_2nic));
    if(rdma_flag){
        s_ctx->gpu_cq = NULL;
        s_ctx->wqbuf = NULL;
        s_ctx->cqbuf = NULL;
        s_ctx->gpu_qp = NULL;


        int num_msg = (unsigned long) atoi(argv[4]);
        int mesg_size = (unsigned long) atoi(argv[5]);
        int num_bufs = (unsigned long) atoi(argv[6]);

        
        struct post_content post_cont, *d_post, host_post;
        struct poll_content poll_cont, *d_poll, host_poll;
        // struct post_content2 /*post_cont2,*/ *d_post2;
        struct server_content_2nic post_cont2, *d_post2;
        struct host_keys keys;
        struct gpu_memory_info gpu_mem;

        int num_iteration = num_msg;
        s_ctx->n_bufs = num_bufs;

        s_ctx->gpu_buf_size = 26*1024*1024*1024llu; // N*sizeof(int)*3llu;
        s_ctx->gpu_buffer = NULL;

        // // remote connection:
        // int ret = connect(argv[2], s_ctx);

        // local connect
        char *mlx_name = "mlx5_0";
        // int ret = local_connect(mlx_name, s_ctx);
        int ret = local_connect_2nic(mlx_name, s_ctx, 0, GPU);

        mlx_name = "mlx5_2";
        // int ret = local_connect(mlx_name, s_ctx);
        ret = local_connect_2nic(mlx_name, s_ctx, 1, GPU);

        ret = prepare_post_poll_content_2nic(s_ctx, &post_cont, &poll_cont, &post_cont2, \
                                        &host_post, &host_poll, &keys, &gpu_mem);
        if(ret == -1) {
            printf("Post and poll contect creation failed\n");    
            exit(-1);
        }

        printf("alloc synDev ret: %d\n", cudaDeviceSynchronize());
        cudaSetDevice(GPU);
        alloc_global_cont(&post_cont, &poll_cont, &post_cont2, gpu_mem);

        // if(cudaSuccess != ){    
        printf("alloc synDev ret1: %d\n", cudaDeviceSynchronize());
            // return -1;
        // }

        ret1 = cudaDeviceSynchronize();
        printf("ret: %d\n", ret1);
        if(cudaSuccess != ret1){    
            return -1;
        }

        size_t restricted_gpu_mem = 16*1024*1024*1024llu;
        // restricted_gpu_mem = restricted_gpu_mem / 3;
        const size_t page_size = REQUEST_SIZE;
        // const size_t numPages = ceil((double)restricted_gpu_mem/page_size);

        printf("function: %s line: %d\n", __FILE__, __LINE__);
        alloc_global_host_content(host_post, host_poll, keys, gpu_mem);
        printf("function: %s line: %d\n", __FILE__, __LINE__);

        ret1 = cudaDeviceSynchronize();
        printf("ret: %d\n", ret1);
        if(cudaSuccess != ret1){    
            return -1;
        }

        ret1 = cudaDeviceSynchronize();
        printf("ret: %d\n", ret1);
        if(cudaSuccess != ret1){    
            return -1;
        }
        
        printf("restricted_gpu_mem: %zu\n", restricted_gpu_mem);
        cudaSetDevice(GPU);
        start_page_queue<<<1, 1>>>(/*s_ctx->gpu_buf_size*/restricted_gpu_mem, page_size);
        ret1 = cudaDeviceSynchronize();
        printf("ret: %d\n", ret1);
        if(cudaSuccess != ret1){    
            return -1;
        }
    }

    
    

    if(rdma_flag){
        // the follwoing for directly reading file in rdma
        // rapids_RDMA(file);
        // following for reading buffers separately
        rapids_RDMA_direct(file);
        // transfer_benchmark();
        cudaFree(s_ctx->gpu_buffer);
    }
    else{
        rapids_uvm_direct(file);
    }

    filter_bin(file);

    // rapids_CUDA(file);

    // rapids_CUDA(file);

    // rapids_CUDA(file);
    
    // printf("oversubs ratio: %d\n", oversubs_ratio_macro-1);
    
	return 0;
}