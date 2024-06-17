#include <cuda.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <stdbool.h>
#include <cuda_profiler_api.h>
#include <vector>
#include <map>
#include "helper.h"
#include "definitions.h"


//static dataType *pages_2MB;
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
        if (code != cudaSuccess)
        {
                fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
                if (abort) exit(code);
        }
}



extern void EvictionSetGenerator(dataType *cacheBuf,
                          dataType *tagetAddr,
                          dataType *EvictionResult);

//extern double ThresholdMeasure(void);
//void * GetMemoryforAddressTranslation(void);
//void IOCTL_try(void);
//std::map<dataType*, uint64_t> CreateMemoryPoolfromGDDR(void);//std::vector<dataType *>& pages, std::vector<uint64_t>& physical_addrs);
//double CompareAddresses(unsigned int *A, unsigned int *B);
//void CheckCacheLinesForConflict(void);
//void CheckCacheLinesWithOtherPages(void);
//bool AddressIsInPollutionBuffer(dataType* A);
//void SeqAccess(void);
//double ThresholdMeasure_optimized(void);
//int device_init();
//double device_find_dram_read_time_optimized( dataType *a, dataType *b);
//double device_find_dram_read_time( dataType *a, dataType *b);