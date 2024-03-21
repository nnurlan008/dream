#define Threshold_BufferSize MB(2)
#define MemoryPool GB(1)
#define PAGE_SIZE (KB(4))
#define CacheBufferSizeMB 26
#define NumOfThreads 1024
#define ComparisonNumber 20
#define AccessBytesSize ((128))
#define UVM_Chunk_Size (MB(2))
#define Threshold_Sample 16384
#define Max_Allowable_Access_Latency 830
#define ExpectedNumberofBanks 2

#define AddressOffset 10000
#define Mask1 0x1d912000
#define Mask2 0x16598000
#define Mask3 0xbc8a000
#define Mask4 0x511e2000
#define Mask5 0x4c8f0000
#define Mask6 0x5ad68000
#define Mask7 0x4747a000

#define MAX_XOR_BITS 25
#define POINTER_SIZE 33//(sizeof(uint64_t) * 8)
#define ADDRESS_ALIGNMENT 8
#define Percentage_ERR_HASH_Functions 90

#define GB(X) ((unsigned long)((X)*1024L*1024L*1024L))
#define MB(X) ((X)*1024L*1024L)
#define KB(X) ((X)*1024L)

#define mainBuffOffsetSz 128//*modified*
#define buffSzTotBig 16// *modified*

#define numRepeatLoops 16

//24 MB is good time buffer size for side channel and for 32 ca
//8 MB is used for cache mapping and would be used for covert channel over multiple thread blocks

#define timeBuffSzDevMB 1 //1 //8 //144

//#define threshold 850//420//
#define locTrojProcThresh 420
#define remSpyProcThresh 800
#define numCheck 3
#define warpSz 32 //*modified*

#define CacheMissThershold 450 //GPU cycle
#define RowConflictThreshold 555 //GPU cycle

#define numIndxEls 18//20
#define numLinesPerSet 16

#define firstNumElsTrav 2800
#define numOfEvSets 1680//1792
#define numLinesPerEvSets 18
#define numLinesToProbe 16

#define GPU_L2_CACHE_LINE_SIZE 128
#define GPU_MAX_OUTER_LOOP 8
#define GPU_MAX_OUTER_LOOP_OPTIMIZED 16
#define THRESHOLD_MULTIPLIER 5
#define OUTLIER_DRAM_PERCENTAGE 10

#define CacheBufferMultiplyer 2
#define numHashedAddr 16//*modified*32//1024//16
#define numHashedAddr_real 16

#define MAX_BUF 1024
#define HEX_BASE 16
#define SUB_STRING "0x"
#define COMMAND_STR "echo %s | sudo -S dmesg -c\n"


typedef float timeType;

typedef uint32_t dataType;
typedef float dummyType;
