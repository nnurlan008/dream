#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>
#include <string>
#include <fstream>
#include <errno.h>
#include <limits.h>
#include <iostream>
#include <unistd.h>
#include <vector>

using namespace std;

long long unsigned int GetPhysicalAddress_KernelLog(void);
vector<uint64_t>* getRandomIndices(uint64_t len, uint64_t nIndices);

static inline uint64_t xorBits(long x) {
    int sum = 0;
    while(x != 0) {
        //x&-x has a binary one where the lest significant one
        //in x was before. By applying XOR to this, the last
        //one becomes a zero.
        //
        //So, this overwrites all ones in x and toggles sum
        //every time until there are no ones left.
        //
        //This looks a bit strange but increases speed. Because
        //this is called very often (once for each mask and each
        //pfn), it should be done this way. Maybe, there is also
        //an even better way.
        sum^=1;
        x ^= (x&-x);
    }
    return sum;
}

static inline int countBits(long x) {
    int sum = 0;
    while(x != 0) {
        sum++;
        x ^= (x&-x);
    }
    return sum;
}