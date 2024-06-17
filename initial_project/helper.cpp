#include<stdio.h>
#include <cstdint>
#include <vector>
#include <algorithm>
#include <iomanip>
#include <mutex>
#include <string.h>

#include<errno.h>
#include<fcntl.h>
#include<string.h>
#include<unistd.h>
#include<sys/mman.h>
#include <vector>

#include "helper.h"

using namespace std;

#define MAX_BUF 1024
#define HEX_BASE 16
#define SUB_STRING "0x"
#define COMMAND_STR "echo %s | sudo -S dmesg -c\n"


long long unsigned int GetPhysicalAddress_KernelLog(void)
{
    FILE *fp;
    char kernel_log[MAX_BUF];
    char password[] = "123456";
    char command[MAX_BUF];
    
    // Construct the command
    sprintf(command, COMMAND_STR , password);

    // Open the command for reading
    fp = popen(command, "r");
    if (fp == NULL) {
        printf("Failed to run command\n");
        exit(1);
    }

    // Read the output a line at a time
    while (fgets(kernel_log, MAX_BUF, fp) != NULL) {
        //printf("%s", kernel_log);
    }

    //printf("%s", kernel_log);

    // Close the pipe
    pclose(fp);
    
    char *substr = SUB_STRING;
    char *result = strstr(kernel_log, substr);
    int index;
    if (result) {
        index = result - kernel_log;
        //printf("Substring found at index: %d\n", index);
    } else {
        printf("No Physical address found!\n");
        return 1;//error code
    }
    //printf("len: %d\n",strlen(path));
    int len_hex = (strlen(kernel_log) - (index));
    char *hex = (char*) malloc(sizeof(char)*len_hex);
    strncpy(hex, kernel_log + (index), len_hex);
    //printf("strlen(hex): %d\n",strlen(hex));
    //printf("strlen(path) - (index): %d\n",strlen(path) - (index));
    //printf("path + index: %c\n", path[index]);
    hex[len_hex - 1] = '\0';

    char *ptr;
    long long unsigned int decimal = strtol(hex, &ptr, HEX_BASE);

    //printf("Hex: %s, Decimal: 0x%llx\n", hex, decimal);
    return decimal;
}

vector<uint64_t>* getRandomIndices(uint64_t len, uint64_t nIndices) {
  vector<uint64_t> *randomIndices = new vector<uint64_t>();
  vector<uint64_t> allIndices;
  for(uint64_t i = 0; i < len; i++) {
    allIndices.push_back(i);
  }

  random_shuffle(allIndices.begin(), allIndices.end());

  for(uint64_t i = 0; i < nIndices && i < len; i++) {
    randomIndices->push_back(allIndices[i]);
  }

  return randomIndices;
}