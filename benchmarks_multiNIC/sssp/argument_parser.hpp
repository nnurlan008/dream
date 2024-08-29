#ifndef ARGUMENT_PARSER_HPP
#define ARGUMENT_PARSER_HPP

// #include "global.hpp"
#include <iostream>
#include <stdio.h>
#include <string>
#include <string.h>
#include <vector>

using namespace std;

typedef unsigned int uint;


class ArgumentParser {
    private:

    public:
        string inputFilePath;
        bool runOnCPU;
        int sourceNode;
        bool hasSourceNode;


    ArgumentParser(int argc, char **argv);


};




#endif