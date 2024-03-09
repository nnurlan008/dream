#!/bin/bash
make
var=13
substring="message from passive/server side"

for i in $(seq 231000 10000000)
do
    # echo "Iteration $i"
    output="$(sudo ./gpu-client read 192.168.1.155 9700 $i)"
    # echo "newline: \n\n"

    if [[ $output == *"$substring"* ]]; then
        echo "Substring '$substring' found in the string"
        echo "The output is: $output"
        echo "newline \n\n"
        echo "Success with iteration: $i"
        break
    else
    	if [[ `expr $i % 1000` -eq 0 ]]; then
    		echo "Iteration $i"
    	fi
        # echo "Substring '$substring' not found in the string"
        
    fi

done
