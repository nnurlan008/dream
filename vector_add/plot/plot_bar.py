import subprocess
import time
import csv
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def make_number_list(inp_list):
    out_list = []
    for i in inp_list:
        out_list.append(float(i))
    return out_list

if __name__=="__main__":
	# Open file  
    # with open('/home/nurlan/Desktop/02_read-write/uvm/uvm_pagefault_Desktop_off_2GHz.csv') as file_obj: 
        
  

        # bar plot below:
        barWidth = 0.25
        fig = plt.subplots(figsize =(5, 4)) 
       
        # set height of bar 
        IT = [61.647, 119.74] 
        # ECE = rdma_lat # [28, 6, 16, 5] 
        # CSE = [29, 3, 24, 25] 
        
        # Set position ar on X axis 
        br1 = ['RDMA', 'UVM'] 
        # br2 = [x + barWidth for x in br1] 
        # br3 = [x + barWidth for x in br2] 
        
        # Make the plot
        plt.bar(br1, IT, color ='r', width = barWidth, 
                edgecolor ='grey') 
       
        for i in range(len(br1)):
            plt.text(i-0.1,IT[i] + 0.4, f'{IT[i]} ms')
        
        # Adding Xticks 
        # plt.xlabel('Data Size in Byte', fontweight='bold', fontsize = 15) 
        plt.ylabel('Execution time in ms', fontweight='bold', fontsize = 15) 
        # plt.xticks([r for r in range(len(IT))], 
        #         br1)
        # plt.xlim(x_axis[0], x_axis[-1])
        # plt.ylim(0, max(remote_bw))
        # plt.legend()
        plt.title("Vector Addition 512KB Request Size")
        plt.show() 








        