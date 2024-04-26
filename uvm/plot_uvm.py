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
    with open('/home/nurlan/Desktop/02_read-write/uvm/uvm_pagefault_Desktop_off_2GHz.csv') as file_obj: 
        
        # Create reader object by passing the file  
        # object to reader method 
        reader_obj = csv.reader(file_obj) 
        reader_obj = pd.DataFrame(reader_obj)
        # print(reader_obj)
        # remote_bw = make_number_list(list(reader_obj[2][1:]))
        # remote_lat = make_number_list(list(reader_obj[3][1:]))
        # print(reader_obj)
        x_axis = range(len(reader_obj))
        # print(remote_lat)
        # x_axis = ['4B','8B','16B','32B','64B','128B','256B','512B','1KB','2KB','4KB','8KB','16KB','32KB',\
        #           '64KB','128KB','256KB','512KB','1MB','2MB','4MB']
        my_list = list(reader_obj[0])
        
        y_axis = [] # [i for i in my_list if i > 1000]
        lat = []
        data_size = []
        index = 0
        index_s = -1
        index_e = 0
        for i in my_list:
            if int(i) > 1000:
                y_axis.append(int(i))
                lat.append(int(i))
                if index_s == -1:
                    index_s = x_axis[index]
                    print(index_s)
                else:
                    index_e = index 
                    data_size.append((index_e-index_s)*32)
                    # print(index_s)
                    # print(index_e)
                    # print(index_e-index_s)
                    index_s = index_e
                    index_e = 0
            else: 
                y_axis.append(0)
            index += 1
        lat = lat[:-1]
        i = 0
        while i < len(lat):
            lat[i] = lat[i]/1815
            i += 1
        # lat = lat/2505000000
        print(data_size)
        print(lat)

        # line-like plot:
        # plt.plot(x_axis, y_axis, linestyle="-")
        # plt.xlim(x_axis[0]-100, x_axis[-1])
        # plt.ylim(0, max(y_axis))
        # plt.show()
        
        # bar plot below:
        barWidth = 0.25
        fig = plt.subplots(figsize =(12, 8)) 
        lat = lat[:20]
        data_size = data_size[0:20]
        # set height of bar 
        IT = lat # [12, 30, 1, 8] 
        # ECE = rdma_lat # [28, 6, 16, 5] 
        # CSE = [29, 3, 24, 25] 
        
        # Set position ar on X axis 
        br1 = np.arange(len(IT)) 
        # br2 = [x + barWidth for x in br1] 
        # br3 = [x + barWidth for x in br2] 
        
        # Make the plot
        plt.bar(br1, IT, color ='r', width = barWidth, 
                edgecolor ='grey') 
        # plt.bar(br2, ECE, color ='g', width = barWidth, 
        #         edgecolor ='grey', label ='Through NIC') 
        # plt.bar(br3, CSE, color ='b', width = barWidth, 
        #         edgecolor ='grey', label ='CSE') 
        for i in range(len(data_size)):
            plt.text(i-0.15,lat[i]+0.9,f'{lat[i]:.3f}')
        
        # Adding Xticks 
        plt.xlabel('Data Size in Byte', fontweight='bold', fontsize = 15) 
        plt.ylabel('Access latency in usec', fontweight='bold', fontsize = 15) 
        plt.xticks([r for r in range(len(IT))], 
                data_size)
        # plt.xlim(x_axis[0], x_axis[-1])
        # plt.ylim(0, max(remote_bw))
        # plt.legend()
        plt.title("Access Latency vs Data Size for UVM with RTX 4060 2.505GHz")
        plt.show() 







        
        # plt.ylabel("BW [GBps]")
        # plt.xlabel("Data [bytes]")
        # plt.title("BW vs Data Size for Remote Transfer with CX4 25Gb")
        # plt.grid(which='major', color='#DDDDDD', linewidth=0.8)
        # plt.grid(which='minor', color='#EEEEEE', linestyle=':', linewidth=0.5)
        # plt.minorticks_on()
       
        # # plt.ticklabel_format(style='plain', axis='x')
        
        # 

        # x_axis = [    4,       8,         16,      32,       64.0,    128.0,    256.0,   512.0,    1024.0,    2048.0,    4096.0,   8192.0, 16384.0,   32768.0, 65536.0,  131072.0,  262144.0,  524288.0,  1048576.0, 2097152.0, 4194304.0, 8388608.0, 16777216.0, 33554432.0, 67108864.0, 134217728]
        # y_axis = [0.000647, 0.001280, 0.002422, 0.005180, 0.011347, 0.019950, 0.041880, 0.089278, 0.162939, 0.356653, 0.645551, 1.360615, 2.184925, 4.441805, 7.553267, 11.805433, 15.775226, 19.340673, 21.738344, 23.264219, 23.953030, 24.328627, 24.544277, 24.720661,   24.658895,  24.815918]
        # x_axis = [4096, 61440, 126976, 258048] # IN BYTE
        # ucm_lat = [1.851, 7.232, 13.921, 27.233] # in useconds
        # rdma_lat = [5.063, 25.369, 44.723, 88.281]
        # print(x_axis)
 

        