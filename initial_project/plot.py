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
    with open('/home/nurlan/Desktop/02_read-write/output.csv') as file_obj: 
        
        # Create reader object by passing the file  
        # object to reader method 
        reader_obj = csv.reader(file_obj) 
        reader_obj = pd.DataFrame(reader_obj)
        x_axis = make_number_list(list(reader_obj[1]))
        y_axis = make_number_list(list(reader_obj[2]))
        # x_axis = [    4,       8,         16,      32,       64.0,    128.0,    256.0,   512.0,    1024.0,    2048.0,    4096.0,   8192.0, 16384.0,   32768.0, 65536.0,  131072.0,  262144.0,  524288.0,  1048576.0, 2097152.0, 4194304.0, 8388608.0, 16777216.0, 33554432.0, 67108864.0, 134217728]
        # y_axis = [0.000647, 0.001280, 0.002422, 0.005180, 0.011347, 0.019950, 0.041880, 0.089278, 0.162939, 0.356653, 0.645551, 1.360615, 2.184925, 4.441805, 7.553267, 11.805433, 15.775226, 19.340673, 21.738344, 23.264219, 23.953030, 24.328627, 24.544277, 24.720661,   24.658895,  24.815918]
        x_axis = [4096, 61440, 126976, 258048] # IN BYTE
        ucm_lat = [1.851, 7.232, 13.921, 27.233] # in useconds
        rdma_lat = [5.063, 25.369, 44.723, 88.281]
        print(x_axis)
        print(len(y_axis))
        
        barWidth = 0.25
        fig = plt.subplots(figsize =(12, 8)) 
        
        # set height of bar 
        IT = ucm_lat # [12, 30, 1, 8] 
        ECE = rdma_lat # [28, 6, 16, 5] 
        # CSE = [29, 3, 24, 25] 
        
        # Set position of bar on X axis 
        br1 = np.arange(len(IT)) 
        br2 = [x + barWidth for x in br1] 
        # br3 = [x + barWidth for x in br2] 
        
        # Make the plot
        plt.bar(br1, IT, color ='r', width = barWidth, 
                edgecolor ='grey', label ='Through UVM') 
        plt.bar(br2, ECE, color ='g', width = barWidth, 
                edgecolor ='grey', label ='Through NIC') 
        # plt.bar(br3, CSE, color ='b', width = barWidth, 
        #         edgecolor ='grey', label ='CSE') 
        for i in range(len(x_axis)):
            plt.text(i-0.08,ucm_lat[i]+0.9,ucm_lat[i])
        for i in range(len(x_axis)):
            plt.text(i+0.15,rdma_lat[i]+0.9,rdma_lat[i])
        # Adding Xticks 
        plt.xlabel('Data Size in Byte', fontweight ='bold', fontsize = 15) 
        plt.ylabel('Transfer latency in usec', fontweight ='bold', fontsize = 15) 
        plt.xticks([r + 0.125 for r in range(len(IT))], 
                x_axis)
        
        plt.legend()
        plt.show() 

        # plt.ylabel("BW [Gbps]")
        # plt.xlabel("Data [bytes]")
        # plt.title("BW vs Data size")
        # plt.grid(which='major', color='#DDDDDD', linewidth=0.8)
        # plt.grid(which='minor', color='#EEEEEE', linestyle=':', linewidth=0.5)
        # plt.minorticks_on()
        # plt.plot(x_axis[:20], y_axis[:20], 'bo-')
        # plt.ticklabel_format(style='plain', axis='x')
        # plt.xlim(0, max(x_axis[:20]))
        # plt.ylim(0, max(y_axis[:20]))
        # plt.bar(br1, x_axis, color ='r', width = barWidth, edgecolor ='grey', label ='IT') 
        # plt.bar(br2, x_axis, color ='g', width = barWidth, edgecolor ='grey', label ='ECE') 

        # plt.xticks(x_axis[:20] + x_axis[:20], [f'{val:.0e}' for val in x_axis[:20] + x_axis[:20]])
        # plt.show()


        # Iterate over each row in the csv  
        # file using reader object 
        # for row in reader_obj: 
        #     print(row)
