import subprocess
import time
import csv
import matplotlib.pyplot as plt
import pandas as pd

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
        print(x_axis)
        print(len(y_axis))
        


        plt.ylabel("BW [Gbps]")
        plt.xlabel("Data [bytes]")
        plt.title("BW vs Data size")
        plt.grid(which='major', color='#DDDDDD', linewidth=0.8)
        plt.grid(which='minor', color='#EEEEEE', linestyle=':', linewidth=0.5)
        plt.minorticks_on()
        plt.plot(x_axis[:20], y_axis[:20], 'bo-')
        plt.ticklabel_format(style='plain', axis='x')
        plt.xlim(0, max(x_axis[:20]))
        plt.ylim(0, max(y_axis[:20]))

        # plt.xticks(x_axis[:20] + x_axis[:20], [f'{val:.0e}' for val in x_axis[:20] + x_axis[:20]])
        plt.show()


        # Iterate over each row in the csv  
        # file using reader object 
        # for row in reader_obj: 
        #     print(row)
