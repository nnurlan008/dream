import subprocess
import time
import csv
from csv import DictWriter

command = ["sudo", "./gpu-client", "read", "192.168.1.155", "read", "1", "4096"]

field_names = ['ID', 'Byte', 'Gbps']

if __name__=="__main__":
	num_test = 10
	output = ""
	bandwidth = 45.4
	msg_size = 2048
	num_msg = 1
	size = 16
	average_bw = []
	while size < 16385:

		command[-1] = str(size)
		command[-2] = str(num_msg)
		print(command)
		bandwidth = 0
		i = 0
		while i < num_test:
			# bandwidth = 0
			
			
			try:
				# Run the command and capture the output as a string
				output = subprocess.check_output(command, text=True)
				if "Function: main line number: 761 bandwidth: " in output:
					index = output.find("Function: main line number: 761 bandwidth: ") + \
							len("Function: main line number: 761 bandwidth: ")
					print(output[index:index+9])
					bandwidth += float(output[index:index+9])
					i += 1
			except subprocess.CalledProcessError as e:
				print(f"Error running the command: {e}")
				# exit()
			time.sleep(1)
		bandwidth = bandwidth/num_test
		print("data in B: ", size, " avg bw: ", bandwidth)
		with open('output.csv', 'a') as f_object:
 
			# Pass the file object and a list
			# of column names to DictWriter()
			# You will get a object of DictWriter
			my_res = {"Byte": size, "Gbps": bandwidth}
			
			dictwriter_object = DictWriter(f_object, fieldnames=field_names)
		
			# Pass the dictionary as an argument to the Writerow()
			dictwriter_object.writerow(my_res)
		
			# Close the file object
			f_object.close()
		size *= 2
	
