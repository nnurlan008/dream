import subprocess

command = "sudo ./gpu-client read 192.168.1.155 read 45"

if __name__=="__main__":
	
	try:
	    # Run the command and capture the output as a string
	    output = subprocess.check_output(command, shell=True, text=True)
	    print("Command output:", output)
	    
	except subprocess.CalledProcessError as e:
	    print(f"Error running the command: {e}")
