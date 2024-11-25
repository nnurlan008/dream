import pandas as pd

# File paths
csv_file = '/mydata/chicago.csv'
bin_file = '/mydata/chicago_python.bin'

# Initialize variables
columns_to_extract = ['Trip Seconds', 'Trip Miles', 'Fare', 'Tips', 'Tolls', 'Extras']
total_lengths = {col: 0 for col in columns_to_extract}

# Open the binary file for writing
with open(bin_file, 'wb') as bin_f:
    # Read the CSV in chunks
    chunksize = 100000  # Adjust chunk size based on your available memory
    for chunk in pd.read_csv(csv_file, usecols=columns_to_extract, chunksize=chunksize):
        # Update column lengths
        for col in columns_to_extract:
            total_lengths[col] += len(chunk[col])
        
        # Convert the chunk to binary and write to file
        chunk.to_records(index=False).tofile(bin_f)

# Print the total length of each column
for col, length in total_lengths.items():
    print(f"Length of {col}: {length}")