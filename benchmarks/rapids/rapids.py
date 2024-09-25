import pandas as pd
import cudf
import time
import glob
import os

# Define the path to your data directory
data_directory = "/mydata/nyc_2015/"
file_pattern = os.path.join(data_directory, "yellow_tripdata_2015-*.parquet")
filenames = glob.glob(file_pattern)

# Debugging: Check if any files are found
if not filenames:
    print("No files found matching the pattern:", file_pattern)
else:
    print("Files found:", filenames)

# Start measuring time
start_time = time.time()

# Initialize an empty DataFrame to concatenate results
all_long_trips = cudf.DataFrame()

# Specify the columns you want to read
required_columns = [
    'trip_distance'
    ,'fare_amount'
    ,'extra'
    ,'mta_tax'
    ,'tip_amount'
    ,'tolls_amount'
    ,'improvement_surcharge'
    ,'congestion_surcharge'
    ,'airport_fee'
]

# Loop through each file and process
for filename in filenames:
    # Load only the necessary columns into GPU memory
    df = cudf.read_parquet(filename, columns=required_columns)
    print(df.head())
    # Filter trips where trip distance >= 30 miles
    long_trips = df[df['trip_distance'] >= 30]
    
    # Append the filtered trips to the all_long_trips DataFrame
    all_long_trips = cudf.concat([all_long_trips, long_trips], ignore_index=True)

# Check the data types of relevant columns
print("Data types before conversion:")
print(all_long_trips.dtypes)

# fare_amount = 1

# Convert relevant columns to numeric types if necessary
trip_distance = all_long_trips['trip_distance'].astype('float32')
fare_amount = all_long_trips['fare_amount'].astype('float32')
extra = all_long_trips['extra'].astype('float32')
mta_tax = all_long_trips['mta_tax'].astype('float32')
tip_amount = all_long_trips['tip_amount'].astype('float32')
tolls_amount = all_long_trips['tolls_amount'].astype('float32')
improvement_surcharge = all_long_trips['improvement_surcharge'].astype('float32')

# Attempt to convert congestion_surcharge and airport_fee
congestion_surcharge = (
    cudf.to_numeric(all_long_trips['congestion_surcharge'], errors='coerce')
)
airport_fee = (
    cudf.to_numeric(all_long_trips['airport_fee'], errors='coerce')
)

# Check for any NaNs that may have resulted from conversion errors
if congestion_surcharge.isnull().any():
    print("There are NaN values in congestion_surcharge after conversion.")
if airport_fee.isnull().any():
    print("There are NaN values in airport_fee after conversion.")

# Q1-Q5: Calculate total earnings for all long trips
b = (
    fare_amount 
    + extra 
    + mta_tax 
    + tip_amount 
    + tolls_amount 
    + improvement_surcharge 
    + congestion_surcharge 
    + airport_fee
)

# Final query: Calculate the average dollar per mile for all long trips
avg_dollar_per_mile = (b / all_long_trips['trip_distance']).mean()

# End measuring time
end_time = time.time()

# Total time taken
total_time = end_time - start_time

print(f"Average dollar/mile for trips >= 30 miles: {avg_dollar_per_mile}")
print(f"Total time taken: {total_time} seconds")
