import pandas as pd
import cudf
import time
import glob
import os
import dask_cudf
import numpy as np
import dask.array as da
import sys
import cupy as cp




# Define the path to your data directory
data_directory = "/mydata/"
file_pattern = os.path.join(data_directory, "chicago.csv")
file_path = '/mydata/chicago_10m.parquet'
filenames = glob.glob(file_pattern)
# file_size = os.path.getsize(file_path)  # File size in bytes

# df = dask_cudf.read_csv(file_path)
# num_rows = df.shape[0].compute()  # Get the number of rows

# # Step 3: Approximate the row size
# row_size = file_size / num_rows  # Bytes per row
# print(f"Approximate size of each row: {row_size:.2f} bytes")

# df = cudf.read_csv('/mydata/chicago.csv', nrows=200000000)

# Read the first 10 rows of the CSV
# Load a sample of the DataFrame
# df = pd.read_csv("/mydata/chicago.csv")

df = cudf.read_csv("/mydata/chicago.csv", nrows=10)

# Get the length of the 'Trip Seconds' column
length_of_trip_seconds = len(df['Trip Seconds'])

# Print the length
print("Length of 'Trip Seconds' column:", length_of_trip_seconds)

# Select the first row (you can select any row you want)
row = df.iloc[0]
 
# Calculate the size of the row in bytes
row_size = row.memory_usage(deep=True).sum()

print(f"Size of one row: {row_size} bytes\nrow {row}")

print(df.columns)

# sys.exit()
# # Select the specific columns
# selected_columns = df[['Trip Seconds', 'Trip Miles', 'Fare', 'Tips', 'Tolls', 'Extras']]

# # Compute and print the result
# print(selected_columns.compute())

# # Print the column names
# print("Column names:", df['Fare'].head(10))

# Debugging: Check if any files are found
if not filenames:
    print("No files found matching the pattern:", file_pattern)
else:
    print("Files found:", filenames)

# Start timing
# start_time = time.time()

df_filtered_miles = 1

df_filtered_fare = 0

df_filtered_tolls = 0

df_filtered_extras = 0


# # Drop rows where 'Trip Seconds' is null
# df_non_null = df.dropna(subset=['Trip Seconds'])

# # Filter rows where 'Trip Seconds' is greater than 1000
# filtered_rows = df_non_null[df_non_null['Trip Seconds'] > 1000]

# # Calculate total number of rows in the dataset
# total_rows = len(df_non_null)

# # Calculate number of rows where 'Trip Seconds' > 1000
# num_filtered_rows = len(filtered_rows)

# # Calculate sparsity
# sparsity = (num_filtered_rows / total_rows)

# # Print sparsity
# print(f"Sparsity of 'Trip Seconds' > 1000: {sparsity}")


# # Measure time for the entire block
# start_time = time.time()

# # Step 1: Read the Trip Seconds column
# df_seconds = dask_cudf.read_csv(
#     "/mydata/chicago.csv", 
#     usecols=['Trip Seconds'], 
#     nrows=10000000
# )S

# # Step 2: Filter rows where 'Trip Seconds' > threshold
# threshold_seconds = 1000  # for example, 10 minutes
# df_filtered_seconds = df_seconds[df_seconds['Trip Seconds'] > threshold_seconds]

# # Step 3: Load the Extras column
# df_extras = dask_cudf.read_csv(
#     "/mydata/chicago.csv", 
#     usecols=['Extras'], 
#     nrows=10000000
# )

# # Step 4: Merge the filtered seconds with the extras column
# df_filtered_combined = df_filtered_seconds.merge(df_extras, left_index=True, right_index=True)

# # Step 5: Calculate the total extras sum from the merged DataFrame
# total_extras_sum = df_filtered_combined['Extras'].sum().compute()  # Compute the total extras sum

# # Print the result
# print(f"Total Extras Sum: {total_extras_sum}")





# Step 3: Count the number of filtered rows
# filtered_indices = df_filtered_seconds.index.compute()



threshold_seconds = 8500  # for example, 10 minutes

# file = "/mydata/chicago_10m.parquet"
# nrows=10_000_000
# file = "/mydata/chicago_30m.parquet"
# nrows=30_000_000
file = "/mydata/chicago_100m.parquet"
nrows=100_000_000

# binary_file_path = "/mydata/chicago.bin"
# nrows=100_000_000
# binary_file_path = "/mydata/chicago_1b.bin"
# nrows=1000_000_000
# binary_file_path = "/mydata/chicago_250m.bin"
# nrows=250_000_000

binary_file_path = "/mydata/chicago_2b_trial.bin"
nrows = 211670894

start_time = time.time()

n_partitions = 1  # Number of partitions for Dask

# Step 1: Read the binary file using NumPy

# trip_seconds = np.empty(nrows, dtype=np.float32)
# trip_miles = np.empty(nrows, dtype=np.float32)
# fare = np.empty(nrows, dtype=np.float32)
# tips = np.empty(nrows, dtype=np.float32)
# tolls = np.empty(nrows, dtype=np.float32)
# extras = np.empty(nrows, dtype=np.float32)


pinned_mem_seconds = cp.cuda.alloc_pinned_memory(nrows * 4)  # 4 bytes per float32
trip_seconds = np.ndarray((nrows,), dtype=np.float32, buffer=pinned_mem_seconds)

pinned_mem_miles = cp.cuda.alloc_pinned_memory(nrows * 4)
trip_miles = np.ndarray((nrows,), dtype=np.float32, buffer=pinned_mem_miles)

pinned_mem_fare = cp.cuda.alloc_pinned_memory(nrows * 4)
fare = np.ndarray((nrows,), dtype=np.float32, buffer=pinned_mem_fare)

pinned_mem_tips = cp.cuda.alloc_pinned_memory(nrows * 4)
tips = np.ndarray((nrows,), dtype=np.float32, buffer=pinned_mem_tips)

pinned_mem_tolls = cp.cuda.alloc_pinned_memory(nrows * 4)
tolls = np.ndarray((nrows,), dtype=np.float32, buffer=pinned_mem_tolls)

pinned_mem_extras = cp.cuda.alloc_pinned_memory(nrows * 4)
extras = np.ndarray((nrows,), dtype=np.float32, buffer=pinned_mem_extras)

# Read the binary file into the NumPy arrays
with open(binary_file_path, 'rb') as f:
    f.readinto(trip_seconds)
    f.readinto(trip_miles)
    f.readinto(fare)
    f.readinto(tips)
    f.readinto(tolls)
    f.readinto(extras)

# Step 4: Start timer
print("Started loading data...")

start_time = time.time()


# Step 2: Convert the NumPy arrays to a cuDF DataFrame
df_trip_data = cudf.DataFrame({
    'Trip Seconds': cp.asarray(trip_seconds),
})

filtered_indices = df_trip_data['Trip Seconds'] > threshold_seconds

# Count the number of elements that satisfy the condition
num_elements = filtered_indices.sum()

# Print the result
print(f"Number of elements where 'Trip Seconds' > {threshold_seconds}: {num_elements}")



# Step 2: Convert the NumPy arrays to a cuDF DataFrame
df_trip_data = cudf.DataFrame({
    # 'Trip Seconds': cp.asarray(trip_seconds),
    'Trip Miles': cp.asarray(trip_miles),
})



# Step 3: Convert the cuDF DataFrame into a Dask cuDF DataFrame with partitions
# df_trip_data_dask = dask_cudf.from_cudf(df_trip_data, npartitions=n_partitions)



# Step 5: Filter the DataFrame based on the condition
df_filtered_miles = df_trip_data[filtered_indices] # df_trip_data[df_trip_data['Trip Seconds'] > threshold_seconds]

# Step 6: Calculate the total sums
total_miles_sum = df_filtered_miles['Trip Miles'].sum() # .compute()

max_trip_miles = df_filtered_miles['Trip Miles'].max()
print(f"Max Trip Miles: {max_trip_miles}")

end_time = time.time()
elapsed_time = end_time - start_time
print(f"Query 1 time - trip miles: {elapsed_time:.2f} seconds")



df_trip_data = cudf.DataFrame({
    # 'Trip Seconds': cp.asarray(trip_seconds),
    'Fare': cp.asarray(fare),
    # 'Trip Seconds': trip_seconds,
    # 'Fare': fare,
})

# Step 3: Convert the cuDF DataFrame into a Dask cuDF DataFrame with partitions
# df_trip_data_dask = dask_cudf.from_cudf(df_trip_data, npartitions=n_partitions)

# Step 4: Start timer
# start_time = time.time()

# Step 5: Filter the DataFrame based on the condition
df_filtered_miles = df_trip_data[filtered_indices] # df_trip_data[df_trip_data['Trip Seconds'] > threshold_seconds]

total_fare_sum = df_filtered_miles['Fare'].sum() # .compute()

end_time = time.time()
elapsed_time = end_time - start_time
print(f"Query 2 time - fare: {elapsed_time:.2f} seconds")
print(total_fare_sum)



df_trip_data = cudf.DataFrame({
    # 'Trip Seconds': cp.asarray(trip_seconds),
    'Tips': cp.asarray(tips),
    # 'Trip Seconds': trip_seconds,
    # 'Tips': tips,
})

# Step 3: Convert the cuDF DataFrame into a Dask cuDF DataFrame with partitions
# df_filtered_miles = dask_cudf.from_cudf(df_trip_data, npartitions=n_partitions)

df_filtered_miles = df_trip_data[filtered_indices] # df_trip_data[df_trip_data['Trip Seconds'] > threshold_seconds]

total_tips_sum = df_filtered_miles['Tips'].sum() #.compute()

end_time = time.time()
elapsed_time = end_time - start_time
print(f"Query 3 time - tip: {elapsed_time:.2f} seconds")


df_trip_data = cudf.DataFrame({
    # 'Trip Seconds': cp.asarray(trip_seconds),
    'Tolls': cp.asarray(tolls),
    # 'Trip Seconds': trip_seconds,
    # 'Tolls': tolls,
})

# Step 3: Convert the cuDF DataFrame into a Dask cuDF DataFrame with partitions
# df_filtered_miles = dask_cudf.from_cudf(df_trip_data, npartitions=n_partitions)

df_filtered_miles = df_trip_data[filtered_indices] #  df_trip_data[df_trip_data['Trip Seconds'] > threshold_seconds]

total_tolls_sum = df_filtered_miles['Tolls'].sum() # .compute()

end_time = time.time()
elapsed_time = end_time - start_time
print(f"Query 4 time - tolls: {elapsed_time:.2f} seconds")


df_trip_data = cudf.DataFrame({
    # 'Trip Seconds': cp.asarray(trip_seconds),
    'Extras': cp.asarray(extras),
    # 'Trip Seconds': trip_seconds,
    # 'Extras': extras,
})

# Step 3: Convert the cuDF DataFrame into a Dask cuDF DataFrame with partitions
# df_filtered_miles = dask_cudf.from_cudf(df_trip_data, npartitions=n_partitions)

df_filtered_miles = df_trip_data[filtered_indices] # df_trip_data[df_trip_data['Trip Seconds'] > threshold_seconds]


total_extras_sum = df_filtered_miles['Extras'].sum() # .compute()

end_time = time.time()
elapsed_time = end_time - start_time
print(f"Query 5 time - extra: {elapsed_time:.2f} seconds")

# Display the results
print(f"Total miles: {total_miles_sum}")
print(f"Total fare: {total_fare_sum}")
print(f"Total tips: {total_tips_sum}")
print(f"Total tolls: {total_tolls_sum}")
print(f"Total extras: {total_extras_sum}")

# Step 7: End timer and print elapsed time
end_time = time.time()
elapsed_time = end_time - start_time
print(f"Query time: {elapsed_time:.2f} seconds")

total_result = total_fare_sum - total_extras_sum - total_tolls_sum + total_tips_sum
dollar_per_mile = total_result / total_miles_sum

print(f"total_result: {total_result}")
print(f"dollar_per_mile: {dollar_per_mile}")


sys.exit(0)





df_miles = dask_cudf.read_parquet(
    file, 
    usecols=['Trip Seconds', 'Trip Miles'], 
    nrows=nrows,
    # chunksize=10_000_000  # Set the chunk size for partitions (adjust based on available memory)
)



# Step 5: Filter the full DataFrame based on the same condition
df_filtered_miles = df_miles[df_miles['Trip Seconds'] > threshold_seconds]

# Step 6: Calculate the total miles sum directly
total_miles_sum = df_filtered_miles['Trip Miles'].sum().compute()  # Compute the total miles sum

end_time = time.time()
elapsed_time = end_time - start_time
print(f"Query 1 trip miles: {elapsed_time:.2f} seconds")


# Step 4: Read both Trip Seconds and Trip Miles columns together to allow filtering
df_fare = dask_cudf.read_parquet(
    file, 
    usecols=['Trip Seconds', 'Fare'], 
    nrows=nrows
)

# Step 5: Filter the full DataFrame based on the same condition
df_filtered_fare = df_fare[df_fare['Trip Seconds'] > threshold_seconds]

# Step 6: Calculate the total miles sum directly
total_fare_sum = df_filtered_fare['Fare'].sum().compute()  # Compute the total miles sum

end_time = time.time()
elapsed_time = end_time - start_time
print(f"Query 2 fare: {elapsed_time:.2f} seconds")
# del df_fare, df_filtered_fare



# Step 4: Read both Trip Seconds and Trip Miles columns together to allow filtering
df_extras = dask_cudf.read_parquet(
    file, 
    usecols=['Trip Seconds', 'Extras'], 
    nrows=nrows
)

# Step 5: Filter the full DataFrame based on the same condition
df_filtered_extras = df_extras[df_extras['Trip Seconds'] > threshold_seconds]

# Step 6: Calculate the total miles sum directly
total_extras_sum = df_filtered_extras['Extras'].sum().compute()  # Compute the total miles sum

end_time = time.time()
elapsed_time = end_time - start_time
print(f"Query 3 Extras: {elapsed_time:.2f} seconds")

# # del df_extras, df_filtered_extras

# Step 4: Read both Trip Seconds and Trip Miles columns together to allow filtering
df_tips = dask_cudf.read_parquet(
    file, 
    usecols=['Trip Seconds', 'Tips'], 
    nrows=nrows
)

# Step 5: Filter the full DataFrame based on the same condition
df_filtered_tips = df_tips[df_tips['Trip Seconds'] > threshold_seconds]

# Step 6: Calculate the total miles sum directly
total_tips_sum = df_filtered_tips['Tips'].sum().compute()  # Compute the total miles sum

end_time = time.time()
elapsed_time = end_time - start_time
print(f"Query 4 Tips: {elapsed_time:.2f} seconds")

# # del df_tips, df_filtered_tips


# Step 4: Read both Trip Seconds and Trip Miles columns together to allow filtering
df_tolls = dask_cudf.read_parquet(
    file, 
    usecols=['Trip Seconds', 'Tolls'], 
    nrows=nrows
)

# Step 5: Filter the full DataFrame based on the same condition
df_filtered_tolls = df_tolls[df_tolls['Trip Seconds'] > threshold_seconds]

# Step 6: Calculate the total miles sum directly
total_tolls_sum = df_filtered_tolls['Tolls'].sum().compute()  # Compute the total miles sum

end_time = time.time()
elapsed_time = end_time - start_time
print(f"Query 5 Tolls: {elapsed_time:.2f} seconds")

# del df_tolls, df_filtered_tolls


# Print results
# print(f"Number of Filtered Rows: {num_filtered_rows}")
print(f"Total Miles Sum: {total_miles_sum}")
print(f"Total Fare Sum: {total_fare_sum}")
print(f"Total Extras Sum: {total_extras_sum}")
print(f"Total Tips Sum: {total_tips_sum}")
print(f"Total Tolls Sum: {total_tolls_sum}")
# print(f"Total Trip Sum: {total_trip_total_sum}")

total_result = total_fare_sum - total_extras_sum - total_tolls_sum + total_tips_sum
dollar_per_mile = total_result / total_miles_sum

print(f"total_result: {total_result}")
print(f"dollar_per_mile: {dollar_per_mile}")


# # Measure end time
# end_time = time.time()

# # Calculate elapsed time
# elapsed_time = end_time - start_time

# print(f"Elapsed Time: {elapsed_time:.2f} seconds")











# threshold_seconds = 1000  # for example, 10 minutes
# nrows=10000000
# # Step 4: Read both Trip Seconds and Trip Miles columns together to allow filtering
# df_miles = dask_cudf.read_csv(
#     "/mydata/chicago.csv", 
#     usecols=['Trip Seconds', 'Trip Miles'], 
#     nrows=nrows
# )

# # Step 5: Filter the full DataFrame based on the same condition
# df_filtered_miles = df_miles[df_miles['Trip Seconds'] > threshold_seconds]

# # Step 6: Calculate the total miles sum directly
# total_miles_sum = df_filtered_miles['Trip Miles'].sum().compute()  # Compute the total miles sum

# end_time = time.time()
# elapsed_time = end_time - start_time
# print(f"Query 1 trip miles: {elapsed_time:.2f} seconds")



# # Step 6: Find the maximum value in the 'Trip Miles' column
# max_trip_miles = df_filtered_miles['Trip Miles'].max().compute()

# # del df_miles, df_filtered_miles
# # Print the result
# print(f"Maximum Trip Miles: {max_trip_miles}")


# # Step 4: Read both Trip Seconds and Trip Miles columns together to allow filtering
# df_fare = dask_cudf.read_csv(
#     "/mydata/chicago.csv", 
#     usecols=['Trip Seconds', 'Fare'], 
#     nrows=nrows
# )

# # Step 5: Filter the full DataFrame based on the same condition
# df_filtered_fare = df_fare[df_fare['Trip Seconds'] > threshold_seconds]

# # Step 6: Calculate the total miles sum directly
# total_fare_sum = df_filtered_fare['Fare'].sum().compute()  # Compute the total miles sum

# end_time = time.time()
# elapsed_time = end_time - start_time
# print(f"Query 2 fare: {elapsed_time:.2f} seconds")
# del df_fare, df_filtered_fare



# # Step 4: Read both Trip Seconds and Trip Miles columns together to allow filtering
# df_extras = dask_cudf.read_csv(
#     "/mydata/chicago.csv", 
#     usecols=['Trip Seconds', 'Extras'], 
#     nrows=nrows
# )

# # Step 5: Filter the full DataFrame based on the same condition
# df_filtered_extras = df_extras[df_extras['Trip Seconds'] > threshold_seconds]

# # Step 6: Calculate the total miles sum directly
# total_extras_sum = df_filtered_extras['Extras'].sum().compute()  # Compute the total miles sum

# end_time = time.time()
# elapsed_time = end_time - start_time
# print(f"Query 3 Extras: {elapsed_time:.2f} seconds")

# # del df_extras, df_filtered_extras

# # Step 4: Read both Trip Seconds and Trip Miles columns together to allow filtering
# df_tips = dask_cudf.read_csv(
#     "/mydata/chicago.csv", 
#     usecols=['Trip Seconds', 'Tips'], 
#     nrows=nrows
# )

# # Step 5: Filter the full DataFrame based on the same condition
# df_filtered_tips = df_tips[df_tips['Trip Seconds'] > threshold_seconds]

# # Step 6: Calculate the total miles sum directly
# total_tips_sum = df_filtered_tips['Tips'].sum().compute()  # Compute the total miles sum

# end_time = time.time()
# elapsed_time = end_time - start_time
# print(f"Query 4 Tips: {elapsed_time:.2f} seconds")

# # del df_tips, df_filtered_tips


# # Step 4: Read both Trip Seconds and Trip Miles columns together to allow filtering
# df_tolls = dask_cudf.read_csv(
#     "/mydata/chicago.csv", 
#     usecols=['Trip Seconds', 'Tolls'], 
#     nrows=nrows
# )

# # Step 5: Filter the full DataFrame based on the same condition
# df_filtered_tolls = df_tolls[df_tolls['Trip Seconds'] > threshold_seconds]

# # Step 6: Calculate the total miles sum directly
# total_tolls_sum = df_filtered_tolls['Tolls'].sum().compute()  # Compute the total miles sum

# end_time = time.time()
# elapsed_time = end_time - start_time
# print(f"Query 5 Tolls: {elapsed_time:.2f} seconds")

# # del df_tolls, df_filtered_tolls


# # Print results
# # print(f"Number of Filtered Rows: {num_filtered_rows}")
# print(f"Total Miles Sum: {total_miles_sum}")
# print(f"Total Fare Sum: {total_fare_sum}")
# print(f"Total Extras Sum: {total_extras_sum}")
# print(f"Total Tips Sum: {total_tips_sum}")
# print(f"Total Tolls Sum: {total_tolls_sum}")
# # print(f"Total Trip Sum: {total_trip_total_sum}")

# total_result = total_fare_sum - total_extras_sum - total_tolls_sum + total_tips_sum
# dollar_per_mile = total_result / total_miles_sum

# print(f"total_result: {total_result}")
# print(f"dollar_per_mile: {dollar_per_mile}")


# # Measure end time
# end_time = time.time()

# # Calculate elapsed time
# elapsed_time = end_time - start_time

# print(f"Elapsed Time: {elapsed_time:.2f} seconds")

