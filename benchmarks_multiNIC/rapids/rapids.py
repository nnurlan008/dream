import pandas as pd
import cudf
import time
import glob
import os
import dask_cudf

# Define the path to your data directory
data_directory = "/mydata/"
file_pattern = os.path.join(data_directory, "chicago.csv")
file_path = '/mydata/chicago.csv'
filenames = glob.glob(file_pattern)
# file_size = os.path.getsize(file_path)  # File size in bytes

# df = dask_cudf.read_csv(file_path)
# num_rows = df.shape[0].compute()  # Get the number of rows

# # Step 3: Approximate the row size
# row_size = file_size / num_rows  # Bytes per row
# print(f"Approximate size of each row: {row_size:.2f} bytes")

# df = cudf.read_csv('/mydata/chicago.csv', nrows=200000000)

# # Read the CSV file
# df = dask_cudf.read_csv("/mydata/chicago.csv", nrows=10)

# # Print the column names
# print("Column names:", df['Fare'].head(10))

# Debugging: Check if any files are found
if not filenames:
    print("No files found matching the pattern:", file_pattern)
else:
    print("Files found:", filenames)

# Start timing
start_time = time.time()

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


threshold_seconds = 1000  # for example, 10 minutes
nrows=10000000
# Step 4: Read both Trip Seconds and Trip Miles columns together to allow filtering
df_miles = dask_cudf.read_csv(
    "/mydata/chicago.csv", 
    usecols=['Trip Seconds', 'Trip Miles'], 
    nrows=nrows
)

# Step 5: Filter the full DataFrame based on the same condition
df_filtered_miles = df_miles[df_miles['Trip Seconds'] > threshold_seconds]

# Step 6: Calculate the total miles sum directly
total_miles_sum = df_filtered_miles['Trip Miles'].sum().compute()  # Compute the total miles sum

end_time = time.time()
elapsed_time = end_time - start_time
print(f"Query 1 trip miles: {elapsed_time:.2f} seconds")



# Step 6: Find the maximum value in the 'Trip Miles' column
max_trip_miles = df_filtered_miles['Trip Miles'].max().compute()

# Print the result
print(f"Maximum Trip Miles: {max_trip_miles}")


# Step 4: Read both Trip Seconds and Trip Miles columns together to allow filtering
df_fare = dask_cudf.read_csv(
    "/mydata/chicago.csv", 
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




# Step 4: Read both Trip Seconds and Trip Miles columns together to allow filtering
df_extras = dask_cudf.read_csv(
    "/mydata/chicago.csv", 
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




# Step 4: Read both Trip Seconds and Trip Miles columns together to allow filtering
df_tips = dask_cudf.read_csv(
    "/mydata/chicago.csv", 
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




# Step 4: Read both Trip Seconds and Trip Miles columns together to allow filtering
df_tolls = dask_cudf.read_csv(
    "/mydata/chicago.csv", 
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





# # Step 4: Read both Trip Seconds and Trip Miles columns together to allow filtering
# df_trip_total = dask_cudf.read_csv(
#     "/mydata/chicago.csv", 
#     usecols=['Trip Seconds', 'Trip Total'], 
#     nrows=10000000
# )

# # Step 5: Filter the full DataFrame based on the same condition
# df_filtered_trip_total = df_trip_total[df_trip_total['Trip Seconds'] > threshold_seconds]

# # Step 6: Calculate the total miles sum directly
# total_trip_total_sum = df_filtered_trip_total['Trip Total'].sum().compute()  # Compute the total miles sum




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


# Measure end time
end_time = time.time()

# Calculate elapsed time
elapsed_time = end_time - start_time

print(f"Elapsed Time: {elapsed_time:.2f} seconds")

