import itertools
import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

is_with_repetitions = False

# Reading all the results from the results folder and concatenating them into a single dataframe
RESULTS_DIR = "results"
df = None
for file in os.listdir(RESULTS_DIR):
    if file.endswith('.csv') and file.startswith("results_node"):
        if df is None:
            df = pd.read_csv(os.path.join(RESULTS_DIR, file))
        else:
            df = pd.concat([df, pd.read_csv(os.path.join(RESULTS_DIR, file))])

# Filtering to have only the relevant columns
time_columns = ['Stage 1 Time', 'Stage 2 Time', 'Stage 3 Time', 'Stage 4 Time']
df = df[['Node Name'] + time_columns]

# Averaging the times for each node
df = df.groupby('Node Name').mean()


# adding a column by stage with the std deviation of the time
for stage in time_columns:
    df[f"{stage} std"] = df[stage].std()

# saving df to a new csv file
df.to_csv(os.path.join(RESULTS_DIR, "results_all.csv"), index=False)


# Generate combinations of nodes for each stage
nodes = df.index.tolist()

if is_with_repetitions:
    # Generate all possible combinations with repetitions
    combinations = list(itertools.product(nodes, repeat=len(time_columns)))

    # Generate DataFrame with combinations
    result_rows = []
    for combination in combinations:
        # Create a combination name (e.g., n5n6n9n1)
        combination_name = "".join(f"n{node.split()[1]}" for node in combination)
        # Match stages to nodes and extract corresponding times
        combination_values = [df.loc[node, stage] for node, stage in zip(combination, time_columns)]
        result_rows.append([combination_name] + combination_values)
else:
    # Generate all possible combinations without repetitions
    combinations = list(itertools.permutations(nodes, len(time_columns)))

    # Generate DataFrame with combinations
    result_rows = []
    for combination in combinations:
        # Create a combination name (e.g., n5n6n9n1)
        combination_name = ",".join(f"VM{node.split()[1]}" for node in combination)
        # Match stages to nodes and extract corresponding times
        combination_values = [df.loc[node, stage] for node, stage in zip(combination, time_columns)]
        result_rows.append([combination_name] + combination_values)


# Create the final DataFrame
columns = ["Combination"] + time_columns
result_df = pd.DataFrame(result_rows, columns=columns)

# Add a column with the total time
result_df["Total Time"] = result_df[time_columns].sum(axis=1)

# Add a column with rank regarding the total time (lower time is better)
result_df["Rank"] = result_df["Total Time"].rank()

# show the mean and std of the Total Time column
print("Total Time AVG",result_df["Total Time"].mean())
print("Total Time STD",result_df["Total Time"].std())

#show the max and min of the Total Time column
print("Total Time MAX",result_df["Total Time"].max())
print("Total Time MIN",result_df["Total Time"].min())

# Sort the DataFrame by rank
result_df = result_df.sort_values("Rank")

# Exporting the result to an Excel file
result_df.to_excel(f"plots/png/results_combinations{"_with_repetitions" if is_with_repetitions else ""}.xlsx", index=False)

plt.figure(figsize=(10, 5))

# Plot a histogram of Total Time
# result_df["Total Time"].plot(kind='hist', bins=15, title="ML Pipeline Time")
plt.hist(result_df["Total Time"], bins='auto', edgecolor='white', linewidth=0.3)
plt.xlabel("Execution Time (seconds)")
plt.ylabel("Frequency")
plt.savefig(f"plots/png/total_time_histogram{"_with_repetitions" if is_with_repetitions else ""}.png", bbox_inches='tight')
plt.savefig(f"plots/pdf/total_time_histogram{"_with_repetitions" if is_with_repetitions else ""}.pdf", bbox_inches='tight')
# plt.show()


# Saving the plot to a file
# Calculate the histogram data
hist, bin_edges = np.histogram(result_df["Total Time"], bins='auto')

# Create a DataFrame from the histogram data
hist_df = pd.DataFrame({'Bin Start': bin_edges[:-1], 'Bin End': bin_edges[1:], 'Frequency': hist})

# Save the DataFrame to a CSV file
hist_df.to_csv(f"plots/png/total_time_histogram_data{'_with_repetitions' if is_with_repetitions else ''}.csv", index=False)

# filtering result_df to have the last 10 rows on the 5 bins with the highest frequency
# Identify the 5 bins with the highest frequency
top_bins = hist_df.nlargest(10, 'Frequency')

# Filter result_df to include only the rows that fall within these bins
filtered_df = result_df[result_df["Total Time"].apply(lambda x: any((x >= row['Bin Start']) & (x < row['Bin End']) for _, row in top_bins.iterrows()))]

# Select the last 10 rows from the filtered DataFrame
six_rows_each_bin = pd.DataFrame()
# Iterate through each of the top bins
for _, row in top_bins.iterrows():
    bin_start = row['Bin Start']
    bin_end = row['Bin End']

    # Filter result_df to include only the rows that fall within the current bin
    bin_df = result_df[(result_df["Total Time"] >= bin_start) & (result_df["Total Time"] < bin_end)]

    # Select the 6 rows from the filtered DataFrame for the current bin
    first_3_rows = bin_df.head(3)
    last_3_rows = bin_df.tail(3)

    # Append the 6 rows to the final DataFrame
    six_rows_each_bin = pd.concat([six_rows_each_bin, first_3_rows, last_3_rows])

# ordering by rank
six_rows_each_bin = six_rows_each_bin.sort_values("Rank")

# removing the rank column
six_rows_each_bin = six_rows_each_bin.drop(columns=["Rank"])

# saving to csv
six_rows_each_bin.to_csv(f"plots/png/six_rows_each_bin{'_with_repetitions' if is_with_repetitions else ''}.csv", index=False)

# print(result_df)