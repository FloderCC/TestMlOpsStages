
import os
import pandas as pd
import matplotlib.pyplot as plt

# Reading all the results from the results folder and concatenating them into a single dataframe
RESULTS_DIR = "results"
df = None
for file in os.listdir(RESULTS_DIR):
    if file.endswith('.csv') and file.startswith("results_node"):
        if df is None:
            df = pd.read_csv(os.path.join(RESULTS_DIR, file))
        else:
            df = pd.concat([df, pd.read_csv(os.path.join(RESULTS_DIR, file))])

datasets_to_use = ['DeepSlice', 'NSR', 'IoT-APD']

# Filtering the results to only include the datasets of interest
df = df[df['Dataset'].isin(datasets_to_use)]

# saving the concatenated dataframe to a new csv file
# df.to_csv(os.path.join(RESULTS_DIR, "results_all.csv"), index=False)

def plot_all():
    # Step 2: Prepare data for plotting
    node_labels = df.apply(lambda row: f"{row['Node Name']} (CPUs: {row['Node CPUs']}, RAM: {row['Node RAM']}GB)",
                           axis=1)
    node_labels = node_labels.drop_duplicates()

    # Select relevant columns
    # time_columns = ['Stage 1 Time', 'Stage 2 Time', 'Stage 3 Time', 'Stage 4 Time', 'Stage 5 Time']
    df_time = df[['Node Name'] + time_columns]

    # Calculate the mean and standard deviation for each stage and node
    node_stage_means = df_time.groupby('Node Name').mean()
    node_stage_stds = df_time.groupby('Node Name').std()


    # Step 3: Plot the data with error bars (use the standard deviation as error)
    ax = node_stage_means.plot(kind='bar', stacked=False, figsize=(8, 6), colormap='tab20', yerr=node_stage_stds,
                               capsize=5)

    # Set the y-axis to logarithmic scale
    # ax.set_yscale('log')

    # Customizing the plot
    plt.title('Execution Times of Stages by Nodes')
    plt.xlabel('Node Name')
    plt.ylabel('Execution Time (seconds)')
    ax.set_xticklabels(node_labels, rotation=45, ha="right")
    plt.legend(title='Stages', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()

    # Saving the plot to a file
    plt.savefig(os.path.join("plots/png", "all_execution_times_avg_by_datasets.png"))
    # Display the plot
    # plt.show()

def plot_all_by_dataset():
    # Updating the dataset names to contain its dimensions between parenthesis
    dataset_dimensions = [
        ['DeepSlice', '63167 x 10'],
        ['NSR', '31583 x 17'],
        ['IoT-APD', '10845 x 17'],
        ['UNAC', '389 x 23'],
        ['KPI-KQI', '165 x 14'],
    ]
    df['Dataset'] = df['Dataset'].apply(lambda dataset: f"{dataset} ({[dim[1] for dim in dataset_dimensions if dim[0] == dataset][0]})")


    # Step 1: Prepare data for plotting
    datasets = df['Dataset'].unique()
    # time_columns = ['Stage 1 Time', 'Stage 2 Time', 'Stage 3 Time', 'Stage 4 Time', 'Stage 5 Time']

    # Step 1: Prepare to calculate the global Y-axis limits
    global_ymin, global_ymax = float('inf'), float('-inf')

    # First loop: Calculate the global ymin and ymax
    for dataset in datasets:
        df_dataset = df[df['Dataset'] == dataset]  # Filter by dataset
        df_time = df_dataset[['Node Name'] + time_columns]
        node_stage_means = df_time.groupby('Node Name').mean()

        # Find the maximum and minimum values (before log scaling)
        global_ymin = min(global_ymin, node_stage_means.min().min())
        global_ymax = max(global_ymax, node_stage_means.max().max())

    # Step 2: Now plot each dataset with the same Y-axis range
    for dataset in datasets:
        df_dataset = df[df['Dataset'] == dataset]  # Filter by dataset

        # Create custom labels for the x-axis
        node_labels = df_dataset.apply(
            lambda row: f"{row['Node Name']} (CPUs: {row['Node CPUs']}, RAM: {row['Node RAM']}GB)",
            axis=1)
        node_labels = node_labels.drop_duplicates()

        # Select relevant columns for the current dataset
        df_time = df_dataset[['Node Name'] + time_columns]

        # Calculate the mean and standard deviation for each stage and node
        node_stage_means = df_time.groupby('Node Name').mean()
        node_stage_stds = df_time.groupby('Node Name').std()

        # Step 3: Plot the data with error bars (use the standard deviation as error)
        ax = node_stage_means.plot(kind='bar', stacked=False, figsize=(8, 6), colormap='tab20', yerr=node_stage_stds,
                                   capsize=5)

        # Set the y-axis to logarithmic scale
        # ax.set_yscale('log')

        # Set the Y-axis range to be the same across all plots
        ax.set_ylim(global_ymin, global_ymax)

        # Customizing the plot
        plt.title(f'Execution Times of Stages by Nodes - Dataset: {dataset}')
        plt.xlabel('Node Name')
        plt.ylabel('Execution Time (seconds)')
        ax.set_xticklabels(node_labels, rotation=45, ha="right")
        plt.legend(title='Stages', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()

        # Saving the plot to a file for the current dataset
        plot_filename = f"execution_times_for_dataset_{dataset}.png"
        plt.savefig(os.path.join("plots/png", plot_filename))

        # Display the plot
        # plt.show()


# time_columns = ['Stage 1 Time', 'Stage 2 Time', 'Stage 3 Time', 'Stage 5 Time']
time_columns = ['Stage 4 Time']

plot_all()
plot_all_by_dataset()