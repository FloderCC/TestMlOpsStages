
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Reading all the results from the results folder and concatenating them into a single dataframe
RESULTS_DIR = "results"
df = None
for file in os.listdir(RESULTS_DIR):
    if file.endswith('.csv') and file.startswith("results_node"):
        if df is None:
            df = pd.read_csv(os.path.join(RESULTS_DIR, file))
        else:
            df = pd.concat([df, pd.read_csv(os.path.join(RESULTS_DIR, file))])


# saving the concatenated dataframe to a new csv file
# df.to_csv(os.path.join(RESULTS_DIR, "results_all.csv"), index=False)

def plot_all():
    # Step 2: Prepare data for plotting

    df_time = df.copy()

    df_time['Node'] = df_time.apply(
        lambda row: f"{row['Node Name'].replace('Node','VM')} (CPUs: {row['Node CPUs']}, RAM: {row['Node RAM']}GB)", axis=1)

    # Select relevant columns
    df_time = df_time[['Node', 'Dataset'] + time_columns]

    # Melt the dataframe
    df_melted = pd.melt(df_time, id_vars=['Node', 'Dataset'], var_name='Stage', value_name='Time')


    # removing the word 'Time' from values of the Stage column
    df_melted['Stage'] = df_melted['Stage'].str.replace(' Time', '')

    # Plot using seaborn
    if len(time_columns) == 1:
        plt.figure(figsize=(3.5, 5))
    else:
        plt.figure(figsize=(8, 5))

    sns.barplot(data=df_melted, x='Stage', y='Time', hue='Node', ci='sd', palette='tab20', capsize=0.2)

    # Customizing the plot
    # plt.title('Execution Times of Stages by Nodes')
    plt.xlabel('Stage')
    plt.ylabel('Execution Time (seconds)')
    if len(time_columns) == 1:
        plt.legend(title='Virtual Machine', loc='upper left', bbox_to_anchor=(1.05, 1))
    else:
        plt.legend(title='Virtual Machine', loc='upper right')

    # Saving the plot to a file
    plot_filename = f"execution_times_by_nodes{"_s3" if len(time_columns)==1 else ""}"
    plt.savefig(os.path.join("plots/png", f"{plot_filename}.png"), bbox_inches='tight')
    plt.savefig(os.path.join("plots/pdf", f"{plot_filename}.pdf"), bbox_inches='tight')
    # Display the plot
    # plt.show()

def plot_all_by_dataset():
    # Updating the dataset names to contain its dimensions between parenthesis
    dataset_dimensions = [
        ['IoT-DNL', '477426 x 14'],
        ['NSL-KDD', '148517 x 42'],
        ['DeepSlice', '63167 x 10'],
        ['NSR', '31583 x 17'],
        ['IoT-APD', '10845 x 17'],
        ['UNAC', '389 x 23'],
        ['KPI-KQI', '165 x 14'],
    ]

    df['Dataset'] = df['Dataset'].apply(lambda dataset: f"{dataset} ({[dim[1] for dim in dataset_dimensions if dim[0] == dataset][0]})")


    # Step 1: Prepare data for plotting
    datasets = df['Dataset'].unique()

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
        df_dataset = df[df['Dataset'] == dataset].copy()  # Filter by dataset

        # Create custom labels for the x-axis
        df_dataset['Node'] = df_dataset.apply(
            lambda row: f"{row['Node Name']} (CPUs: {row['Node CPUs']}, RAM: {row['Node RAM']}GB)", axis=1)

        # selecting the necesary columns
        df_time = df_dataset[['Node'] + time_columns]

        # removing the word 'Time' from values of the time_columns
        df_time.columns = df_time.columns.str.replace(' Time', '')

        # Melt the dataframe
        df_melted = pd.melt(df_time, id_vars=['Node'], var_name='Stage', value_name='Time')

        # Plot using seaborn
        plt.figure(figsize=(7, 4))
        sns.barplot(data=df_melted, x='Stage', y='Time', hue='Node', palette='tab20')

        # Customizing the plot
        # plt.ylim(global_ymin, global_ymax)
        plt.title(f'Execution Times of Stages by Nodes for {dataset}')
        plt.xlabel('Stage')
        plt.ylabel('Execution Time (seconds)')
        plt.legend(title='Node Name', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()

        # Saving the plot to a file for the current dataset
        plot_filename = f"execution_times_for_dataset_{dataset}{"_s3" if len(time_columns)==1 else ""}"
        plt.savefig(os.path.join("plots/png", f"{plot_filename}.png"), bbox_inches='tight')
        plt.savefig(os.path.join("plots/pdf", f"{plot_filename}.pdf"), bbox_inches='tight')

        # Display the plot
        # plt.show()

# datasets_to_use = ['DeepSlice', 'NSR', 'IoT-APD']
# Filtering the results to only include the datasets of interest
# df = df[df['Dataset'].isin(datasets_to_use)]

# time_columns = ['Stage 1 Time', 'Stage 2 Time', 'Stage 4 Time']
time_columns = ['Stage 3 Time']
plot_all()
# plot_all_by_dataset()