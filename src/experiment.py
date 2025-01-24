import os
import csv
import gc
from sklearn.model_selection import train_test_split

from mlops_stages import *

dataset_list = [
    ['KPI-KQI', [], 'Service'],  # 165 x 14
    ['UNAC', ['file'], 'output'],  # 389 x 23
    ['IoT-APD', ['second'], 'label'],  # 10845 x 17
    ['NSR', [], 'slice Type'],  # 31583 x 17
    ['DeepSlice', ['no'], 'slice Type'],  # 63167 x 10
    ['NSL-KDD', [], 'class'],  # 148517 x 42
    ['IoT-DNL', [], 'normality'],  # 477426 x 14
]

def run_experiment(experiment_name: str, node_cpus: str, node_ram: str, node_name: str) -> None:
    # if node RAM is 1GB then we limit the numbers of processes to 1 due ram memory limitations
    if node_ram == "1":
        setNJobs(1)

    output_csv_header = [
        'Node Name', 'Node CPUs', 'Node RAM',
        'Dataset', 'Stage 1 Time', 'Stage 1 Removed Columns', 'Stage 1 Qty of Removed Rows', 'Stage 2 Time', 'Stage 2 Encoded Columns', 'Stage 2 Number PCA Components',
        'Stage 3 Time', 'Stage 4 Time', 'Stage 4 Best Model Name', 'Stage 4 Best Model Params'
    ]

    output_csv = []

    for dataset_setup in dataset_list:
        dataset_name = dataset_setup[0]
        unuseful_columns = dataset_setup[1]
        class_name = dataset_setup[2]

        print(f"Running experiment for dataset: {dataset_name}")

        # loading the dataset
        dataset_folder = f"./datasets/{dataset_name}"
        full_df = pd.read_csv(
            f"{dataset_folder}/{[file for file in os.listdir(dataset_folder) if file.endswith('.csv')][0]}")


        # stage 1: data cleaning
        print("Stage 1: Data Cleaning")
        st1_time, _, (full_df, st1_qty_of_removed_rows) = clean_data(full_df, unuseful_columns)

        # stage 2: data preprocessing
        print("Stage 2: Data Preprocessing")
        st2_time, _, (full_df, st2_encoded_columns) = preprocess_data(full_df, class_name)
        st2_number_pca_components = full_df.shape[1] - 1

        # stage 3: model tuning
        print("Stage 3: Model Tuning")
        #   dividing the dataframe into two dataframes: one with 20% of the data and the other with the remaining 80%
        df_train, df_test = train_test_split(full_df, test_size=0.2, random_state=global_random_seed, stratify=full_df[class_name])
        #   tuning the models
        st3_time, _, tuned_models = tune_models(df_train, class_name)

        # stage 4: model evaluation and selection
        print("Stage 4: Model Evaluation and Selection\n")
        st4_time, _, (best_model_name, best_model_params, best_model_mcc) = evaluate_models(df_test, class_name, tuned_models)

        # saving to output_csv
        output_csv.append([
            node_name, node_cpus, node_ram,
            dataset_name, st1_time, unuseful_columns, st1_qty_of_removed_rows, st2_time, st2_encoded_columns,
            st2_number_pca_components,
            # st3_time, results, st3_time, st4_time, best_model_name, best_model_params
            st3_time, st4_time, best_model_name, best_model_params
        ])

        # saving the output to a csv file
        output_csv_file = f"./results/{experiment_name}.csv"
        with open(output_csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(output_csv_header)
            writer.writerows(output_csv)

        print(f"Results saved to {output_csv_file}")

        # cleaning ram memory
        del full_df, df_train, df_test, tuned_models, best_model_name, best_model_params, best_model_mcc
        gc.collect()







