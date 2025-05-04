import os
import sys

import pandas as pd
# from pandas.core.interchange.from_dataframe import primitive_column_to_ndarray
#
# from ExperimentsPerformanceAnalysis import std_val_roc_per_generation

# Define the path to the main directory
main_dir = 'results'  # Update this with the actual path


def average_metrics_across_files(folder_path):
    # List to store DataFrames
    dfs = []

    # Iterate through all files in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith(".csv"):
            file_path = os.path.join(folder_path, filename)
            # Read each CSV file into a DataFrame
            df = pd.read_csv(file_path)
            dfs.append(df)

    combined_df = pd.concat(dfs, axis=0)

    # Group by epoch and calculate the mean of all metrics
    averaged_df = combined_df.groupby(combined_df.index).mean()

    return averaged_df


def get_baseline_averages():
    # Initialize lists to collect values from each 'R_' subfolder
    val_loss_list = []
    val_acc_list = []
    val_roc_list = []
    fitness_list = []

    # Walk through each experiment folder and each R_ subfolder
    for exp_folder in os.listdir(main_dir):
        if exp_folder.endswith('.png'):
            continue
        elif 'CSV' in exp_folder:
            continue
        elif exp_folder.endswith('.xlsx'):
            continue
        exp_path = os.path.join(main_dir, exp_folder)

        # Check if the current directory is an experiment folder
        if os.path.isdir(exp_path) and exp_folder.startswith('T_'):
            dfs = []
            # Iterate over subfolders in each experiment folder
            for sub_folder in os.listdir(exp_path):
                sub_folder_path = os.path.join(exp_path, sub_folder)
                # Check if the subfolder name starts with 'R_'
                if os.path.isdir(sub_folder_path) and sub_folder.startswith('R_'):
                    # Look for CSV files with 'averages' in their name
                    for file in os.listdir(sub_folder_path):
                        if file.endswith('.csv') and 'averages' in file:
                            csv_file_path = os.path.join(sub_folder_path, file)
                            # Read the CSV file
                            df = pd.read_csv(csv_file_path, header=None)
                            df.set_index(0, inplace=True)
                            df = df.apply(pd.to_numeric, errors='coerce')
                            dfs.append(df)
            combined_df = pd.concat(dfs, axis=1)
            averaged_df = combined_df.groupby(combined_df.columns, axis=1).mean()
            test_metrics = {
                'Metric': ['Min Val Loss', 'Max Val Accuracy', 'Max Val ROC', 'Max Fitness'],
                'Value': [
                    round(averaged_df.loc['Val Losses'].min(), 4),
                    round(averaged_df.loc['Val Accuracy'].max(), 4),
                    round(averaged_df.loc['Val ROC'].max(), 4),
                    round(averaged_df.loc['Fitness'].max(), 4)
                ]
            }
            test_metrics_df = pd.DataFrame(test_metrics)
            test_metrics_df.to_excel(f'results/{exp_folder}.xlsx', index=False)






            std_df = combined_df.groupby(combined_df.columns, axis=1).std()

            val_losses_row = averaged_df.loc['Val Losses']
            min_val_loss_index = val_losses_row.idxmin()
            std_val_losses_value = std_df.loc['Val Losses', min_val_loss_index]


            val_acc_row =averaged_df.loc['Val Accuracy']
            max_val_acc_index = val_acc_row.idxmax()
            std_val_acc_value = std_df.loc['Val Accuracy', max_val_acc_index]

            val_roc_row = averaged_df.loc['Val ROC']
            max_val_roc_index = val_roc_row.idxmax()
            std_val_roc_value = std_df.loc['Val ROC', max_val_roc_index]

            fitness_row = averaged_df.loc['Fitness']
            max_fitness_index = fitness_row.idxmax()
            std_fitness_value = std_df.loc['Fitness', max_fitness_index]


            averaged_df = averaged_df.dropna(how='all')

            val_loss = float(averaged_df.loc['Val Losses'].min())
            val_acc = float(averaged_df.loc['Val Accuracy'].max())
            val_roc = float(averaged_df.loc['Val ROC'].max())
            fitenss = float(averaged_df.loc['Fitness'].max())

    averages_df = pd.DataFrame({
        'Metric': ['Avg Val Loss', 'Avg Val Accuracy', 'Avg Val ROC AUC', 'Avg Fitness', 'STD Val Loss', 'STD Val Accuracy', 'STD Val ROC AUC', 'STD Fitness'],
        'Value': [val_loss, val_acc, val_roc, fitenss, std_val_losses_value, std_val_acc_value, std_val_roc_value, std_fitness_value]
    })
    averages_df['Value'] = averages_df['Value'].round(4)
    #
    output_path = 'results/baseline_averages.xlsx'  # Update this with your desired output path
    #
    # # Save the DataFrame to an Excel file
    averages_df.to_excel(output_path, index=False)
get_baseline_averages()


def get_individual_averages(main_dir):
    results = []

    # Walk through each experiment folder
    for exp_folder in os.listdir(main_dir):
        if exp_folder.endswith('.png') or 'CSV' in exp_folder:
            continue

        exp_path = os.path.join(main_dir, exp_folder)

        # Check if the current directory is an experiment folder
        if os.path.isdir(exp_path) and exp_folder.startswith('T_'):
            # Look for a folder containing 'plots'
            for sub_folder in os.listdir(exp_path):
                if 'plots' in sub_folder:
                    sub_folder_path = os.path.join(exp_path, sub_folder)

                    # Check if the subfolder is indeed a directory
                    if os.path.isdir(sub_folder_path):
                        # Look for the specific CSV file
                        csv_file_path = os.path.join(sub_folder_path,
                                                     'avg_values_of_all_individuals_per_generation.csv')

                        std_file_path = os.path.join(sub_folder_path, 'std_values_of_all_individuals_per_generation.csv')

                        if os.path.isfile(csv_file_path):
                            # Read the CSV file
                            df = pd.read_csv(csv_file_path)
                            std_df = pd.read_csv(std_file_path)
                            std_val_loss_row = std_df.loc[df['0'] == 'Val Loss', std_df.columns[1:]]
                            val_loss_row = df.loc[df['0'] == 'Val Loss', df.columns[1:]]
                            val_loss_index = val_loss_row.idxmin(axis=1).values[0]
                            val_loss = float(val_loss_row.min(axis=1).values[0])

                            std_val_loss_at_index = std_val_loss_row.iloc[0, int(val_loss_index) - 1]

                            std_val_acc_row = std_df.loc[df['0'] == 'Val Acc', std_df.columns[1:]]
                            val_acc_row = df.loc[df['0'] == 'Val Acc', df.columns[1:]]
                            val_acc_index = val_acc_row.idxmax(axis=1).values[0]
                            val_acc = float(val_acc_row.max(axis=1).values[0])
                            std_val_acc_at_index = std_val_acc_row.iloc[0, int(val_acc_index) - 1]

                            std_val_roc_row = std_df.loc[df['0'] == 'Val ROC', std_df.columns[1:]]
                            val_roc_row = df.loc[df['0'] == 'Val ROC', df.columns[1:]]
                            val_roc_index = val_roc_row.idxmax(axis=1).values[0]
                            val_roc = float(val_roc_row.max(axis=1).values[0])
                            std_val_roc_at_index = std_val_roc_row.iloc[0, int(val_roc_index) - 1]

                            std_fitness_row = std_df.loc[df['0'] == 'Fitness', std_df.columns[1:]]
                            fitness_row = df.loc[df['0'] == 'Fitness', df.columns[1:]]
                            fitness_index = fitness_row.idxmax(axis=1).values[0]
                            fitness = float(fitness_row.max(axis=1).values[0])
                            std_fitness_at_index = std_fitness_row.iloc[0, int(fitness_index) - 1]





                            val_loss = float(df.loc[df['0'] == 'Val Loss', df.columns[1:]].min(axis=1).values[0])
                            val_acc = float(df.loc[df['0'] == 'Val Acc', df.columns[1:]].max(axis=1).values[0])
                            val_roc = float(df.loc[df['0'] == 'Val ROC', df.columns[1:]].max(axis=1).values[0])
                            fitness = float(df.loc[df['0'] == 'Fitness', df.columns[1:]].max(axis=1).values[0])
                            results.append([exp_folder, val_loss, val_acc, val_roc, fitness, std_val_loss_at_index, std_val_acc_at_index, std_val_roc_at_index, std_fitness_at_index])

    # Create a DataFrame from results
    results_df = pd.DataFrame(results, columns=['Experiment', 'Val loss', 'Val Acc', 'ROC AUC', 'Fitness', 'Val loss STD', 'Val Acc STD', 'ROC AUC STD', 'Fitness STD'])

    # Compute the averages, skipping non-numeric values
    avg_values = results_df[['Val loss', 'Val Acc', 'ROC AUC', 'Fitness', 'Val loss STD', 'Val Acc STD', 'ROC AUC STD', 'Fitness STD']].mean().values
    avg_row = ['Exp Avg'] + list(avg_values)

    # Append the averages row
    results_df.loc[len(results_df)] = avg_row

    # Round values
    results_df = results_df.round(4)

    # Define the output path for the Excel file
    output_path = 'results/all_experiments_individual_averages.xlsx'  # Update this with your desired output path

    # Save the DataFrame to an Excel file
    results_df.to_excel(output_path, index=False)
# Call the function with your main directory
get_individual_averages(main_dir)
import csv

def get_best_individual(main_dir):
    final_values_dict = {}
    for exp_folder in os.listdir(main_dir):
        if exp_folder.endswith('.png') or 'CSV' in exp_folder:
            continue
        exp_path = os.path.join(main_dir, exp_folder)

        # Check if the current directory is an experiment folder
        if os.path.isdir(exp_path) and exp_folder.startswith('T_'):
            # Look for a folder containing 'plots'

            for sub_folder in os.listdir(exp_path):
                if 'plots' in sub_folder:
                    df = pd.read_csv(os.path.join(exp_path, sub_folder, 'values_of_top1_individuals_per_generation.csv'))
                    min_val_loss = round(df.iloc[0, 1:].min(),4)
                    max_acc = round(df.iloc[1, 1:].max(),4)
                    max_auc = round(df.iloc[2, 1:].max(), 4)
                    max_fitness = round(df.iloc[3, 1:].max(), 4)
                    final_values_dict[exp_folder] = [min_val_loss, max_acc, max_auc, max_fitness]
    with open('results/top1_of_all_tests.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        # Writing the rows where keys are row names and values are the data
        for key, values in final_values_dict.items():
            writer.writerow([key] + values)


get_best_individual(main_dir)