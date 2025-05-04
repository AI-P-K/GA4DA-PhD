import os
import sys

import pandas as pd
base_path = 'results'
all_tests = os.listdir(base_path)
all_tests_full_path = [os.path.join(base_path, i) for i in all_tests if os.path.isdir(os.path.join(base_path, i))]
#
#
#
# def process_experiment_csv_file(file_path):
#     hyperparameters_dict = {}
#     file = pd.read_csv(file_path)
#     hyperparameters = [x for x in file.columns if x.strip != ' ']
#     cleaned_hyperparams = list(filter(lambda x: x.strip() != '', hyperparameters))
#     cleaned_hyperparams = list(filter(lambda x: x.strip(), map(str.strip, cleaned_hyperparams)))
#
#     for i in cleaned_hyperparams:
#         hyper_description = i.split(':')[0]
#         hyper_value = i.split(':')[-1]
#         hyperparameters_dict[hyper_description] = hyper_value
#
#     return hyperparameters_dict
#
# for path in all_tests_full_path:
#     print(path)
#     if 'Sorted' in path or 'CSVs' in path:
#         continue
#     experiment_file = [x for x in os.listdir(path) if 'description.txt' in x][0]
#     experiment_file_path = os.path.join(path, experiment_file)
#     hyperparameters_dict = process_experiment_csv_file(experiment_file_path)
#     df = pd.DataFrame([hyperparameters_dict])
#     df.to_excel(os.path.join(base_path, 'CSVs', experiment_file.split('.')[0] + '.xlsx'))

# import os
# import pandas as pd


def process_experiment_csv_file(file_path):
    hyperparameters_dict = {}
    file = pd.read_csv(file_path)
    hyperparameters = [x for x in file.columns if x.strip() != ' ']
    cleaned_hyperparams = list(filter(lambda x: x.strip() != '', hyperparameters))
    cleaned_hyperparams = list(filter(lambda x: x.strip(), map(str.strip, cleaned_hyperparams)))

    for i in cleaned_hyperparams:
        hyper_description = i.split(':')[0]
        hyper_value = i.split(':')[-1]
        hyperparameters_dict[hyper_description] = hyper_value

    return hyperparameters_dict


for path in all_tests_full_path:
    if 'Sorted' in path or 'CSVs' in path:
        continue
    experiment_file = [x for x in os.listdir(path) if 'description.txt' in x][0]
    experiment_file_path = os.path.join(path, experiment_file)
    test_id = path.split('/')[-1]

    # Process the file and retrieve the hyperparameters
    hyperparameters_dict = process_experiment_csv_file(experiment_file_path)

    # Add the 'TestID' key and set its value to 'EXP'
    hyperparameters_dict['TestID'] = test_id

    # Create a DataFrame with the updated hyperparameters dictionary
    df = pd.DataFrame([hyperparameters_dict])

    # Ensure the 'TestID' is the first column
    df = df[['TestID'] + [col for col in df.columns if col != 'TestID']]

    # Save the DataFrame as an Excel file
    df.to_excel(os.path.join(base_path, 'CSVs', experiment_file.split('.')[0] + '.xlsx'), index=False)
