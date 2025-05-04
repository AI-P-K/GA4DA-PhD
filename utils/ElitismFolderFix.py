import sys
from sys import prefix

from ultralytics.utils.ops import process_mask

from GetMetricsFromCsv import GetMetricsFromCsv
import os
import pandas as pd
from collections import defaultdict
import shutil


def custom_sort(item):
    # Split the item by the underscore
    parts = item.split('_')
    # Extract the first part and convert it to an integer
    try:
        first_number = int(parts[0])
    except ValueError:
        # If conversion to int fails, return a very large number
        return float('inf')
    return first_number

result_root_folder = '../results'
for test in os.listdir(result_root_folder):
    if "T_" in test:
        all_solutions_in_test_folder = sorted(os.listdir(os.path.join(result_root_folder, test)), key=custom_sort)
        full_path_solutions_folder = [os.path.join(result_root_folder, test, i) for i in all_solutions_in_test_folder]
        full_path_solutions_folder = [item for item in full_path_solutions_folder if "description.txt" not in item]

        def extract_all_solutions_best_fitness():
            all_solutions_best_fitness = defaultdict(float)
            for solution in full_path_solutions_folder:
                if "R_" in solution or os.path.isfile(solution) or 'plots' in solution:
                    continue
                else:
                    for filename in os.listdir(solution):
                        if 'metrics' in filename:
                            metrics_averages = pd.read_csv(os.path.join(solution, filename), index_col=0)
                            val_losses = metrics_averages.loc['Val Losses']
                            fitness = 1/val_losses
                            best_fitness_epoch = fitness.idxmax()
                            best_fitness_value = fitness[best_fitness_epoch]
                            solution_id = solution.split('/')[-1]
                            all_solutions_best_fitness[solution_id] = best_fitness_value

            return all_solutions_best_fitness


        all_solutions_best_fitness = extract_all_solutions_best_fitness()


        def identify_the_best_individual_in_each_generation(all_solutions_best_fitness):
            max_values = {}
            # Iterate over dictionary keys
            for key, value in all_solutions_best_fitness.items():
                # Extract the prefix (e.g., '1', '2', '3', ...)
                prefix = key.split('_')[0]
                # Check if prefix exists in max_values
                if prefix in max_values:
                    # Update max value if current value is greater
                    if value > max_values[prefix][1]:
                        max_values[prefix] = (key, value)
                else:
                    # If prefix not in max_values, add it
                    max_values[prefix] = (key, value)
            return max_values

        best_individuals_id_and_value = identify_the_best_individual_in_each_generation(all_solutions_best_fitness)

        for solution in full_path_solutions_folder:
            if "R_" in solution or os.path.isfile(solution) or 'plots' in solution:
                continue
            else:
                generation = solution.split('/')[-1].split('_')[0]
                metrics_file_path = os.path.join(solution, 'Fold_0.csv')
                if os.path.exists(metrics_file_path):
                    continue
                else:
                    generation_missing_individual = int(generation)
                    previous_generation_elite = generation_missing_individual - 1
                    previous_generation_elite_id = best_individuals_id_and_value[f'{previous_generation_elite}'][0]
                    no_metrics_individual_full_path = solution
                    print(f'Individual{no_metrics_individual_full_path} does not have metrics')
                    test_base_path = os.path.dirname(no_metrics_individual_full_path)

                    previous_generation_elite_individual_full_path = os.path.join(test_base_path, previous_generation_elite_id)
                    elite_individual_files_to_be_copied = os.listdir(previous_generation_elite_individual_full_path)
                    print("Copying files: ", elite_individual_files_to_be_copied)
                    for file in elite_individual_files_to_be_copied:
                        source_file = os.path.join(test_base_path, previous_generation_elite_id, file)
                        destination_file = os.path.join(no_metrics_individual_full_path, file)
                        # print('Destination', destination_file)
                        if os.path.isfile(source_file):
                            shutil.copy(source_file, destination_file)
                    all_solutions_best_fitness = extract_all_solutions_best_fitness()
                    best_individuals_id_and_value = identify_the_best_individual_in_each_generation(
                        all_solutions_best_fitness)





