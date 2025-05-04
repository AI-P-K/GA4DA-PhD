import sys
from utils.GetMetricsFromCsv import GetMetricsFromCsv
import os
from utils.GAMetricsCalculator import GAMetricsCalculator, GAMetricsPlotter
import copy
import matplotlib.pyplot as plt
font_size = 20
plt.rcParams.update({'font.size': font_size})

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

result_root_folder = 'results'

"""
!!!!!!!! Experiments pairs should be fixed at index 0 and variable at index 1 !!!!!!!!!!!!!!!
"""

btmri_pairs_20 = {
    "1_A": ['T_dcdafb', 'T_b15191'],
    "1_B": ['T_3bcc1c', 'T_9cf90f'],
}

btmri_pairs_10 = {
    "1_A":['T_5b7331', 'T_264f5d'],
    "1_B":['T_0d43c6', 'T_d16dd8'],
    "2_A":['T_017b0e', 'T_9c7b16'],
    "2_B":['T_b7b09b', 'T_0053ca']
}

btmri_pairs_5 = {
    "1_A":['T_9e1bdd', 'T_e6280c'],
    "1_B":['T_28fe84', 'T_b4d5ec'],
}

breast_resnet = {
    "1_A":['T_3247ad', 'T_5d1e01'],
    "1_B":['T_179900', 'T_453958'],
}

breast_evocnn = {
    "1_A":['T_a9b4d0', 'T_9b5f74'],
    "1_B":['T_83550e', 'T_7bd86a'],
}

breast_evocnn_2 = {
    "1_A":['T_c40848', 'T_335814'],
    "1_B":['T_85db7c', 'T_d07418' ],
}



cifar10 = {
    "1_A":['T_db0a72', 'T_10d30d'],
    "1_B":['T_0ede21', 'T_ad8502']
}

soyleaf = {
    "1_A":['T_d8ca14', 'T_e50450'],
    "1_B":['T_dd1d27', 'T_ceb4da']
}

soyleaf2 = {
    "1_A":['T_2f3c99', 'T_e95e83'],
    "1_B":['T_e2f8ed', 'T_f2fd8f']
}


y_lim = (1.5, 2.2)
baseline = 1.9085


def comparison_plot_averaged_best_metrics_and_std_of_each_generation_error_bar(avg_metric_dictionary_fixed,
                                                                               std_metric_dictionary_fixed,
                                                                               avg_metric_dictionary_variable,
                                                                               std_metric_dictionary_variable,
                                                                               alphabet,
                                                                               title=None,
                                                                               xlabel=None,
                                                                               ylabel=None):

    fig_size = (12, 7)
    font = {'size': font_size}
    plt.figure(figsize=fig_size)
    keys = list(map(int, avg_metric_dictionary_fixed.keys()))
    avg_values_fixed = list(avg_metric_dictionary_fixed.values())
    std_values_fixed = list(std_metric_dictionary_fixed.values())
    avg_values_variable = list(avg_metric_dictionary_variable.values())
    std_values_variable = list(std_metric_dictionary_variable.values())

    # plt.errorbar(keys, avg_values_fixed, yerr=std_values_fixed, fmt='o-', capsize=5, label='Fixed')
    # plt.errorbar(keys, avg_values_variable, yerr=std_values_variable, fmt='o-', capsize=5, label='Variable')
    if 'A' in alphabet:
        plt.errorbar(keys, avg_values_fixed, yerr=std_values_fixed, fmt='o-', capsize=5, label='Fixed', color='blue')
        plt.errorbar(keys, avg_values_variable, yerr=std_values_variable, fmt='o-', capsize=5, label='Variable', color='red')
    elif 'B' in alphabet:
        plt.errorbar(keys, avg_values_fixed, yerr=std_values_fixed, fmt='o-', capsize=5, label='Fixed', color='green')
        plt.errorbar(keys, avg_values_variable, yerr=std_values_variable, fmt='o-', capsize=5, label='Variable', color='purple')

    plt.rc('font', **font)

    # max_value = max(max(avg_values_fixed), max(avg_values_variable))
    #
    # min_value = min(min(avg_values_fixed), min(avg_values_variable))
    plt.axhline(y=baseline, linestyle='--', color='orange', label='Baseline')
    # plt.ylim(min_value*0.7, max_value*1.3)
    # plt.ylim(min_value * 0.9, max_value * 1.1)
    plt.ylim(y_lim[0], y_lim[1])
    plt.xlabel(f'{xlabel}')
    plt.ylabel(f'{ylabel}')
    plt.legend()
    plt.xticks(keys)
    plt.savefig(f'{result_root_folder}/{title}.png', bbox_inches='tight')
    plt.clf()
    plt.close()

def fixed_vs_variable(pairs_dict):
    for key, value in pairs_dict.items():
        genes_type = key.split('_')[-1]
        test_fixed = value[0]
        test_variable = value[1]


        all_solutions_in_test_fixed = sorted(os.listdir(os.path.join(result_root_folder, test_fixed)), key=custom_sort)
        all_solutions_in_test_variable = sorted(os.listdir(os.path.join(result_root_folder, test_variable)), key=custom_sort)
        full_path_solutions_fixed = [os.path.join(result_root_folder, test_fixed, i) for i in all_solutions_in_test_fixed]
        full_path_solutions_fixed = [item for item in full_path_solutions_fixed if "description.txt" not in item and "time.csv" not in item]

        full_path_solutions_variable = [os.path.join(result_root_folder, test_variable, i) for i in all_solutions_in_test_variable]
        full_path_solutions_variable = [item for item in full_path_solutions_variable if "description.txt" not in item and "time.csv" not in item]


        ##################### GET FIXED MUTATION RATE AVERAGE FITNESS SCORE ################
        ##################### GET FIXED MUTATION RATE AVERAGE FITNESS SCORE ################
        ##################### GET FIXED MUTATION RATE AVERAGE FITNESS SCORE ################
        get_metrics_fixed = GetMetricsFromCsv(full_path_solutions_fixed)
        (averaged_kfolds_train_loss_of_each_solution_per_generation,
         averaged_kfolds_val_loss_of_each_solution_per_generation,
         averaged_kfolds_train_acc_of_each_solution_per_generation,
         averaged_kfolds_val_acc_of_each_solution_per_generation,
         averaged_kfolds_val_roc_of_each_solution_per_generation,
         averaged_kfolds_train_loss_of_each_baseline,
         averaged_kfolds_val_loss_of_each_baseline,
         averaged_kfolds_train_acc_of_each_baseline,
         averaged_kfolds_val_acc_of_each_baseline,
         averaged_kfolds_val_roc_of_each_baseline) = get_metrics_fixed.put_averaged_metrics_csv_in_dict()

        # sys.exit()
        FIXED_averaged_kfolds_fitness_of_each_solution_per_generation_per_generation = copy.deepcopy(
            averaged_kfolds_val_loss_of_each_solution_per_generation)
        for gen, val_loss_dataframe_list in FIXED_averaged_kfolds_fitness_of_each_solution_per_generation_per_generation.items():
            for i, df in enumerate(val_loss_dataframe_list):
                FIXED_averaged_kfolds_fitness_of_each_solution_per_generation_per_generation[gen][i] = 1 / df

        FIXED_averaged_fitness_of_each_generation = get_metrics_fixed.do_average_all_solutions_in_generation_by_metric(
            FIXED_averaged_kfolds_fitness_of_each_solution_per_generation_per_generation)
        ga_metrics = GAMetricsCalculator()

        # print(FIXED_averaged_fitness_of_each_generation[0])
        # print(FIXED_averaged_fitness_of_each_generation[1])
        # sys.exit()
        FIXED_best_averaged_fitness_of_each_generation = ga_metrics.get_best_metric_value_of_each_generation_dict_with_list(
            FIXED_averaged_fitness_of_each_generation[0], metric_type='roc')

        (best_fitness_of_every_individual_in_each_generation,
         avg_fitness_roc_of_generation,
         FIXED_std_fitness_per_generation) = get_metrics_fixed.get_best_metric_value_of_each_solution(
            FIXED_averaged_kfolds_fitness_of_each_solution_per_generation_per_generation, metric_type="roc")


        ##################### GET VARIABLE MUTATION RATE AVERAGE FITNESS SCORE ################
        ##################### GET VARIABLE MUTATION RATE AVERAGE FITNESS SCORE ################
        ##################### GET VARIABLE MUTATION RATE AVERAGE FITNESS SCORE ################

        get_metrics_variable = GetMetricsFromCsv(full_path_solutions_variable)
        (averaged_kfolds_train_loss_of_each_solution_per_generation,
         averaged_kfolds_val_loss_of_each_solution_per_generation,
         averaged_kfolds_train_acc_of_each_solution_per_generation,
         averaged_kfolds_val_acc_of_each_solution_per_generation,
         averaged_kfolds_val_roc_of_each_solution_per_generation,
         averaged_kfolds_train_loss_of_each_baseline,
         averaged_kfolds_val_loss_of_each_baseline,
         averaged_kfolds_train_acc_of_each_baseline,
         averaged_kfolds_val_acc_of_each_baseline,
         averaged_kfolds_val_roc_of_each_baseline) = get_metrics_variable.put_averaged_metrics_csv_in_dict()

        VARIABLE_averaged_kfolds_fitness_of_each_solution_per_generation_per_generation = copy.deepcopy(
            averaged_kfolds_val_loss_of_each_solution_per_generation)
        for gen, val_loss_dataframe_list in VARIABLE_averaged_kfolds_fitness_of_each_solution_per_generation_per_generation.items():
            for i, df in enumerate(val_loss_dataframe_list):
                VARIABLE_averaged_kfolds_fitness_of_each_solution_per_generation_per_generation[gen][i] = 1 / df

        VARIABLE_averaged_fitness_of_each_generation = get_metrics_fixed.do_average_all_solutions_in_generation_by_metric(
            VARIABLE_averaged_kfolds_fitness_of_each_solution_per_generation_per_generation)
        ga_metrics = GAMetricsCalculator()
        VARIABLE_best_averaged_fitness_of_each_generation = ga_metrics.get_best_metric_value_of_each_generation_dict_with_list(
            VARIABLE_averaged_fitness_of_each_generation[0], metric_type='roc')

        (best_fitness_of_every_individual_in_each_generation,
         avg_fitness_roc_of_generation,
         VARIABLE_std_fitness_per_generation) = get_metrics_variable.get_best_metric_value_of_each_solution(
            VARIABLE_averaged_kfolds_fitness_of_each_solution_per_generation_per_generation, metric_type="roc")
        comparison_plot_averaged_best_metrics_and_std_of_each_generation_error_bar(FIXED_best_averaged_fitness_of_each_generation,
                                                                                   FIXED_std_fitness_per_generation,
                                                                                   VARIABLE_best_averaged_fitness_of_each_generation,
                                                                                   VARIABLE_std_fitness_per_generation,
                                                                                   key,
                                                                                   title=f'Genes type {genes_type} {test_fixed} vs {test_variable}',
                                                                                   xlabel='Generations',
                                                                                   ylabel='Fitness')

fixed_vs_variable(cifar10)




breast_resnet_encoding = {
    "fixed":['T_3247ad', 'T_179900'],
    "variable":['T_5d1e01', 'T_453958'],
}

breast_evocnn_encoding = {
    "fixed":['T_a9b4d0', 'T_83550e'],
    "variable":['T_9b5f74', 'T_7bd86a'],
}


soyleaf_encoding = {
    "fixed":['T_d8ca14', 'T_dd1d27'],
    "variable":['T_e50450', 'T_ceb4da']
}

cifar10_encoding = {
    "fixed":['T_db0a72', 'T_0ede21'],
    "variable":['T_10d30d', 'T_ad8502']
}

btmri_pairs_20_encoding = {
    "fixed": ['T_dcdafb', 'T_3bcc1c'],
    "variable": ['T_b15191', 'T_9cf90f'],
}


btmri_pairs_10_encoding = {
    "fixed":['T_5b7331', 'T_0d43c6'],
    "variable":['T_264f5d', 'T_d16dd8'],
    "fixed1":['T_017b0e', 'T_b7b09b'],
    "variable1":['T_9c7b16', 'T_0053ca']
}

btmri_pairs_5_encoding = {
    "fixed":['T_9e1bdd', 'T_28fe84'],
    "variable":['T_e6280c', 'T_b4d5ec'],
}


breast_evocnn_2_encoding = {
    "fixed":['T_c40848', 'T_85db7c'],
    "variable":['T_335814', 'T_d07418' ],
}



soyleaf2_encoding = {
    "fixed":['T_2f3c99', 'T_e2f8ed'],
    "variable":['T_e95e83', 'T_f2fd8f']
}

def comparison_plot_averaged_best_metrics_and_std_of_each_generation_error_bar_ENCODING(avg_metric_dictionary_fixed,
                                                                               std_metric_dictionary_fixed,
                                                                               avg_metric_dictionary_variable,
                                                                               std_metric_dictionary_variable,
                                                                               mutation,
                                                                               title=None,
                                                                               xlabel=None,
                                                                               ylabel=None):


    fig_size = (12, 7)
    font = {'size': font_size}
    plt.figure(figsize=fig_size)
    keys = list(map(int, avg_metric_dictionary_fixed.keys()))
    avg_values_fixed = list(avg_metric_dictionary_fixed.values())
    std_values_fixed = list(std_metric_dictionary_fixed.values())
    avg_values_variable = list(avg_metric_dictionary_variable.values())
    std_values_variable = list(std_metric_dictionary_variable.values())

    # plt.errorbar(keys, avg_values_fixed, yerr=std_values_fixed, fmt='o-', capsize=5, label='Fixed')
    # plt.errorbar(keys, avg_values_variable, yerr=std_values_variable, fmt='o-', capsize=5, label='Variable')
    if 'fix' in mutation:
        plt.errorbar(keys, avg_values_fixed, yerr=std_values_fixed, fmt='o-', capsize=5, label='A', color='blue')
        plt.errorbar(keys, avg_values_variable, yerr=std_values_variable, fmt='o-', capsize=5, label='B', color='green')
    elif 'var' in mutation:
        plt.errorbar(keys, avg_values_fixed, yerr=std_values_fixed, fmt='o-', capsize=5, label='A', color='red')
        plt.errorbar(keys, avg_values_variable, yerr=std_values_variable, fmt='o-', capsize=5, label='B', color='purple')

    plt.rc('font', **font)
    plt.axhline(y=baseline, linestyle='--', color='orange', label='Baseline')

    plt.ylim(y_lim[0], y_lim[1])
    plt.xlabel(f'{xlabel}')
    plt.ylabel(f'{ylabel}')
    plt.legend()
    plt.xticks(keys)
    plt.savefig(f'{result_root_folder}/{title}.png', bbox_inches='tight')
    plt.clf()
    plt.close()


def A_vs_B(pairs_dict):
    for key, value in pairs_dict.items():
        genes_type = key.split('_')[-1]



        test_fixed = value[0]
        test_variable = value[1]


        all_solutions_in_test_fixed = sorted(os.listdir(os.path.join(result_root_folder, test_fixed)), key=custom_sort)
        all_solutions_in_test_variable = sorted(os.listdir(os.path.join(result_root_folder, test_variable)), key=custom_sort)
        # print('full_path_solutions_fixed',os.path.join(result_root_folder, test_fixed))
        # print('full_path_solutions_variable',os.path.join(result_root_folder, test_variable))
        # sys.exit()
        full_path_solutions_fixed = [os.path.join(result_root_folder, test_fixed, i) for i in all_solutions_in_test_fixed]
        full_path_solutions_fixed = [item for item in full_path_solutions_fixed if "description.txt" not in item and "time.csv" not in item]

        full_path_solutions_variable = [os.path.join(result_root_folder, test_variable, i) for i in all_solutions_in_test_variable]
        full_path_solutions_variable = [item for item in full_path_solutions_variable if "description.txt" not in item and "time.csv" not in item]


        ##################### GET FIXED MUTATION RATE AVERAGE FITNESS SCORE ################
        ##################### GET FIXED MUTATION RATE AVERAGE FITNESS SCORE ################
        ##################### GET FIXED MUTATION RATE AVERAGE FITNESS SCORE ################

        get_metrics_fixed = GetMetricsFromCsv(full_path_solutions_fixed)
        (averaged_kfolds_train_loss_of_each_solution_per_generation,
         averaged_kfolds_val_loss_of_each_solution_per_generation,
         averaged_kfolds_train_acc_of_each_solution_per_generation,
         averaged_kfolds_val_acc_of_each_solution_per_generation,
         averaged_kfolds_val_roc_of_each_solution_per_generation,
         averaged_kfolds_train_loss_of_each_baseline,
         averaged_kfolds_val_loss_of_each_baseline,
         averaged_kfolds_train_acc_of_each_baseline,
         averaged_kfolds_val_acc_of_each_baseline,
         averaged_kfolds_val_roc_of_each_baseline) = get_metrics_fixed.put_averaged_metrics_csv_in_dict()

        FIXED_averaged_kfolds_fitness_of_each_solution_per_generation_per_generation = copy.deepcopy(
            averaged_kfolds_val_loss_of_each_solution_per_generation)
        for gen, val_loss_dataframe_list in FIXED_averaged_kfolds_fitness_of_each_solution_per_generation_per_generation.items():
            for i, df in enumerate(val_loss_dataframe_list):
                FIXED_averaged_kfolds_fitness_of_each_solution_per_generation_per_generation[gen][i] = 1 / df

        FIXED_averaged_fitness_of_each_generation = get_metrics_fixed.do_average_all_solutions_in_generation_by_metric(
            FIXED_averaged_kfolds_fitness_of_each_solution_per_generation_per_generation)
        ga_metrics = GAMetricsCalculator()
        FIXED_best_averaged_fitness_of_each_generation = ga_metrics.get_best_metric_value_of_each_generation_dict_with_list(
            FIXED_averaged_fitness_of_each_generation[0], metric_type='roc')

        (best_fitness_of_every_individual_in_each_generation,
         avg_fitness_roc_of_generation,
         FIXED_std_fitness_per_generation) = get_metrics_fixed.get_best_metric_value_of_each_solution(
            FIXED_averaged_kfolds_fitness_of_each_solution_per_generation_per_generation, metric_type="roc")


        ##################### GET VARIABLE MUTATION RATE AVERAGE FITNESS SCORE ################
        ##################### GET VARIABLE MUTATION RATE AVERAGE FITNESS SCORE ################
        ##################### GET VARIABLE MUTATION RATE AVERAGE FITNESS SCORE ################

        get_metrics_variable = GetMetricsFromCsv(full_path_solutions_variable)
        (averaged_kfolds_train_loss_of_each_solution_per_generation,
         averaged_kfolds_val_loss_of_each_solution_per_generation,
         averaged_kfolds_train_acc_of_each_solution_per_generation,
         averaged_kfolds_val_acc_of_each_solution_per_generation,
         averaged_kfolds_val_roc_of_each_solution_per_generation,
         averaged_kfolds_train_loss_of_each_baseline,
         averaged_kfolds_val_loss_of_each_baseline,
         averaged_kfolds_train_acc_of_each_baseline,
         averaged_kfolds_val_acc_of_each_baseline,
         averaged_kfolds_val_roc_of_each_baseline) = get_metrics_variable.put_averaged_metrics_csv_in_dict()

        VARIABLE_averaged_kfolds_fitness_of_each_solution_per_generation_per_generation = copy.deepcopy(
            averaged_kfolds_val_loss_of_each_solution_per_generation)
        for gen, val_loss_dataframe_list in VARIABLE_averaged_kfolds_fitness_of_each_solution_per_generation_per_generation.items():
            for i, df in enumerate(val_loss_dataframe_list):
                VARIABLE_averaged_kfolds_fitness_of_each_solution_per_generation_per_generation[gen][i] = 1 / df

        VARIABLE_averaged_fitness_of_each_generation = get_metrics_fixed.do_average_all_solutions_in_generation_by_metric(
            VARIABLE_averaged_kfolds_fitness_of_each_solution_per_generation_per_generation)
        ga_metrics = GAMetricsCalculator()
        VARIABLE_best_averaged_fitness_of_each_generation = ga_metrics.get_best_metric_value_of_each_generation_dict_with_list(
            VARIABLE_averaged_fitness_of_each_generation[0], metric_type='roc')

        (best_fitness_of_every_individual_in_each_generation,
         avg_fitness_roc_of_generation,
         VARIABLE_std_fitness_per_generation) = get_metrics_variable.get_best_metric_value_of_each_solution(
            VARIABLE_averaged_kfolds_fitness_of_each_solution_per_generation_per_generation, metric_type="roc")

        comparison_plot_averaged_best_metrics_and_std_of_each_generation_error_bar_ENCODING(FIXED_best_averaged_fitness_of_each_generation,
                                                                                   FIXED_std_fitness_per_generation,
                                                                                   VARIABLE_best_averaged_fitness_of_each_generation,
                                                                                   VARIABLE_std_fitness_per_generation,
                                                                                   key,
                                                                                   title=f'Mutation type {genes_type} {test_fixed} vs {test_variable}',
                                                                                   xlabel='Generations',
                                                                                   ylabel='Fitness')

A_vs_B(cifar10_encoding)