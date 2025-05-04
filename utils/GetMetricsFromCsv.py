import os
import statistics
import sys
import numpy as np
import pandas as pd
from collections import defaultdict


class GetMetricsFromCsv:
    def __init__(self, full_path_sorted_solutions):
        self.full_path_sorted_solutions = full_path_sorted_solutions
    def put_averaged_metrics_csv_in_dict(self):
        averaged_kfolds_train_loss_of_each_solution_per_generation = {}
        averaged_kfolds_val_loss_of_each_solution_per_generation = {}
        averaged_kfolds_train_acc_of_each_solution_per_generation = {}
        averaged_kfolds_val_acc_of_each_solution_per_generation = {}
        averaged_kfolds_val_roc_of_each_solution_per_generation = {}
        averaged_kfolds_train_loss_of_each_baseline = {}
        averaged_kfolds_val_loss_of_each_baseline = {}
        averaged_kfolds_train_acc_of_each_baseline = {}
        averaged_kfolds_val_acc_of_each_baseline = {}
        averaged_kfolds_val_roc_of_each_baseline = {}
        for solution in self.full_path_sorted_solutions:
            if "R_" in solution or os.path.isfile(solution):
                baseline_key = solution.split("/")[-1]
                averaged_kfolds_train_loss_of_each_baseline[baseline_key] = []
                averaged_kfolds_val_loss_of_each_baseline[baseline_key] = []
                averaged_kfolds_train_acc_of_each_baseline[baseline_key] = []
                averaged_kfolds_val_acc_of_each_baseline[baseline_key] = []
                averaged_kfolds_val_roc_of_each_baseline[baseline_key] = []
            else:
                key = solution.split('/')[-1].split('_')[0]
                for filename in os.listdir(solution):
                    if 'metrics-averages' in filename:
                        averaged_kfolds_train_loss_of_each_solution_per_generation[key] = []
                        averaged_kfolds_val_loss_of_each_solution_per_generation[key] = []
                        averaged_kfolds_train_acc_of_each_solution_per_generation[key] = []
                        averaged_kfolds_val_acc_of_each_solution_per_generation[key] = []
                        averaged_kfolds_val_roc_of_each_solution_per_generation[key] = []
        for solution in self.full_path_sorted_solutions:
            if "R_" in solution:
                baseline_key = solution.split("/")[-1]
                for filename in os.listdir(solution):
                    if "metrics-averages" in filename:
                        metrics_averages = pd.read_csv(os.path.join(solution, filename), index_col=0)
                        train_losses = metrics_averages.loc["Train Losses"]
                        val_losses = metrics_averages.loc["Val Losses"]
                        train_accuracies = metrics_averages.loc["Train Accuracy"]
                        val_accuracies = metrics_averages.loc["Val Accuracy"]
                        val_rocs = metrics_averages.loc["Val ROC"]
                        averaged_kfolds_train_loss_of_each_baseline[baseline_key] = train_losses
                        averaged_kfolds_val_loss_of_each_baseline[baseline_key] = val_losses
                        averaged_kfolds_train_acc_of_each_baseline[baseline_key] = train_accuracies
                        averaged_kfolds_val_acc_of_each_baseline[baseline_key] = val_accuracies
                        averaged_kfolds_val_roc_of_each_baseline[baseline_key] = val_rocs

            else:
                key = solution.split('/')[-1].split('_')[0]
                for filename in os.listdir(solution):
                    if 'metrics-averages' in filename:
                        metrics_averages = pd.read_csv(os.path.join(solution, filename), index_col=0)
                        train_losses = metrics_averages.loc["Train Losses"]
                        val_losses = metrics_averages.loc["Val Losses"]
                        train_accuracies = metrics_averages.loc["Train Accuracy"]
                        val_accuracies = metrics_averages.loc["Val Accuracy"]
                        val_rocs = metrics_averages.loc["Val ROC"]
                        averaged_kfolds_train_loss_of_each_solution_per_generation[key].append(train_losses)
                        averaged_kfolds_val_loss_of_each_solution_per_generation[key].append(val_losses)
                        averaged_kfolds_train_acc_of_each_solution_per_generation[key].append(train_accuracies)
                        averaged_kfolds_val_acc_of_each_solution_per_generation[key].append(val_accuracies)
                        averaged_kfolds_val_roc_of_each_solution_per_generation[key].append(val_rocs)

        return (averaged_kfolds_train_loss_of_each_solution_per_generation,
                averaged_kfolds_val_loss_of_each_solution_per_generation,
                averaged_kfolds_train_acc_of_each_solution_per_generation,
                averaged_kfolds_val_acc_of_each_solution_per_generation,
                averaged_kfolds_val_roc_of_each_solution_per_generation,
                averaged_kfolds_train_loss_of_each_baseline,
                averaged_kfolds_val_loss_of_each_baseline,
                averaged_kfolds_train_acc_of_each_baseline,
                averaged_kfolds_val_acc_of_each_baseline,
                averaged_kfolds_val_roc_of_each_baseline
                )

    def get_metrics_from_separate_folds_csvs_calculate_their_average(self):
        averaged_kfolds_train_loss_of_each_solution_per_generation = defaultdict(list)
        averaged_kfolds_val_loss_of_each_solution_per_generation = defaultdict(list)
        averaged_kfolds_train_acc_of_each_solution_per_generation = defaultdict(list)
        averaged_kfolds_val_acc_of_each_solution_per_generation = defaultdict(list)
        averaged_kfolds_val_roc_of_each_solution_per_generation = defaultdict(list)
        for solution in self.full_path_sorted_solutions:
            if "R_" in solution:
                continue
            else:
                solution_train_losses = []
                solution_val_losses = []
                solution_train_accuracies = []
                solution_val_accuracies = []
                solution_val_rocs = []
                solution_from_generation = solution.split('/')[-1].split('_')[0]
                for filename in sorted(os.listdir(solution)):
                    if "Fold" in filename:
                        fold_metrics = pd.read_csv(os.path.join(solution, filename), index_col=0)
                        fold_train_losses = fold_metrics.loc["Train Losses"]
                        fold_val_losses = fold_metrics.loc["Val Losses"]
                        fold_train_accuracies = fold_metrics.loc["Train Accuracy"]
                        fold_val_accuracies = fold_metrics.loc["Val Accuracy"]
                        fold_val_rocs = fold_metrics.loc["Val ROC"]
                        solution_train_losses.append(fold_train_losses)
                        solution_val_losses.append(fold_val_losses)
                        solution_train_accuracies.append(fold_train_accuracies)
                        solution_val_accuracies.append(fold_val_accuracies)
                        solution_val_rocs.append(fold_val_rocs)
                try:
                    # Concatenate the DataFrames into a single DataFrame
                    combined_train_losses = pd.concat(solution_train_losses)
                    averaged_train_losses = combined_train_losses.groupby(combined_train_losses.index).mean()
                    combined_val_losses = pd.concat(solution_val_losses)
                    averaged_val_losses = combined_val_losses.groupby(combined_val_losses.index).mean()
                    combined_train_accuracies = pd.concat(solution_train_accuracies)
                    averaged_train_accuracies = combined_train_accuracies.groupby(combined_train_accuracies.index).mean()
                    combined_val_accuracies = pd.concat(solution_val_accuracies)
                    averaged_val_accuracies = combined_val_accuracies.groupby(combined_val_accuracies.index).mean()
                    combined_val_rocs = pd.concat(solution_val_rocs)
                    averaged_val_rocs = combined_val_rocs.groupby(combined_val_rocs.index).mean()
                    averaged_kfolds_train_loss_of_each_solution_per_generation[solution_from_generation].append(averaged_train_losses)
                    averaged_kfolds_val_loss_of_each_solution_per_generation[solution_from_generation].append(averaged_val_losses)
                    averaged_kfolds_train_acc_of_each_solution_per_generation[solution_from_generation].append(averaged_train_accuracies)
                    averaged_kfolds_val_acc_of_each_solution_per_generation[solution_from_generation].append(averaged_val_accuracies)
                    averaged_kfolds_val_roc_of_each_solution_per_generation[solution_from_generation].append(averaged_val_rocs)
                except:
                    continue

        return (averaged_kfolds_train_loss_of_each_solution_per_generation,
                averaged_kfolds_val_loss_of_each_solution_per_generation,
                averaged_kfolds_train_acc_of_each_solution_per_generation,
                averaged_kfolds_val_acc_of_each_solution_per_generation,
                averaged_kfolds_val_roc_of_each_solution_per_generation)



    def average_baselines(self, metric_dict):
        # Concatenate dataframes into a single dataframe
        df_concatenated = pd.concat(metric_dict.values(), axis=1)
        # Compute the mean along the columns (axis=1)
        averaged_df = df_concatenated.mean(axis=1)
        return averaged_df.tolist()

    def average_solution_by_epoch(self, metric_list):
        # print(metric_list)
        concatenated_df = pd.concat(metric_list, axis=1)
        # print(concatenated_df)
        transposed_df = concatenated_df.T
        # print(transposed_df)
        average_by_epoch = transposed_df.groupby(level=0).mean()
        std_by_epoch = transposed_df.groupby(level=0).std()

        # print(average_by_epoch)
        # sys.exit()
        return average_by_epoch, std_by_epoch



    # def average_solution_by_epoch(self, metric_list):
    #     # print(metric_list)
    #     concatenated_df = pd.concat(metric_list, axis=1)
    #     average_by_epoch = concatenated_df.mean(axis=1)
    #     # average_by_epoch = transposed_df.groupby(level=0).mean()
    #     return average_by_epoch



    def do_average_all_solutions_in_generation_by_metric(self, averaged_kfolds_metric_of_each_generation_dict):
        metric_average_by_generation_dict = {}
        metric_std_by_generation_dict = {}
        for key, value in averaged_kfolds_metric_of_each_generation_dict.items():
            averaged_solutions_metric_by_generation, std_solutions_metric_by_generation = self.average_solution_by_epoch(value)
            metric_average_by_generation_dict[key] = averaged_solutions_metric_by_generation.values.tolist()[0]
            metric_std_by_generation_dict[key] = std_solutions_metric_by_generation.values.tolist()[0]
        return metric_average_by_generation_dict, metric_std_by_generation_dict


    def get_top_n_performing_solutions(self, averaged_kfolds_metric_of_each_solution_per_generation,
                                       top_n=None,
                                       metric_type=None):
        """
        Args:
            averaged_kfolds_metric_of_each_solution_per_generation: dictionary of the averaged kfolds metrics
            top_n: integer
            metric_type: loss or accuracy

        Returns:
            top_n_performing_solutions: dictionary of the top performing solutions with generation number as keys
            and as values the solution that record best performance in terms of metric.
        """

        averaged_kfolds_metric_of_topn_solutions_per_generation = {key: None for key in
                                                                   averaged_kfolds_metric_of_each_solution_per_generation}

        if metric_type == "loss":
            for generation in averaged_kfolds_metric_of_each_solution_per_generation:
                sorted_generation_losses_dfs = sorted(
                    averaged_kfolds_metric_of_each_solution_per_generation[generation], key=lambda df: df.min())
                lowest_generation_loss_dfs = sorted_generation_losses_dfs[:top_n]
                averaged_kfolds_metric_of_topn_solutions_per_generation[generation] = lowest_generation_loss_dfs

        elif metric_type == "accuracy" or metric_type == "roc":
            for generation in averaged_kfolds_metric_of_each_solution_per_generation:
                sorted_generation_accuracy_dfs = sorted(
                    averaged_kfolds_metric_of_each_solution_per_generation[generation], key=lambda df: df.max(),
                    reverse=True)
                highest_generation_accuracy_dfs = sorted_generation_accuracy_dfs[:top_n]
                averaged_kfolds_metric_of_topn_solutions_per_generation[generation] = highest_generation_accuracy_dfs

        return averaged_kfolds_metric_of_topn_solutions_per_generation


    def get_values_of_last_epoch_of_each_solution_per_generation(self, averaged_kfolds_metric_of_each_solution_per_generation):
        """
        Args:
            averaged_kfolds_metric_of_each_solution_per_generation: dictionary of the averaged kfolds metrics

        Returns:
            get_values_of_last_epoch_of_each_solution_per_generation: dictionary of the metrics values recorded at last
            epoch of CNN training of each solution.
        """


        values_of_last_epoch_of_each_solution_per_generation = {key: None for key in averaged_kfolds_metric_of_each_solution_per_generation}
        min_max_avg_std_values_of_each_generation = {key: None for key in averaged_kfolds_metric_of_each_solution_per_generation}

        for generation in averaged_kfolds_metric_of_each_solution_per_generation:
            last_epoch_values = [df.iloc[-1] for df in averaged_kfolds_metric_of_each_solution_per_generation[generation]]
            values_of_last_epoch_of_each_solution_per_generation[generation] = last_epoch_values

        for generation in values_of_last_epoch_of_each_solution_per_generation:
            generation_values = values_of_last_epoch_of_each_solution_per_generation[generation]
            avg = np.mean(generation_values)
            std = np.std(generation_values)
            min_val = np.min(generation_values)
            max_val = np.max(generation_values)
            q1_val = avg - std;
            q3_val = avg + std;
            min_max_avg_std_values_of_each_generation[generation] = [min_val, q1_val, avg, std, q3_val, max_val]

        return values_of_last_epoch_of_each_solution_per_generation, min_max_avg_std_values_of_each_generation



    def get_best_solution_of_best_generation(self,
                                             averaged_kfolds_metric_of_topn_solutions_per_generation,
                                             metric_type=None):
        best_generation = None
        best_metric_df = None
        best_metric_value = None

        if metric_type == "loss":
            min_loss_df_key = min(averaged_kfolds_metric_of_topn_solutions_per_generation,
                                  key=lambda k: averaged_kfolds_metric_of_topn_solutions_per_generation[k][0].min())
            best_metric_df = averaged_kfolds_metric_of_topn_solutions_per_generation[min_loss_df_key]
            best_generation = min_loss_df_key
            # best_metric_value = best_metric_df[0].min()
        elif metric_type == "accuracy" or metric_type == "roc":
            max_acc_df_key = max(averaged_kfolds_metric_of_topn_solutions_per_generation,
                                  key=lambda k: averaged_kfolds_metric_of_topn_solutions_per_generation[k][0].max())
            best_metric_df = averaged_kfolds_metric_of_topn_solutions_per_generation[max_acc_df_key]
            best_generation = max_acc_df_key

        return best_metric_df, best_generation

    def get_best_baseline(self, baselines_dictionary, metric_type=None):
        best_metric_df, best_beseline_code = None, None
        if metric_type == 'loss':
            min_loss_baseline = min(baselines_dictionary, key=lambda k: baselines_dictionary[k].min())
            best_metric_df = baselines_dictionary[min_loss_baseline]
            best_beseline_code = min_loss_baseline
        elif metric_type == 'accuracy' or metric_type == 'roc':
            max_baseline = max(baselines_dictionary, key=lambda k: baselines_dictionary[k].max())
            best_metric_df = baselines_dictionary[max_baseline]
            best_beseline_code = max_baseline
        return best_metric_df, best_beseline_code





    def get_best_metric_value_of_each_generation(self, solutions_dictionary, metric_type=None):
        best_metric_value_of_each_generation = {}
        if metric_type == 'loss':
            for key, value in solutions_dictionary.items():
                best_metric_value = min(value[0].tolist())
                best_metric_value_of_each_generation[key] = best_metric_value
        elif metric_type == 'accuracy' or metric_type == 'roc':
            for key, value in solutions_dictionary.items():
                best_metric_value = max(value[0].tolist())
                best_metric_value_of_each_generation[key] = best_metric_value

        return best_metric_value_of_each_generation

    def get_worst_metric_value_of_each_generation(self, solutions_dictionary, metric_type=None):
        best_metric_value_of_each_generation = {}
        if metric_type == 'loss':
            for key, value in solutions_dictionary.items():
                best_metric_value = max(value[0].tolist())
                best_metric_value_of_each_generation[key] = best_metric_value
        elif metric_type == 'accuracy' or metric_type == 'roc':
            for key, value in solutions_dictionary.items():
                best_metric_value = min(value[0].tolist())
                best_metric_value_of_each_generation[key] = best_metric_value

        return best_metric_value_of_each_generation

    def sort_best_metrics_of_each_individual_in_each_generation(self,
                                                                best_metrics_of_every_individual_in_each_generation_dict,
                                                                metric_type,
                                                                top_n):
        sorted_best_metrics_of_every_individual_in_each_generation_dict = {}

        for gen, values in best_metrics_of_every_individual_in_each_generation_dict.items():
            if metric_type == 'loss':
                sorted_best_metrics_of_every_individual_in_each_generation_dict[gen] = sorted(values)[:top_n]
            elif metric_type == 'accuracy' or metric_type == 'roc':
                sorted_best_metrics_of_every_individual_in_each_generation_dict[gen] = sorted(values)[-top_n:]
        return sorted_best_metrics_of_every_individual_in_each_generation_dict
    def get_best_metric_value_of_each_solution(self, solutions_dictionary, metric_type=None):
        best_metric_value_of_each_solution_per_generation = {}
        averaged_best_metric_value_of_generation = {}
        std_best_metric_per_generation = {}
        if metric_type == 'loss':
            for generation, population in solutions_dictionary.items():
                best_metric_value_of_each_solution_per_generation[generation] = []
                for individual in population:
                    min_individual_loss = min(individual.tolist())
                    best_metric_value_of_each_solution_per_generation[generation].append(min_individual_loss)
                averaged_best_metric_value_of_generation[generation] = sum(
                    best_metric_value_of_each_solution_per_generation[generation]) / len(
                    best_metric_value_of_each_solution_per_generation[generation])
                std_best_metric_per_generation[generation] = np.std(
                    best_metric_value_of_each_solution_per_generation[generation])

        elif metric_type == 'accuracy' or metric_type == 'roc':
            for generation, population in solutions_dictionary.items():
                best_metric_value_of_each_solution_per_generation[generation] = []
                for individual in population:
                    max_individual_accuracy = max(individual.tolist())
                    best_metric_value_of_each_solution_per_generation[generation].append(max_individual_accuracy)
                averaged_best_metric_value_of_generation[generation] = sum(
                    best_metric_value_of_each_solution_per_generation[generation]) / len(
                    best_metric_value_of_each_solution_per_generation[generation])
                std_best_metric_per_generation[generation] = np.std(
                    best_metric_value_of_each_solution_per_generation[generation])

        return (best_metric_value_of_each_solution_per_generation,
                averaged_best_metric_value_of_generation,
                std_best_metric_per_generation)

    def calculate_fitness_score(self, averaged_kfolds_metrics_of_each_solution_per_generation):
        for gen, df_list in averaged_kfolds_metrics_of_each_solution_per_generation.items():
            for i, df in enumerate(df_list):
                averaged_kfolds_metrics_of_each_solution_per_generation[gen][i] = 1 / df

        last_epoch_values = {}
        for gen, df_list in averaged_kfolds_metrics_of_each_solution_per_generation.items():
            last_epoch_values[gen] = [df.iloc[-1] for df in df_list]

        last_epoch_averaged_values = {}
        for gen, list_values in last_epoch_values.items():
            last_epoch_averaged_values[gen] = sum(list_values) / len(list_values)

        last_epoch_std = {}
        for gen, list_values in last_epoch_values.items():
            last_epoch_std[gen] = np.std(list_values)



        minimum_fitness = {}
        maximum_fitness = {}
        for gen, list_values in last_epoch_values.items():
            minimum_fitness[gen] = min(list_values)
            maximum_fitness[gen] = max(list_values)

        return last_epoch_values, last_epoch_averaged_values, last_epoch_std, minimum_fitness, maximum_fitness












