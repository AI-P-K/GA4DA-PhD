import os
import sys
from collections import OrderedDict

from scipy.stats import linregress
import matplotlib.pyplot as plt
from pathlib import Path
import os
import numpy as np

class GAMetricsCalculator:
    def get_best_metric_value_of_each_generation_dict_with_list(self, solutions_dictionary, metric_type=None):
        """
        Args:
            solutions_dictionary:
            metric_type:

        Returns:
        """
        best_metric_value_of_each_generation_dict = {}
        if metric_type == 'loss':
            for key, value in solutions_dictionary.items():
                best_metric_value = min(value)
                best_metric_value_of_each_generation_dict[key] = best_metric_value
        elif metric_type == 'accuracy' or metric_type == 'roc':
            for key, value in solutions_dictionary.items():
                best_metric_value = max(value)
                best_metric_value_of_each_generation_dict[key] = best_metric_value
        return best_metric_value_of_each_generation_dict


class GAMetricsPlotter:
    def __init__(self, figure_path, fig_size, font):
        self.figure_path = figure_path
        self.fig_size = fig_size
        self.font = font

    def plot_ga_all_top1_top5_baseline_lineplot(self, best_averaged_train_losses_of_each_generation,
                              top_1_train_losses_of_every_individual_in_each_generation,
                              top_5_train_losses_of_every_individual_in_each_generation,
                              baseline,
                              title=None,
                              xlabel=None,
                              ylabel=None
                              ):

            average_of_all_individuals_per_generation = list(best_averaged_train_losses_of_each_generation.values())
            average_of_top_5_individuals_per_generation = []
            top_1_individual_per_generation = []

            for gen in list(top_5_train_losses_of_every_individual_in_each_generation.values()):
                generation_average = sum(gen) / len(gen)
                average_of_top_5_individuals_per_generation.append(generation_average)

            for gen in list(top_1_train_losses_of_every_individual_in_each_generation.values()):
                top_1_individual_per_generation.append(gen[0])

            generations = list(best_averaged_train_losses_of_each_generation.keys())
            plt.rc('font', **self.font)
            plt.figure(figsize=self.fig_size)
            positions = range(1, len(generations) + 1)
            plt.axhline(y=baseline, color='r', linestyle='--', label='Baseline')
            plt.plot(positions, average_of_all_individuals_per_generation, marker='o', color='blue', label='All Individuals')
            plt.plot(positions, average_of_top_5_individuals_per_generation, marker='o', color='red', label='Top 5 Individuals')
            plt.plot(positions, top_1_individual_per_generation, marker='o', color='green', label='Top 1 Individual')

            plt.xlabel(f'{xlabel}')
            plt.ylabel(f'{ylabel}')
            # plt.title(f'{title}')
            plt.xticks(positions)
            plt.legend()

            # for i, txt in enumerate(average_of_all_individuals_per_generation):
            #     plt.annotate(round(txt, 4), (positions[i], average_of_all_individuals_per_generation[i]),
            #                  textcoords="offset points", xytext=(0, 10), ha='center', fontsize=8)
            #
            # for i, txt in enumerate(average_of_top_5_individuals_per_generation):
            #     plt.annotate(round(txt, 4), (positions[i], average_of_top_5_individuals_per_generation[i]),
            #                  textcoords="offset points", xytext=(0, 10), ha='center', fontsize=8)
            #
            # for i, txt in enumerate(top_1_individual_per_generation):
            #     plt.annotate(round(txt, 4), (positions[i], top_1_individual_per_generation[i]),
            #                  textcoords="offset points", xytext=(0, 10), ha='center', fontsize=8)
            #
            # plt.annotate(f'{baseline:.4f}', xy=(1, baseline), xytext=(8, 0),
            #              xycoords=('axes fraction', 'data'), textcoords='offset points', fontsize=8)

            plt.savefig(f'{self.figure_path}/{title}.png', bbox_inches='tight')
            plt.clf()
            plt.close()

    def plot_averaged_best_metrics_of_each_generation_box_plot(self,
                                                               min_max_avg_std_values_of_each_generation_dictionary,
                                                               title=None,
                                                               xlabel=None,
                                                               ylabel=None):

        generations = list(min_max_avg_std_values_of_each_generation_dictionary.keys())
        plt.figure(figsize=self.fig_size)
        # Create positions for each boxplot
        positions = range(1, len(generations) + 1)
        # Iterate over generations and plot a box for each
        for i, (key, values) in enumerate(min_max_avg_std_values_of_each_generation_dictionary.items()):
            plt.boxplot(values, positions=[positions[i]], labels=[key])
        plt.rc('font', **self.font)
        plt.xlabel(f'{xlabel}')
        plt.ylabel(f'{ylabel}')
        # plt.title(f'{title}')
        plt.grid(False)
        plt.savefig(f'{self.figure_path}/{title}.png', bbox_inches='tight')
        plt.clf()
        plt.close()

    def plot_averaged_best_metrics_and_std_of_each_generation_error_bar(self,
                                                                        avg_metric_dictionary,
                                                                        std_metric_dictionary,
                                                                        title=None,
                                                                        xlabel=None,
                                                                        ylabel=None):
        plt.figure(figsize=self.fig_size)
        keys = list(map(int, avg_metric_dictionary.keys()))
        avg_values = list(avg_metric_dictionary.values())
        std_values = list(std_metric_dictionary.values())
        plt.errorbar(keys, avg_values, yerr=std_values, fmt='o-', capsize=10)
        plt.rc('font', **self.font)
        # plt.ylim(2.5, 3)
        plt.xlabel(f'{xlabel}')
        plt.ylabel(f'{ylabel}')
        plt.xticks(keys)
        plt.savefig(f'{self.figure_path}/{title}.png', bbox_inches='tight')
        plt.clf()
        plt.close()

    def plot_best_metric_of_all_individuals_scatter_plot(self,
                                                         metric_dictionary,
                                                         title=None,
                                                         xlabel=None,
                                                         ylabel=None):
        num_sets = len(metric_dictionary)
        colors = plt.cm.viridis(np.linspace(0, 1, num_sets))
        plt.figure(figsize=self.fig_size)
        for idx, (key, values) in enumerate(metric_dictionary.items()):
            x_values = np.full(len(values), float(key))  # Use the key as x-axis value for each set
            plt.scatter(x_values, values, label=key, color=colors[idx])
        plt.rc('font', **self.font)
        # plt.title(f'{title}')
        plt.xlabel(f'{xlabel}')  # Update x-axis label as per your preference
        plt.ylabel(f'{ylabel}')
        plt.xticks(range(1, len(metric_dictionary) + 1))  # Set x-axis ticks to be integers
        plt.grid(True)
        plt.savefig(f'{self.figure_path}/{title}.png', bbox_inches='tight')
        plt.clf()
        plt.close()

    def plot_averaged_metrics_of_each_generation_and_baseline(self,
                                                 averaged_solution_metric_by_generation_dict,
                                                 averaged_baselines,
                                                 title=None,
                                                 xlabel=None,
                                                 ylabel=None):
        metrics_range = range(1, len(averaged_solution_metric_by_generation_dict['1']) + 1)
        plt.figure(figsize=self.fig_size)
        plt.plot(metrics_range, averaged_baselines, marker='s', linestyle='--', label=f'Baseline')

        for key, value in averaged_solution_metric_by_generation_dict.items():
            # plt.plot(metrics_range, value, marker='o', linestyle='-', label=f'Generation {key}')
            plt.plot(metrics_range, value, marker='o', linestyle='-')

        plt.plot([], [], marker='o', linestyle='-', label='Generation')
        plt.rc('font', **self.font)
        # plt.title(f'{title}')
        plt.xlabel(f'{xlabel}')
        plt.ylabel(f'{ylabel}')
        plt.legend()
        plt.grid(True)
        plt.xticks(metrics_range)
        plt.savefig(f'{self.figure_path}/{title}.png', bbox_inches='tight')
        plt.clf()
        plt.close()


