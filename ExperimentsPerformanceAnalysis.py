import sys
from utils.GetMetricsFromCsv import GetMetricsFromCsv
import os
from pathlib import Path
import shutil
from utils.GAMetricsCalculator import GAMetricsCalculator, GAMetricsPlotter
import copy
import csv
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

for test in os.listdir(result_root_folder):
    if "T_" in test:
        fig_size = (12, 7)
        font = {'size': 20}
        plot_folder = f'{test}-plots'
        figure_path = Path(os.path.join(result_root_folder, test, plot_folder))
        try:
            shutil.rmtree(os.path.join(result_root_folder, test, plot_folder))
            print('Old plots directory removed from all_solutions_in_test_folder')
        except:
            print("Nothing to remove")

        all_solutions_in_test_folder = sorted(os.listdir(os.path.join(result_root_folder, test)), key=custom_sort)

        if not os.path.exists(os.path.join(result_root_folder, test, plot_folder)):
            os.makedirs(os.path.join(result_root_folder, test, plot_folder))
            print(f'New {plot_folder} directory created')


        full_path_solutions_folder = [os.path.join(result_root_folder, test, i) for i in all_solutions_in_test_folder]
        full_path_solutions_folder = [item for item in full_path_solutions_folder if "description.txt" not in item and "time.csv" not in item]
        get_metrics = GetMetricsFromCsv(full_path_solutions_folder)

        """
        1. Load data from "-metrics-averages.csv" into dictionaries.
        averaged_kfolds_val_loss_of_each_baseline={'R_abcdef': val loss dataframe, 
                                                   'R_ghijklm': val loss dataframe}
                                                   
        averaged_kfolds_val_loss_of_each_solution_per_generation={'1': val loss dataframes of all individuals in generation 1, 
                                                                  '2': val loss dataframes of all individuals in generation 2}
        """
        (averaged_kfolds_train_loss_of_each_solution_per_generation,
         averaged_kfolds_val_loss_of_each_solution_per_generation,
         averaged_kfolds_train_acc_of_each_solution_per_generation,
         averaged_kfolds_val_acc_of_each_solution_per_generation,
         averaged_kfolds_val_roc_of_each_solution_per_generation,
         averaged_kfolds_train_loss_of_each_baseline,
         averaged_kfolds_val_loss_of_each_baseline,
         averaged_kfolds_train_acc_of_each_baseline,
         averaged_kfolds_val_acc_of_each_baseline,
         averaged_kfolds_val_roc_of_each_baseline) = get_metrics.put_averaged_metrics_csv_in_dict()



        """
        averaged_kfolds_val_loss_of_top_5_solutions_per_generation={'1': val loss dataframes of top 5 individuals in generation 1, 
                                                                  '2': val loss dataframes of top 5 individuals in generation 2}
        """
        averaged_kfolds_train_loss_of_top_5_solutions_per_generation = get_metrics.get_top_n_performing_solutions(
                                                          averaged_kfolds_train_loss_of_each_solution_per_generation,
                                                          top_n=5,
                                                          metric_type='loss')

        averaged_kfolds_val_loss_of_top_5_solutions_per_generation = get_metrics.get_top_n_performing_solutions(
                                                          averaged_kfolds_val_loss_of_each_solution_per_generation,
                                                          top_n=5,
                                                          metric_type='loss')

        averaged_kfolds_train_acc_of_top_5_solutions_per_generation = get_metrics.get_top_n_performing_solutions(
                                                          averaged_kfolds_train_acc_of_each_solution_per_generation,
                                                          top_n=5,
                                                          metric_type='accuracy')
        averaged_kfolds_val_acc_of_top_5_solutions_per_generation = get_metrics.get_top_n_performing_solutions(
                                                          averaged_kfolds_val_acc_of_each_solution_per_generation,
                                                          top_n=5,
                                                          metric_type='accuracy')

        averaged_kfolds_val_roc_of_top_5_solutions_per_generation = get_metrics.get_top_n_performing_solutions(
                                                          averaged_kfolds_val_roc_of_each_solution_per_generation,
                                                          top_n=5,
                                                          metric_type='roc')

        """
        kfolds_val_loss_of_top_1_solutions_per_generation={'1': val loss dataframes of top 1 individual in generation 1, 
                                                           '2': val loss dataframes of top 1 individual in generation 2}
        """
        kfolds_train_loss_of_top_1_solutions_per_generation = get_metrics.get_top_n_performing_solutions(
            averaged_kfolds_train_loss_of_each_solution_per_generation,
            top_n=1,
            metric_type='loss')

        kfolds_val_loss_of_top_1_solutions_per_generation = get_metrics.get_top_n_performing_solutions(
            averaged_kfolds_val_loss_of_each_solution_per_generation,
            top_n=1,
            metric_type='loss')

        kfolds_train_acc_of_top_1_solutions_per_generation = get_metrics.get_top_n_performing_solutions(
            averaged_kfolds_train_acc_of_each_solution_per_generation,
            top_n=1,
            metric_type='accuracy')
        kfolds_val_acc_of_top_1_solutions_per_generation = get_metrics.get_top_n_performing_solutions(
            averaged_kfolds_val_acc_of_each_solution_per_generation,
            top_n=1,
            metric_type='accuracy')

        kfolds_val_roc_of_top_1_solutions_per_generation = get_metrics.get_top_n_performing_solutions(
            averaged_kfolds_val_roc_of_each_solution_per_generation,
            top_n=1,
            metric_type='roc')


        ##### CALCULATE THE FITNESS SCORE BASED ON THE VALIDATION LOSS ####
        averaged_kfolds_fitness_of_each_solution_per_generation_per_generation = copy.deepcopy(averaged_kfolds_val_loss_of_each_solution_per_generation)
        for gen, val_loss_dataframe_list in averaged_kfolds_fitness_of_each_solution_per_generation_per_generation.items():
            for i, df in enumerate(val_loss_dataframe_list):
                averaged_kfolds_fitness_of_each_solution_per_generation_per_generation[gen][i] = 1/df

        averaged_kfolds_fitness_of_top_5_solutions_per_generation = get_metrics.get_top_n_performing_solutions(
            averaged_kfolds_fitness_of_each_solution_per_generation_per_generation,
            top_n=5,
            metric_type='roc')

        kfolds_fitness_of_top_1_solutions_per_generation = get_metrics.get_top_n_performing_solutions(
            averaged_kfolds_fitness_of_each_solution_per_generation_per_generation,
            top_n=1,
            metric_type='roc')




        """
        1.1. Load data from "Fold_0.csv, ... ,Fold_n.csv", do the average and compare it with the data loaded previously
        from "-metrics-averages.csv".
        """

        (manually_averaged_kfolds_train_loss_of_each_solution_per_generation,
         manually_averaged_kfolds_val_loss_of_each_solution_per_generation,
         manually_averaged_kfolds_train_acc_of_each_solution_per_generation,
         manually_averaged_kfolds_val_acc_of_each_solution_per_generation,
         manually_averaged_kfolds_val_roc_of_each_solution_per_generation) = get_metrics.get_metrics_from_separate_folds_csvs_calculate_their_average()



        """
        2. We can use the loaded data from point 1 or 1.1 to average the metrics of all individuals in a generation.
        Therefore each generation will be represented by a single list of values for each metric.
        """

        averaged_train_loss_of_each_generation, std_train_loss_of_each_generation = get_metrics.do_average_all_solutions_in_generation_by_metric(
            averaged_kfolds_train_loss_of_each_solution_per_generation)

        averaged_val_loss_of_each_generation, std_val_loss_of_each_generation = get_metrics.do_average_all_solutions_in_generation_by_metric(
            averaged_kfolds_val_loss_of_each_solution_per_generation)

        averaged_train_accuracy_of_each_generation, std_train_accuracy_of_each_generation = get_metrics.do_average_all_solutions_in_generation_by_metric(
            averaged_kfolds_train_acc_of_each_solution_per_generation)

        averaged_val_accuracy_of_each_generation, std_val_accuracy_of_each_generation = get_metrics.do_average_all_solutions_in_generation_by_metric(
            averaged_kfolds_val_acc_of_each_solution_per_generation)

        averaged_val_roc_of_each_generation, std_val_roc_of_each_generation = get_metrics.do_average_all_solutions_in_generation_by_metric(
            averaged_kfolds_val_roc_of_each_solution_per_generation)

        averaged_fitness_of_each_generation, std_fitness_of_each_generation = get_metrics.do_average_all_solutions_in_generation_by_metric(
            averaged_kfolds_fitness_of_each_solution_per_generation_per_generation)



        """
        averaged_val_loss_of_each_generation - epoch 1 of all individuals is averaged, and so on for all epochs. 
        results in a list with length as number of epochs.
        """


        max_averaged_fitness_of_all_individuals_in_generation = {gen: max(fitness_list) for generation, fitness_list in averaged_fitness_of_each_generation.items()}


        """
        Average of top 5 individuals in generations over epochs.
        """

        averaged_train_loss_of_top_5_each_generation, std_train_loss_of_top_5_each_generation = get_metrics.do_average_all_solutions_in_generation_by_metric(
            averaged_kfolds_train_loss_of_top_5_solutions_per_generation)
        averaged_val_loss_of_top_5_each_generation, std_val_loss_of_top_5_each_generation = get_metrics.do_average_all_solutions_in_generation_by_metric(
            averaged_kfolds_val_loss_of_top_5_solutions_per_generation)
        averaged_train_accuracy_of_top_5_each_generation, std_train_accuracy_of_top_5_each_generation = get_metrics.do_average_all_solutions_in_generation_by_metric(
            averaged_kfolds_train_acc_of_top_5_solutions_per_generation)
        averaged_val_accuracy_of_top_5_each_generation, std_val_accuracy_of_top_5_each_generation = get_metrics.do_average_all_solutions_in_generation_by_metric(
            averaged_kfolds_val_acc_of_top_5_solutions_per_generation)
        averaged_val_roc_of_top_5_each_generation, std_val_roc_of_top_5_each_generation = get_metrics.do_average_all_solutions_in_generation_by_metric(
            averaged_kfolds_val_roc_of_top_5_solutions_per_generation)
        averaged_fitness_of_top_5_each_generation, std_fitness_of_top_5_each_generation = get_metrics.do_average_all_solutions_in_generation_by_metric(
            averaged_kfolds_fitness_of_top_5_solutions_per_generation)


        def write_csv_data(avg_val_loss, avg_val_acc, avg_val_roc, avg_val_fitness, file_name):
            full_best_val_loss_of_each_generation = {'0': 'Val Loss'}
            full_best_val_accuracy_of_each_generation = {'0': 'Val Acc'}
            full_best_val_roc_of_each_generation = {'0': 'Val ROC'}
            full_best_fitness_of_each_generation = {'0': 'Fitness'}

            best_val_loss_of_each_generation = {key: min(value) for key, value in avg_val_loss.items()}
            best_val_acc_of_each_generation = {key: max(value) for key, value in avg_val_acc.items()}
            best_val_roc_of_each_generation = {key: max(value) for key, value in avg_val_roc.items()}
            best_fitness_of_each_generation = {key: max(value) for key, value in avg_val_fitness.items()}

            full_best_val_loss_of_each_generation.update(best_val_loss_of_each_generation)
            full_best_val_accuracy_of_each_generation.update(best_val_acc_of_each_generation)
            full_best_val_roc_of_each_generation.update(best_val_roc_of_each_generation)
            full_best_fitness_of_each_generation.update(best_fitness_of_each_generation)

            csv_file_path = os.path.join(figure_path, file_name)


            with open(csv_file_path, mode='w', newline='') as file:
                writer = csv.writer(file)
                # Write the keys (column names)
                writer.writerow(full_best_val_loss_of_each_generation.keys())
                # Write the values (under the keys)
                writer.writerow(full_best_val_loss_of_each_generation.values())
                writer.writerow(full_best_val_accuracy_of_each_generation.values())
                writer.writerow(full_best_val_roc_of_each_generation.values())
                writer.writerow(full_best_fitness_of_each_generation.values())

        """
        min_avg_values_of_all_individuals_per_generation.csv - contains the minimum value at recorded at 
        a certain epoch for each generation
        """

        write_csv_data(averaged_val_loss_of_each_generation,
                       averaged_val_accuracy_of_each_generation,
                       averaged_val_roc_of_each_generation,
                       averaged_fitness_of_each_generation,
                       'avg_values_of_all_individuals_per_generation.csv')

        write_csv_data(std_val_loss_of_each_generation,
                       std_val_accuracy_of_each_generation,
                       std_val_roc_of_each_generation,
                       std_fitness_of_each_generation,
                       'std_values_of_all_individuals_per_generation.csv')

        write_csv_data(averaged_val_loss_of_top_5_each_generation,
                       averaged_val_accuracy_of_top_5_each_generation,
                       averaged_val_roc_of_top_5_each_generation,
                       averaged_fitness_of_top_5_each_generation,
                       'avg_values_of_top5_individuals_per_generation.csv')





        max_averaged_fitness_of_top_5_individuals_in_generation = {gen: max(fitness_list) for generation, fitness_list in
                                                                 averaged_fitness_of_top_5_each_generation.items()}




        """
        Values of top 1 individuals in each generation over epochs transformed from dataframes to list
        """
        train_loss_of_top_1_each_generation = {key: value[0].tolist() for key, value in
                                               kfolds_train_loss_of_top_1_solutions_per_generation.items()}
        val_loss_of_top_1_each_generation = {key: value[0].tolist() for key, value in
                                               kfolds_val_loss_of_top_1_solutions_per_generation.items()}
        train_acc_of_top_1_each_generation = {key: value[0].tolist() for key, value in
                                               kfolds_train_acc_of_top_1_solutions_per_generation.items()}
        val_acc_of_top_1_each_generation = {key: value[0].tolist() for key, value in
                                               kfolds_val_acc_of_top_1_solutions_per_generation.items()}
        val_roc_of_top_1_each_generation = {key: value[0].tolist() for key, value in
                                               kfolds_val_roc_of_top_1_solutions_per_generation.items()}
        fitness_of_top_1_each_generation = {key: value[0].tolist() for key, value in
                                            kfolds_fitness_of_top_1_solutions_per_generation.items()}

        max_fitness_of_top_1_individual_in_generation = {gen: max(fitness_list) for generation, fitness_list
                                                                   in
                                                                   fitness_of_top_1_each_generation.items()}

        write_csv_data(val_loss_of_top_1_each_generation,
                       val_acc_of_top_1_each_generation,
                       val_roc_of_top_1_each_generation,
                       fitness_of_top_1_each_generation,
                       'values_of_top1_individuals_per_generation.csv')


        # manually_averaged_train_loss_of_each_generation = get_metrics.do_average_all_solutions_in_generation_by_metric(
        #     manually_averaged_kfolds_train_loss_of_each_solution_per_generation)
        # manually_averaged_val_loss_of_each_generation = get_metrics.do_average_all_solutions_in_generation_by_metric(
        #     manually_averaged_kfolds_val_loss_of_each_solution_per_generation)
        # manually_averaged_train_accuracy_of_each_generation = get_metrics.do_average_all_solutions_in_generation_by_metric(
        #     manually_averaged_kfolds_train_acc_of_each_solution_per_generation)
        # manually_averaged_val_accuracy_of_each_generation = get_metrics.do_average_all_solutions_in_generation_by_metric(
        #     manually_averaged_kfolds_val_acc_of_each_solution_per_generation)
        # manually_averaged_val_roc_of_each_generation = get_metrics.do_average_all_solutions_in_generation_by_metric(
        #     manually_averaged_kfolds_val_roc_of_each_solution_per_generation)

        """
        2.1 We can use the loaded data from point 1 or 1.1 to average the metrics of all baselines. Therefore there 
        will be a single list of values for each metric representing the baseline performance.
        """
        baselines_averaged_train_loss_over_epochs = get_metrics.average_baselines(averaged_kfolds_train_loss_of_each_baseline)
        baselines_averaged_val_loss_over_epochs = get_metrics.average_baselines(averaged_kfolds_val_loss_of_each_baseline)
        baselines_averaged_train_acc_over_epochs = get_metrics.average_baselines(averaged_kfolds_train_acc_of_each_baseline)
        baselines_averaged_val_acc_over_epochs = get_metrics.average_baselines(averaged_kfolds_val_acc_of_each_baseline)
        baselines_averaged_val_roc_over_epochs = get_metrics.average_baselines(averaged_kfolds_val_roc_of_each_baseline)
        baselines_averaged_fitness_over_epochs = [1/val_loss for val_loss in baselines_averaged_val_loss_over_epochs]


        """
        2.2 Identify the best recorded metric of the averaged baselines
        """

        best_averaged_train_loss = min(baselines_averaged_train_loss_over_epochs)
        best_averaged_val_loss = min(baselines_averaged_val_loss_over_epochs)
        best_averaged_train_acc = max(baselines_averaged_train_acc_over_epochs)
        best_averaged_val_acc = max(baselines_averaged_val_acc_over_epochs)
        best_averaged_val_roc = max(baselines_averaged_val_roc_over_epochs)
        best_averaged_fitness = max(baselines_averaged_fitness_over_epochs)

        """
        3. Identify the epoch of the recorded best metric. 
        """
        ga_metrics = GAMetricsCalculator()
        best_averaged_train_losses_of_each_generation = ga_metrics.get_best_metric_value_of_each_generation_dict_with_list(averaged_train_loss_of_each_generation, metric_type='loss')
        best_averaged_val_losses_of_each_generation = ga_metrics.get_best_metric_value_of_each_generation_dict_with_list(averaged_val_loss_of_each_generation, metric_type='loss')
        best_averaged_train_acc_of_each_generation = ga_metrics.get_best_metric_value_of_each_generation_dict_with_list(averaged_train_accuracy_of_each_generation, metric_type='accuracy')
        best_averaged_val_acc_of_each_generation = ga_metrics.get_best_metric_value_of_each_generation_dict_with_list(averaged_val_accuracy_of_each_generation, metric_type='accuracy')
        best_averaged_val_roc_of_each_generation = ga_metrics.get_best_metric_value_of_each_generation_dict_with_list(averaged_val_roc_of_each_generation, metric_type='roc')
        best_averaged_fitness_of_each_generation = ga_metrics.get_best_metric_value_of_each_generation_dict_with_list(averaged_fitness_of_each_generation, metric_type='roc')

        """
        4. Identify the best values of each metric of every individual per generation.
        """
        (best_train_losses_of_every_individual_in_each_generation,
         avg_best_train_losses_of_generation,
         std_train_loss_per_generation) = get_metrics.get_best_metric_value_of_each_solution(
            averaged_kfolds_train_loss_of_each_solution_per_generation, metric_type="loss")


        (best_val_losses_of_every_individual_in_each_generation,
         avg_best_val_losses_per_generation,
         std_val_loss_per_generation) = get_metrics.get_best_metric_value_of_each_solution(
            averaged_kfolds_val_loss_of_each_solution_per_generation, metric_type="loss")

        (best_train_accuracies_of_every_individual_in_each_generation,
         avg_best_train_accuracies_of_generation,
         std_train_accuracy_per_generation) = get_metrics.get_best_metric_value_of_each_solution(
            averaged_kfolds_train_acc_of_each_solution_per_generation, metric_type="accuracy")

        (best_val_accuracies_of_every_individual_in_each_generation,
         avg_best_val_accuracies_of_generation,
         std_val_accuracy_per_generation) = get_metrics.get_best_metric_value_of_each_solution(
            averaged_kfolds_val_acc_of_each_solution_per_generation, metric_type="accuracy")

        (best_val_rocs_of_every_individual_in_each_generation,
         avg_best_val_roc_of_generation,
         std_val_roc_per_generation) = get_metrics.get_best_metric_value_of_each_solution(
            averaged_kfolds_val_roc_of_each_solution_per_generation, metric_type="roc")

        (best_fitness_of_every_individual_in_each_generation,
         avg_fitness_roc_of_generation,
         std_fitness_per_generation) = get_metrics.get_best_metric_value_of_each_solution(
            averaged_kfolds_fitness_of_each_solution_per_generation_per_generation, metric_type="roc")

        """
        4.1 Identify the top 1 performers of each generation
        """

        top_1_train_losses_of_every_individual_in_each_generation = get_metrics.sort_best_metrics_of_each_individual_in_each_generation(best_train_losses_of_every_individual_in_each_generation, metric_type="loss", top_n=1)
        top_1_val_losses_of_every_individual_in_each_generation = get_metrics.sort_best_metrics_of_each_individual_in_each_generation(best_val_losses_of_every_individual_in_each_generation, metric_type="loss", top_n=1)
        top_1_train_accuracies_of_every_individual_in_each_generation = get_metrics.sort_best_metrics_of_each_individual_in_each_generation(best_train_accuracies_of_every_individual_in_each_generation, metric_type="accuracy", top_n=1)
        top_1_val_accuracies_of_every_individual_in_each_generation = get_metrics.sort_best_metrics_of_each_individual_in_each_generation(best_val_accuracies_of_every_individual_in_each_generation, metric_type="accuracy", top_n=1)
        top_1_val_roc_of_every_individual_in_each_generation = get_metrics.sort_best_metrics_of_each_individual_in_each_generation(best_val_rocs_of_every_individual_in_each_generation, metric_type="roc", top_n=1)
        top_1_fitness_of_every_individual_in_each_generation = get_metrics.sort_best_metrics_of_each_individual_in_each_generation(best_fitness_of_every_individual_in_each_generation, metric_type="roc", top_n=1)

        """
        4.2 Identify the top 5 performers of each generation
        """
        top_5_train_losses_of_every_individual_in_each_generation = get_metrics.sort_best_metrics_of_each_individual_in_each_generation(
            best_train_losses_of_every_individual_in_each_generation, metric_type="loss", top_n=5)
        top_5_val_losses_of_every_individual_in_each_generation = get_metrics.sort_best_metrics_of_each_individual_in_each_generation(
            best_val_losses_of_every_individual_in_each_generation, metric_type="loss", top_n=5)
        top_5_train_accuracies_of_every_individual_in_each_generation = get_metrics.sort_best_metrics_of_each_individual_in_each_generation(
            best_train_accuracies_of_every_individual_in_each_generation, metric_type="accuracy", top_n=5)
        top_5_val_accuracies_of_every_individual_in_each_generation = get_metrics.sort_best_metrics_of_each_individual_in_each_generation(
            best_val_accuracies_of_every_individual_in_each_generation, metric_type="accuracy", top_n=5)
        top_5_val_roc_of_every_individual_in_each_generation = get_metrics.sort_best_metrics_of_each_individual_in_each_generation(
            best_val_rocs_of_every_individual_in_each_generation, metric_type="roc", top_n=5)
        top_5_fitness_of_every_individual_in_each_generation = get_metrics.sort_best_metrics_of_each_individual_in_each_generation(
            best_fitness_of_every_individual_in_each_generation, metric_type="roc", top_n=5)

        """
        4.3 Plot GA evolution performance. An average of all individuals, vs average of top 5 individuals 
        vs top 1 individual of each generation.
        """
        ############### SIMPLE LINE PLOTS of Combined Averages of all, top 5 and top 1 individuals in a generation ################
        ############### SIMPLE LINE PLOTS of Combined Averages of all, top 5 and top 1 individuals in a generation ################
        ############### SIMPLE LINE PLOTS of Combined Averages of all, top 5 and top 1 individuals in a generation ################

        ga_metric_plotter = GAMetricsPlotter(figure_path=figure_path, fig_size=fig_size, font=font)
        ga_metric_plotter.plot_ga_all_top1_top5_baseline_lineplot(best_averaged_train_losses_of_each_generation,
                                          top_1_train_losses_of_every_individual_in_each_generation,
                                          top_5_train_losses_of_every_individual_in_each_generation,
                                          best_averaged_train_loss,
                                          title=f"{test} individuals averaged training loss per generation",
                                          xlabel='Generations',
                                          ylabel='Training Loss')

        ga_metric_plotter.plot_ga_all_top1_top5_baseline_lineplot(best_averaged_val_losses_of_each_generation,
                                          top_1_val_losses_of_every_individual_in_each_generation,
                                          top_5_val_losses_of_every_individual_in_each_generation,
                                          best_averaged_val_loss,
                                          title=f"{test} individuals averaged validation loss per generation",
                                          xlabel='Generations',
                                          ylabel='Validation Loss')

        ga_metric_plotter.plot_ga_all_top1_top5_baseline_lineplot(best_averaged_train_acc_of_each_generation,
                                          top_1_train_accuracies_of_every_individual_in_each_generation,
                                          top_5_train_accuracies_of_every_individual_in_each_generation,
                                          best_averaged_train_acc,
                                          title=f"{test} individuals averaged training accuracy per generation",
                                          xlabel='Generations',
                                          ylabel='Training Accuracy')

        ga_metric_plotter.plot_ga_all_top1_top5_baseline_lineplot(best_averaged_val_acc_of_each_generation,
                                          top_1_val_accuracies_of_every_individual_in_each_generation,
                                          top_5_val_accuracies_of_every_individual_in_each_generation,
                                          best_averaged_val_acc,
                                          title=f"{test} individuals averaged validation accuracy per generation",
                                          xlabel='Generations',
                                          ylabel='Validation Accuracy')

        ga_metric_plotter.plot_ga_all_top1_top5_baseline_lineplot(best_averaged_val_roc_of_each_generation,
                                          top_1_val_roc_of_every_individual_in_each_generation,
                                          top_5_val_roc_of_every_individual_in_each_generation,
                                          best_averaged_val_roc,
                                          title=f"{test} individuals averaged validation ROCs per generation",
                                          xlabel='Generations',
                                          ylabel='Validation ROC')

        ga_metric_plotter.plot_ga_all_top1_top5_baseline_lineplot(best_averaged_fitness_of_each_generation,
                                          top_1_fitness_of_every_individual_in_each_generation,
                                          top_5_fitness_of_every_individual_in_each_generation,
                                          best_averaged_fitness,
                                          title=f"{test} individuals averaged fitness score per generation",
                                          xlabel='Generations',
                                          ylabel='Fitness')


        #######################  BOX PLOT Best Metric of All Individuals in Generations  ##############################
        #######################  BOX PLOT Best Metric of All Individuals in Generations  ##############################
        #######################  BOX PLOT Best Metric of All Individuals in Generations  ##############################
        ga_metric_plotter.plot_averaged_best_metrics_of_each_generation_box_plot(
            best_train_losses_of_every_individual_in_each_generation,
            title=f"{test} train loss box plot for each generation",
            xlabel="Generations",
            ylabel="Training Loss")

        ga_metric_plotter.plot_averaged_best_metrics_of_each_generation_box_plot(
            best_val_losses_of_every_individual_in_each_generation,
            title=f"{test} validation loss box plot for each generation",
            xlabel="Generations",
            ylabel="Validation Loss")

        ga_metric_plotter.plot_averaged_best_metrics_of_each_generation_box_plot(
            best_train_accuracies_of_every_individual_in_each_generation,
            title=f"{test} training accuracy box plot for each generation",
            xlabel="Generations",
            ylabel="Training Accuracy")

        ga_metric_plotter.plot_averaged_best_metrics_of_each_generation_box_plot(
            best_val_accuracies_of_every_individual_in_each_generation,
            title=f"{test} validation accuracy box plot for each generation",
            xlabel="Generations",
            ylabel="Validation Accuracy")

        ga_metric_plotter.plot_averaged_best_metrics_of_each_generation_box_plot(
            best_val_rocs_of_every_individual_in_each_generation,
            title=f"{test} validation ROC box plot for each generation",
            xlabel="Generations",
            ylabel="Validation ROC")

        ga_metric_plotter.plot_averaged_best_metrics_of_each_generation_box_plot(
            best_fitness_of_every_individual_in_each_generation,
            title=f"{test} fitness score box plot for each generation",
            xlabel="Generations",
            ylabel="Fitness")

        #######################  BOX PLOT Best Metric of Top 5 Individuals in Generations  #############################
        #######################  BOX PLOT Best Metric of Top 5 Individuals in Generations  #############################
        #######################  BOX PLOT Best Metric of Top 5 Individuals in Generations  #############################

        ga_metric_plotter.plot_averaged_best_metrics_of_each_generation_box_plot(
            top_5_train_losses_of_every_individual_in_each_generation,
            title=f"{test} top 5 train losses box plot for each generation",
            xlabel="Generations",
            ylabel="Training Loss")

        ga_metric_plotter.plot_averaged_best_metrics_of_each_generation_box_plot(
            top_5_val_losses_of_every_individual_in_each_generation,
            title=f"{test} top 5 validation losses box plot for each generation",
            xlabel="Generations",
            ylabel="Validation Loss")

        ga_metric_plotter.plot_averaged_best_metrics_of_each_generation_box_plot(
            top_5_train_accuracies_of_every_individual_in_each_generation,
            title=f"{test} top 5 training accuracies box plot for each generation",
            xlabel="Generations",
            ylabel="Training Accuracy")

        ga_metric_plotter.plot_averaged_best_metrics_of_each_generation_box_plot(
            top_5_val_accuracies_of_every_individual_in_each_generation,
            title=f"{test} top 5 validation accuracies box plot for each generation",
            xlabel="Generations",
            ylabel="Validation Accuracy")

        ga_metric_plotter.plot_averaged_best_metrics_of_each_generation_box_plot(
            top_5_val_roc_of_every_individual_in_each_generation,
            title=f"{test} top 5 validation ROCs box plot for each generation",
            xlabel="Generations",
            ylabel="Validation ROC")


        ga_metric_plotter.plot_averaged_best_metrics_of_each_generation_box_plot(
            top_5_fitness_of_every_individual_in_each_generation,
            title=f"{test} top 5 fitness scores box plot for each generation",
            xlabel="Generations",
            ylabel="Fitness")

        #################### ERROR BAR Averaged Best Metrics and STD of All Individual in Generations ##################
        #################### ERROR BAR Averaged Best Metrics and STD of All Individual in Generations ##################
        #################### ERROR BAR Averaged Best Metrics and STD of All Individual in Generations ##################

        ga_metric_plotter.plot_averaged_best_metrics_and_std_of_each_generation_error_bar(
                                                      best_averaged_train_losses_of_each_generation,
                                                      std_val_loss_per_generation,
                                                      title=f"{test} averaged validation losses and STD of each generation",
                                                      xlabel="Generations", ylabel="Training Loss")

        ga_metric_plotter.plot_averaged_best_metrics_and_std_of_each_generation_error_bar(
                                                      best_averaged_val_losses_of_each_generation,
                                                      std_train_loss_per_generation,
                                                      title=f"{test} averaged training losses and STD of each generation",
                                                      xlabel="Generations", ylabel="Validation Loss")

        ga_metric_plotter.plot_averaged_best_metrics_and_std_of_each_generation_error_bar(
                                                    best_averaged_train_acc_of_each_generation,
                                                    std_train_accuracy_per_generation,
                                                    title=f"{test} averaged train accuracy and STD of each generation",
                                                    xlabel="Generations", ylabel="Training Accuracy")


        ga_metric_plotter.plot_averaged_best_metrics_and_std_of_each_generation_error_bar(
                                                      best_averaged_val_acc_of_each_generation,
                                                      std_val_accuracy_per_generation,
                                                      title=f"{test} averaged validation accuracy and STD of each generation",
                                                      xlabel="Generations", ylabel="Validation Accuracy")


        ga_metric_plotter.plot_averaged_best_metrics_and_std_of_each_generation_error_bar(
                                                      best_averaged_val_roc_of_each_generation,
                                                      std_val_roc_per_generation,
                                                      title=f"{test} averaged validation ROC and STD of each generation",
                                                      xlabel="Generations", ylabel="Validation ROC")

        ga_metric_plotter.plot_averaged_best_metrics_and_std_of_each_generation_error_bar(
                                                    best_averaged_fitness_of_each_generation,
                                                    std_fitness_per_generation,
                                                    title=f"{test} averaged fitness scores and STD of each generation",
                                                    xlabel="Generations", ylabel="Fitness")

        ######################### Scatter Plot best metric of all individuals in generations ###########################
        ######################### Scatter Plot best metric of all individuals in generations ###########################
        ######################### Scatter Plot best metric of all individuals in generations ###########################


        ga_metric_plotter.plot_best_metric_of_all_individuals_scatter_plot(
                                            best_val_losses_of_every_individual_in_each_generation,
                                            title=f'{test} validation losses of each individual in each generation',
                                            xlabel='Generations',
                                            ylabel='Validation Loss'
                                            )

        ga_metric_plotter.plot_best_metric_of_all_individuals_scatter_plot(
                                            best_train_losses_of_every_individual_in_each_generation,
                                            title=f'{test} train losses of each individual in each generation',
                                            xlabel='Generations',
                                            ylabel='Training Loss'
                                            )

        ga_metric_plotter.plot_best_metric_of_all_individuals_scatter_plot(
                                            best_val_accuracies_of_every_individual_in_each_generation,
                                            title=f'{test} validation accuracies of each individual in each generation',
                                            xlabel='Generations',
                                            ylabel='Validation Accuracy'
                                            )

        ga_metric_plotter.plot_best_metric_of_all_individuals_scatter_plot(
                                            best_train_accuracies_of_every_individual_in_each_generation,
                                            title=f'{test} train accuracies of each individual in each generation',
                                            xlabel='Generations',
                                            ylabel='Training Accuracy'
                                            )

        ga_metric_plotter.plot_best_metric_of_all_individuals_scatter_plot(
                                            best_val_rocs_of_every_individual_in_each_generation,
                                            title=f'{test} validation ROCs of each individual in each generation',
                                            xlabel='Generations',
                                            ylabel='Validation ROC'
                                            )

        ga_metric_plotter.plot_best_metric_of_all_individuals_scatter_plot(
                                            best_fitness_of_every_individual_in_each_generation,
                                            title=f'{test} fitness of each individual in each generation',
                                            xlabel='Generations',
                                            ylabel='Fitness')

        ################### Line plot average of all solutions vs baselines on all epochs ######################
        ################### Line plot average of all solutions vs baselines on all epochs ######################
        ################### Line plot average of all solutions vs baselines on all epochs ######################


        ga_metric_plotter.plot_averaged_metrics_of_each_generation_and_baseline(averaged_train_loss_of_each_generation,
                                                 baselines_averaged_train_loss_over_epochs,
                                                 title=f'{test} averaged train loss of all solutions vs averaged baselines',
                                                 xlabel='Epochs',
                                                 ylabel='Training Loss')
        ga_metric_plotter.plot_averaged_metrics_of_each_generation_and_baseline(averaged_val_loss_of_each_generation,
                                                 baselines_averaged_val_loss_over_epochs,
                                                 title=f'{test} averaged validation loss of all solutions vs averaged baselines',
                                                 xlabel='Epochs',
                                                 ylabel='Validation Loss')

        ga_metric_plotter.plot_averaged_metrics_of_each_generation_and_baseline(averaged_train_accuracy_of_each_generation,
                                                 baselines_averaged_train_acc_over_epochs,
                                                 title=f'{test} averaged train accuracy of all solutions vs averaged baselines',
                                                 xlabel='Epochs',
                                                 ylabel='Training Accuracy')

        ga_metric_plotter.plot_averaged_metrics_of_each_generation_and_baseline(averaged_val_accuracy_of_each_generation,
                                                 baselines_averaged_val_acc_over_epochs,
                                                 title=f'{test} averaged validation accuracy of all solutions vs averaged baselines',
                                                 xlabel='Epochs',
                                                 ylabel='Validation Accuracy')

        ga_metric_plotter.plot_averaged_metrics_of_each_generation_and_baseline(averaged_val_roc_of_each_generation,
                                                 baselines_averaged_val_roc_over_epochs,
                                                 title=f'{test} averaged validation ROC of all solutions vs averaged baselines',
                                                 xlabel='Epochs',
                                                 ylabel='Validation ROC')

        ga_metric_plotter.plot_averaged_metrics_of_each_generation_and_baseline(averaged_fitness_of_each_generation,
                                                baselines_averaged_fitness_over_epochs,
                                                title=f'{test} averaged fitness of all solutions vs averaged baselines',
                                                xlabel='Epochs',
                                                ylabel='Fitness')

        ################### Line plot average of top 5 solutions vs baselines on all epochs ######################
        ################### Line plot average of top 5 solutions vs baselines on all epochs ######################
        ################### Line plot average of top 5 solutions vs baselines on all epochs ######################

        ga_metric_plotter.plot_averaged_metrics_of_each_generation_and_baseline(
            averaged_train_loss_of_top_5_each_generation,
            baselines_averaged_train_loss_over_epochs,
            title=f'{test} averaged train loss of top 5 solutions vs averaged baselines',
            xlabel='Epochs',
            ylabel='Training Loss')
        ga_metric_plotter.plot_averaged_metrics_of_each_generation_and_baseline(
            averaged_val_loss_of_top_5_each_generation,
            baselines_averaged_val_loss_over_epochs,
            title=f'{test} averaged validation loss of top 5 solutions vs averaged baselines',
            xlabel='Epochs',
            ylabel='Validation Loss')

        ga_metric_plotter.plot_averaged_metrics_of_each_generation_and_baseline(
            averaged_train_accuracy_of_top_5_each_generation,
            baselines_averaged_train_acc_over_epochs,
            title=f'{test} averaged train accuracy of top 5 solutions vs averaged baselines',
            xlabel='Epochs',
            ylabel='Training Accuracy')

        ga_metric_plotter.plot_averaged_metrics_of_each_generation_and_baseline(
            averaged_val_accuracy_of_top_5_each_generation,
            baselines_averaged_val_acc_over_epochs,
            title=f'{test} averaged validation accuracy of top 5 solutions vs averaged baselines',
            xlabel='Epochs',
            ylabel='Validation Accuracy')

        ga_metric_plotter.plot_averaged_metrics_of_each_generation_and_baseline(
            averaged_val_roc_of_top_5_each_generation,
            baselines_averaged_val_roc_over_epochs,
            title=f'{test} averaged validation ROC of top 5 solutions vs averaged baselines',
            xlabel='Epochs',
            ylabel='Validation ROC')

        ga_metric_plotter.plot_averaged_metrics_of_each_generation_and_baseline(
            averaged_fitness_of_top_5_each_generation,
            baselines_averaged_fitness_over_epochs,
            title=f'{test} averaged fitness of top 5 solutions vs averaged baselines',
            xlabel='Epochs',
            ylabel='Fitness')



        ################### Line plot average of top 1 solution vs baselines on all epochs ######################
        ################### Line plot average of top 1 solution vs baselines on all epochs ######################
        ################### Line plot average of top 1 solution vs baselines on all epochs ######################

        ga_metric_plotter.plot_averaged_metrics_of_each_generation_and_baseline(
            train_loss_of_top_1_each_generation,
            baselines_averaged_train_loss_over_epochs,
            title=f'{test} averaged train loss of top 1 solution vs averaged baselines',
            xlabel='Epochs',
            ylabel='Training Loss')
        ga_metric_plotter.plot_averaged_metrics_of_each_generation_and_baseline(
            val_loss_of_top_1_each_generation,
            baselines_averaged_val_loss_over_epochs,
            title=f'{test} averaged validation loss of top 1 solution vs averaged baselines',
            xlabel='Epochs',
            ylabel='Validation Loss')

        ga_metric_plotter.plot_averaged_metrics_of_each_generation_and_baseline(
            train_acc_of_top_1_each_generation,
            baselines_averaged_train_acc_over_epochs,
            title=f'{test} averaged train accuracy of top 1 solution vs averaged baselines',
            xlabel='Epochs',
            ylabel='Training Accuracy')

        ga_metric_plotter.plot_averaged_metrics_of_each_generation_and_baseline(
            val_acc_of_top_1_each_generation,
            baselines_averaged_val_acc_over_epochs,
            title=f'{test} averaged validation accuracy of top 1 solution vs averaged baselines',
            xlabel='Epochs',
            ylabel='Validation Accuracy')

        ga_metric_plotter.plot_averaged_metrics_of_each_generation_and_baseline(
            val_roc_of_top_1_each_generation,
            baselines_averaged_val_roc_over_epochs,
            title=f'{test} averaged validation ROC of top 1 solution vs averaged baselines',
            xlabel='Epochs',
            ylabel='Validation ROC')

        ga_metric_plotter.plot_averaged_metrics_of_each_generation_and_baseline(
            fitness_of_top_1_each_generation,
            baselines_averaged_fitness_over_epochs,
            title=f'{test} averaged fitness of top 1 solution vs averaged baselines',
            xlabel='Epochs',
            ylabel='Fitness')









