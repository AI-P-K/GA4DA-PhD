import csv
import os.path
import sys


class MetricsCalculator:
    def __init__(self, folder_path=None, filename=None):
        self.folder_path = folder_path
        self.filename = filename

    def calculate_epochs_metrics_averages(self,
                                          kFolds_train_losses,
                                          kFolds_train_accuracies,
                                          kFolds_val_losses,
                                          kFolds_val_accuracies,
                                          kFolds_val_rocs,
                                          kFolds_fitness_scores):

        avg_train_losses = [sum(col) / len(col) for col in zip(*kFolds_train_losses)]
        avg_train_accuracies = [sum(col) / len(col) for col in zip(*kFolds_train_accuracies)]
        avg_val_losses = [sum(col) / len(col) for col in zip(*kFolds_val_losses)]
        avg_val_accuracies = [sum(col) / len(col) for col in zip(*kFolds_val_accuracies)]
        avg_val_rocs = [sum(col) / len(col) for col in zip(*kFolds_val_rocs)]
        avg_fitness = [sum(col) / len(col) for col in zip(*kFolds_fitness_scores)]

        # We consider the validation loss as our guiding metric, identify the minimum validation loss,
        # than its index(epoch) and then we register all our metrics from that epoch.
        best_validation_loss = min(avg_val_losses)
        best_validation_loss_index = avg_val_losses.index(best_validation_loss)

        best_fitness = max(avg_fitness)
        best_fitness_index = avg_fitness.index(best_fitness)

        if 1/best_validation_loss != best_fitness:
            sys.exit("Inverse best validation loss not equal to best fitness")


        avg_metrics_lists = [avg_train_losses, avg_val_losses, avg_train_accuracies, avg_val_accuracies, avg_val_rocs, avg_fitness]

        return best_validation_loss, avg_metrics_lists

    def save_fold_metrics(self, metrics_lists):
        full_path = os.path.join(self.folder_path, self.filename)
        with open(full_path, mode="w", newline="") as file:
            writer = csv.writer(file)

            # Define custom labels for each row
            row_labels = ["Train Losses", "Val Losses", "Train Accuracy", "Val Accuracy", "Val ROC", "Fitness"]

            # Write the header row
            writer.writerow(["Metric"] + [f"Epoch {i + 1}" for i in range(len(metrics_lists[0]))])

            # Write each row of metrics with custom labels
            for label, metrics in zip(row_labels, metrics_lists):
                writer.writerow([label] + metrics)





