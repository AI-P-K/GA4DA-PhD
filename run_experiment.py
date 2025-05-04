import sys
import uuid
from os import pread
from traceback import print_last

import cv2
import matplotlib.pyplot as plt
from utils.DataProcessor import *
from utils.ModelInitializer import *
from utils.ImageDataLoader import *
from utils.ModelTrainer import *
from tqdm import tqdm
from utils.GA import GeneticAlgorithm
import csv
from utils.Metrics import MetricsCalculator
import random
from utils.CategorizedAugmentations import get_categorized_augmentations
from utils.VisualizeData import visualize_training_samples

def run_experiment(genes_type, mutation, baseline_augmentation, fixed_mutation_rate, ranked_crossover):
    # Genes type A - first tests with all augmentations automatically processed conditioning only parameters such a p,
    # always_apply, input_size and also with an exclusion list.
    # Genes type B -
    dataset_folder = 'Datasets/CIFAR10-Reduced'
    dataset_name = dataset_folder.split('/')[-1]
    # Describe the purpose of the val in few words

    # Define dataset-related parameters
    validation_set_size = 0.3

    # Define image preprocessing parameters
    input_size = 150
    # Define model parameters
    architecture = 'resnet18-weights'
    feature_extract = False
    batch_size = 32
    workers = 8
    num_epochs = 15
    loss_function = 'CrossEntropyLoss'
    optimisation_algorithm = 'SGD'
    kFolds = 1
    linear_mutation_rate_function = False
    number_baseline_trainings = 20
    # population size a number between 2 and alpha to the power of l.
    population_size = 30
    parents = int(population_size/2)
    fixed_rate_value = 1.2
    variable_rate_value = -5
    elitism = True
    target_fitness_score = 9999999999999
    number_offsprings = int(parents)
    generations = 20
    ga_operators_order = 'crossover-mutation'

    test_description = (f"Dataset:{dataset_name}, "
                        f"Genes Type:{genes_type}, "
                        f"Mutation:{mutation}, "
                        f"Baseline Oversampled:{baseline_augmentation}, "
                        f"Ranked Crossover:{ranked_crossover}, "
                        f"Feature Extraction:{feature_extract}, "
                        f"Optimizer:{optimisation_algorithm}, "
                        f"Generations:{generations}, "
                        f"Population:{population_size}, "
                        f"Input Size:{input_size}, "
                        f"Validation Set Size:{validation_set_size}, "
                        f"K-folds:{kFolds}, "
                        f"Fixed Mutation Rate:{fixed_mutation_rate}, "
                        f"Epochs:{num_epochs}, "
                        f"Architecture:{architecture}, "
                        f"Linear Mutation Function:{linear_mutation_rate_function}",
                        f"Fixed rate value:{fixed_rate_value}",
                        f"Variable rate value:{variable_rate_value}",)

    testID = "T_" + str(uuid.uuid4().hex[:6])
    test_root_path = os.path.join('results', testID)
    os.makedirs(test_root_path, exist_ok=True)

    # Save the val description
    file_name = os.path.join(test_root_path, f'{testID}-description.txt')
    with open(file_name, "w") as file:
        file.write(str(test_description))
    print(f"Text description saved to {file_name}")

    # Split the dataset into training and validation sets
    dataset_splitter = DatasetSplitter(dataset_folder)

    (train_images_filepaths_splits,
     val_images_filepaths_splits,
     classes_names, num_classes) = dataset_splitter.get_train_val_filepaths(validation_set_size=validation_set_size,
                                                                            n_splits=kFolds)

    set_of_all_categorized_augmentations = get_categorized_augmentations()

    all_processed_augmentations = set_of_all_categorized_augmentations['class1_color_adjustments'] + \
                                  set_of_all_categorized_augmentations['class2_geometric_transformations'] + \
                                  set_of_all_categorized_augmentations['class3_noise_distortion'] + \
                                  set_of_all_categorized_augmentations['class4_affine_transformations'] + \
                                  set_of_all_categorized_augmentations['class5_cropping_transformations'] + \
                                  set_of_all_categorized_augmentations['class6_resizing_transformations'] + \
                                  set_of_all_categorized_augmentations['class7_rotation_transformations']


    # Select a random augmentation for the baseline oversampling if needed
    random_global_baseline_processed_augmentation = random.choice(all_processed_augmentations)
    if genes_type == 'A':
        all_processed_augmentations = all_processed_augmentations
    elif genes_type == 'B':
        all_processed_augmentations = set_of_all_categorized_augmentations

    # sys.exit()
    # Initialize the required Albumentations function to process the dataset (Resize, Normalize, ToTensor)
    transform_provider = AlbumentationsImageTransformations(input_size=input_size)
    required_image_transformations = transform_provider.get_transforms()


    # Initialize Albumentation function to globally augment the baseline dataset if needed
    global_baseline_transformations = transform_provider.\
        get_baseline_global_augmentation(random_global_baseline_processed_augmentation)



    # Initialize the model based on the provided architecture and settings

    model_initializer = ModelInitializer(architecture, num_classes, feature_extract)

    def train_one_baseline():
        baselineID = "R_" + uuid.uuid4().hex[:6]
        fold_path = os.path.join('results', testID, baselineID)
        os.mkdir(fold_path)

        # these hold the lists of metrics for each fold at every epoch. for overall average of the baseline do average over
        # all folds for each epoch individually resulting in single list of metrics.
        start_training_time = time.time()
        kFolds_train_losses = []
        kFolds_train_accuracies = []
        kFolds_val_losses = []
        kFolds_val_accuracies = []
        kFolds_val_rocs = []
        kFolds_fitness_scores = []


        for index, (train_fold, val_fold) in tqdm(enumerate(zip(train_images_filepaths_splits, val_images_filepaths_splits)), desc="K-Folds Training", leave=False):
            filename = 'Fold_' + str(index) + '.csv'

            model = model_initializer.set_model()
            params_to_update = model.parameters()

            criterion = model_initializer.set_criterion(loss_function=loss_function)
            optimizer = model_initializer.initialize_optimizer(optimizer_name=optimisation_algorithm,
                                                               params_to_update=params_to_update)

            if baseline_augmentation:
                def save_dataset_for_visualisation(dataset, type):
                    from torchvision import transforms
                    import torch
                    save_dir = os.path.join(test_root_path, baselineID, type)
                    for idx, (image, label) in enumerate(dataset):
                        if isinstance(image, torch.Tensor):
                            mean = np.array([0.485, 0.456, 0.406])
                            std = np.array([0.229, 0.224, 0.225])
                            unnormalized_tensor = image * std[:, None, None] + mean[:, None, None]
                            unnormalized_image = torch.clamp(unnormalized_tensor, 0, 1)
                            image = transforms.ToPILImage()(unnormalized_image)
                            os.makedirs(save_dir, exist_ok=True)
                            filename = os.path.join(save_dir, f'image_{idx}.png')
                        image.save(filename)

                globlal_augmented_train_dataset = ClassSpecificTransformationsDataset(images_filepaths=train_fold,
                                                                         classes_names=classes_names,
                                                                         ga_solution_transforms=None,
                                                                         input_size=input_size,
                                                                         required_transforms=global_baseline_transformations,
                                                                         save_path=fold_path)


                baseline_train_dataset = ClassSpecificTransformationsDataset(images_filepaths=train_fold,
                                                                             classes_names=classes_names,
                                                                             ga_solution_transforms=None,
                                                                             input_size=input_size,
                                                                             required_transforms=required_image_transformations,
                                                                             save_path=fold_path)

                # save_dataset_for_visualisation(globlal_augmented_train_dataset, 'global_augmented')
                # save_dataset_for_visualisation(baseline_train_dataset, 'baseline')

                baseline_train_dataset = ConcatDataset([globlal_augmented_train_dataset, baseline_train_dataset])

            else:
                baseline_train_dataset = ClassSpecificTransformationsDataset(images_filepaths=train_fold,
                                                                             classes_names=classes_names,
                                                                             ga_solution_transforms=None,
                                                                             input_size=input_size,
                                                                             required_transforms=required_image_transformations,
                                                                             save_path=fold_path)
                # save_dataset_for_visualisation(baseline_train_dataset, 'baseline')

            baseline_validation_dataset = ClassSpecificTransformationsDataset(images_filepaths=val_fold,
                                                                              classes_names=classes_names,
                                                                              ga_solution_transforms=None,
                                                                              input_size=input_size,
                                                                              required_transforms=required_image_transformations,
                                                                              save_path=fold_path)

            # save_dataset_for_visualisation(baseline_validation_dataset, 'validation')

            dataloaders_dict = {
                'train': DataLoader(baseline_train_dataset, batch_size=batch_size, shuffle=True, num_workers=workers),
                'val': DataLoader(baseline_validation_dataset, batch_size=batch_size, shuffle=False, num_workers=workers),
            }

            if len(baseline_validation_dataset) != dataloaders_dict['val'].sampler.__len__():
                sys.exit('Number of validation samples does not match between dataset and dataloader.')

            model_trainer = ModelTrainer(model, dataloaders_dict, criterion, optimizer, num_epochs=num_epochs)

            (fold_train_losses,
             fold_val_losses,
             fold_train_accs,
             fold_val_accs,
             fold_val_rocs,
             fold_fitness) = model_trainer.train_model()

            calculate_metrics = MetricsCalculator(fold_path, filename)
            metrics_lists = [fold_train_losses, fold_val_losses, fold_train_accs, fold_val_accs, fold_val_rocs, fold_fitness]
            calculate_metrics.save_fold_metrics(metrics_lists)

            kFolds_train_losses.append(fold_train_losses)
            kFolds_train_accuracies.append(fold_train_accs)
            kFolds_val_losses.append(fold_val_losses)
            kFolds_val_accuracies.append(fold_val_accs)
            kFolds_val_rocs.append(fold_val_rocs)
            kFolds_fitness_scores.append(fold_fitness)

        best_validation_loss, \
        avg_metrics_lists = calculate_metrics.calculate_epochs_metrics_averages(kFolds_train_losses,
                                                                                kFolds_train_accuracies,
                                                                                kFolds_val_losses,
                                                                                kFolds_val_accuracies,
                                                                                kFolds_val_rocs,
                                                                                kFolds_fitness_scores)
        finish_trainig_time = time.time() - start_training_time
        calculate_metrics = MetricsCalculator(fold_path, f'{baselineID}-metrics-averages.csv')
        calculate_metrics.save_fold_metrics(avg_metrics_lists)
        time_file_path = os.path.join(fold_path, f'{baselineID}-training-time.csv')

        with open(time_file_path, 'w', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            if baseline_augmentation:
                csvwriter.writerow([finish_trainig_time, str(random_global_baseline_processed_augmentation)])
            else:
                csvwriter.writerow([finish_trainig_time])


    for i in tqdm(range(number_baseline_trainings), desc='Baseline training'):
        train_one_baseline()

    # Save the time and the solution  components
    ga = GeneticAlgorithm(all_transformations=all_processed_augmentations,
                          generations=generations,
                          population_size=population_size,
                          number_of_parents_for_reproduction=parents,
                          elitism=elitism,
                          target_fitness_score=target_fitness_score,
                          mutation_operator=mutation,
                          num_classes=num_classes,
                          classes_names=classes_names,
                          number_offsprings=number_offsprings,
                          train_images_filepaths_splits=train_images_filepaths_splits,
                          val_images_filepaths_splits=val_images_filepaths_splits,
                          required_image_transformations=required_image_transformations,
                          input_size=input_size,
                          batch_size=batch_size,
                          loss_function=loss_function,
                          optimisation_algorithm=optimisation_algorithm,
                          num_epochs=num_epochs,
                          results_path=f'results/{testID}',
                          genes_type=genes_type,
                          num_workers=workers,
                          fixed_mutation_rate=fixed_mutation_rate,
                          ranked_crossover=ranked_crossover,
                          operator_order=ga_operators_order,
                          model_initializer=model_initializer,
                          linear_mutation_rate_function=linear_mutation_rate_function,
                          fixed_rate_value=fixed_rate_value,
                          variable_rate_value=variable_rate_value)

    since_ga_start_time = time.time()
    time_file_name = os.path.join(test_root_path, f'{testID}-time.csv')
    ga.GA()
    ga_evolution_time = time.time() - since_ga_start_time

    with open(time_file_name, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([ga_evolution_time])

