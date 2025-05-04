import math
import random
import sys
import os
import csv
from cProfile import label

import numpy as np
import cv2
from numpy.linalg.linalg import solve
from six import print_
from tqdm import tqdm
import uuid
from torch.utils.data import ConcatDataset
from utils.ImageDataLoader import CustomDataLoader
from torch.utils.data import DataLoader
from utils.ModelTrainer import ModelTrainer
from utils.Metrics import MetricsCalculator
import matplotlib.pyplot as plt
from utils.DataProcessor import ClassSpecificTransformationsDataset
import time
from utils.VisualizeData import visualize_training_samples


class GeneticAlgorithm:
    def __init__(self, all_transformations,
                 generations,
                 population_size,
                 number_of_parents_for_reproduction,
                 elitism,
                 target_fitness_score,
                 mutation_operator,
                 num_classes,
                 classes_names,
                 number_offsprings,
                 train_images_filepaths_splits,
                 val_images_filepaths_splits,
                 required_image_transformations,
                 input_size,
                 batch_size,
                 loss_function,
                 optimisation_algorithm,
                 num_epochs,
                 results_path,
                 genes_type,
                 num_workers,
                 fixed_mutation_rate,
                 ranked_crossover,
                 operator_order,
                 model_initializer,
                 linear_mutation_rate_function,
                 fixed_rate_value,
                 variable_rate_value
                 ):
        self.generations = generations
        self.all_callable_transformations = all_transformations
        self.population_size = population_size
        self.number_of_parents_for_reproduction = number_of_parents_for_reproduction
        self.elitism = elitism
        self.target_fitness_score = target_fitness_score
        self.mutation_operator = mutation_operator
        self.num_classes = num_classes
        self.classes_names = classes_names
        self.number_offsprings = number_offsprings
        self.train_images_filepaths_splits = train_images_filepaths_splits
        self.val_images_filepaths_splits = val_images_filepaths_splits
        self.required_image_transformations = required_image_transformations
        self.model_initializer = model_initializer
        self.input_size = input_size
        self.batch_size = batch_size
        self.loss_function = loss_function
        self.optimisation_algorithm = optimisation_algorithm
        self.num_epochs = num_epochs
        self.results_path = results_path
        self.genes_type = genes_type
        self.num_workers = num_workers
        self.fixed_mutation_rate = fixed_mutation_rate
        self.ranked_crossover = ranked_crossover
        self.operator_order = operator_order
        self.linear_mutation_rate_function = linear_mutation_rate_function
        self.fixed_rate_value = fixed_rate_value,
        self.variable_rate_value = variable_rate_value

    def _ranked_pairing_crossover(self, parents, offsprings_target):
        """
        Takes in parents list and creates offsprings until the target length is reached
        :param parents: List of parent tuples
        :param offsprings_target: Desired number of offsprings
        :return: List of offspring tuples
        """
        unique_offsprings = set()
        odd_parents = parents[::2]
        even_parents = parents[1::2]
        iteration = 0
        max_iterations = 1000
        while len(unique_offsprings) < offsprings_target:
            for odd_parent, even_parent in zip(odd_parents, even_parents):
            # Select crossover point randomly
                crossover_point = random.choice(range(len(odd_parent)))
                # Create offsprings by swapping genetic material
                odd_offspring = list(odd_parent[:crossover_point] + even_parent[crossover_point:])
                even_offspring = list(even_parent[:crossover_point] + odd_parent[crossover_point:])
                # Add offsprings to the set if they are not duplicates
                if tuple(odd_offspring) not in unique_offsprings and tuple(odd_offspring) not in set(parents):
                    unique_offsprings.add(tuple(odd_offspring))
                if tuple(even_offspring) not in unique_offsprings and tuple(even_offspring) not in set(parents):
                    unique_offsprings.add(tuple(even_offspring))
                # Break the loop if we have enough offsprings
                if len(unique_offsprings) >= offsprings_target:
                    break
            iteration += 1
            if iteration >= max_iterations:
                random.shuffle(odd_parents)
                random.shuffle(even_parents)
            unique_offsprings_list = list(unique_offsprings)
            unique_offsprings_list.extend(parents)
        return unique_offsprings_list

    def _pool_pairing_crossover(self, parents, num_classes, offsprings_target):
        # Create a new list to store all parents transformations
        parents_transformations_pool = []
        for parent in parents:
            # Extend the new list with the individual image transformations
            parents_transformations_pool.extend(parent)
        # Create newGenSolutions by selecting elements randomly from best_solutions
        unique_offsprings = set()
        while len(unique_offsprings) < offsprings_target:
            for _ in range(offsprings_target):
                solution = tuple([random.choice(parents_transformations_pool) for _ in range(num_classes)])
                if solution not in unique_offsprings and solution not in parents:
                    unique_offsprings.add(solution)
                if len(unique_offsprings) >= offsprings_target:
                    break
        offsprings = list(unique_offsprings)
        offsprings.extend(parents)
        return offsprings

    def calculate_offspring_mutation_rate(self, fitness_score, min_fitness, max_fitness):
        """
            Calculate the offspring mutation rate based on normalized fitness score.

            Parameters:
            - fitness_score (float): The fitness score of the offspring.
            - min_fitness (float): The minimum possible fitness score.
            - max_fitness (float): The maximum possible fitness score.
            - min_mutation_rate (float): The minimum mutation rate.
            - max_mutation_rate (float): The maximum mutation rate.

            Returns:
            - float: The calculated offspring mutation rate.
        """
        if max_fitness < min_fitness:
            sys.exit('Max fitness score must be higher than min fitness score.')
        # Normalize fitness score to [0, 1] range
        relative_fitness = (fitness_score - min_fitness) / (max_fitness - min_fitness)
        # convert this number into a mutation rate (e.g. rate = 1-relative_fitness) (1-CDF = cummulative distribution function)

        # # Linear interpolation for mutation rate based on normalized fitness score
        # Linear fitness function
        if self.linear_mutation_rate_function:
            offspring_mutation_rate = abs(1 - relative_fitness)
        else:
            offspring_mutation_rate = math.exp(self.variable_rate_value*relative_fitness)
        return offspring_mutation_rate

    def _mutate_A(self, parents):
        """
        Args:
            parents: List with top n parents chosen for reproduction; [(f1,(g1,g2,g3)), ..., (fn,(gx,gy,gz))].
            Parents come with their respective fitness this allows us to calculate the mutation rate
            based on their fitness score calculate_offspring_mutation_rate. However, if the fixed mutation rate
            parameter is True than the formula for mutation rate calculation is 1/# of genes in the parent.
        Returns: List of mutated individuals(offsprings)
        """
        mutated_solutions = []
        unique_solutions = set()
        num_parents = len(parents)
        parent_index = 0
        while len(mutated_solutions) < num_parents:
            parent = parents[parent_index]
            if self.fixed_mutation_rate == False:
                offspring_fitness_score = parent[0]
                max_fitness = parents[0][0]
                min_fitness = parents[-1][0]
                mutation_rate = self.calculate_offspring_mutation_rate(fitness_score=offspring_fitness_score,
                                                                       min_fitness=min_fitness,
                                                                       max_fitness=max_fitness)
            else:
                mutation_rate = 1 / (len(parent[-1])*self.fixed_rate_value)
            mutated_offspring = []
            offspring_for_mutation = list(parent[-1])
            for gene in offspring_for_mutation:
                dice_roll = random.random()
                if dice_roll < mutation_rate:
                    gene_pool_idx = self.all_callable_transformations.index(gene)
                    self.all_callable_transformations.remove(gene)
                    mutated_gene = random.choice(self.all_callable_transformations)
                    self.all_callable_transformations.insert(gene_pool_idx, gene)
                else:
                    mutated_gene = gene
                mutated_offspring.append(mutated_gene)
            mutated_offspring_tuple = tuple(mutated_offspring)
            if mutated_offspring_tuple not in unique_solutions:
                mutated_solutions.append(mutated_offspring_tuple)
                unique_solutions.add(mutated_offspring_tuple)
            parent_index = (parent_index + 1) % num_parents
        return mutated_solutions

    def _mutate_B(self, parents):
        mutated_solutions = []
        unique_solutions = set()
        num_parents = len(parents)
        parent_index = 0
        while len(mutated_solutions) < num_parents:
            parent = parents[parent_index]
            if self.fixed_mutation_rate == False:
                # What is high fitness and what is low fitness?
                offspring_fitness_score = parent[0]
                max_fitness = parents[0][0]
                min_fitness = parents[-1][0]
                offspring_mutation_rate = self.calculate_offspring_mutation_rate(fitness_score=offspring_fitness_score,
                                                                                 min_fitness=min_fitness,
                                                                                 max_fitness=max_fitness)
            else:
                offspring_mutation_rate = 1 / (len(parent[-1])*2)
            mutated_offspring = []
            parent_genes = parent[1]
            for gene in parent_genes:
                gene_dice_roll = random.random()
                if gene_dice_roll < offspring_mutation_rate:
                    gene_category_dice_roll = random.random()
                    if gene_category_dice_roll < offspring_mutation_rate:
                        found_category = None
                        for category, possible_values in self.all_callable_transformations.items():
                            if gene in possible_values:
                                found_category = category
                        # Randomly choose a dictionary key excluding the found category
                        other_categories = [key for key in self.all_callable_transformations.keys() if
                                            key != found_category]
                        randomly_chosen_category = random.choice(other_categories)
                        selected_transformation = random.choice(
                            self.all_callable_transformations[randomly_chosen_category])
                        mutated_offspring.append(selected_transformation)
                    else:
                        found_category = None
                        for category, possible_values in self.all_callable_transformations.items():
                            if gene in possible_values:
                                found_category = category
                                # Exclude the current gene from the possible transformations
                        possible_transformations = [t for t in self.all_callable_transformations[found_category] if
                                                    t != gene]
                        selected_transformation = random.choice(possible_transformations)
                        mutated_offspring.append(selected_transformation)
                else:
                    mutated_offspring.append(gene)
            mutated_offspring_tuple = tuple(mutated_offspring)
            # Add to list if not already encountered
            if mutated_offspring_tuple not in unique_solutions:
                mutated_solutions.append(mutated_offspring_tuple)
                unique_solutions.add(mutated_offspring_tuple)
            # Move to the next parent cyclically
            parent_index = (parent_index + 1) % num_parents
        return mutated_solutions

    def _calculate_fitness(self, solution,
                           train_images_filepaths_splits, val_images_filepaths_splits,
                           solutionID, solution_path, generation):

        kFolds_train_losses = []
        kFolds_train_accuracies = []
        kFolds_val_losses = []
        kFolds_val_accuracies = []
        kFolds_val_rocs = []
        kFolds_val_fitness = []

        calculate_metrics = MetricsCalculator()

        for fold_index, (train_data, val_data) in tqdm(enumerate(zip(
                train_images_filepaths_splits,
                val_images_filepaths_splits)),
                desc='Solution_K_Fold', leave=False):
            solution_fold_filename = 'Fold_' + str(fold_index) + '.csv'
            model = self.model_initializer.set_model()
            params_to_update = model.parameters()
            criterion = self.model_initializer.set_criterion(loss_function=self.loss_function)
            optimizer = self.model_initializer.initialize_optimizer(optimizer_name=self.optimisation_algorithm,
                                                                    params_to_update=params_to_update)

            def visualize_ga_datasets(dataset, type):
                from torchvision import transforms
                import torch
                for idx, (image, label) in enumerate(dataset):
                    if isinstance(image, torch.Tensor):
                        mean = np.array([0.485, 0.456, 0.406])
                        std = np.array([0.229, 0.224, 0.225])
                        unnormalized_tensor = image * std[:, None, None] + mean[:, None, None]
                        unnormalized_image = torch.clamp(unnormalized_tensor, 0, 1)
                        image = transforms.ToPILImage()(unnormalized_image)
                        if type == 'solution':
                            save_dir = os.path.join(solution_path, type, self.classes_names[label])
                        else:
                            save_dir = os.path.join(solution_path, type)
                        os.makedirs(save_dir, exist_ok=True)
                        filename = os.path.join(save_dir, f'image_{idx}.png')
                    image.save(filename)


            solution_train_dataset = ClassSpecificTransformationsDataset(images_filepaths=train_data,
                                                                         classes_names=self.classes_names,
                                                                         ga_solution_transforms=solution,
                                                                         input_size=self.input_size,
                                                                         required_transforms=None,
                                                                         save_path=solution_path)

            # visualize_ga_datasets(solution_train_dataset, 'solution')


            baseline_train_dataset = ClassSpecificTransformationsDataset(images_filepaths=train_data,
                                                                         classes_names=self.classes_names,
                                                                         ga_solution_transforms=None,
                                                                         input_size=self.input_size,
                                                                         required_transforms=self.required_image_transformations,
                                                                         save_path=solution_path)
            # visualize_ga_datasets(baseline_train_dataset, 'solution_baseline_train')

            validation_dataset = ClassSpecificTransformationsDataset(images_filepaths=val_data,
                                                                     classes_names=self.classes_names,
                                                                     ga_solution_transforms=None,
                                                                     input_size=self.input_size,
                                                                     required_transforms=self.required_image_transformations,
                                                                     save_path=solution_path)
            # visualize_ga_datasets(validation_dataset, 'solution_baseline_validation')

            combined_augmented_train_dataset = ConcatDataset([solution_train_dataset, baseline_train_dataset])

            dataloaders_dict = {
                'train': DataLoader(combined_augmented_train_dataset,
                                    batch_size=self.batch_size,
                                    shuffle=True,
                                    num_workers=self.num_workers,
                                    pin_memory=True),

                'val': DataLoader(validation_dataset,
                                  batch_size=self.batch_size,
                                  shuffle=False,
                                  num_workers=self.num_workers,
                                  pin_memory=True),
            }

            model_trainer = ModelTrainer(model,
                                 dataloaders_dict,
                                 criterion,
                                 optimizer,
                                 self.num_epochs)

            solution_fold_train_losses, \
            solution_fold_val_losses, \
            solution_fold_train_accs, \
            solution_fold_val_accs, \
            solution_fold_val_roc,\
            solution_fold_fitness = model_trainer.train_model()


            metrics_lists = [solution_fold_train_losses,
                             solution_fold_val_losses,
                             solution_fold_train_accs,
                             solution_fold_val_accs,
                             solution_fold_val_roc,
                             solution_fold_fitness]

            calculate_metrics = MetricsCalculator(solution_path, solution_fold_filename)
            calculate_metrics.save_fold_metrics(metrics_lists)

            kFolds_train_losses.append(solution_fold_train_losses)
            kFolds_train_accuracies.append(solution_fold_train_accs)
            kFolds_val_losses.append(solution_fold_val_losses)
            kFolds_val_accuracies.append(solution_fold_val_accs)
            kFolds_val_rocs.append(solution_fold_val_roc)
            kFolds_val_fitness.append(solution_fold_fitness)


        best_validation_loss, \
        avg_metrics_lists = calculate_metrics.calculate_epochs_metrics_averages(kFolds_train_losses,
                                                                                kFolds_train_accuracies,
                                                                                kFolds_val_losses,
                                                                                kFolds_val_accuracies,
                                                                                kFolds_val_rocs,
                                                                                kFolds_val_fitness)
        calculate_metrics = MetricsCalculator(solution_path, f'{solutionID}-metrics-averages.csv')
        calculate_metrics.save_fold_metrics(avg_metrics_lists)
        fitness_score = 1 / best_validation_loss
        return fitness_score

    def generate_unique_A_solutions(self, population_size, num_items_per_solution, genes_pool):
        """
        Generates unique solutions based on population size and number of genes per individual from a pool of genes.
        The function uses while loop to generate unique solutions. We make sure that each solution is unique by using
        set instead of list and hashing the solution tuple.
        A solution is formed by randomly choosing a transformation(gene) from a pool of transformations(genes).
        Args:
            population_size: Population size set in the run_experiment.py
            num_items_per_solution: Number of genes per individual. Genes in our case are image transformations.
            genes_pool: All available image transformations mixed in the same list
        Returns: List of unique solutions
        """
        unique_solutions = set()  # Set to store unique tuples
        while len(unique_solutions) < population_size:
            new_solution = tuple(random.sample(genes_pool, num_items_per_solution))  # Generate a new solution
            if new_solution not in unique_solutions:  # Check if the tuple is unique
                unique_solutions.add(new_solution)  # Add the tuple to the set of unique tuples
        return list(unique_solutions)

    def generate_unique_B_solutions(self, population_size, num_items_per_solution, categorized_genes_dict):
        """
        Function logic is the same as generate_unique_A_solutions. In this function however the genes are categorised.
        They are organized in a dictionary, which maps each gene to its corresponding category.
        We use while loop to create as many unique solutions as possible. This time we random choose category,
        and a transformation within that category. We do this for every gene of the individual.
        Args:
            population_size: Population size set in the run_experiment.py
            num_items_per_solution: Number of genes per individual. Genes in our case are image transformations.
            categorized_genes_dict: dict = {'cat1':[g1,g2,g3], ... ,'cat7':[g4,g5,g6]}
        Returns: List of unique solutions

        """
        unique_solutions = set()  # Set to store unique tuples
        while len(unique_solutions) < population_size:
            new_solution = []
            for _ in range(num_items_per_solution):
                # Randomly select a category
                selected_category = random.choice(list(categorized_genes_dict.keys()))
                # Randomly select an image transformation from the selected class
                selected_transformation = random.choice(categorized_genes_dict[selected_category])
                new_solution.append(selected_transformation)
            new_solution = tuple(new_solution)
            if new_solution not in unique_solutions:  # Check if the tuple is unique
                unique_solutions.add(new_solution)  # Add the tuple to the set of unique tuples
        return list(unique_solutions)

    def GA(self):
        ## # 1. Create first population of solutions(individuals)
        # Initialize an empty list where we store the formed solutions as tuples
        solutions = None
        if self.genes_type == 'A':
            solutions = self.generate_unique_A_solutions(self.population_size,
                                                         self.num_classes,
                                                         self.all_callable_transformations)

        elif self.genes_type == 'B':
            solutions = self.generate_unique_B_solutions(self.population_size,
                                                         self.num_classes,
                                                         self.all_callable_transformations)

        ## # 2. Initialize the GA
        # initialize generation_no to 0 a variable that holds account of the generation number and we pass it into the fitness() -> run_ga_solutions() -> ga_training_records.csv
        generation_number = 0
        # initialize average_fitnessScores_per_generation list which holds the average fitness score per generation (sum(solutionsScores) / len(solutionsScores))
        average_fitnessScores_per_generation = []

        # initialize top_one_elitism as None which after the first generation of solutions
        elite_solution = None
        for i in tqdm(range(self.generations), desc='Generations'):
            # first the generation_no increments by one at each generation
            generation_number += 1
            # rankedSolutions a list that holds all tested solutions and their corresponding fitness score in tuples (fitness score, (solution))
            # e.g. (39.38024949675888, (RandomSizedCrop(always_apply=True, p=1, min_max_height=(224, 224), height=224, width=224, w2h_ratio=1.0, interpolation=1), RandomShadow(always_apply=True, p=1, shadow_roi=(0, 0.5, 1, 1), num_shadows_lower=1, num_shadows_upper=2, shadow_dimension=5), RandomCropFromBorders(always_apply=True, p=1, crop_left=0.1, crop_right=0.1, crop_top=0.1, crop_bottom=0.1)))
            # e.g. (39.38024949675888, (RandomSizedCrop(parameters), RandomShadow(parameters), RandomCropFromBorders(parameters))
            rankedsolutions = []
            # solutionsScores a list that hold only the fitness scores of all solutions in a generation used to calculate the avg fitness score per generation
            solutionsScores = []
            for solution in tqdm(solutions, desc='Solutions', leave=False):
                solutionID = str(generation_number) + "_" + (uuid.uuid4().hex[:6])
                solution_path = os.path.join(self.results_path, solutionID)
                os.makedirs(solution_path, exist_ok=True)
                with open(os.path.join(solution_path, f'{solutionID}-transformations.csv'), 'w', newline='') as csvfile:
                    csvwriter = csv.writer(csvfile)
                    for item in solution:
                        csvwriter.writerow([str(item)])

                # we check if the solution has already fitness score attached if yes then we skip testing it again;
                # we append it to the ranked solutions and its score into solutions scores as this is the elite top one solution from previous generation

                if isinstance(solution[0], float):
                    rankedsolutions.append(solution)
                    solutionsScores.append(solution[0])
                    continue

                fitnesScore = self._calculate_fitness(solution,
                                                      self.train_images_filepaths_splits,
                                                      self.val_images_filepaths_splits,
                                                      solutionID,
                                                      solution_path,
                                                      generation_number)
                rankedsolutions.append((fitnesScore, solution))
                solutionsScores.append(fitnesScore)

            # calculate the average fitness score per generation and append it to the average_fitnessScores_per_generation
            averageFitnessScorePerGeneration = sum(list(solutionsScores)) / len(list(solutionsScores))
            average_fitnessScores_per_generation.append(averageFitnessScorePerGeneration)

            # The sorting will be performed in ascending order, considering the first element of each tuple and then the second if the first elements are equal.
            rankedsolutions = sorted(rankedsolutions, key=lambda x: x[0], reverse=True)


            # best solutions are the selected solutions that will produce new offsprings
            parents = rankedsolutions[:self.number_of_parents_for_reproduction]
            parents_scores = [x[0] for x in parents]
            # print('Selecte parents scores: ', parents_scores)

            elite_solution = parents[0]
            # print(f'Elite solution is {elite_solution[0]} at generation {generation_number}')

            if elite_solution[0] >= self.target_fitness_score:
                print('Perfect solution found', elite_solution)
                break

            if self.mutation_operator:
                if self.genes_type == 'A':
                    mutated_offsprings = self._mutate_A(parents)
                    solutions = mutated_offsprings
                elif self.genes_type == 'B':
                    mutated_offsprings = self._mutate_B(parents)
                    solutions = mutated_offsprings
            else:
                solutions = parents

            if self.ranked_crossover:
                # Perform crossover using the _ranked_pairing_crossovers function
                crossed_offsprings = self._ranked_pairing_crossover(solutions, self.number_offsprings)
                solutions = crossed_offsprings
            else:
                crossed_offsprings = self._pool_pairing_crossover(solutions, self.num_classes, self.number_offsprings)
                solutions = crossed_offsprings

            if elite_solution is not None:
                solutions.insert(0, elite_solution)









