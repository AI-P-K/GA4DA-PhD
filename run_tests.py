import sys

from run_experiment import run_experiment

"""
If crossover is True than we can not use the fitness based mutation rate
"""
run_experiment(genes_type='B',
               mutation=True,
               baseline_augmentation=False,
               fixed_mutation_rate=False,
               ranked_crossover=True)

# run_experiment(genes_type='B',
#                mutation=True,
#                baseline_augmentation=True,
#                fixed_mutation_rate=True,
#                ranked_crossover=True)

# run_experiment(genes_type='A',
#                mutation=True,
#                baseline_augmentation=True,
#                fixed_mutation_rate=True,
#                ranked_crossover=True)

# run_experiment(genes_type='A',
#                mutation=True,
#                baseline_augmentation=True,
#                fixed_mutation_rate=False,
#                ranked_crossover=True)


