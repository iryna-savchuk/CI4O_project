from random import randint, sample
import random


def inversion_mut(individual):
    """Inversion mutation for a GA individual. Reverts a portion of the representation.

    Args:
        individual (Individual): A GA individual from charles.py

    Returns:
        Individual: Mutated Individual
    """
    mut_indexes = sample(range(0, len(individual)), 2)
    mut_indexes.sort()
    # The subsection of the chromosome between the two indices is reversed
    individual[mut_indexes[0]:mut_indexes[1]] = individual[mut_indexes[0]:mut_indexes[1]][::-1]
    return individual


def arithmetic_mut(individual, lower_bound=-0.1, upper_bound=0.1):
    """Arithmetic mutation for a GA individual. Each gene is altered by a random value.

    Args:
        individual (Individual): A GA individual
        lower_bound (float): The lower bound for the random value
        upper_bound (float): The upper bound for the random value

    Returns:
        Individual: Mutated Individual
    """
    for i in range(len(individual)):
        # Generating and adding a random value to the gene
        mutation_value = random.uniform(lower_bound, upper_bound)
        individual[i] += mutation_value

        # Ensure the gene is within the desired range
        individual[i] = max(min(individual[i], 1.0), -1.0)

    return individual
















