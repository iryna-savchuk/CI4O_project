from random import randint, sample


def binary_mutation(individual):
    """Binary mutation for a GA individual. Flips the bits.

    Args:
        individual (Individual): A GA individual from charles.py

    Raises:
        Exception: When individual is not binary encoded.py

    Returns:
        Individual: Mutated Individual
    """
    mut_index = randint(0, len(individual) - 1)

    if individual[mut_index] == 0:
        individual[mut_index] = 1
    elif individual[mut_index] == 1:
        individual[mut_index] = 0
    else:
        raise Exception(
            f"Trying to do binary mutation on {individual}. But it's not binary.")
    return individual


def swap_mutation(individual):
    """Swap mutation for a GA individual. Swaps the bits.

    Args:
        individual (Individual): A GA individual from charles.py

    Returns:
        Individual: Mutated Individual
    """
    mut_indexes = sample(range(0, len(individual)), 2)
    individual[mut_indexes[0]], individual[mut_indexes[1]] = individual[mut_indexes[1]], individual[mut_indexes[0]]
    return individual


def inversion_mutation(individual):
    """Inversion mutation for a GA individual. Reverts a portion of the representation.
    We choose an interval and invert that interval

    Args:
        individual (Individual): A GA individual from charles.py

    Returns:
        Individual: Mutated Individual
    """
    mut_indexes = sample(range(0, len(individual)), 2)
    #mut_indexes = [0,3]
    mut_indexes.sort()
    individual[mut_indexes[0]:mut_indexes[1]] = individual[mut_indexes[0]:mut_indexes[1]][::-1]
    return individual


# This code iterates over every weight in the individual and, with a probability of mutation_rate, adds a small value drawn from a normal distribution to the weight.
# For mutation, we might want to perturb a weight by a small amount.
def mutation(individual, mutation_rate=0.01):
    """Apply Gaussian mutation to the weights of the individual."""
    for i in range(len(individual)):
        for j in np.nditer(individual[i], op_flags=['readwrite']):
            if np.random.random() < mutation_rate:
                j[...] += np.random.normal(0, 1)
    return individual








