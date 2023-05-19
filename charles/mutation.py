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


# It randomly selects two positions in an individual's representation and swaps the values at those positions.
# This introduces variation into the population by changing the genetic material of an individual.
def swap_mutation(individual): # The function takes an individual as input, represented as a list.
    """Swap mutation for a GA individual. Swaps the bits.

    Args:
        individual (Individual): A GA individual from charles.py

    Returns:
        Individual: Mutated Individual
    """
    mut_element = 0  # the first element index to mutate, the element within the individual where the mutation will occur.
    # It uses the sample function from the random module to randomly select two indexes (mut_indexes) within the range of the individual.
    mut_indexes = sample(range(0, len(individual[0])), 2)
    # It performs swap mutation by swapping the values at the selected indexes in the individual. The values are swapped using a tuple assignment operation:
    individual[mut_element][mut_indexes[0]], individual[mut_element][mut_indexes[1]] = individual[mut_element][mut_indexes[1]], individual[mut_element][mut_indexes[0]]
    return individual # the function returns the mutated individual.


def inversion_mutation(individual):
    """Inversion mutation for a GA individual. Reverts a portion of the representation.
    We choose an interval and invert that interval

    Args:
        individual (Individual): A GA individual from charles.py

    Returns:
        Individual: Mutated Individual
    """
    mut_element = 0  # the first element index to mutate, the element within the individual where the mutation will occur.
    mut_indexes = sample(range(0, len(individual[0])), 2)
    mut_indexes.sort()
    individual[mut_element][mut_indexes[0]:mut_indexes[1]] = individual[mut_element][mut_indexes[0]:mut_indexes[1]][::-1]
    return individual






