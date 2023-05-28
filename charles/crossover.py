from random import randint, sample, uniform, choices


def single_point_xo(p1, p2):
    """Implementation of single point crossover.

    Args:
        p1 (Individual): First parent for crossover.
        p2 (Individual): Second parent for crossover.

    Returns:
        Individuals: Two offspring, resulting from the crossover.
    """
    co_point = randint(1, len(p1)-2)

    offspring1 = p1[:co_point] + p2[co_point:]
    offspring2 = p2[:co_point] + p1[co_point:]

    return offspring1, offspring2


def arithmetic_xo(p1, p2):
    """Implementation of arithmetic crossover/geometric crossover with constant alpha.

    Args:
        p1 (Individual): First parent for crossover.
        p2 (Individual): Second parent for crossover.

    Returns:
        Individuals: Two offspring, resulting from the crossover.
    """
    alpha = uniform(0, 1)
    o1 = [None] * len(p1)
    o2 = [None] * len(p1)
    for i in range(len(p1)):
        o1[i] = p1[i] * alpha + (1-alpha) * p2[i]
        o2[i] = p2[i] * alpha + (1-alpha) * p1[i]
    return o1, o2


def uniform_xo(p1, p2):
    """ Implementation of uniform crossover.
    Each gene in the parent, by random choice, decides if it will be swapped.
    This is done by generating a random binary mask of 1s and 0s, where a 1 indicates the gene will be swapped and a 0 - will not.
    Therefore, it is necessary to iterate over every gene and decide whether to swap it.

    Args:
        p1 (Individual): First parent for crossover.
        p2 (Individual): Second parent for crossover.

    Returns:
        Individuals: Two offspring, resulting from the crossover.
    """
    mask = choices([0, 1], k=len(p1))  # random binary mask
    offspring1, offspring2 = p1[:], p2[:]  # copies of parents
    for i in range(len(p1)):
        if mask[i] == 1:  # if the mask at position i is 1, swap the genes at position i
            offspring1[i], offspring2[i] = offspring2[i], offspring1[i]
    return offspring1, offspring2
