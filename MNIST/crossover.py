from random import randint, sample, uniform


def single_point_co(p1, p2):
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


def cycle_xo(p1, p2):
    """Implementation of cycle crossover.

    Args:
        p1 (Individual): First parent for crossover.
        p2 (Individual): Second parent for crossover.

    Returns:
        Individuals: Two offspring, resulting from the crossover.
    """
    # offspring placeholders
    offspring1 = [None] * len(p1)
    offspring2 = [None] * len(p1)

    while None in offspring1:
        index = offspring1.index(None)
        val1 = p1[index]
        val2 = p2[index]

        # copy the cycle elements
        while val1 != val2:
            offspring1[index] = p1[index]
            offspring2[index] = p2[index]
            val2 = p2[index]
            index = p1.index(val2)

        # copy the rest
        for element in offspring1:
            if element is None:
                index = offspring1.index(None)
                if offspring1[index] is None:
                    offspring1[index] = p2[index]
                    offspring2[index] = p1[index]

    return offspring1, offspring2


def pmx(p1, p2):
    """Implementation of partially matched/mapped crossover.

    Args:
        p1 (Individual): First parent for crossover.
        p2 (Individual): Second parent for crossover.

    Returns:
        Individuals: Two offspring, resulting from the crossover.
    """
    xo_points = sample(range(len(p1)), 2)  # to define randomly the segment
    xo_points = [3,6]  # it will be a list like this
    # xo_points.sort() # sort it to create a range


    def pmx_offspring(x,y):
        o = [None] * len(x)  # create an offspring full of none
        # offspring2
        # copy the segment from the other parent
        o[xo_points[0]:xo_points[1]]  = x[xo_points[0]:xo_points[1]] # from 3 to 6, please get your list segments from the other parent
        z = set(y[xo_points[0]:xo_points[1]]) - set(x[xo_points[0]:xo_points[1]]) # find the unique numbers that exist in the segment of y and does not exist in the segment of x

        # numbers that exist in the segment
        # get tge pair of the numbers in the segment
        for i in z:
            temp = i
            index = y.index(x[y.index(temp)])
            while o[index] is not None:
                temp = index
                index = y.index(x[temp])
            o[index] = i

        # numbers that doesn't exist in the segment
        while None in o:
            index = o.index(None)
            o[index] = y[index]
        return o

    o1, o2 = pmx_offspring(p1, p2), pmx_offspring(p2, p1)
    return o1, o2

# we might want to swap all weights for one or more layers between two parents.
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
        o1 = [alpha * w1 + (1 - alpha) * w2 for w1, w2 in zip(p1, p2)]
        o2 = [alpha * w2 + (1 - alpha) * w1 for w1, w2 in zip(p1, p2)]
    return o1, o2



# to swap all weights for one or more layers between two parents.
# This code iterates over each layer and, with a 50% chance, swaps the weights for that layer between the two parents.
def crossover(p1, p2):
    """Implement uniform crossover."""
    o1, o2 = deepcopy(p1), deepcopy(p2)
    for i in range(len(p1)):
        if np.random.rand() < 0.5:
            o1[i], o2[i] = o2[i].copy(), o1[i].copy()
    return o1, o2












