from random import randint, sample, uniform
from copy import deepcopy


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

# geometric crossover
# we might want to swap all weights for one or more layers between two parents.
def arithmetic_xo_aula(p1, p2): # takes two individuals (p1 and p2) as input, representing the parents for crossover.
    """Implementation of arithmetic crossover/geometric crossover with constant alpha.

    Args:
        p1 (Individual): First parent for crossover.
        p2 (Individual): Second parent for crossover.

    Returns:
        Individuals: Two offspring, resulting from the crossover.
    """
    alpha = uniform(0, 1) # It initializes the alpha value by generating a random number between -1 and 1 using the uniform function from the random module.
    # the alpha value determines the degree of influence each parent has on the offspring.
    # If alpha is closer to 0, the offspring will resemble the average of the parents.
    # If alpha is closer to 1, the offspring will resemble one of the parents more.
    o1 = [None] * len(p1) # It creates two empty lists, o1 and o2, to store the offspring.
    o2 = [None] * len(p1)
    # performs arithmetic crossover between the parents
    for i in range(len(p1)):
        # For each component (weight or bias) of the individuals, it calculates the corresponding component for the offspring using the following formulas:
        # p1[i] and p2[i] represent the i-th component of the parents
        # o1[i] and o2[i] represent the i-th component of the offspring.
        # the calculations are performed for each component in a loop, using the zip function to iterate over corresponding components of p1 and p2.
        o1[i] = alpha * p1[i] + (1 - alpha) * p2[i]
        o2[i] = alpha * p2[i] + (1 - alpha) * p1[i]
        # o1 = [alpha * w1 + (1 - alpha) * w2 for w1, w2 in zip(p1, p2)]
        # o2 = [alpha * w2 + (1 - alpha) * w1 for w1, w2 in zip(p1, p2)]
    return o1, o2 # the function returns the two offspring individuals (o1 and o2).

# geometric crossover
# we might want to swap all weights for one or more layers between two parents.
def arithmetic_xo(p1, p2): # takes two individuals (p1 and p2) as input, representing the parents for crossover.
    """Implementation of arithmetic crossover/geometric crossover with constant alpha.

    Args:
        p1 (Individual): First parent for crossover.
        p2 (Individual): Second parent for crossover.

    Returns:
        Individuals: Two offspring, resulting from the crossover.
    """
    alpha = uniform(0, 1) # It initializes the alpha value by generating a random number between -1 and 1 using the uniform function from the random module.
    # the alpha value determines the degree of influence each parent has on the offspring.
    # If alpha is closer to 0, the offspring will resemble the average of the parents.
    # If alpha is closer to 1, the offspring will resemble one of the parents more.
    o1 = [None] * len(p1) # It creates two empty lists, o1 and o2, to store the offspring.
    o2 = [None] * len(p1)
    # performs arithmetic crossover between the parents
    for i in range(len(p1)):
        # For each component (weight or bias) of the individuals, it calculates the corresponding component for the offspring using the following formulas:
        # p1[i] and p2[i] represent the i-th component of the parents
        # o1[i] and o2[i] represent the i-th component of the offspring.
        # the calculations are performed for each component in a loop, using the zip function to iterate over corresponding components of p1 and p2.
        #o1[i] = alpha * p1[i] + (1 - alpha) * p2[i]
        #o2[i] = alpha * p2[i] + (1 - alpha) * p1[i]
        o1 = [alpha * w1 + (1 - alpha) * w2 for w1, w2 in zip(p1, p2)]
        o2 = [alpha * w2 + (1 - alpha) * w1 for w1, w2 in zip(p1, p2)]
    return o1, o2 # the function returns the two offspring individuals (o1 and o2).


# uniform crossover
# Uniform crossover randomly selects corresponding components from the parents and exchanges them between the offspring.
# to swap all weights for one or more layers between two parents.
# This code iterates over each layer and, swaps the weights for that layer between the two parents.
def uniform_xo(p1, p2): # takes two individuals (p1 and p2) as input, representing the parents for crossover.
    """Implement uniform crossover."""
    o1, o2 = deepcopy(p1), deepcopy(p2) # It creates two deep copies of the parents, o1 and o2, using the deepcopy function from the copy module.
                                        # This step ensures that the original parents are not modified during the crossover process.
    for i in range(len(p1)): # It then performs uniform crossover between the parents. For each component (weight or bias) of the individuals, it swaps the components between o1 and o2.
                            # The swapping is performed for each component in a loop, using the range(len(p1)) to iterate over the indices.
        o1[i], o2[i] = o2[i].copy(), o1[i].copy() # The swapping is done using the .copy() method, which creates a copy of the component, ensuring that the original component is not modified.
    return o1, o2 # the function returns the two offspring individuals (o1 and o2), which have undergone uniform crossover.












