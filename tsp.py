from charles.charles import Population, Individual
from charles.search import hill_climb, sim_annealing
from data.tsp_data import distance_matrix
from random import choices
from copy import deepcopy
from charles.crossover import cycle_xo, pmx
from charles.mutation import swap_mutation, inversion_mutation
from charles.selection import tournament_sel, fps


def get_fitness(self):
    """A simple objective function to calculate distances
    for the TSP problem.

    Returns:
        int: the total distance of the path
    """
    fitness = 0
    for i in range(len(self.representation)):
        fitness += distance_matrix[self.representation[i - 1]][self.representation[i]]
    return int(fitness)


def get_neighbours(self):
    """A neighbourhood function for the TSP problem. Switches
    indexes around in pairs.

    Returns:
        list: a list of individuals
    """
    n = [deepcopy(self.representation) for i in range(len(self.representation) - 1)]

    for count, i in enumerate(n):
        i[count], i[count + 1] = i[count + 1], i[count]

    n = [Individual(i) for i in n]
    return n


# Monkey patching
Individual.get_fitness = get_fitness
Individual.get_neighbours = get_neighbours


pop = Population(
    size=50,
    sol_size=len(distance_matrix[0]),
    valid_set=[i for i in range(len(distance_matrix[0]))],
    replacement=False,
    optim="min")

pop.evolve(gens=100, select=tournament_sel, mutate=inversion_mutation, crossover=pmx,
           mut_prob=0.05, xo_prob=0.9, elitism=True)


#hill_climb(pop)
#sim_annealing(pop)

