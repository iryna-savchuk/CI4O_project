from random import shuffle, choice, sample, random
from operator import attrgetter
from copy import deepcopy
from numpy.random import normal
import numpy as np

# This file will keep the definition of classes, Individual and Population
# Individual is a possible solution, for an optimization problem, one possible solution for our optimization problem
# Population is a search space

# The class, every object will have a representation
# Every possible solution, in the search space, will belong to a class
# we need to modify the Individual class to hold the weights of the neural network

# The weights of a Convolutional Neural Network (CNN) aren't a flat list; they're a list of arrays,
# where each array corresponds to the weights and biases for each layer.
# Therefore, in the Individual class, you need to adjust the way you initialize the representation variable to reflect the architecture of the CNN.
class Individual:
    def __init__(self, train_data, train_labels, validation_data, validation_labels, layer_shapes=None, representation=None):
        if representation is None:
            # class initializes its representation (the weights of the neural network) with a normal distribution, which is more appropriate for neural networks.
            self.representation = [np.random.normal(0, 1, size) for size in layer_shapes]
            # Initialize weights with normal distribution
        else:
            # if we pass an argument like Individual(my_path)
            # fitness will be assigned to the individual
            self.representation = representation
        self.train_data = train_data
        self.train_labels = train_labels
        self.validation_data = validation_data
        self.validation_labels = validation_labels
        self.fitness = self.get_fitness(train_data, train_labels, validation_data, validation_labels)

    def get_fitness(self):
        raise Exception("You need to monkey patch the fitness path.")

    def get_neighbours(self, func, **kwargs):
        raise Exception("You need to monkey patch the neighbourhood function.")

    def __len__(self):
        return len(self.representation)

    def __getitem__(self, position):
        return self.representation[position]

    def __setitem__(self, position, value):
        self.representation[position] = value

    def __repr__(self):
        # return individual size and fitness, we can just keep the fitness
        return f"Individual(size={len(self.representation)}); Fitness: {self.fitness}"

# layer_shapes is a list of tuples, where each tuple is the shape of the weight array for a layer in the CNN.
# For example, for a simple CNN with one convolutional layer with 32 filters of size 3x3 and one dense layer with 128 units, layer_shapes would be [(32, 3, 3), (128, )].
# we're ignoring biases for simplicity. If we want to include biases, you would need to adjust layer_shapes accordingly and modify the GA to handle both weights and biases.

# When you initialize a population, the Population class currently creates a population of individuals where each individual's weights are initialized randomly.
# This is fine, but remember that these weights need to match the structure of the CNN's weights (a list of arrays).
# This version of Population is similar to your original one, but it creates Individuals with the appropriate layer_shapes
# and it uses deepcopy when creating the offspring to ensure that the offspring are new Individuals and not just references to the parents.
class Population:
    def __init__(self, size, optim, layer_shapes, train_data, train_labels, validation_data, validation_labels, **kwargs):
        self.individuals = []
        self.size = size
        self.optim = optim
        for _ in range(size):
            self.individuals.append(Individual(train_data, train_labels, validation_data, validation_labels, layer_shapes=layer_shapes))

    def evolve(self, gens, xo_prob, mut_prob, select, mutate, crossover, elitism):
        for i in range(gens):
            new_pop = []
            if elitism:
                if self.optim == "max":
                    elite = deepcopy(max(self.individuals, key=attrgetter("fitness")))
                elif self.optim == "min":
                    elite = deepcopy(min(self.individuals, key=attrgetter("fitness")))
            while len(new_pop) < self.size:
                parent1, parent2 = select(self), select(self)
                if random() < xo_prob:
                    offspring1, offspring2 = crossover(parent1, parent2)
                else:
                    offspring1, offspring2 = deepcopy(parent1), deepcopy(parent2)
                if random() < mut_prob:
                    offspring1 = mutate(offspring1)
                if random() < mut_prob:
                    offspring2 = mutate(offspring2)
                new_pop.append(offspring1)
                if len(new_pop) < self.size:
                    new_pop.append(offspring2)
            if elitism:
                if self.optim == "max":
                    worst = min(new_pop, key=attrgetter("fitness"))
                    if elite.fitness > worst.fitness:
                        new_pop.pop(new_pop.index(worst))
                        new_pop.append(elite)
                elif self.optim == "min":
                    worst = max(new_pop, key=attrgetter("fitness"))
                    if elite.fitness < worst.fitness:
                        new_pop.pop(new_pop.index(worst))
                        new_pop.append(elite)
            self.individuals = new_pop
            if self.optim == "max":
                print(f'Best Individual: {max(self, key=attrgetter("fitness"))}')
            elif self.optim == "min":
                print(f'Best Individual: {min(self, key=attrgetter("fitness"))}')

    def __len__(self):
        return len(self.individuals)

    def __getitem__(self, position):
        return self.individuals[position]


