from random import random  # random() - Returns a random float number between 0 and 1
from operator import attrgetter
from copy import deepcopy

import numpy as np


# defines a class called Individual, which represents an individual in a genetic algorithm
# this code defines a basic structure for an individual in an optimization problem.
class Individual:
    def __init__( # initializes an individual with a representation, which consists of four parts: weights_1, biases_1, weights_2, and biases_2
        self,
        representation=None,
        input=28*28, # The shapes of the weights and biases are determined by the input, hidden, and output parameters.
        hidden=512,
        output=10
    ):
        if representation is None: # If no representation is provided, random values are generated for these components using np.random.rand.
            weights_1 = np.random.rand(input * hidden).reshape(input, hidden)
            biases_1 = np.random.rand(hidden).reshape(hidden, )
            weights_2 = np.random.rand(hidden * output).reshape(hidden, output)
            biases_2 = np.random.rand(output).reshape(output, )
            self.representation = [weights_1, biases_1, weights_2, biases_2]
        else:
            self.representation = representation
        self.fitness = self.get_fitness()
    # The fitness value represents the quality or performance of the individual
    def get_fitness(self): # This method should be overridden in a subclass or monkey-patched to calculate and return the fitness value of the individual.
        raise Exception("You need to monkey patch the fitness path.") # method is not implemented in the class but raises an exception

    def get_neighbours(self, func, **kwargs):
        raise Exception("You need to monkey patch the neighbourhood function.")

    def index(self, value): # returns the index of a value within the representation of the individual.
        return self.representation.index(value)

    def __len__(self): # length of the representation, which is the number of components (four in this case).
        return len(self.representation)

    def __getitem__(self, position): # allows accessing the components of the representation using indexing.
        return self.representation[position]

    def __setitem__(self, position, value): # allows modifying the components of the representation using indexing.
        self.representation[position] = value

    def __repr__(self): # provides a string representation of the individual. It prints the shapes of the weights and biases in the representation, along with the fitness value.
        weights_print = ""
        for weights in self.representation:
            weights_print = weights_print+" "+str(weights.shape)
        return f"Individual: {weights_print}; Fitness: {self.fitness}\n"


# defines a class called Population, which represents a population of individuals in a genetic algorithm
# this code provides a framework for evolving a population of individuals over multiple generations using genetic operators such as selection, crossover, and mutation
class Population:
    def __init__(self, size, optim, **kwargs): #  constructor of the class. It initializes a population with a given size and optimization direction (min or max)
        self.individuals = [] # creates an empty list to store individuals (self.individuals) and initializes an empty list to store the best fitness value for each generation (self.best_fitnesses)
        self.size = size # creates size number of Individual objects and adds them to the individuals
        self.optim = optim
        self.best_fitnesses = [] # list to store the best fitness for each generation
        for _ in range(size):
            self.individuals.append(
                Individual()
            )

    def evolve(self, gens, xo_prob, mut_prob, select, mutate, crossover, elitism, tourn_size): #  performs the evolution process for the population
        # gens - number of generations
        # xo_prob - crossover probability
        # mut_prob - mutation probability
        # select - selection function
        # mutate - mutation function
        # crossover - crossover function
        # elitism - elitism flag

        for i in range(gens):
            new_pop = [] # In each generation, it creates a new population (new_pop) by selecting individuals from the current population using the select function.

            if elitism: # If elitism is enabled, it selects the best individual from the current population (elite) based on the optimization direction (self.optim).
                if self.optim == "max":
                    # It uses the attrgetter function from the operator module to get the fitness attribute of the individuals for comparison.
                    elite = deepcopy(max(self.individuals, key=attrgetter("fitness")))
                elif self.optim == "min":
                    elite = deepcopy(min(self.individuals, key=attrgetter("fitness")))

            while len(new_pop) < self.size:
                parent1, parent2 = select(self, tourn_size), select(self, tourn_size)

                if random() < xo_prob: # It generates offspring by performing crossover between two parents (parent1 and parent2) with a probability of xo_prob.
                                       # If crossover occurs, the crossover function is called to generate two offspring (offspring1 and offspring2).
                    offspring1, offspring2 = crossover(parent1, parent2)
                else: # Otherwise it keeps the parents
                    offspring1, offspring2 = parent1, parent2

                if random() < mut_prob: # It applies mutation to the offspring with a probability of mut_prob using the mutate function.
                    offspring1 = mutate(offspring1)
                if random() < mut_prob:
                    offspring2 = mutate(offspring2)

                new_pop.append(Individual(representation=offspring1)) # The offspring individuals are added to the new population (new_pop).
                if len(new_pop) < self.size: # in case of odd individuals in the population
                    new_pop.append(Individual(representation=offspring2))

            if elitism: # If elitism is enabled, it checks if the elite individual has a better fitness than the worst individual in the new population
                        # If so, it replaces the worst individual with the elite individual.
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

            self.individuals = new_pop # The new population becomes the current population (self.individuals).

            if self.optim == "max":
                print(f'Gen {i+1}, Best Individual: {max(self, key=attrgetter("fitness"))}')
            elif self.optim == "min":
                print(f'Gen {i+1}, Best Individual: {min(self, key=attrgetter("fitness"))}')

            # Update best_fitnesses list
            # The best individual in the current population is printed based on the optimization direction (self.optim).
            if self.optim == "max":
                best_individual = max(self, key=attrgetter("fitness"))
                self.best_fitnesses.append(best_individual.fitness) # The fitness value of the best individual is added to the best_fitnesses list.
            elif self.optim == "min":
                best_individual = min(self, key=attrgetter("fitness"))
                self.best_fitnesses.append(best_individual.fitness)

    def __len__(self): # returns the number of individuals in the population.
        return len(self.individuals)

    def __getitem__(self, position): # method allows accessing the individuals in the population using indexing.
        return self.individuals[position]