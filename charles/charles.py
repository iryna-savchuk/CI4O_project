from random import random  # random() - Returns a random float number between 0 and 1
from operator import attrgetter
from copy import deepcopy

import numpy as np

class Individual:
    def __init__(
        self,
        representation=None,
        input=28*28,
        hidden=512,
        output=10
    ):
        if representation is None:
            weights_1 = np.random.rand(input * hidden).reshape(input, hidden)
            biases_1 = np.random.rand(hidden).reshape(hidden, )
            weights_2 = np.random.rand(hidden * output).reshape(hidden, output)
            biases_2 = np.random.rand(output).reshape(output, )
            self.representation = [weights_1, biases_1, weights_2, biases_2]
        else:
            self.representation = representation
        self.fitness = self.get_fitness()

    def get_fitness(self):
        raise Exception("You need to monkey patch the fitness path.")

    def get_neighbours(self, func, **kwargs):
        raise Exception("You need to monkey patch the neighbourhood function.")

    def index(self, value):
        return self.representation.index(value)

    def __len__(self):
        return len(self.representation)

    def __getitem__(self, position):
        return self.representation[position]

    def __setitem__(self, position, value):
        self.representation[position] = value

    def __repr__(self):
        weights_print = ""
        for weights in self.representation:
            weights_print = weights_print+" "+str(weights.shape)
        return f"Individual: {weights_print}; Fitness: {self.fitness}\n"


class Population:
    def __init__(self, size, optim, **kwargs):
        self.individuals = []
        self.size = size
        self.optim = optim
        self.best_fitnesses = [] # list to store the best fitness for each generation
        for _ in range(size):
            self.individuals.append(
                Individual()
            )

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
                    offspring1, offspring2 = parent1, parent2

                if random() < mut_prob:
                    offspring1 = mutate(offspring1)
                if random() < mut_prob:
                    offspring2 = mutate(offspring2)

                new_pop.append(Individual(representation=offspring1))
                if len(new_pop) < self.size:
                    new_pop.append(Individual(representation=offspring2))

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
                print(f'Gen {i+1}, Best Individual: {max(self, key=attrgetter("fitness"))}')
            elif self.optim == "min":
                print(f'Gen {i+1}, Best Individual: {min(self, key=attrgetter("fitness"))}')

            # Update best_fitnesses list
            if self.optim == "max":
                best_individual = max(self, key=attrgetter("fitness"))
                self.best_fitnesses.append(best_individual.fitness)
            elif self.optim == "min":
                best_individual = min(self, key=attrgetter("fitness"))
                self.best_fitnesses.append(best_individual.fitness)

    def __len__(self):
        return len(self.individuals)

    def __getitem__(self, position):
        return self.individuals[position]