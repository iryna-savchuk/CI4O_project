# Imports
import random
from operator import attrgetter
from copy import deepcopy
from tqdm import tqdm # Python library that produces a progress bar

random.seed(42)  # Ensuring reproducibility of the results

# Define classes

class Individual: # Class defining an individual for the Genetic Algorithm
    # Each individual is characterized by a set of weights (representing a neural network's weights),
    # the number of input/hidden/output neurons (the neural network has 1 hidden layer)
    # and has a fitness score which is intended to be determined by the external function in mnist.py (get_fitness)
    def __init__( # Constructor method that gets called upon creation of a new instance of the class
            self,
            representation=None,
            input=None,
            hidden=None,
            output=None,
            valid_range=[]
    ):
        if representation is None: # If no representation is provided, generate random weights in the given valid_range
            size = input * hidden + hidden + hidden * output + output
            random_weights = [random.uniform(valid_range[0], valid_range[1]) for _ in range(size)]
            self.representation = {
                "weights": random_weights,
                "input": input,
                "hidden": hidden,
                "output": output
            }
        else: # If representation is provided, assign it directly
            self.representation = representation
        self.fitness = self.get_fitness()

    def get_fitness(self): # Fitness function evaluates the quality or performance of an individual solution
        # Raises an exception indicating that you need to monkey patch it
        raise Exception("You need to monkey patch the fitness path.")

    def index(self, value): # Returns the index of the provided value in the weights of the representation
        return self.representation["weights"].index(value)

    def __len__(self): # Returns the length of the weights in the representation
        return len(self.representation["weights"])

    def __getitem__(self, position): # Returns the value of the weight at the specified position
        return self.representation["weights"][position]

    def __setitem__(self, position, value): # Sets the weight at the specified position to the provided value
        self.representation["weights"][position] = value

    def __repr__(self): # Returns a string representation of the object, called whenever the object is printed
        # In this case, the returned string contains details about the individual and its fitness
        representation_print = str(self.representation["input"])+", " + str(self.representation["hidden"])+", " + str(self.representation["output"])
        return f"\nIndividual: {representation_print}; Fitness: {self.fitness}"


class Population: # Class defining Population of individuals in Genetic Algorithm
    def __init__(self, size, optim, **kwargs):
        print("Generating initial population...")
        self.individuals = []
        self.size = size # initial population size is determined by the parameter size
        self.optim = optim # parameter to determine whether an optimization problem is maximization or minimization
        self.best_fitnesses = []  # list to store the best fitness for each generation
        for _ in tqdm(range(size)):
            self.individuals.append(
                Individual( # **kwargs is used to pass the parameters to the Individual class to initialize each individual
                    input=kwargs["sol_input"],
                    hidden=kwargs["sol_hidden"],
                    output=kwargs["sol_output"],
                    valid_range=kwargs["valid_range"],
                )
            )

    def evolve(self, gens, xo_prob, mut_prob, select, mutate, crossover, elitism, tourn_size):
        # Method to evolve the population for a given number of generations,
        # with a given selection method, crossover, mutation, and elitism.

        # Parameters:
        # gens (int): The number of generations to evolve the population for
        # xo_prob (float): Crossover probability
        # mut_prob (float): Mutation probability
        # select (func): The selection method to use
        # mutate (func): The mutation method to use
        # crossover (func): The crossover method to use
        # elitism (bool): Whether to use elitism

        for i in tqdm(range(gens)):
            # Printing out the current generation number each time a new generation starts
            print("Starting Generation ", i + 1, "...")
            new_pop = [] # at the beginning of each generation, new population is empty

            if elitism:
                if self.optim == "max":
                    elite = deepcopy(max(self.individuals, key=attrgetter("fitness")))
                elif self.optim == "min":
                    elite = deepcopy(min(self.individuals, key=attrgetter("fitness")))

            while len(new_pop) < self.size: # generate new individuals until the population size is reached
                parent1, parent2 = select(self, tourn_size), select(self, tourn_size)

                if random.random() < xo_prob:
                    offspring1, offspring2 = crossover(parent1, parent2)
                else:
                    offspring1, offspring2 = parent1, parent2

                if random.random() < mut_prob:
                    offspring1 = mutate(offspring1)
                if random.random() < mut_prob:
                    offspring2 = mutate(offspring2)

                # The new individuals are of the same architecture (input, hidden, and output) as their parents
                new_pop.append(Individual(representation={
                    "weights": offspring1,
                    "input": parent1.representation["input"],
                    "hidden": parent1.representation["hidden"],
                    "output": parent1.representation["output"],
                }))
                if len(new_pop) < self.size:
                    new_pop.append(Individual(representation={
                        "weights": offspring2,
                        "input": parent1.representation["input"],
                        "hidden": parent1.representation["hidden"],
                        "output": parent1.representation["output"],
                    }))

            # If elitism is enabled, the worst individual in the new population is replaced with
            # the elite individual from the old population.
            # This ensures that the best solution is not lost during evolution.
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

            # Replacing the population with the new one - completing the transition to the next generation
            self.individuals = new_pop

            # Printing out information about the generation
            print("Results for Generation >>> ", i+1)
            if self.optim == "max":
                print(f'Best Individual: {max(self, key=attrgetter("fitness"))}')
            elif self.optim == "min":
                print(f'Best Individual: {min(self, key=attrgetter("fitness"))}')

            # Updating the best_fitnesses list
            if self.optim == "max":
                best_individual = max(self, key=attrgetter("fitness"))
                self.best_fitnesses.append(best_individual.fitness)
            elif self.optim == "min":
                best_individual = min(self, key=attrgetter("fitness"))
                self.best_fitnesses.append(best_individual.fitness)

    def __len__(self): # Returns the number of individuals in the population
        return len(self.individuals)

    def __getitem__(self, position): # Returns an individual at the given position in the population
        return self.individuals[position]
