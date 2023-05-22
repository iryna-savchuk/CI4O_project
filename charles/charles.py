#from random import shuffle, choice, sample
import random
from operator import attrgetter
from copy import deepcopy
from tqdm import tqdm

random.seed(42)  #Ensuring reproducibility of the results

class Individual: # represent an individual in a population for the genetic algorithm
    # where each individual is characterized by a set of weights (representing a neural network's weights),
    # and has a fitness score which is intended to be determined by an external function in mnist.pt (get_fitness)
    def __init__( # constructor method that gets called when create a new instance of the class
            self,
            representation=None,
            input=None,
            hidden=None,
            output=None,
            valid_range=[]
    ):
        if representation is None: #  If no representation is provided, it generates random weights in the given valid_range
            size = input * hidden + hidden + hidden * output + output
            random_weights = [random.uniform(valid_range[0], valid_range[1]) for _ in range(size)]
            self.representation = {
                "weights": random_weights,
                "input": input,
                "hidden": hidden,
                "output": output
            }
        else: # If representation is provided, it assigns it directly
            self.representation = representation
        self.fitness = self.get_fitness()

    def get_fitness(self): # fitness function evaluates the quality or performance of an individual solution
        # raises an exception indicating that you need to monkey patch it
        raise Exception("You need to monkey patch the fitness path.")

    def index(self, value): # the index of the provided value in the weights of the representation
        return self.representation["weights"].index(value)

    def __len__(self): # It returns the length of the weights in the representation.
        return len(self.representation["weights"])

    def __getitem__(self, position): # returns the weight at the specified position.
        return self.representation["weights"][position]

    def __setitem__(self, position, value): # It sets the weight at the specified position to the provided value.
        self.representation["weights"][position] = value

    def __repr__(self): # calls when you try to print the object or convert it to a string. It returns a string that represents the object, in this case, details about the individual and its fitness.
        representation_print = str(self.representation["input"])+", " + str(self.representation["hidden"])+", " + str(self.representation["output"])
        return f"\nIndividual: {representation_print}; Fitness: {self.fitness}"


class Population: # defines the class Population, used to initialize a new population of individuals
    def __init__(self, size, optim, **kwargs):
        print("Generating initial population...")
        self.individuals = []
        self.size = size # The initial population size is determined by the parameter size
        self.optim = optim # parameter to determine whether the optimization problem is maximization or minimization
        for _ in tqdm(range(size)):
            self.individuals.append(
                Individual( # **kwargs is used to pass the parameters to the Individual class to initialize each individual.
                    input=kwargs["sol_input"],
                    hidden=kwargs["sol_hidden"],
                    output=kwargs["sol_output"],
                    valid_range=kwargs["valid_range"],
                )
            )

    def evolve(self, gens, xo_prob, mut_prob, select, mutate, crossover, elitism, tourn_size):
        # where the evolutionary process happens
        # It's responsible for evolving the population for a certain number of generations, gen
        # Evvolution involves selection (choosing individuals based on their fitness), crossover (combining two parents to create offspring), and mutation (randomly altering parts of an individual).
        # These operations are determined by provided parameters:
        # select - selection function
        # crossover - crossover function
        # mutate - mutation function
        #xo_prob - crossover probability
        # mut_prob - mutation probability
        for i in tqdm(range(gens)): # Python library that produces a progress bar.
            # display a progress bar in the console that shows how many iterations have been completed and how many total iterations there are,
            # giving a visual indication of how much longer the loop will continue to run.
            # Inside the loop, print statement that will output the current generation number each time a new generation starts.
            print("Starting Generation ", i + 1, "...")
            new_pop = [] # is initializing an empty list that will be used to hold the individuals of the new generation. It is reset to an empty list at the beginning of each generation.

            if elitism: # If elitism is set to true, the best individual from the current generation (either with maximum or minimum fitness based on optim) is preserved into the next generation.
                if self.optim == "max":
                    # It uses the attrgetter function from the operator module to get the fitness attribute of the individuals for comparison.
                    elite = deepcopy(max(self.individuals, key=attrgetter("fitness")))
                elif self.optim == "min":
                    elite = deepcopy(min(self.individuals, key=attrgetter("fitness")))

            while len(new_pop) < self.size: # generate new individuals until the population size
                # Select a new population based on the selection function chosen and the tournament size
                parent1, parent2 = select(self, tourn_size), select(self, tourn_size)

                if random.random() < xo_prob: # It generates offspring by performing crossover between two parents (parent1 and parent2) with a probability of xo_prob.
                                              # If crossover occurs, the crossover function is called to generate two offspring (offspring1 and offspring2).
                    offspring1, offspring2 = crossover(parent1, parent2)
                else:  # Otherwise it keeps the parents
                    offspring1, offspring2 = parent1, parent2

                if random.random() < mut_prob: # It applies mutation to the offspring with a probability of mut_prob using the mutate function.
                    offspring1 = mutate(offspring1)
                if random.random() < mut_prob:
                    offspring2 = mutate(offspring2)
                # After the crossover and mutation operations, the offspring are added to the new_pop list as new individuals.
                # The new individuals use the same architecture (input, hidden, and output) as parent1.
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

            # After generating the new population, if elitism is enabled, the worst individual in the new population is replaced with the elite individual saved from the old population.
            # This ensures that the best solution is not lost from one generation to the next.
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
            # replaces the old population with the new one, completing the transition to the next generation.
            self.individuals = new_pop
            #  the function prints out the results of the current generation, including the best individual according to the optimization type (optim).
            print("Results for Generation >>> ", i+1)
            if self.optim == "max":
                print(f'Best Individual: {max(self, key=attrgetter("fitness"))}')
            elif self.optim == "min":
                print(f'Best Individual: {min(self, key=attrgetter("fitness"))}')

    def __len__(self): # returns the length of the population, i.e., the number of individuals.
        return len(self.individuals)

    def __getitem__(self, position): # returns an individual at the given position in the population.
        return self.individuals[position]
