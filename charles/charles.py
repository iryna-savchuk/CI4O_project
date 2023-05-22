#from random import shuffle, choice, sample
import random
from operator import attrgetter
from copy import deepcopy
from tqdm import tqdm

random.seed(42)

class Individual:
    def __init__(
            self,
            representation=None,
            input=None,
            hidden=None,
            output=None,
            valid_range=[]
    ):
        if representation is None:
            size = input * hidden + hidden + hidden * output + output
            #random_weights = np.random.uniform(valid_range[0], valid_range[1], size=size)
            random_weights = [random.uniform(valid_range[0], valid_range[1]) for _ in range(size)]
            self.representation = {
                "weights": random_weights,
                "input": input,
                "hidden": hidden,
                "output": output
            }
        else:
            self.representation = representation
        self.fitness = self.get_fitness()

    def get_fitness(self):
        raise Exception("You need to monkey patch the fitness path.")

    def get_neighbours(self, func, **kwargs):
        raise Exception("You need to monkey patch the neighbourhood function.")

    def index(self, value):
        return self.representation["weights"].index(value)

    def __len__(self):
        return len(self.representation["weights"])

    def __getitem__(self, position):
        return self.representation["weights"][position]

    def __setitem__(self, position, value):
        self.representation["weights"][position] = value

    def __repr__(self):
        #return f"Individual: {self.representation};" #Fitness: {self.fitness}\n"
        #return '\nIndividual:'+'\n'.join(f"{k}: {v}" for k, v in self.representation.items() if k!="weights")+'\nFitness: '+str(self.fitness)+'\n'
        representation_print = str(self.representation["input"])+", " + str(self.representation["hidden"])+", " + str(self.representation["output"])
        return f"\nIndividual: {representation_print}; Fitness: {self.fitness}"
class Population:
    def __init__(self, size, optim, **kwargs):
        print("Generating initial population...")
        self.individuals = []
        self.size = size
        self.optim = optim
        for _ in tqdm(range(size)):
            self.individuals.append(
                Individual(
                    input=kwargs["sol_input"],
                    hidden=kwargs["sol_hidden"],
                    output=kwargs["sol_output"],
                    valid_range=kwargs["valid_range"],
                )
            )

    def evolve(self, gens, xo_prob, mut_prob, select, mutate, crossover, elitism):

        for i in tqdm(range(gens)):
            new_pop = []

            if elitism:
                if self.optim == "max":
                    elite = deepcopy(max(self.individuals, key=attrgetter("fitness")))
                elif self.optim == "min":
                    elite = deepcopy(min(self.individuals, key=attrgetter("fitness")))

            while len(new_pop) < self.size:
                parent1, parent2 = select(self), select(self)

                if random.random() < xo_prob:
                    offspring1, offspring2 = crossover(parent1, parent2)
                else:
                    offspring1, offspring2 = parent1, parent2

                if random.random() < mut_prob:
                    offspring1 = mutate(offspring1)
                if random.random() < mut_prob:
                    offspring2 = mutate(offspring2)

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

            print("\nGeneration >>> ", i+1)
            if self.optim == "max":
                print(f'Best Individual: {max(self, key=attrgetter("fitness"))}')
            elif self.optim == "min":
                print(f'Best Individual: {min(self, key=attrgetter("fitness"))}')

    def __len__(self):
        return len(self.individuals)

    def __getitem__(self, position):
        return self.individuals[position]
