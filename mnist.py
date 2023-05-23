from charles.charles import Population, Individual
from data.mnist_data import train_images, train_labels
from charles.crossover import cycle_xo, pmx, single_point_co, arithmetic_xo
from charles.mutation import swap_mutation, inversion_mutation, arithmetic_mutation
from charles.selection import tournament_sel, fps
import numpy as np
from tensorflow.keras import Sequential, layers
import csv
import sys

def get_fitness(self):
    input = self.representation["input"]
    hidden = self.representation["hidden"]
    output = self.representation["output"]

    # Defining the architecture of a neural network
    nn = Sequential([
        layers.Dense(hidden, activation="relu", input_shape=(input,)),
        layers.Dense(output, activation="softmax")
    ])
    current_weights = np.array(self.representation["weights"])
    weights_1 = current_weights[:input * hidden].reshape(input, hidden)
    biases_1 = current_weights[input * hidden:input * hidden + hidden].reshape(hidden, )
    weights_2 = current_weights[input * hidden + hidden:input * hidden + hidden + hidden * output].reshape(hidden, output)
    biases_2 = current_weights[input * hidden + hidden + hidden * output:].reshape(output, )

    # Setting the weights of the model
    nn.set_weights([weights_1, biases_1, weights_2, biases_2])

    # Compiling the model
    nn.compile(loss='categorical_crossentropy', metrics=["accuracy"])

    train_loss, train_acc = nn.evaluate(train_images, train_labels)
    return train_acc


# Monkey patching
Individual.get_fitness = get_fitness

all_runs_best_fitness = []
def run_evolution(run, pop_size, gens, select, crossover, mutate, xo_prob, mut_prob, elitism, tourn_size, filename):
    for i in range(run):
        print(f"Running GA iteration: {i + 1}")
        # Generating Population
        pop = Population(
            size=pop_size,
            sol_input=28 * 28,
            sol_hidden=512,
            sol_output=10,
            valid_range=[-1, 1],
            optim="max")

        print(pop.individuals)
        # print(pop.individuals[0].representation)

        # Running evolution iterations
        print('Evolving...')
        pop.evolve(gens=gens,
                   select=select, mutate= mutate, crossover=crossover,
                   mut_prob=mut_prob, xo_prob=xo_prob,
                   elitism=elitism,
                   tourn_size=tourn_size)
        all_runs_best_fitness.append(pop.best_fitnesses)

    # Writing to CSV
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Best_fitness'])  # header
        for fitness in all_runs_best_fitness:
            writer.writerow([fitness])

    print(f"All best fitness, from each generation, from each run have been saved to {filename}")

# we can define the name of the file, to have different files with different parameters
run_evolution(run = 4, pop_size = 4, gens = 4, select = tournament_sel, crossover= arithmetic_xo, mutate= arithmetic_mutation,
              xo_prob= 0.9, mut_prob= 0.3, elitism= True, tourn_size= 4, filename = 'best_fitness_test.csv')

# How the list, we will store as csv file, looks like:
print('----------------------------------------------------------------------------------------------')
print('-----------List with the best fitness, from each generation, from each run -------------------')
print(all_runs_best_fitness)

# based on the list, if we want to calculate the Average Best Fitness, instead of store all_runs_best_fitness, we can store this
print('-----------------------------------------------------------------')
print('-----------List with the Average Best Fitness -------------------')
abf = [np.mean([run[i] for run in all_runs_best_fitness]) for i in range(len(all_runs_best_fitness[0]))]
print(abf)

# Or the Median Best Fitness, best if we have outliers:
print('----------------------------------------------------------------')
print('-----------List with the Median Best Fitness -------------------')
mbf = [np.median([run[i] for run in all_runs_best_fitness]) for i in range(len(all_runs_best_fitness[0]))]
print(mbf)

# Calling run_evolution() and printing/storing the results
#output_to_console = False
#if output_to_console:
#    run_evolution()
#else:  # if want to store results to file
#    with open("output.cvs", "w") as file:
#        # Redirecting standard output to the file
#        sys.stdout = file
#        run_evolution()
#    # Restoring the standard output
#    sys.stdout = sys.__stdout__