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
    nn.compile(loss='sparse_categorical_crossentropy', metrics=["categorical_accuracy"])

    train_loss, train_acc = nn.evaluate(train_images, train_labels, verbose=0)
    return train_acc


# Monkey patching
Individual.get_fitness = get_fitness


def run_evolution(runs, pop_size, gens, select, crossover, mutate, xo_prob, mut_prob, elitism, output_dir, **kwargs):
    for r in range(runs):
        print(f"RUN #{r + 1}")

        # Generating Initial Population
        pop = Population(
            size=pop_size,
            sol_input=28 * 28,
            sol_hidden=512,
            sol_output=10,
            valid_range=[-1, 1],
            optim="max")
        print(pop.individuals)

        # Running evolution iterations
        print('Evolving...')
        best_fitness_lst = pop.evolve(gens=gens,
                                      select=select, crossover=crossover, mutate= mutate,
                                      xo_prob=xo_prob, mut_prob=mut_prob,
                                      elitism=elitism, **kwargs)

        print(best_fitness_lst)


# we can define the name of the file, to have different files with different parameters
run_evolution(runs=2, pop_size=5, gens=3,
              select= fps, #tournament_sel,
              crossover=arithmetic_xo,
              mutate=arithmetic_mutation,
              tourn_size=3,
              xo_prob=0.9, mut_prob= 0.1, elitism=True,
              output_dir='output/',
              )

'4_50_50_tournament_sel_arithmetic_xo_mutation_90_30_4.csv'