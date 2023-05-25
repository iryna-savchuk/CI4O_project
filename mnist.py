from charles.charles import Population, Individual
from data.mnist_data import train_images, train_labels
from charles.crossover import cycle_xo, pmx, single_point_co, arithmetic_xo
from charles.mutation import swap_mutation, inversion_mutation, arithmetic_mutation
from charles.selection import tournament_sel, fps
import numpy as np
from tensorflow.keras import Sequential, layers
import os
import csv


# Defining parameters for Image Classification problem
INPUT_SIZE = 28 * 28
HIDDEN_SIZE = 512
OUTPUT_SIZE = 10
ACTIVATION_1 = "relu"
ACTIVATION_2 = "softmax"
ACCURACY = 'categorical_accuracy'


def get_fitness(self):
    """Method to calculate Fitness for MNIST image classification problem.

    Note:
        This function is intended to be monkey patched as the 'get_fitness' method into Individual class.
    Returns:
        float: Accuracy computed for a given instance of Individual class.
        In case of MNIST image classification, this function returns categorical_accuracy
        calculated on the training dataset (60,000 handwritten digits).
    """
    input = self.representation["input"]
    hidden = self.representation["hidden"]
    output = self.representation["output"]

    # Defining the architecture of a neural network
    nn = Sequential([
        layers.Dense(hidden, activation=ACTIVATION_1, input_shape=(input,)),
        layers.Dense(output, activation=ACTIVATION_2)
    ])

    current_weights = np.array(self.representation["weights"])
    weights_1 = current_weights[:input * hidden].reshape(input, hidden)
    biases_1 = current_weights[input * hidden:input * hidden + hidden].reshape(hidden, )
    weights_2 = current_weights[input * hidden + hidden:input * hidden + hidden + hidden * output].reshape(hidden, output)
    biases_2 = current_weights[input * hidden + hidden + hidden * output:].reshape(output, )

    # Setting the weights of the model
    nn.set_weights([weights_1, biases_1, weights_2, biases_2])

    # Compiling the model
    nn.compile(metrics=[ACCURACY])

    # Evaluating the model on training dataset
    train_loss, train_acc = nn.evaluate(train_images, train_labels, verbose=0)
    return train_acc


# Monkey patching
Individual.get_fitness = get_fitness


def run_evolution(runs, pop_size, gens, select, crossover, mutate, xo_prob, mut_prob, elitism, output_file, **kwargs):
    """
    Function to perform a certain number of runs for the evolution algorithm of specific configuration.
    During the function execution, details about the best fitness values for each generation are printed out
    and progress bars are shown to track iterations.
    Upon each run completion, the results are appended into 'output_file'.

    Args:
        runs (int): Number of runs to perform evolution algorithm with the specified parameters
        pop_size (int): Population size
        gens (int): The number of generations to evolve the population for
        select (func): The selection method to use
        crossover (func): The crossover method to use
        mutate (func): The mutation method to use
        xo_prob (float): Crossover probability
        mut_prob (float): Mutation probability
        elitism (bool): Whether to use elitism
        output_file (str): File name to store the output results
        **kwargs: Optionally, may contain 'tourn_size' (int), when selection is 'tournament_sel'

    """

    for r in range(runs):
        print(f"RUN #{r + 1}")

        # Generating Initial Population
        pop = Population(
            size=pop_size,
            sol_input=INPUT_SIZE,
            sol_hidden=HIDDEN_SIZE,
            sol_output=OUTPUT_SIZE,
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

        # Merging configuration and results of the current run into the 'output_row' array
        output_row = [r+1, runs, pop_size, gens, select.__name__, crossover.__name__, mutate.__name__, xo_prob, mut_prob]
        if (select.__name__ == 'tournament_sel') and ('tourn_size' in kwargs):
            output_row.append(kwargs['tourn_size'])
        elif (select.__name__ == 'tournament_sel') and ('tourn_size' not in kwargs):
            output_row.append(4)
        else:
            output_row.append("")
        output_row.append(best_fitness_lst)

        # Storing results of the current run into the 'output_file'
        if not os.path.exists(output_file):  # File does not exist, create it and write headers
            with open(output_file, 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(["run_number", "total_runs", "pop_size", "gens", "select", "crossover", "mutate",
                                 "xo_prob", "mut_prob", "tourn_size", "best_fitness_lst"])
        # Append new row of data to the 'output_file'
        with open(output_file, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(output_row)


###############################################################
###### Running evolution with the desired configurations ######
###############################################################

# TOURNAMENT SELECTION
# tourn_size = 4 (default)
#1
run_evolution(runs=10, pop_size=20, gens=20,
              select=tournament_sel,
              crossover=arithmetic_xo,
              mutate=arithmetic_mutation,
              xo_prob=0.9, mut_prob=0.1, elitism=True,
              output_file='output/10_runs.csv')

#2
run_evolution(runs=10, pop_size=20, gens=20,
              select=tournament_sel,
              crossover=arithmetic_xo,
              mutate=inversion_mutation,
              xo_prob=0.9, mut_prob=0.1, elitism=True,
              output_file='output/10_runs.csv')

#3
run_evolution(runs=10, pop_size=20, gens=20,
              select=tournament_sel,
              crossover=single_point_co,
              mutate=arithmetic_mutation,
              xo_prob=0.9, mut_prob=0.1, elitism=True,
              output_file='output/10_runs.csv')

#4
run_evolution(runs=10, pop_size=20, gens=20,
              select=tournament_sel,
              crossover=single_point_co,
              mutate=inversion_mutation,
              xo_prob=0.9, mut_prob=0.1, elitism=True,
              output_file='output/10_runs.csv')

# tourn_size = 2
#5
run_evolution(runs=10, pop_size=20, gens=20,
              select=tournament_sel,
              crossover=arithmetic_xo,
              mutate=arithmetic_mutation,
              xo_prob=0.9, mut_prob=0.1, elitism=True,
              output_file='output/10_runs.csv',
              tourn_size=2)

#6
run_evolution(runs=10, pop_size=20, gens=20,
              select=tournament_sel,
              crossover=arithmetic_xo,
              mutate=inversion_mutation,
              xo_prob=0.9, mut_prob=0.1, elitism=True,
              output_file='output/10_runs.csv',
              tourn_size=2)

#7
run_evolution(runs=10, pop_size=20, gens=20,
              select=tournament_sel,
              crossover=single_point_co,
              mutate=arithmetic_mutation,
              xo_prob=0.9, mut_prob=0.1, elitism=True,
              output_file='output/10_runs.csv',
              tourn_size=2)

#8
run_evolution(runs=10, pop_size=20, gens=20,
              select=tournament_sel,
              crossover=single_point_co,
              mutate=inversion_mutation,
              xo_prob=0.9, mut_prob=0.1, elitism=True,
              output_file='output/10_runs.csv',
              tourn_size=2)


# Fitness proportionate selection (FPS)
#9
run_evolution(runs=10, pop_size=20, gens=20,
              select=fps,
              crossover=arithmetic_xo,
              mutate=arithmetic_mutation,
              xo_prob=0.9, mut_prob=0.1, elitism=True,
              output_file='output/10_runs.csv')

#10
run_evolution(runs=10, pop_size=20, gens=20,
              select=fps,
              crossover=arithmetic_xo,
              mutate=inversion_mutation,
              xo_prob=0.9, mut_prob=0.1, elitism=True,
              output_file='output/10_runs.csv')

#11
run_evolution(runs=10, pop_size=20, gens=20,
              select=fps,
              crossover=single_point_co,
              mutate=arithmetic_mutation,
              xo_prob=0.9, mut_prob=0.1, elitism=True,
              output_file='output/10_runs.csv')

#12
run_evolution(runs=10, pop_size=20, gens=20,
              select=fps,
              crossover=single_point_co,
              mutate=inversion_mutation,
              xo_prob=0.9, mut_prob=0.1, elitism=True,
              output_file='output/10_runs.csv')

 