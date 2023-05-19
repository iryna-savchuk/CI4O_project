import numpy as np
from copy import deepcopy
import pandas as pd

from charles.charles import Population, Individual
from charles.selection import tournament_sel
from charles.crossover import arithmetic_xo, uniform_xo, arithmetic_xo_aula
from charles.mutation import swap_mutation, inversion_mutation


from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist


(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
# Reshape and normalize the input images
train_images = train_images.reshape(-1, 784) / 255.0
test_images = test_images.reshape(-1, 784) / 255.0

def get_fitness(self):
    # Defining the architecture of a neural network
    nn = keras.Sequential([
        layers.Dense(512, activation="relu", input_shape=(784,)),
        layers.Dense(10, activation="softmax")
    ])

    # Setting the weights of the model
    nn.set_weights(self.representation)

    # Compiling the model
    nn.compile(optimizer="rmsprop",
               loss="sparse_categorical_crossentropy",
               metrics=["accuracy"])

    # Train model
    nn.fit(train_images, train_labels, epochs=1, verbose=0)

    test_loss, test_acc = nn.evaluate(test_images, test_labels)

    return test_loss


# Monkey patching
Individual.get_fitness = get_fitness



all_runs_best_fitness = []

for i in range(4):
    print(f"Running GA iteration: {i + 1}")
    # Create a population
    pop = Population(size=4, optim='min')
    # Reset best_fitness_list for each run
    best_fitness_list = []
    # Train the population
    pop.evolve(gens=4,
               select=tournament_sel,
               mutate=inversion_mutation,
               crossover=arithmetic_xo,
               mut_prob=0.7,
               xo_prob=0.8,
               elitism=True,
               tourn_size = 4)
    all_runs_best_fitness.append(pop.best_fitnesses)

# Calculate the mean fitness for each generation across all runs
inversion_mean_fitness_per_generation = [np.mean([run[i] for run in all_runs_best_fitness]) for i in range(len(all_runs_best_fitness[0]))]
inversion_median_fitness_per_generation = [np.median([run[i] for run in all_runs_best_fitness]) for i in range(len(all_runs_best_fitness[0]))]