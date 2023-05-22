from charles.charles import Population, Individual
from data.mnist_data import train_images, train_labels
from copy import deepcopy
from charles.crossover import cycle_xo, pmx, single_point_co, arithmetic_xo
from charles.mutation import swap_mutation, inversion_mutation
from charles.selection import tournament_sel, fps

import numpy as np
from tensorflow import keras
from tensorflow.keras import layers


def get_fitness(self):
    input = self.representation["input"]
    hidden = self.representation["hidden"]
    output = self.representation["output"]

    # Defining the architecture of a neural network
    nn = keras.Sequential([
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
    nn.compile(metrics=["accuracy"])

    _, train_acc = nn.evaluate(train_images, train_labels, verbose=0)
    return train_acc


# Monkey patching
Individual.get_fitness = get_fitness

pop = Population(
    size=30,
    sol_input=28*28,
    sol_hidden=512,
    sol_output=10,
    valid_range=[-1, 1],
    optim="max")

print(pop.individuals)
#print(pop.individuals[0].representation)


pop.evolve(gens=20,
           select=tournament_sel, mutate=inversion_mutation, crossover=arithmetic_xo,
           mut_prob=0.3, xo_prob=0.9,
           elitism=True)


