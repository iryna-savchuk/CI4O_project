from charles.charles import Population, Individual
from data.mnist_data import train_images, train_labels
from charles.crossover import cycle_xo, pmx, single_point_co, arithmetic_xo
from charles.mutation import swap_mutation, inversion_mutation
from charles.selection import tournament_sel, fps
import numpy as np
from tensorflow.keras import Sequential, layers
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

def run_evolution():
    # Generating Population
    pop = Population(
        size=20,
        sol_input=28 * 28,
        sol_hidden=512,
        sol_output=10,
        valid_range=[-1, 1],
        optim="max")

    print(pop.individuals)
    # print(pop.individuals[0].representation)

    # Running evolution iterations
    print('Evolving...')
    pop.evolve(gens=100,
               select=tournament_sel, mutate=inversion_mutation, crossover=arithmetic_xo,
               mut_prob=0.3, xo_prob=0.9,
               elitism=True, tourn_size =4)


# Calling run_evolution() and printing/storing the results
output_to_console = True
if output_to_console:
    run_evolution()
else:  # if want to store results to file
    with open("output.txt", "w") as file:
        # Redirecting standard output to the file
        sys.stdout = file
        run_evolution()
    # Restoring the standard output
    sys.stdout = sys.__stdout__