from charles.charles import Population, Individual
from data.tsp_data import distance_matrix
from copy import deepcopy
from charles.crossover import cycle_xo, pmx
from charles.mutation import swap_mutation, inversion_mutation
from charles.selection import tournament_sel, fps
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
    fitness = 1
    test_loss, test_acc = nn.evaluate(train_images, train_labels)
    return test_loss


# Monkey patching
Individual.get_fitness = get_fitness


pop = Population(
    size=2,
    optim="min")


print(pop.individuals)
#print(pop.individuals[0].representation)

#pop.evolve(gens=5, select=tournament_sel, mutate=inversion_mutation, crossover=pmx,
#           mut_prob=0.05, xo_prob=0.9, elitism=True)


