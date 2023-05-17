import tensorflow as tf
from tensorflow.keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.utils import np_utils
from selection import tournament_sel
from crossover import crossover
from mutation import mutation
from charles import Individual, Population
import numpy as np

# Define a simple CNN architecture:
def create_model(weights):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    model.set_weights(weights)
    return model

# In this example, calculate_fitness takes an individual and data as arguments, creates a model with the individual's weights,
# trains the model, and sets the individual's fitness to the negative validation loss (since we want to minimize loss but maximize fitness).
#Update the fitness function:
def get_fitness(individual, train_data, train_labels, validation_data, validation_labels):
    """Calculate the fitness of an individual."""
    # Create model
    model = Sequential()
    model.add(Dense(10, input_shape=(784,), activation='softmax'))

    # Set weights
    model.set_weights(individual.representation)

    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])

    # Train model
    model.fit(train_data, train_labels, epochs=1, verbose=0)

    # Evaluate model on validation data
    val_loss, val_acc = model.evaluate(validation_data, validation_labels, verbose=0)

    # Fitness is negative validation loss so that lower loss corresponds to higher fitness
    individual.fitness = -val_loss

    return individual

# Monkey patch the get_fitness method
Individual.get_fitness = get_fitness

# Adam is being used as an optimizer in the training of a model for each individual's set of weights
# This is different from using Genetic Algorithms (GAs) to find the optimal weights of the network.
# Each individual in the GA population represents a possible solution, i.e., a set of weights for the neural network.
# When calculating the fitness of an individual, we need to evaluate how well these weights perform in the neural network.
# To do this, we create a model with these weights, train the model on the training data, and evaluate the model's performance (validation loss) on the validation data.
# The model training process involves adjusting the weights slightly to minimize the loss on the training data.
# This is where the Adam optimizer comes in - it's used to make these adjustments during training.
# The better the weights of an individual (i.e., the lower the validation loss), the higher the fitness of the individual in the GA.
# The GA then uses this fitness to guide the search for better solutions (sets of weights).
# The GA continues to evolve the population over generations, promoting the survival and reproduction of individuals with higher fitness,
# until it finds the optimal weights for the neural network.

# The Adam optimizer is used in the model training process to fine-tune the weights, while the GA is used to search the weight space for the optimal weights.
# The GA's fitness function is essentially evaluating how well the model with a certain set of weights performs on the validation data,
# and this involves training the model on the training data, which requires an optimizer like Adam.

# Load MNIST data
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Preprocess input data
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

# Preprocess class labels
Y_train = np_utils.to_categorical(y_train, 10)
Y_test = np_utils.to_categorical(y_test, 10)

# Define the shape of the layers of the model
layer_shapes = [(28, 28, 1, 32), (32,), (12*12*32, 128), (128,), (128, 10), (10,)]

# Create a population
pop = Population(size=10, optim='min', train_data=X_train, train_labels=Y_train, validation_data=X_test, validation_labels=Y_test, layer_shapes=layer_shapes)

# Train the population
pop.evolve(gens=100,
           xo_prob=0.8,
           mut_prob=0.2,
           select=tournament_sel,
           mutate=mutation,
           crossover=crossover,
           elitism=True)

# After the training, you can get the best individual with:
best_individual = min(pop.individuals, key=attrgetter("fitness"))
