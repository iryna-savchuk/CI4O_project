import numpy as np
import random
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.utils import to_categorical
import tensorflow as tf


# Load the CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Preprocess the data, normalizing it
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# Convert class labels to one-hot encoded vectors
num_classes = 10
y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)

# Define the CNN architecture
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(10, activation='softmax'))

# Define fitness function
def get_fitness(weights):
    model.set_weights(weights)
    predictions = model.predict(x_train)
    loss = tf.keras.losses.categorical_crossentropy(y_train, predictions)
    mean_loss = tf.reduce_mean(loss)
    return 1 / mean_loss.numpy()  # Return inverse of mean loss as fitness


# Genetic Algorithm parameters
population_size = 20
mutation_rate = 0.01
num_generations = 10

# Generate initial population
population = []
for _ in range(population_size):
    weights = model.get_weights()
    population.append(weights)

# Main loop for the genetic algorithm
for generation in range(num_generations):
    # Evaluate fitness for each individual in the population
    fitness_scores = []
    for individual in population:
        fitness_scores.append(get_fitness(individual))

    # Select parents for mating (tournament selection)
    parents = []
    for _ in range(population_size):
        random_indices = random.sample(range(population_size), 2)
        parent1 = population[random_indices[0]]
        parent2 = population[random_indices[1]]
        if fitness_scores[random_indices[0]] > fitness_scores[random_indices[1]]:
            parents.append(parent1)
        else:
            parents.append(parent2)

    # Create offspring through crossover and mutation
    offspring = []
    for parent in parents:
        child = parent.copy()
        for i in range(len(child)):
            if random.random() < mutation_rate:
                mutation = np.random.randn(*child[i].shape) * 0.1  # Gaussian mutation
                child[i] += mutation
        offspring.append(child)

    # Replace old population with offspring
    population = offspring

    # Print the best fitness score in the current generation
    best_fitness = np.max(fitness_scores)
    print(f"Generation {generation + 1}: Best Fitness = {best_fitness}")




