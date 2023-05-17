from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.datasets import mnist
from keras.utils import to_categorical
import numpy as np
from deap import creator, base, tools, algorithms
from sklearn.model_selection import train_test_split

# Function to create model, CNN model using Keras
# with two sets of convolution + pooling layers, followed by a flattening step and two dense layers.
def create_cnn_model():
    #  initializes a new Sequential model.
    model = Sequential()
    # This line adds a 2D convolution layer to the model. The layer has 32 output filters, each with a size of 3x3.
    # # It uses the ReLU (Rectified Linear Unit) activation function, and expects input tensors of shape (28, 28, 1) - this is the shape of each input image.
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)))
    # adds a max pooling layer to the model. Max pooling is a downsampling strategy often used in convolutional neural networks.
    # The pool size of (2, 2) means it looks at 2x2 windows and keeps only the max value in each window.
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # adds another 2D convolution layer to the model, this time with 64 output filters.
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # This is needed because fully connected layers (like the upcoming Dense layer) expect 1D inputs, but the output of convolutional and pooling layers is 3D.
    model.add(Flatten())
    # adds a fully connected layer (Dense layer) with 128 neurons and ReLU activation to the model.
    model.add(Dense(128, activation='relu'))
    # adds another fully connected layer with 10 neurons and softmax activation. Softmax activation is often used in the final layer of a neural network model
    # for multi-class classification, as it outputs a probability distribution over the classes.
    model.add(Dense(10, activation='softmax'))
    #  This line configures the model for training. It uses categorical crossentropy as the loss function (which is suitable for multi-class classification),
    #  Adam as the optimizer (a variant of stochastic gradient descent), and it will report accuracy as the performance metric.
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# Function to take a list of weights and set them into layers of model
# It allows the algorithm to 'apply' an individual's genes to a model.
# After applying an individual's genes, the model can then be evaluated to see how good those genes are (i.e., how good the corresponding weights are).
def set_model_weights(individual, model):
    # initializes an empty list to store the reshaped weights.
    weights = []
    # keeps track of where in the individual's list of genes we are.
    start = 0

    for layer in model.layers:
        # This line gets the shape of each weight tensor in the current layer.
        layer_weights_shapes = [w.shape for w in layer.get_weights()]
        # This line calculates the total number of elements in each weight tensor.
        layer_weights_sizes = [np.prod(shape) for shape in layer_weights_shapes]
        # loop iterates over each weight tensor in the current layer.
        for size, shape in zip(layer_weights_sizes, layer_weights_shapes):
            # extracts the parameters for the current weight tensor from the individual's genes.
            params = individual[start:start + size]
            # reshapes the extracted parameters to match the weight tensor's shape, then adds them to the weights list.
            weights.append(np.array(params).reshape(shape))
            # increments the start index for the next iteration.
            start += size
    # sets the model's weights to the new weights.
    model.set_weights(weights)

# Function to create an individual
#  initializes an individual for the population in the genetic algorithm.
#  Each individual represents a possible solution (or a set of parameters/weights for the neural network model).
# this function creates a new individual with random genes (parameters).
# This individual can then be evaluated and used in the genetic algorithm's operations (like mutation and crossover).
def initIndividual(icls, total_params):
    # icls - This is a DEAP framework class that represents an individual in the population.
    # In this case, it is likely an alias for creator.Individual, which represents a single individual in the population.
    # total_params - This is the total number of parameters in the model.
    # This value is the sum of all the parameters (weights and biases) in the model layers.
    return icls(np.random.uniform(-1, 1, total_params))
    #  This creates an array of random numbers drawn from a uniform distribution between -1 and 1.
    #  The size of this array is total_params, so there's one random number for each parameter in the model.

# Function to create a fitness function
# This function is creating a fitness function that will be used to evaluate how well each individual (i.e., each set of weights for the neural network) performs.
# The fitness of each individual is the accuracy of the neural network model when the weights of the model are set to the weights represented by the individual.
def create_fitness(model, x_train, y_train, x_val, y_val):
    def fitness(individual):
        # This sets the weights of the model to be the weights represented by the individual. This allows us to evaluate how well this particular set of weights performs.
        set_model_weights(individual, model)
        # This evaluates the model on the training data and returns the loss and accuracy.
        # The loss is a measure of how well the model is predicting the training labels, and accuracy is the percentage of labels it gets correct.
        loss, accuracy = model.evaluate(x_train, y_train, verbose=0)
        # This does the same thing as the previous line, but for the validation data instead of the training data.
        val_loss, val_accuracy = model.evaluate(x_val, y_val, verbose=0)
        #  This returns the training accuracy and validation accuracy as a tuple. This is the fitness of the individual.
        #  The genetic algorithm will try to find the individual (i.e., the set of weights) that maximizes this fitness.
        return accuracy, val_accuracy
    return fitness

def main():
    # Load MNIST data
    # The MNIST dataset is loaded and reshaped to fit the requirements of the Keras Conv2D layer. The labels are one-hot encoded.
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)

    # Split train data into actual train and validation
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42)

    # Create model
    # is called to create the Convolutional Neural Network model that will be used.
    model = create_cnn_model()

    # Calculate total parameters
    # The total number of parameters (weights and biases) in the model is calculated. This is used to initialize the genetic algorithm population.
    total_params = sum([np.prod(w.shape) for layer in model.layers for w in layer.get_weights()])

    # Setting up the genetic algorithm: The DEAP library is used to set up the genetic algorithm. This includes creating the fitness and individual classes,
    # setting up the toolbox (which includes the genetic operators like mutation and crossover), and defining the fitness function.
    # Create types
    creator.create("FitnessMax", base.Fitness, weights=(1.0, 1.0))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    # Create toolbox
    toolbox = base.Toolbox()
    toolbox.register("attr_float", np.random.uniform, -1, 1)
    toolbox.register("individual", initIndividual, creator.Individual, total_params=total_params)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # Register evaluate function
    toolbox.register("evaluate", create_fitness(model, x_train, y_train, x_val, y_val))

    # Register mate, mutate, and select operators
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.1)
    toolbox.register("select", tools.selTournament, tournsize=3)

    #Defining statistics: DEAP's Statistics class is used to keep track of various statistics of the population through generations.
    #Here, it records the average, standard deviation, minimum, and maximum fitness in the population.
    # Initialize statistics object
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    # Initialize hall of fame and population
    # stores the best individual found throughout the evolution, while the population is a list of individuals.
    hof = tools.HallOfFame(1)
    pop = toolbox.population(n=10)

    # Evolve the population
    # The genetic algorithm is run using DEAP's eaSimple function, which implements the simplest form of a genetic algorithm.
    # This function evolves the population over 40 generations, with a crossover probability of 0.5 and a mutation probability of 0.2.
    pop, log = algorithms.eaSimple(pop, toolbox, cxpb=0.9, mutpb=0.1, ngen=40, stats=stats, halloffame=hof, verbose=True)

if __name__ == "__main__":
    main()

