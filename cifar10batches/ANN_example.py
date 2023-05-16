import random
import numpy as np
from numpy import load

# Activation Functions:
# computes the sigmoid activation function for a given input x.
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
# Softmax
def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)
# Relu
def relu(x):
    return np.maximum(0, x)

# the creation of the class “genetic_algorithm” that holds all the functions that concerns the genetic algorithm and how it is supposed to function.
# The main function is the execute function, that takes pop_size,generations,threshold,X,y,network as parameters.
# pop_size (population size), generations (number of generations/epochs), threshold (the desired loss value), X (input data), y (target labels),
# # and network (the structure of the neural network).

class genetic_algorithm:

    def execute(pop_size, generations, threshold, X, y, network):
        class Agent: # Each agent represents an individual in the population and contains a neural network.
            def __init__(self, network):
                # describes the initialization of the weights and the propagation of the network for each agent’s neural network.
                # The neural network is defined within the Agent class. It initializes the weights randomly and stores the activation functions for each layer.
                class neural_network:
                    def __init__(self, network):
                        self.weights = [] # This initializes an empty list weights, which will store the weights of each layer in the neural network.
                        self.activations = [] # an empty list activations, which will store the activation functions for each layer in the neural network.
                        for layer in network: # iterates over each layer in the network list, which represents the structure of the neural network.
                            if layer[0] != None: # This condition checks if the first element of the current layer is not None. If it's not None, it means that the layer has a specified input size.
                                input_size = layer[0] # This assigns the input size of the current layer to the variable input_size.
                            else: # f the first element of the current layer is None, it means that the layer doesn't have a specified input size and should inherit it from the previous layer.
                                input_size = network[network.index(layer) - 1][1] # This assigns the input size of the current layer to the output size of the previous layer.
                            output_size = layer[1] # This assigns the output size of the current layer to the variable output_size.
                            activation = layer[2] # This assigns the activation function of the current layer to the variable activation
                            self.weights.append(np.random.randn(input_size, output_size)) # generates random weights for the current layer using NumPy's np.random.randn function. The weights are initialized with a normal distribution and added to the weights list.
                            self.activations.append(activation) # adds the activation function for the current layer to the activations list.

                    # performs forward propagation through the network given input data
                    # takes the input data, performs forward propagation by iterating through each layer of the neural network,
                    # calculating the weighted sum (z), applying the activation function (a),
                    # and passing the output of one layer as the input to the next layer.
                    def propagate(self, data):
                        input_data = data
                        for i in range(len(self.weights)): # loop iterates over each layer in the neural network.
                            z = np.dot(input_data, self.weights[i]) # calculates the weighted sum of the input data (input_data) by performing a dot product between the input data and the weights of the current layer (self.weights[i])
                            a = self.activations[i](z) # This applies the activation function (self.activations[i]) corresponding to the current layer to the weighted sum (z).
                            input_data = a # This updates the input_data variable with the output (a) of the current layer, which becomes the input for the next layer in the loop.
                        yhat = a # This assigns the final output (a) of the last layer to the yhat variable. It represents the predicted output of the neural network.
                        return yhat # This returns the predicted output (yhat) of the neural network.

                self.neural_network = neural_network(network)
                self.fitness = 0

        # creates a population of agents with random neural networks based on the specified network structure.
        # generates a population of agents for the genetic algorithm by creating Agent objects.
        # The number of agents in the population is determined by the population parameter,
        # and each agent is initialized with the provided network structure.
        def generate_agents(population, network):
            # population (representing the size of the population) and network (representing the structure of the neural network).
            return [Agent(network) for _ in range(population)];
            #  uses a list comprehension to create a list of Agent objects.
            #  It iterates population times (the range of population) and creates a new Agent object for each iteration,
            #  passing the network structure as an argument to the Agent constructor. The resulting list of Agent objects is then returned.

        # calculates the fitness of each agent in the population. It propagates the input data through the agent's neural network
        # and computes the mean squared error between the predicted output (yhat) and the actual output (y).
        def fitness(agents, X, y): # agents (a list of agent objects), X (the input data), and y (the target values).
            for agent in agents: # This loop iterates over each agent in the agents list.
                yhat = agent.neural_network.propagate(X) # This line uses the propagate method of the agent's neural network to propagate the input data X through the network and obtain the predicted output yhat.
                cost = (yhat - y) ** 2 # This calculates the squared difference between the predicted output yhat and the target values y. It represents the cost or error for each agent.
                agent.fitness = sum(cost)  # This assigns the sum of the squared differences (cost) to the fitness attribute of the agent. The fitness value represents how well the agent's neural network performs on the given input data.
            return agents # the function returns the updated list of agents, where each agent now has its fitness value assigned.

        #  sorts the agents based on their fitness and selects the top 20% best-performing agents to keep for the next generation.
        # selecting a subset of agents from the population based on their fitness values.
        # This process helps to prioritize and retain the best-performing agents for the next steps of the genetic algorithm, such as crossover and mutation.
        def selection(agents):
            agents = sorted(agents, key=lambda agent: agent.fitness, reverse=False) # sorts the agents list based on their fitness values in ascending order. The key parameter specifies that the sorting should be based on the fitness attribute of each agent. The reverse parameter is set to False to sort the agents in ascending order.
            # print('\n'.join(map(str, agents)))
            agents = agents[:int(0.2 * len(agents))]  # selects the top 20% of agents from the sorted list. It uses list slicing to create a new list agents containing only the first 20% of the sorted agents.
            return agents  # the function returns the selected subset of agents.

        # converts a flattened array back into its original shape based on the provided shapes.
        #
        def unflatten(flattened, shapes):
            newarray = []
            index = 0
            for shape in shapes:
                size = np.product(shape)
                newarray.append(flattened[index: index + size].reshape(shape))
                index += size
            return newarray

        # performs crossover between pairs of agents to create offspring. Randomly selected genes (weights)
        # from the parents are exchanged to create new child agents.
        def crossover(agents, network, pop_size):
            offspring = []
            for _ in range((pop_size - len(agents)) // 2):
                parent1 = random.choice(agents)
                parent2 = random.choice(agents)
                child1 = Agent(network)
                child2 = Agent(network)

                shapes = [a.shape for a in parent1.neural_network.weights]

                genes1 = np.concatenate([a.flatten() for a in parent1.neural_network.weights])
                genes2 = np.concatenate([a.flatten() for a in parent2.neural_network.weights])

                split = random.randint(0, len(genes1) - 1)
                child1_genes = np.array(genes1[0:split].tolist() + genes2[split:].tolist())
                child2_genes = np.array(genes2[0:split].tolist() + genes1[split:].tolist())

                child1.neural_network.weights = unflatten(child1_genes, shapes)
                child2.neural_network.weights = unflatten(child2_genes, shapes)

                offspring.append(child1)
                offspring.append(child2)
            agents.extend(offspring)
            return agents

        # introduces random mutations to the agents' weights. Each weight has a 10% chance of being randomly changed.
        def mutation(agents):
            for agent in agents:
                if random.uniform(0.0, 1.0) <= 0.1:
                    weights = agent.neural_network.weights
                    shapes = [a.shape for a in weights]
                    flattened = np.concatenate([a.flatten() for a in weights])
                    randintg = random.randint(0, len(flattened) - 1)
                    flattened[randintg] = np.random.randn()
                    newarray = []
                    indeweights = 0
                    for shape in shapes:
                        size = np.product(shape)
                        newarray.append(flattened[indeweights: indeweights + size].reshape(shape))
                        indeweights += size
            agent.neural_network.weights = newarray
            return agents
        # The main loop in the execute function runs for the specified number of generations.
        # It generates the initial population, evaluates fitness, performs selection, crossover, and mutation operations.
        # It also checks if the threshold has been reached and prints information about the best and worst agent in each generation.
        for i in range(generations):
            if i == 0:
                agents = generate_agents(pop_size, network)

            agents = fitness(agents, X, y)
            agents = selection(agents)
            agents = crossover(agents, network, pop_size)
            agents = mutation(agents)
            agents = fitness(agents, X, y)
            if any(agent.fitness < threshold for agent in agents):
                print('Threshold met at generation ' + str(i) + ' !')

            if i % 10 == 0:
                print('Generation', str(i), ':')
                print('The Best agent has fitness ' + str(agents[0].fitness) + 'at generation ' + str(i) + '.')
                print('The Worst agent has fitness ' + str(agents[-1].fitness) + 'at generation ' + str(i) + '.')
        # function returns the best agent from the last generation, which has the lowest fitness value.
        return agents[0]


#############################################################################
#################### Some attempts for our dataset ##########################
#############################################################################
#def unpickle(file):
#    import pickle
#    with open(file, 'rb') as fo:
#        dict = pickle.load(fo, encoding='bytes')
#    return dict

#batch_1 = unpickle('data_batch_1')
#X = np.array(batch_1[b'data'][:3])
#X = X.astype('float32')
#X /=255
#y = np.array(batch_1[b'labels'][:3])
#y = np.array([[0, 1, 1]]).T

#X_train_matrix = load('X_train_matrix.npz')
#X_train_matrix = X_train_matrix['arr_0']
#X_train_matrix

#y_train_matrix = load('y_train_matrix.npz')
#y_train_matrix = y_train_matrix['arr_0']
#y_train_matrix

#X = X_train_matrix[:4]
#y = y_train_matrix[:4]


###########################################################################
##################### The given Example ###################################
###########################################################################

# Each row of the array corresponds to a data point with three features.
X = np.array([[0, 0, 1], [1, 1, 1], [1, 0, 1], [0, 1, 1]])
# It is a column vector with four rows, each row corresponding to the target value for the corresponding data point in X.
y = np.array([[0, 1, 1, 0]]).T
# This defines the structure of the neural network.
# It is a list of layers, where each layer is specified by a list [input_size, output_size, activation_function].
# the network has two layers: the input layer with three neurons, a hidden layer with ten neurons using the sigmoid activation function,
# and an output layer with one neuron also using the sigmoid activation function.
network = [[3, 10, sigmoid], [None, 1, sigmoid]]
# This assigns the genetic_algorithm class to the variable ga.
ga = genetic_algorithm
# This executes the genetic algorithm by calling the execute method of the genetic_algorithm class.
# It creates a population of 100 agents and runs the genetic algorithm for 100 generations
# The threshold is set to 0.1. The input data X, target values y, and the network structure are also passed as arguments
# The returned agent represents the best agent found by the genetic algorithm.
agent = ga.execute(100, 100, 0.1, X, y, network)
# assigns the weights of the neural network of the best agent to the variable weights.
weights = agent.neural_network.weights
# This prints the fitness value of the best agent, which represents how well the neural network performs on the provided data.
print(agent.fitness)
# This uses the neural network of the best agent to propagate the input data X through the network and obtain the predicted output. It prints the predicted output values.
print(agent.neural_network.propagate(X))




