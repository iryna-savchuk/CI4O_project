from random import uniform, choice, sample
from operator import attrgetter


def fps(population): # The individuals are selected based on their fitness scores
    """Fitness proportionate selection implementation.

    Args:
        population (Population): The population we want to select from.

    Returns:
        Individual: selected individual.
    """

    if population.optim == "max": # This function only implements the "maximization" scenario

        # Sum total fitness
        # the total sum of the fitness of all individuals in the population is calculated.
        total_fitness = sum([i.fitness for i in population])
        # This is like creating a roulette wheel where each individual occupies a portion of the wheel proportional to its fitness.
        # Get a 'position' on the wheel
        spin = uniform(0, total_fitness) # A random number between 0 and the total sum of fitness is generated. This can be likened to spinning the roulette wheel.
        position = 0
        # Find individual in the position of the spin
        # It then iterates over the population in order, keeping a running total of the fitnesses.
        # When the running total exceeds the random number, the current individual is selected.
        # individuals with higher fitness have a higher chance of being selected.
        for individual in population:
            position += individual.fitness
            if position > spin:
                return individual

    elif population.optim == "min":
        raise NotImplementedError

    else:
        raise Exception("No optimization specified (min or max).")


# Selection doesn't need to be changed as it doesn't directly operate on the individuals' representation.
# In tournament selection, a group of individuals (the tournament) is randomly chosen from the population,
# and the best individual within that group is selected as the winner.
# Tournament selection is a way to balance exploration and exploitation in genetic algorithms.
# By randomly selecting individuals and choosing the best among them,
# tournament selection promotes diversity and increases the chance of preserving good individuals in the population.
# tourn_size - the number of individuals in the tournament, was set a parameter to be set in the model
# it establish the selection pressure, how strong is the probability of the next individuals to be selected
# small value - low selection pressure - give more diversity and allow to scape from the local optimum
# large value - high selection pressure - it is more prompt to the only best survives
def tournament_sel(population, tourn_size):
    """Tournament selection implementation.

    Args:
        population (Population): The population we want to select from.
        size (int): Size of the tournament.

    Returns:
        Individual: The best individual in the tournament.
    """

    # Select individuals based on tournament size
    # with choice, there is a possibility of repetition in the choices,
    # so every individual has a chance of getting selected
    tournament = [choice(population.individuals) for _ in range(tourn_size)]

    # with sample, there is no repetition of choices
    # tournament = sample(population.individuals, size)
    if population.optim == "max":
        return max(tournament, key=attrgetter("fitness"))
    if population.optim == "min":
        return min(tournament, key=attrgetter("fitness"))