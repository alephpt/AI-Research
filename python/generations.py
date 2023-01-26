import random
import math
import matplotlib.pyplot as plt

# Define the population size and the number of generations
pop_size = 5000
num_generations = 500

# Define the mutation rate
mutation_rate = 0.001

# Define the fitness function
def fitness(individual):
    return sum(individual)

# Generate the initial population
population = [[random.randint(0, 1) for _ in range(10)] for _ in range(pop_size)]

# Track the best fitness value for each generation
best_fitness_values = []

# Run the genetic algorithm
for generation in range(num_generations):
    # Evaluate the fitness of each individual
    fitness_values = [fitness(individual) for individual in population]
    
    # Select the parents for the next generation
    parents = [population[i] for i in range(pop_size) if fitness_values[i] > random.random() * random.random()]
    
    # Crossover and mutation
    children = []
    while len(children) < pop_size:
        parent1 = random.choice(parents)
        parent2 = random.choice(parents)
        child = parent1[:5] + parent2[5:]
        for i in range(len(child)):
            if random.random() < mutation_rate:
                child[i] = 1 - child[i]
        children.append(child)
    population = children

    
    # Track the best fitness value for this generation
    best_fitness_values.append(max(fitness_values))

# Plot the best fitness values for each generation
plt.plot(best_fitness_values)
plt.xlabel('Generation')
plt.ylabel('Best Fitness')
plt.show()
