import random
import string

# Target string we want to guess
TARGET = "Hello, World!"
TARGET_LEN = len(TARGET)

# Genetic algorithm parameters
POPULATION_SIZE = 100
MUTATION_RATE = 0.01
MAX_GENERATIONS = 10000

def random_string(length):
    return ''.join(random.choice(string.printable[:95]) for _ in range(length))

def fitness(candidate):
    return sum(c1 == c2 for c1, c2 in zip(candidate, TARGET))

def mutate(candidate):
    candidate = list(candidate)
    for i in range(len(candidate)):
        if random.random() < MUTATION_RATE:
            candidate[i] = random.choice(string.printable[:95])
    return ''.join(candidate)

def crossover(parent1, parent2):
    idx = random.randint(0, len(parent1) - 1)
    child = parent1[:idx] + parent2[idx:]
    return child

def main():
    population = [random_string(TARGET_LEN) for _ in range(POPULATION_SIZE)]
    generation = 0

    while generation < MAX_GENERATIONS:
        population = sorted(population, key=lambda x: fitness(x), reverse=True)
        if fitness(population[0]) == TARGET_LEN:
            break
        
        next_generation = population[:POPULATION_SIZE // 2]

        while len(next_generation) < POPULATION_SIZE:
            parent1 = random.choice(population[:POPULATION_SIZE // 2])
            parent2 = random.choice(population[:POPULATION_SIZE // 2])
            child = mutate(crossover(parent1, parent2))
            next_generation.append(child)
        
        population = next_generation
        generation += 1
        print(f"Generation {generation}: Best match so far: {population[0]}")
    
    print(f"Target achieved in {generation} generations! The string is: {population[0]}")

if __name__ == "__main__":
    # Logging Start and finish times
    import time
    start_time = time.time()
    main()
    print("--- %s seconds ---" % (time.time() - start_time))
