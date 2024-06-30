# Hello Genetic Learning

## Description

### Overview
This Python script demonstrates a genetic algorithm approach to evolve a population of strings to match a target string.

### Algorithm Details
- **Target String:** "Hello, Genetic Learning!"
- **Population Size:** 100
- **Mutation Rate:** 0.01
- **Max Generations:** 10000

### Functions
- **random_string(length):** Generates a random string of specified length.
- **fitness(candidate):** Evaluates the fitness of a candidate string by comparing it with the target string.
- **mutate(candidate):** Introduces mutations in the candidate string based on the mutation rate.
- **crossover(parent1, parent2):** Performs crossover between two parent strings to generate a child string.

### Execution
1. Initializes a population of random strings.
2. Evolves the population across generations:
   - Sorts the population based on fitness.
   - Selects top performers as parents for the next generation.
   - Applies crossover and mutation operations to generate the next generation.
3. Stops when the target string is achieved or after reaching the maximum number of generations.

### Output
- Prints each generation's best match.
- Displays the generation count and the evolved string that matches the target.

## Usage
Run the script to observe the evolution process of the strings towards the target "Hello, Genetic Learning!".

## Dependencies
- Python 3.x
- No external libraries required beyond Python's standard library.

## Author
- Replace with author information or organization details if applicable.
