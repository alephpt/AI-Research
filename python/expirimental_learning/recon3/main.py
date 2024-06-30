import pygame
import random
from grid import Grid
from agent import Agent
from genetic import GeneticAlgorithm

def calculate_statistics(agents):
    ages = [agent.age for agent in agents]
    avg_age = sum(ages) / len(ages) if ages else 0
    max_age = max(ages) if ages else 0
    min_age = min(ages) if ages else 0
    return avg_age, max_age, min_age

def main():
    pygame.init()
    screen = pygame.display.set_mode((1200, 800))
    pygame.display.set_caption('Genetic Algorithm Simulation')
    clock = pygame.time.Clock()

    grid = Grid(100, 100)
    ga = GeneticAlgorithm()

    for i in range(100):
        x, y = random.randint(0, 99), random.randint(0, 99)
        sex = 'M' if random.random() < 0.5 else 'F'
        grid.add_agent(Agent(i, x, y, sex))

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        grid.update()

        if not any(agent.alive for agent in grid.agents):
            grid.agents = ga.new_generation(grid.agents)

        avg_age, max_age, min_age = calculate_statistics(grid.agents)
        grid.draw(screen, ga.generation, avg_age, max_age, min_age)
        pygame.display.flip()
        clock.tick(30)

    pygame.quit()

if __name__ == '__main__':
    main()
