import pygame
import random
from grid import Grid
from unit import Unit
from genetic import GeneticAlgorithm

def calculate_statistics(units):
    ages = [unit.age for unit in units]
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
        grid.add_unit(Unit(i, x, y, sex))

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        grid.update()

        if not any(unit.alive for unit in grid.units):
            grid.units = ga.new_generation(grid.units)

        avg_age, max_age, min_age = calculate_statistics(grid.units)
        grid.draw(screen, ga.generation, avg_age, max_age, min_age)
        pygame.display.flip()
        clock.tick(30)

    pygame.quit()

if __name__ == '__main__':
    main()
