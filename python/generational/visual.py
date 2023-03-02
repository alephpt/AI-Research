import pygame
import random

SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
POPULATION_COUNT = 100
MUTATION_RATE = 0.05
GENERATIONS = 500

pygame.init()
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Generational Learning")
clock = pygame.time.Clock()
running = True

class Individual:
    def __init__(self):
        self.x = random.randint(0, SCREEN_WIDTH)
        self.y = random.randint(0, SCREEN_HEIGHT)
        self.color = (random.randint(0, 122), random.randint(0, 255), random.randint(0, 255))

    def mutate(self):
        self.x += random.randint(-10, 10)
        self.y += random.randint(-10, 10)
        
        # if mate is found then change color and do something else

    def draw(self):
        pygame.draw.circle(screen, self.color, (self.x, self.y), 10)

class Population:
    def __init__(self):
        self.individuals = [Individual() for _ in range(POPULATION_COUNT)]
        self.generation = 1

    def draw(self):
        for individual in self.individuals:
            individual.draw()

    def mutate(self):
        for individual in self.individuals:
            if random.random() < MUTATION_RATE:
                individual.mutate()


population = Population()


while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    screen.fill((0, 0, 0))
    
    population.draw()
    population.mutate()

    pygame.display.update()
    clock.tick(60)

pygame.quit()
