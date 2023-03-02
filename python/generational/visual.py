import pygame
import random
import math

SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
POPULATION_COUNT = 100
MUTATION_RATE = 0.05
GENERATIONS = 500
INIT_RADIUS = 3

pygame.init()
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Generational Learning")
clock = pygame.time.Clock()
running = True

class Individual:
    def __init__(self, id):
        self.id = id
        self.partner = None
        self.taken = False
        self.is_female = random.choice([True, False])
        self.r = INIT_RADIUS
        self.x = random.randint(0, SCREEN_WIDTH)
        self.y = random.randint(0, SCREEN_HEIGHT)
        self.color = (random.randint(122, 255), random.randint(0, 73), random.randint(0, 73)) if self.is_female else (random.randint(0, 73), random.randint(0, 73), random.randint(122, 255)) 

    def mutate(self):
        self.x += random.randint(-3, 3)
        self.y += random.randint(-3, 3)
        
        # if mate is found then change color and do something else

    def draw(self):
        pygame.draw.circle(screen, self.color, (self.x, self.y), self.r)

class Population:
    def __init__(self):
        self.individuals = [Individual(i) for i in range(POPULATION_COUNT)]
        self.generation = 1

    def draw(self):
        for individual in self.individuals:
            individual.draw()

    def mutate(self):
        for individual in self.individuals:
            if random.random() < MUTATION_RATE:
                if individual.partner == None:
                    individual.mutate()
                    for others in self.individuals:
                        if others.partner == None:
                            if individual.is_female != others.is_female:
                                distance = math.sqrt((others.x - individual.x) ** 2 + (others.y -individual.y) ** 2)
                                if distance < (individual.r + others.r):
                                    individual.partner = others.id
                                    others.partner = individual.id
                                    individual.color = (individual.color[0], random.randint(122,255), individual.color[2])
                                    others.color = (others.color[0], random.randint(122,255), others.color[2])
                                    individual.taken = True
                                    others.taken = True


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
