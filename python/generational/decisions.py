import pygame
import random
from enum import Enum
import math

SCREEN_WIDTH = 1200
SCREEN_HEIGHT = 800
INIT_EMPLOYERS = 2
INIT_EMPLOYER_SIZE = 16
INIT_POPULATION = 4
INIT_INDIVIDUAL_RADIUS = 5
INIT_FOOD = 2
INIT_FOOD_SIZE = 4

pygame.init()
pygame.font.init()
pygame.display.set_caption("Genetic Desicion Making")
text = pygame.font.SysFont('Comic Sans MS', 16)
clock = pygame.time.Clock()
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))

ACTIONS = {
    "forward": (0.2, 0),
    "reverse": (-0.2, 0),
    "left": (0, -0.0349),
    "right": (0, 0.0349),
    "forward_left": (0.2, -0.0349),
    "forward_right": (0.2, 0.0349),    
    "reverse_left": (-0.2, -0.0349),
    "reverse_right": (-0.2, 0.0349),
}

TARGETS = {
    "working": 0,   # need money to get food, and to be more attractive
    "eating": 1,    # need food to have energy, and to not die
    "mating": 2     # need energy to mate, as well as attraction
}

class Work():
    def __init__(self):
        self.x = random.randint(INIT_EMPLOYER_SIZE, SCREEN_WIDTH - INIT_EMPLOYER_SIZE)
        self.y = random.randint(INIT_EMPLOYER_SIZE, SCREEN_HEIGHT - INIT_EMPLOYER_SIZE)
        self.s = INIT_EMPLOYER_SIZE
        self.c = (255, 125, 0)
    
    def draw(self):
        pygame.draw.rect(screen, self.c, (self.x, self.y, self.s, self.s), 1)

class Food():
    def __init__(self):
        self.x = random.randint(INIT_FOOD_SIZE, SCREEN_WIDTH - INIT_FOOD_SIZE)
        self.y = random.randint(INIT_FOOD_SIZE, SCREEN_HEIGHT - INIT_FOOD_SIZE)
        self.s = INIT_FOOD_SIZE
        self.c = (0, 183, 69)
        
    def draw(self):
        pygame.draw.rect(screen, self.c, (self.x, self.y, self.s, self.s), 1)

class Individual():
    def __init__(self, id):
        self.x = random.randint(INIT_INDIVIDUAL_RADIUS, SCREEN_WIDTH - INIT_INDIVIDUAL_RADIUS)
        self.y = random.randint(INIT_INDIVIDUAL_RADIUS, SCREEN_HEIGHT - INIT_INDIVIDUAL_RADIUS)
        self.s = INIT_INDIVIDUAL_RADIUS
        self.sex = id % 2   # 0 for female, 1 for male
        self.c = (255, 125, 125) if self.sex == 0 else (125, 125, 255)
    
    def draw(self):
        pygame.draw.circle(screen, self.c, (self.x, self.y), self.s)
        
class Society():
    def __init__(self):
        self.population = [Individual(n) for n in range(INIT_POPULATION)]
        self.employers = [Work() for _ in range(INIT_EMPLOYERS)]
        self.food_supply = [Food() for _ in range(INIT_FOOD)]
    
    def draw(self):
        for individual in self.population:
            individual.draw()
        
        for food in self.food_supply:
            food.draw()
        
        for employer in self.employers:
            employer.draw()

class World():
    def __init__(self):
        self.society = Society()
    
    def render(self):
        self.society.draw()


def main():
    world = World()
    running = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        screen.fill((0, 0, 0))
        
        world.render()
        
        pygame.display.update()
        clock.tick(15)

if __name__ == "__main__":
    main()