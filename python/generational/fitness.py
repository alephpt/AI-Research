import pygame
import random
import math


SCREEN_WIDTH = 1200
SCREEN_HEIGHT = 800
INIT_POP_N = 10
INIT_SIZE = 5
total_population = INIT_POP_N

pygame.init()
pygame.font.init()
pygame.display.set_caption("Genetic Learning")
text = pygame.font.SysFont('Comic Sans MS', 16)
clock = pygame.time.Clock()
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))


def maxFood():
    global total_population
    return math.floor(math.e ** -(total_population ** 0.25 / 2) * total_population)

class Food():
    def __init__(self):
        self.x = random.random() * SCREEN_WIDTH
        self.y = random.random() * SCREEN_HEIGHT
        self.r = INIT_SIZE
        self.color = (0, 255, 0)
    
    def draw(self):
        pygame.draw.rect(screen, self.color, (self.x, self.y, self.r, self.r), 1)

class Individual():
    actions = {
        0: [-1, 0], # left
        1: [1, 0],  # right
        2: [0, 1],  # up
        3: [0, -1]  # down
    }
    
    def __init__(self):
        self.x = random.random() * SCREEN_WIDTH
        self.y = random.random() * SCREEN_HEIGHT
        self.r = INIT_SIZE
        self.color = (255, 0, 0)
        self.energy_total = 2000
        self.energy = 1000
        self.fitness = 0
        self.success = False
    
    def energyConservation(self):
        return self.energy / self.energy_total
    
    def targetFound(self, target):
        distance = math.sqrt((target.x - self.x) ** 2 + (target.y - self.y) ** 2)
        return distance < (self.r + target.r)
    
    def navigate(self, target):
        
    
    def findTarget(self, target):
        init_distance = math.sqrt((target.x - self.x) ** 2 + (target.y - self.y) ** 2)
        self.navigate(self, target)
        new_distance = math.sqrt((target.x - self.x) ** 2 + (target.y - self.y) ** 2)
        reward += init_distance - new_distance
        
        if self.targetFound(self, target):
            reward = reward * self.energyConservation()
    
    def draw(self):
        pygame.draw.circle(screen, self.color, (self.x, self.y), self.r)

class Society():
    def __init__(self): 
        self.population = [Individual() for _ in range(INIT_POP_N)]

class World():
    def __init__(self):
        self.society = Society();
        self.food = [Food() for _ in range(maxFood())]
    
    def draw(self):
        for individual in self.society.population:
            individual.draw()
        
        for edible in self.food:
            edible.draw()

def main():
    running = True
    world = World()

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        screen.fill((0, 0, 0))

        world.draw()
        
        pygame.display.update()
        clock.tick(60)


if __name__ == "__main__":
    main()