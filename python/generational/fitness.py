import pygame
import random
import math


SCREEN_WIDTH = 1600
SCREEN_HEIGHT = 1200
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
        "turn_right": (0, 1),
        "turn_left": (0, -1),
        "accelerate": (1, 0),
        "decelerate": (-1, 0),
        "accel_right": (1, 1),
        "accel_left": (1, -1),
        "decel_right": (-1, 1),
        "decel_left": (-1, -1)
    }
    
    def __init__(self):
        self.alive = True
        self.x = random.random() * SCREEN_WIDTH
        self.y = random.random() * SCREEN_HEIGHT
        self.velocity = 0
        self.direction = math.radians(random.randint(0, 360))
        self.r = INIT_SIZE
        self.color = (255, 0, 0)
        self.energy_total = 2000
        self.energy = 1000
        self.perspective = INIT_SIZE
        self.reward = 0
        self.success = False
    
    def energyConservation(self):
        return self.energy / self.energy_total

    def distance(a, b):
        return math.sqrt((b.x - a.x) ** 2 + (b.y - a.y) ** 2)
    
    def targetFound(self, target):
        return self.distance(target) < (self.r + target.r)
    
    def navigate(self, target):
        self.energy -= 1
        best_action = None
        max_fitness = 0
        current_x = self.x
        current_y = self.y
        current_vel = self.velocity
        current_dir = self.direction

        for action in self.actions:
            current_action = action
            current_direction = self.actions[current_action]

            self.velocity += current_direction[0]
            self.direction += current_direction[1]
            self.x += self.velocity * math.cos(self.direction)
            self.y += self.velocity * math.sin(self.direction)

            current_distance = self.distance(target)
            current_fitness = 1 / current_distance

            if max_fitness < current_fitness:
                max_fitness = current_fitness
                best_action = current_action
                print("best action: ", best_action)

            self.direction = current_dir
            self.velocity = current_vel
            self.x = current_x
            self.y = current_y

        self.velocity += self.actions[best_action][0]
        self.direction += self.actions[best_action][1]
        self.x += self.velocity * math.cos(self.direction)
        self.y += self.velocity * math.sin(self.direction)

        return
    
    def findTarget(self, target):
        init_distance = self.distance(target)
        self.navigate(target)
        new_distance = self.distance(target)
        self.reward += (init_distance - new_distance) * self.energyConservation()
        
        if self.targetFound(target):
            self.reward = (self.reward * 2) * self.energyConservation()
            self.energy += 10
            self.perspective = INIT_SIZE
            return True
        
        self.perspective -= init_distance - new_distance + 1
        return False

    def roam(self):
        self.velocity += random.randint(-1, 1)
        self.direction += random.randint(-1, 1)

        self.energy -= 1
        self.perspective += self.perspective

    def locateTarget(self, target):
        return self.distance(target) < self.perspective
    
    def die(self):
        self.velocity = 0
        self.color = (75, 75, 75)
        self.alive = False
        global total_population 
        total_population -= 1

    def draw(self):
        pygame.draw.circle(screen, self.color, (self.x, self.y), self.r)

class Society():
    evolution_rate = 0.05

    def __init__(self): 
        self.population = [Individual() for _ in range(INIT_POP_N)]

    def maintainHealth(self, food):
        for individual in self.population:
            if individual.alive:
                if individual.energy <= 0:
                    individual.die()
                    return

                if random.random() < self.evolution_rate:
                    target_located = False
                    for edible in food:
                        if individual.locateTarget(edible):
                            target_located = True
                            if individual.findTarget(edible):
                                food.remove(edible)
                
                    if not target_located:
                        individual.roam()

class World():
    def __init__(self):
        self.society = Society();
        self.food = [Food() for _ in range(maxFood())]
        self.food_delay = 0
    
    def maintainFoodSupply(self):
        if len(self.food) < maxFood():
            if self.food_delay >= (maxFood() / 2):
                self.food.append(Food())
                self.food_delay = 0
                return

            self.food_delay += 1

    def evolve(self):
        self.society.maintainHealth(self.food)
        self.society.maintainFoodSupply()

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
        world.evolve()
        
        pygame.display.update()
        clock.tick(60)


if __name__ == "__main__":
    main()