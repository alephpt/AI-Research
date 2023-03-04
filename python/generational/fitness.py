from functools import total_ordering
from re import L
import pygame
import random
import math


SCREEN_WIDTH = 1200
SCREEN_HEIGHT = 800
INIT_POP_N = 300
FOOD_SIZE = 5
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
        self.r = FOOD_SIZE
        self.color = (0, random.randint(178, 255), 0)
    
    def draw(self):
        pygame.draw.rect(screen, self.color, (self.x, self.y, self.r, self.r), 1)

class Individual():
    evolution_rate = 0.75
    actions = {
        "turn_right": (0, 0.25),
        "turn_left": (0, -0.25),
        "accelerate": (0.375, 0),
        "decelerate": (-0.375, 0),
        "accel_right": (0.375, 0.25),
        "accel_left": (0.375, -0.25),
        "decel_right": (-0.375, 0.25),
        "decel_left": (-0.375, -0.25)
    }
    
    def __init__(self):
        self.alive = True
        self.lifetime = 0
        self.x = random.random() * SCREEN_WIDTH
        self.y = random.random() * SCREEN_HEIGHT
        self.r = INIT_SIZE
        self.color = (random.randint(178, 255), 0, 0)
        self.direction = math.radians(random.randint(0, 360))  # inherits avg from parents
        self.velocity = 0
        self.acceleration = 0
        self.max_energy = 1000                                 # inherits average on next generation
        self.energy = 1000                                     # inherits average on next generation <- should be used for target
        self.perspective = INIT_SIZE                           #
        self.threshold_accel = 0                               # inherits avg_vel on next generation <- should be used for target
        self.threshold_velocity = 0                            # inherits avg_vel on next generation <- should be used for max + 1/2 accel
        self.threshold_energy = 1000                           # inherits (max + avg / 2) on next generation <- should be used for target
        self.threshold_energy_conserv = 0                      # TODO: maybe we create a reward to optimize maintaining energy at converservation level?
        self.threshold_perspective = INIT_SIZE                 # inherits from parents and becomes new default perspective
        self.avg_accel = 0
        self.avg_vel = 0
        self.avg_direction = 0
        self.avg_perspective = INIT_SIZE
        self.avg_energy_conservation = 1
        self.change_in_acceleration = 0
        self.change_in_direction = 0
        self.reward = 0
        self.fitness = 0
        self.targets_reached = 0
    
    def energyConservation(self):
        return math.sqrt((self.energy / self.max_energy) ** 2) 

    def distance(a, b):
        return math.sqrt((b.x - a.x) ** 2 + (b.y - a.y) ** 2)
    
    #randomly accelerates/decelerates
    def roam(self):
        self.acceleration = random.uniform(-1, 1)
        self.direction += random.uniform(-10, 10)
        self.energy -= self.perspective / INIT_SIZE
        self.reward -= self.perspective / INIT_SIZE
        self.perspective += self.perspective

    def targetFound(self, target):
        return self.distance(target) < (self.r + target.r)
    
    def determineMovement(self, target):
        self.energy -= 1
        best_action = None
        max_fitness = 0
        current_x = self.x
        current_y = self.y
        current_vel = self.velocity
        current_dir = self.direction

        for action in self.actions:
            current_action = action
            current_movement = self.actions[current_action]

            self.acceleration += current_movement[0]
            self.velocity += self.acceleration
            self.direction += current_movement[1]
            self.x += self.velocity * math.cos(self.direction)
            self.y += self.velocity * math.sin(self.direction)

            current_distance = self.distance(target)
            current_fitness = 1 / current_distance

            if max_fitness < current_fitness:
                max_fitness = current_fitness
                best_action = current_action

            self.direction = current_dir
            self.velocity = current_vel
            self.x = current_x
            self.y = current_y

        self.change_in_acceleration = self.actions[best_action][0]
        self.change_in_direction = self.actions[best_action][1]
        self.acceleration += self.change_in_acceleration
        self.direction += self.change_in_direction
    
    def updateAverages(self):
        self.avg_vel = (self.avg_vel + self.velocity) / 2
        self.avg_accel = (self.avg_accel + self.acceleration) / 2
        self.avg_direction = (self.avg_direction + self.direction) / 2
        self.avg_perspective = (self.avg_perspective + self.perspective) / 2
        self.avg_energy_conservation = (self.avg_energy_conservation + self.energyConservation()) / 2

    def updateLocation(self):
        self.velocity += self.acceleration
        
        new_x = self.x + (self.velocity * math.cos(self.direction))
        new_y = self.y + (self.velocity * math.sin(self.direction))
        
        if new_x < 0 or new_x > SCREEN_WIDTH:
            self.direction = math.pi - self.direction

        if new_y < 0 or new_y > SCREEN_HEIGHT:
            self.direction = 2 * math.pi - self.direction

        self.x += self.velocity * math.cos(self.direction)
        self.y += self.velocity * math.sin(self.direction)
        self.updateAverages()
    
    # finds the initial distance, updates location
    # adjusts reward +/- new distance to the degree of energy consumed
    # uses energy to accelerate
    def approachTarget(self, target):
        init_distance = self.distance(target)
        self.updateLocation()
        new_distance = self.distance(target)   
        distance_traveled = init_distance - new_distance

        change = math.sqrt((self.change_in_acceleration + self.change_in_direction) ** 2) / 2 
        self.r -= change if self.r > 2 else 0
        self.energy -= math.floor(math.sqrt(distance_traveled ** 2) * self.r)
        self.reward += distance_traveled * self.energyConservation()
        
        self.perspective -= init_distance - new_distance + 1
        return False

    # returns true if the target is within scope
    def locateTarget(self, target):
        return self.distance(target) < self.perspective



    def navigate(self, targets):
        target_located = False

        # for all possible target inputs
        for target in targets:
            # if a target is within our perspective
            if self.locateTarget(target):
                # set trigger to not roam
                target_located = True

                # determine if we should choose a new action
                if random.random() < self.evolution_rate:
                    self.determineMovement(target)

                # attempt to approach our target updating location and rewards/energy
                self.approachTarget(target)

                # execute if we can
                # TODO: we need to be able to have different executions per target type
                #       maybe just return bool to parent function call
                if self.targetFound(target):
                    if self.energy > self.max_energy:
                        self.max_energy = self.energy
                    self.energy += 500
                    self.reward = (50 + self.reward) * self.energyConservation()
                    self.r += FOOD_SIZE
                    self.perspective = INIT_SIZE + self.r
                    

                    
                    self.targets_reached += 1
                    targets.remove(target)


        
        # if no targets are without our perspective
        if not target_located:
            # randomly move and increase perspective
            self.roam()
            # update location and averages
            self.updateLocation()

    def determineFitness(self):
        return self.reward / self.lifetime * self.avg_energy_conservation

    def die(self):
        self.fitness = self.determineFitness()
        self.velocity = 0
        self.color = (75, 75, 75)
        self.alive = False
        global total_population 
        total_population -= 1

    def printStatus(self, id):
        print("#################### \tIndividual", id, " \t#################### ")
        print("\t - fitness: \t\t", self.fitness)
        print("\t - total reward: \t", self.reward)
        print("\t - total lifetime: \t", self.lifetime)
        print("\t - targets reached: \t", self.targets_reached)
        print("\t - avg acceleration: \t", self.avg_accel)
        print("\t - avg velocity: \t", self.avg_vel)
        print("\t - avg direction: \t", self.avg_direction)
        print("\t - avg perspective: \t", self.avg_perspective)
        print("\t - avg energy conserved: ", self.avg_energy_conservation)
        print("#################### \t~~~~~~~~~~~~ \t#################### \n")

    def draw(self):
        pygame.draw.circle(screen, self.color, (self.x, self.y), self.r)

class Society():
    def __init__(self): 
        self.population = [Individual() for _ in range(INIT_POP_N)]
        self.sorted_population = []

    def maintainHealth(self, food):
        for individual in self.population:
            if individual.alive:
                if individual.energy <= 0:
                    individual.die()
                    #individual.printStatus(self.population.index(individual))
                    return

                individual.navigate(food)

                individual.lifetime += 1

    def sortPopulation(self):
        self.sorted_population = sorted(self.population, key=lambda x: x.fitness, reverse=True)

        for individual in self.sorted_population:
            individual.printStatus(self.population.index(individual))



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
        global total_population
        
        if len(self.society.sorted_population) < 1 and total_population == 0:
            self.society.sortPopulation()
            return
        elif total_population > 0:
            self.society.maintainHealth(self.food)
            self.maintainFoodSupply()


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
        clock.tick(2)


if __name__ == "__main__":
    main()
