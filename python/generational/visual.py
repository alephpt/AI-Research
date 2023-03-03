from tkinter.tix import MAX
import pygame
from enum import Enum
import random
import math

SCREEN_WIDTH = 1600
SCREEN_HEIGHT = 1200
INIT_POPULATION_COUNT = 100
MUTATION_RATE = 0.05
INIT_SPEED = 2
INIT_RADIUS = 3
total_alive = INIT_POPULATION_COUNT

def max_food():
   global total_alive
   return math.floor(math.e ** -(total_alive ** 0.25 / 2) * total_alive)


#############################
## Initialize PyGame State ##
#############################

pygame.init()
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Generational Learning")
pygame.font.init()
text = pygame.font.SysFont('Comic Sans MS', 30)
clock = pygame.time.Clock()
running = True

class Gender(Enum):
    Female = 0
    Male = 1

class Sexuality(Enum):
    Virgin = 0
    Heterosexual = 1
    Bisexual = 2
    Homosexual = 3
    Abstainant = 4

class Food():
    def __init__(self):
        self.x = random.randint(0, SCREEN_WIDTH)
        self.y = random.randint(0, SCREEN_HEIGHT)
        self.r = INIT_RADIUS * 2
        self.color = (255, 255, 0)

    def draw(self):
        pygame.draw.rect(screen, self.color, (self.x, self.y, self.r, self.r), 2)

## Todo: Set Adjustable Thresholds for Values unique to each individual
class Individual:
    def __init__(self, id):
        self.id = id
        self.alive = True
        self.generation = 0
        self.lifetime = 0
        self.speed = INIT_SPEED
        self.search_radius = INIT_RADIUS
        self.r = INIT_RADIUS
        self.x = random.randint(0, SCREEN_WIDTH)
        self.y = random.randint(0, SCREEN_HEIGHT)
        self.father = None
        self.mother = None
        self.gender = random.choice([Gender.Male, Gender.Female])
        self.sexuality = Sexuality.Virgin
        self.partner = None
        self.committed = False
        self.max_energy = 3500
        self.energy_threshold = 2000
        self.energy = 3000
        self.max_satisfaction = 100
        self.satisfaction_threshold = 0
        self.satisfaction = 0
        self.money = 0
        self.color = (random.randint(122, 255), random.randint(0, 73), random.randint(0, 73))  \
                      if self.gender == Gender.Female else \
                      (random.randint(0, 73), random.randint(0, 73), random.randint(122, 255)) 

    def moveTo(self, target):
        distance = math.sqrt((target.x - self.x) ** 2 + (target.y - self.y) ** 2)
        movement = (self.speed * self.energy) / self.max_energy

        if distance < self.search_radius:
            self.x +=  movement if self.x < target.x else -movement
            self.y +=  movement if self.y < target.y else -movement
            self.search_radius -= 1
            self.satisfaction += 1
        else:
            self.x += random.uniform(-movement, movement)
            self.y += random.uniform(-movement, movement)
            self.search_radius += movement
            self.satisfaction -= 1

        if self.x < 0 or self.x > SCREEN_WIDTH or \
           self.y < 0 or self.y > SCREEN_HEIGHT:
            self.die()

    def findFood(self, edible):
        self.energy -= 1
        self.moveTo(edible)

    def findPartner(self, other):
        self.energy -= 1
        self.moveTo(other)

    def foundTarget(self, other):
        distance = math.sqrt((other.x - self.x) ** 2 + (other.y - self.y) ** 2)
        
        if distance < (self.r + other.r):
            return True
        
        return False

    def consume(self, edible):
        self.energy += 100



    ## TODO: Add Attraction Function
    def mate(self, other):
        self.energy -= 1

        if self.sexuality == Sexuality.Abstainant:
            return

        self.partner = other.id
        self.color = (self.color[0], random.randint(122,255), self.color[2])

        if self.gender == other.gender and self.sexuality != Sexuality.Homosexual:
            if self.sexuality == Sexuality.Virgin:
                ## TODO: determine likelihood to mate
                self.sexuality == Sexuality.Homosexual
            elif self.sexuality == Sexuality.Heterosexual:
                ## TODO: determine willingness to change orientation
                self.sexuality == Sexuality.Bisexual
        elif self.sexuality != Sexuality.Heterosexual:
            if self.sexuality == Sexuality.Virgin:
                ## TODO: determine likelihood to mate
                self.sexuality == Sexuality.Heterosexual
            elif self.sexuality == Sexuality.Homosexual:
                ## TODO: determine willingness to change orientation
                self.sexuality = Sexuality.Bisexual
        
        self.satisfaction += 1
        ## TODO: determine likelihood to reproduce
        ## TODO: determine satisfaction value

    def breakUp(self):
        self.energy -= 1
        self.partner = None
        self.color = (self.color[0], random.randint(0, 73), self.color[2])
        
    def die(self):
        global total_alive
        self.alive = False
        self.color = (122, 122, 122)
        total_alive -= 1

    def draw(self):
        pygame.draw.circle(screen, self.color, (self.x, self.y), self.r)

class Population:
    def __init__(self):
        self.individuals = [Individual(p) for p in range(INIT_POPULATION_COUNT)]
        self.food = [Food() for _ in range(max_food())]
        self.food_spawn_time = 0
        self.food_spawn_delay = 100

    def draw(self):
        for individual in self.individuals:
            individual.draw()

        for edible in self.food:
            edible.draw()

    def reproduce(self, a, b):
        newborn = Individual(len(self.individuals))

        if a.gender == Gender.Male:
            newborn.father = a.id
            newborn.mother = b.id
        else:
            newborn.mother = a.id
            newborn.father = b.id

        newborn.generation = a.generation + 1 \
                             if a.generation > b.generation else \
                             b.generation + 1
        newborn.satisfaction = (a.satisfaction + b.satisfaction) / 2
        newborn.x = (a.x + b.x) / 2
        newborn.y = (a.y + b.y) / 2

        print (a.id + " and " + b.id + " had a baby: generation " + newborn.generation)
        self.individuals.append(newborn)
    
    def spawnFood(self):
        if len(self.food) < max_food:
            if self.food_spawn_time < self.food_spawn_delay:
                self.food_spawn_time += 1
            else:
                self.food.append(Food())
                self.food_spawn_time = 0

    def mutate(self):
        # go through all the individuals
        for individual in self.individuals:
            if individual.alive:
                if individual.energy <= 0 and individual.satisfaction <= 0:
                    individual.die()
                    return

                individual.lifetime += 1
                
                # check against mutation rate
                if random.random() < MUTATION_RATE:
                    if individual.energy < individual.energy_threshold:
                        for edible in self.food:
                            individual.findFood(edible)

                            if individual.foundTarget(edible):
                                individual.consume(edible)
                                self.food.remove(edible)
                
                    # if the individual does NOT have a partner
                    else:
                        if individual.partner == None:
                            # check against all the other individuals
                            for other in self.individuals:
                                # if the other also does not have a partner that isn't the parent
                                if other.alive and individual != other \
                                   and other.partner == None \
                                   and other.id != individual.father \
                                   and other.id != individual.mother:

                                   if individual.sexuality == other.sexuality or random.random() < MUTATION_RATE:
                                       individual.findPartner(other)

                                       if individual.foundTarget(other):
                                            if random.random() < MUTATION_RATE:
                                                individual.mate(other)
                                                other.mate(individual)
                                                continue
                        else:
                            for other in self.individuals:
                                if other.id == individual.partner:
                                    if random.random() < MUTATION_RATE and individual.gender != other.gender:
                                        self.reproduce(individual, other)

                                    if random.random() < MUTATION_RATE:
                                        other.breakUp()
                                        individual.breakUp()



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
