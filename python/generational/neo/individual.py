import random
import pygame
import numpy as np
from colors import colorMap, criticalRaceTheory

MALE_REPRODUCTIVE_CHANCE = 0.4
FEMALE_REPRODUCTIVE_CHANCE = 0.8
ATTRACTIVE_REDUCTION_FACTOR = 0.5
BIOLOGY = ["Male", "Female"]
Q_TABLE = np.zeros((3, 3)) # Can be Hungry, Not Hungry and Horny or Not Horny


# Do we want to add these, or do we want to see what happens without them?
#
#    Add factor for the age men and women want to have sex ?
#    Add factor for age that men and women are most fertile ? 
#    Add factor for age that men and women are most attractive ?
#    Add factor for the age that men and women are most likely to want to reproduce ?

# Define the Individual class
class Individual:
    def __init__(self, id, w, h):
        self.id = id
        self.x = random.randint(0, w - 10)
        self.y = random.randint(0, h - 10)
        self.energy = 100
        self.satisfaction = 100             # Goal is to have the highest satisfaction possible
        self.fitness = 100
        self.lifetime = 0
        self.money = 0
        self.alive = True
        self.sex = random.choice(BIOLOGY)   # Random
        self.partner = None                 # Pointer to the Partner for Mating purposes
        self.father = None                  # Pointer to the Father for Reproductive purposes
        self.mother = None                  # Pointer to the Mother for Reproductive purposes
        self.generation = 0
        self.color = random.choice(list(colorMap.keys()))
        self.color_rgb = colorMap[self.color]
 
    # We want to add 360 to the movement and the ability to accelerate and decelerate       
    def moveTo(self, target):
        self.x += 1 if self.x < target.x else -1
        self.y += 1 if self.y < target.y else -1
    
    def eat(self, food):
        self.money -= food.cost
        self.energy += food.energy
        self.fitness += food.fitness
        self.satisfaction += food.satisfaction
    
    def work(self, job):
        self.money += job.pay
        self.energy += job.energy
        self.satisfaction += job.satisfaction
        self.fitness += job.fitness
    
    def mate(self, partner):
        # Add a factor for the attractiveness of the partner based on fitness and money
        self.satisfaction += (partner.satisfaction + partner.fitness + partner.energy) // 6
        self.energy -= (partner.fitness + partner.energy) // 3
        # Add a factor for reproduction based on male and female reproductive chances and the attractiveness of the partner, plus fertility and age
        
    def die(self):
        self.alive = False
        # Add inheritance for the children
        
    def draw(self, screen):
        if self.sex == "Male":
            pygame.draw.polygon(screen, self.color_rgb, [(self.x, self.y), (self.x + 7, self.y + 14), (self.x - 7, self.y + 14)])
        else:    
            pygame.draw.circle(screen, self.color_rgb, (self.x, self.y), 7)