import pygame
import random
import math

SCREEN_WIDTH = 1200
SCREEN_HEIGHT = 800
INIT_POP_COUNT = 10

total_population = INIT_POP_COUNT

def maxFood():
    global total_population
    return math.floor(math.e ** -(total_population ** 0.25 / 2) * total_population)

class Food():
    def __init__(self):
        self.x = random.random() * SCREEN_WIDTH
        self.y = random.random() * SCREEN_HEIGHT

class Individual():
    def __init__(self):
        self.x = random.random() * SCREEN_WIDTH
        self.y = random.random() * SCREEN_HEIGHT

class Society():
    def __init__(self): 
        self.population = [Individual() for _ in range(INIT_POP_COUNT)]

class World():
    def __init__(self):
        self.people = Society();
        self.food = [Food() for _ in range(maxFood())]

def main():
    derp