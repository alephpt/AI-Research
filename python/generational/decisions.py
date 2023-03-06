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
    "turn_left": (0, -0.0349),
    "turn_right": (0, 0.0349),
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


class Work:
    def __init__(self):
        self.x = random.randint(INIT_EMPLOYER_SIZE, SCREEN_WIDTH - INIT_EMPLOYER_SIZE)
        self.y = random.randint(INIT_EMPLOYER_SIZE, SCREEN_HEIGHT - INIT_EMPLOYER_SIZE)
        self.size = INIT_EMPLOYER_SIZE
        self.color = (255, 125, 0)
        self.energy = -100
        self.pay = 100
    
    def draw(self):
        pygame.draw.rect(screen, self.color, (self.x, self.y, self.size, self.size), 1)


class Food:
    def __init__(self):
        self.x = random.randint(INIT_FOOD_SIZE, SCREEN_WIDTH - INIT_FOOD_SIZE)
        self.y = random.randint(INIT_FOOD_SIZE, SCREEN_HEIGHT - INIT_FOOD_SIZE)
        self.size = INIT_FOOD_SIZE
        self.color = (0, 183, 69)
        self.energy = 500
        self.cost = 10
        
    def draw(self):
        pygame.draw.rect(screen, self.color, (self.x, self.y, self.size, self.size), 1)


class Individual:
    def __init__(self, _id):
        self.fitness = 0                                       # used for genetic evolution
        self.lifetime = 0
        self.total_actions = 0                                 # used for averaging actions
        self.target = None
        self.targets_reached = 0                               # fitness scalar
        self.alive = True
        self.father = None
        self.mother = None
        self.partner = None
        self.children = 0
        self.sex = _id % 2   # 0 for female, 1 for male
        self.x = random.randint(INIT_INDIVIDUAL_RADIUS, SCREEN_WIDTH - INIT_INDIVIDUAL_RADIUS)
        self.y = random.randint(INIT_INDIVIDUAL_RADIUS, SCREEN_HEIGHT - INIT_INDIVIDUAL_RADIUS)
        self.size = INIT_INDIVIDUAL_RADIUS
        self.color = (255, 125, 125) if self.sex == 0 else (125, 125, 255)
        self.perception = INIT_INDIVIDUAL_RADIUS               # inherits from parent
        self.avg_perception = self.perception                  # inherits from parents and becomes new default perspective
        ## Energy
        self.energy = 1250                                     # inherits average on next generation
        self.avg_energy = self.energy                          # used for threshold and next gen
        self.threshold_energy = 1000                           # inherits (max + avg) / 2 on next generation
        self.max_energy = 2500                                 # inherits (max + avg) / 2 on next generation
        self.energy_score = 0.5                                # used for target_learning table -> passes to nextgen
        ## Energy Conservation
        self.avg_energy_conservation = 1                       # used for threshold in next gen
        self.threshold_energy_conserv = 0                      # used for threshold reward
        ## Direction
        self.direction = math.radians(random.randint(0, 360))  # inherits avg from parents
        self.change_in_direction = 0                           # used for energy deduction
        self.avg_direction = 0                                 # used for next gen
        ## Acceleration
        self.acceleration = 0
        self.change_in_acceleration = 0                        # used for energy deduction
        self.avg_acceleration = 0                              # used for threshold and next gen
        self.threshold_acceleration = 0                        # inherits avg_acceleration on next generation
        ## Velocity
        self.velocity = 0
        self.avg_velocity = 0                                  # inherits avg on nextgen (used as Velocity + Avg / 2)
        self.target_velocity = 0                               # inherits avg_vel + max / 2 on nextgen (used for threshrold)
        self.max_velocity = 40                                 # needs to be adjustable
        ## Satisfaction
        self.satisfaction = 0
        self.avg_satisfaction = 0                              # passed to children as new target
        self.target_satisfaction = 100                         # inherits parents avg on nextgen
        self.satisfaction_score = 0                            # used for target_learning table
        ## Money
        self.money = 0                                         # used to buy food and determine attraction <- mutate this
        self.money_threshold = 100                             # used to determine score, passed to children
        self.money_score = 0                                   # used for target_learning
        ## Rewards
        self.attractability = 0                                # used to attracting mates - Needs to incorporate energy conservation
        self.reward = 0                                        # main fitness factor
        self.threshold_reward = 0                              # used to scale fitness
        ## Action Tables
        self.target_bias = {
            "working": ((self.satisfaction_score, self.energy_score, self.money_score), 1),
            "eating": ((self.satisfaction_score, self.energy_score, self.money_score), 1),
            "mating": ((self.satisfaction_score, self.energy_score, self.money_score), 1)
        }
        self.action_history = {
            "forward": 0,
            "reverse": 0,
            "turn_left": 0,
            "turn_right": 0,
            "forward_left": 0,
            "forward_right": 0,    
            "reverse_left": 0,
            "reverse_right": 0
        }
        self.action_bias = {
            "forward": 1,
            "reverse": 1,
            "turn_left": 1,
            "turn_right": 1,
            "forward_left": 1,
            "forward_right": 1,    
            "reverse_left": 1,
            "reverse_right": 1
        }
    
    def chooseTarget(self):
        return
    
    def chooseAction(self):
        return
    
    def draw(self):
        pygame.draw.circle(screen, self.color, (self.x, self.y), self.size)


class Society:
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


class World:
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
