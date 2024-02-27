import random
import pygame
import numpy as np
from colors import colorMap, criticalRaceTheory

MALE_REPRODUCTIVE_CHANCE = 0.4
FEMALE_REPRODUCTIVE_CHANCE = 0.8
ATTRACTIVE_REDUCTION_FACTOR = 0.5
BIOLOGY = ["Male", "Female"]
Q_TABLE = np.zeros((3, 3)) # Can want to eat, work, or mate


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
        self.reward = 0
        self.energy = 100
        self.fitness = 100
        self.wakefulness = 100
        self.satisfaction = 100             # Goal is to have the highest satisfaction possible
        self.lifetime = 0
        self.money = 0
        self.sex_appeal = (self.fitness * 0.5 + self.money * 0.5 + self.satisfaction * 0.5 + self.energy * 0.5) // 2
        self.alive = True
        self.sleeping = False               # Introduce 5 options?  eat, work, sleep, mate, nothing
        self.sex = random.choice(BIOLOGY)   # Random
        self.partner = None                 # Pointer to the Partner for Mating purposes
        self.father = None                  # Pointer to the Father for Reproductive purposes
        self.mother = None                  # Pointer to the Mother for Reproductive purposes
        self.generation = 0
        self.color = random.choice(list(colorMap.keys()))
        self.color_rgb = colorMap[self.color]
        self.q_table = Q_TABLE
        self.actions = ["eat", "work", "mate"]
        self.chosen_action = None
        self.chosen_meal = None
        self.age = 0
        
    # UTILITY FUNCTIONS        
    def draw(self, screen):
        if self.sex == "Male":
            pygame.draw.polygon(screen, self.color_rgb, [(self.x, self.y), (self.x + 7, self.y + 14), (self.x - 7, self.y + 14)])
        else:    
            pygame.draw.circle(screen, self.color_rgb, (self.x, self.y), 7)
    
    def chooseBestAction(self):
        flat_index = np.argmax(self.q_table)
        row, col = np.unravel_index(flat_index, self.q_table.shape)
        return self.actions[col] if self.q_table.any() else None
    
    def chooseRandomAction(self):
        self.chosen_action = random.choice(self.actions)
 
    # We want to add 360 to the movement and the ability to accelerate and decelerate       
    def moveTo(self, target):
        self.x += 1 if self.x < target.x else -1 if self.x > target.x else 0
        self.y += 1 if self.y < target.y else -1 if self.y > target.y else 0
    
    def years(self):
        return self.lifetime / 365
    
    # ACTIONS
    def eat(self, food):
        self.money -= food.cost
        self.energy += food.energy * random.randint(-1, 3)
        self.fitness += food.fitness * random.randint(-2, 1)
        self.satisfaction += food.satisfaction * random.randint(0, 1)
    
    def work(self, job):
        self.money += job.pay
        self.energy += job.energy * random.randint(-2, 1)
        self.satisfaction += job.satisfaction
        self.fitness += job.fitness
    
    def mate(self, partner):
        # Add a factor for the attractiveness of the partner based on fitness and money
        self.satisfaction += (partner.satisfaction + partner.fitness + partner.energy) // 6
        self.energy -= (partner.fitness + partner.energy) // 3
        self.fitness += partner.fitness * random.randint(0, 1)
        # Add a factor for reproduction based on male and female reproductive chances and the attractiveness of the partner, plus fertility and age
        
    def die(self):
        self.alive = False
        # Add inheritance for the children of the individual based on the wealth and satisfaction of the individual

    #####################
    ## Logic Functions ##
    #####################
    
    # Determining Food Choices
    def lowEndurance(self):
        return self.energy < self.fitness + self.satisfaction    # if our energy is less than our fitness level, we are low on endurance
    
    def lowFunds(self):
        return self.money < self.energy + self.fitness           # if we have no money, we are low on funds
    
    def hungered(self):
        return self.satisfaction < self.fitness + self.energy    # if our satisfaction is less than our fitness and energy, we are hungry

    
    def attractedTo(self, partner):
        return ((self.sex_appeal < (partner.sex_appeal * 1.5)) and (self.sex_appeal > (partner.sex_appeal / 1.5)))
    
    
    # Standard Loop Flow
    def sleep(self):
        self.wakefulness += 10                          # While we sleep, we gain wakefulness
        self.satisfaction += random.randint(-1, 3)      # we have some random satisfaction +/-
        self.energy += 1                                # we gain some energy
        
        if self.wakefulness >= 93:                      # 93 - 103 is the max range for wakefulness       
            self.fitness -= 100 - self.wakefulness      # fitness is reduced by the amount of wakefulness that is under 100
            self.sleeping = False                       # if we are awake, we are not sleeping
            return
    
    def exist(self):
        if self.alive:
            # Increase the lifetime, have some random satisfaction and check if they are sleeping
            self.lifetime += 1
            self.satisfaction += np.random.randint(-1, 3)

            if self.sleeping or self.wakefulness < 0:
                return self.sleep()
                
            self.wakefulness -= 1
            self.energy -= 1        # should this be based on fitness, age, diet, sleep, and satisfaction
            self.satisfaction -= 1  # Should we introduce randomness here each turn, or based on some factor like 'emotion' ??
            self.fitness -= 1       # Should this be factored by age && diet && exercise/work amounts?
            self.money -= 1         # Should we add some logic here?
            
            # Update Sex Appeal
            self.sex_appeal = (self.fitness * 0.5 + self.money * 0.5 + self.satisfaction * 0.5 + self.energy * 0.5) // 2

            if self.energy < 0 or self.satisfaction < 0 or self.fitness < 0: # they can die from lack of energy, satisfaction, or fitness
                self.age = self.years()
                self.reward *= (self.satisfaction * self.age)
                self.die()
        
        return self.alive
    
    # Main Loop Function
    def update(self, foods, jobs, partners):
        # Exit if dead
        if not self.exist():
            return
        
        # gain reward based on fitness, energy, money, and lifetime
        self.reward += self.fitness + self.energy + self.money + self.lifetime + self.satisfaction
            
        # If we do not have a chosen action, we will choose a random action, or we will choose the action with the highest Q value
        if self.chosen_action is None:
            self.chosen_action = self.chooseRandomAction() if np.random.uniform(0, 1) < 0.5 else self.chooseBestAction() # Need to ramp this up to 1.0 over time
            
        ### EAT ###
        # If we have a chosen action, we will perform that action
        if self.chosen_action == "eat":
            if not self.chosen_meal:
                closest_meal = min(foods, key=lambda food: abs(self.x - food.x) + abs(self.y - food.y))
                best_meal = max(foods, key=lambda food: food.energy)
                cheapest_meal = min(foods, key=lambda food: food.cost)
                
                # TODO: Determine how to 'learn' which way to go to get the best meal based on distance/time, desire, and financial 'mindset'
                #       Maybe we add another row to the Q_TABLE for the individuals options to choose meals
                # determine desire to eat based on energy, fitness, and satisfaction
                if self.lowEndurance():
                    self.chosen_meal = closest_meal
                elif self.lowFunds():
                    self.chosen_mean = cheapest_meal
                elif self.hungered():
                    self.chosen_meal = best_meal
                else:
                    self.chosen_meal = random.choice([closest_meal, cheapest_meal, best_meal])
                    
            if abs(self.x - self.chosen_meal.x) < 10 and abs(self.y - self.chosen_meal.y) < 10:
                self.eat(self.chosen_meal)
                self.chosen_action = None
                self.chosen_meal = None
            else:
                self.moveTo(self.chosen_meal)
        
        ### WORK ###
        elif self.chosen_action == "work":
            closest_job = min(jobs, key=lambda job: abs(self.x - job.x) + abs(self.y - job.y))
            
            # In theory we could establish an economy based on work experience, market demands, and desire to work
            # likely would be correlated to fitness and energy
            
            
            # check if the job is close enough to work
            if abs(self.x - closest_job.x) < 10 and abs(self.y - closest_job.y) < 10:
                self.work(jobs)
                self.chosen_action = None
            else:
                self.moveTo(closest_job)
        
        ### SEX ###
        elif self.chosen_action == "mate":
            # if we don't have a partner, we will choose a random partner that is not ourselves
            if not self.partner:
                random_partner = random.choice([partner for partner in partners if partner != self])
                
                if self.attractedTo(random_partner) and random_partner.attractedTo(self):
                    self.partner = random_partner
                    self.partner.partner = self
            
            # if we still don't have a partner or our partner is not interested 
            #  - we will choose a random action
            #        and exit early
            if not self.partner or self.partner.chosen_action != "mate":
                self.chooseRandomAction()
                return
            
            # if we have a partner, who also wants to mate we will move to the partner
            if abs(self.x - self.partner.x) < 10 and abs(self.y - self.partner.y) < 10:
                self.mate(self.partner) # TODO: Add Reproductivity and other factors like finance, fitness, busy-ness, etc
                self.chosen_action = None
                
                # Break up if not attracted (# maybe we add factors like kids, age, etc)
                if not self.attractedTo(self.partner) or not self.partner.attractedTo(self):
                    self.satisfaction -= random.uniform(-20, 5)
                    self.partner.satisfaction -= random.uniform(-20, 5)
                    self.partner.partner = None
                    self.partner = None
                
                return
            
            # If we got here, we are not close enough to our partner to mate
            self.moveTo(self.partner)