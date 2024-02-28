import random
import pygame
import numpy as np
from colors import colorMap, criticalRaceTheory

MALE_REPRODUCTIVE_CHANCE = 0.4
FEMALE_REPRODUCTIVE_CHANCE = 0.8
ATTRACTIVE_REDUCTION_FACTOR = 0.5
BIOLOGY = ["Male", "Female"]
Q_TABLE = np.zeros((3, 2)) # Can want to eat, work, or mate


# Do we want to add these, or do we want to see what happens without them?
#
#    Add factor for the age men and women want to have sex ?
#    Add factor for age that men and women are most fertile ? 
#    Add factor for age that men and women are most attractive ?
#    Add factor for the age that men and women are most likely to want to reproduce ?

# Define the Individual class
class Individual:
    
    def __init__(self, id, w, h):
        #print("Creating Individual", id)
        self.id = id
        self.x = random.randint(0, w - 10)
        self.y = random.randint(0, h - 10)
        self.reward = 0
        self.energy = 100
        self.wakefulness = 100
        self.satisfaction = 100             # Goal is to have the highest satisfaction possible
        self.lifetime = 0
        self.money = 0
        self.fitness = 0
        self.sex_appeal = self.sexAppeal()
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
        self.chosen_target = None
        self.age = 0
        self.learning_rate = 0.1
        self.size = 7
        
    # UTILITY FUNCTIONS        
    def draw(self, screen):
        if self.sex == "Male":
            pygame.draw.polygon(screen, self.color_rgb, [(self.x, self.y), (self.x + self.size, self.y + (self.size * 2)), (self.x - self.size, self.y + (self.size * 2))])
        else:    
            pygame.draw.circle(screen, self.color_rgb, (self.x, self.y), self.size)
    
    def chooseBestAction(self):
        flat_index = np.argmax(self.q_table)
        row, col = np.unravel_index(flat_index, self.q_table.shape)
        return self.actions[col] if self.q_table.any() else None
    
    def chooseRandomAction(self):
        self.chosen_action = random.choice(self.actions)
        return self.chosen_action
 
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
        food.ate = True
    
    def work(self, job):
        self.money += job.pay
        self.energy += job.energy * random.randint(-2, 1)
        self.satisfaction += job.satisfaction
        self.fitness += job.fitness
    
    def mate(self, partner):
        self.satisfaction += (partner.satisfaction + partner.fitness + partner.energy) // 6
        self.energy -= (partner.fitness + partner.energy) // 3
        self.fitness += partner.fitness * random.randint(0, 1)
        
    def die(self):
        self.alive = False
        self.updateQTable()
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

    # our partner needs to at least be more than half as attractive as we are
    def sexAppeal(self):
        return (self.fitness * 0.5 + self.money * 0.5 + self.satisfaction * 0.5 + self.energy * 0.5) // 2
    
    def attractedTo(self, partner):
        return partner.sex_appeal > self.sex_appeal * ATTRACTIVE_REDUCTION_FACTOR
    
    def updateQTable(self):
        action_index = self.actions.index(self.chosen_action)
        current_q_value = self.q_table[action_index]
        max_future_q_value = np.max(self.q_table)
        new_q_value = (1 - self.learning_rate) * current_q_value + self.learning_rate * (self.reward + 0.9 * max_future_q_value)
        self.q_table[action_index] = new_q_value 
        self.reward = 0
        self.chosen_action = None
        self.chosen_target = None
        
    
    # Standard Loop Flow
    def sleep(self):
        self.wakefulness += random.randint(7, 11)       # While we sleep, we gain wakefulness
        self.satisfaction += random.randint(-2, 3)      # we have some random satisfaction +/-
        self.energy += random.randint(5, 12)            # we gain some energy
        
        if self.wakefulness >= 93:                      # 93 - 103 is the max range for wakefulness    
            self.fitness -= 100 - self.wakefulness      # fitness is reduced by the amount of wakefulness that is under 100
            self.sleeping = False                       # if we are awake, we are not sleeping
            return
    
    def exist(self):
        if self.alive:
            # Increase the lifetime, have some random satisfaction and check if they are sleeping
            self.lifetime += 1

            if self.sleeping or self.wakefulness < 0:
                return self.sleep()
                
            self.wakefulness -= 1
            
            self.sex_appeal = self.sexAppeal()

            if self.energy < 0 or self.satisfaction < 0 or self.fitness < 0: # they can die from lack of energy, satisfaction, or fitness
                self.age = self.years()
                self.reward = (self.satisfaction * self.age) + self.fitness + self.money # Add factor for lineage / n^descendants
                self.die()
        
        return self.alive
    
    # Main Loop Function
    def update(self, foods, jobs, partners):
        # Exit if dead
        if not self.exist():
            return
        
        #print("Individual", self.id, "is updating")
        
        # gain reward based on fitness, energy, money, and lifetime
            
        # If we do not have a chosen action, we will choose a random action, or we will choose the action with the highest Q value
        if self.chosen_action is None:
            self.chosen_action = self.chooseRandomAction() if np.random.uniform(0, 1) < 0.5 else self.chooseBestAction() # Need to ramp this up to 1.0 over time
            #print("Individual", self.id, "is choosing an action", self.chosen_action)
            
        ### EAT ###
        # If we have a chosen action, we will perform that action
        if self.chosen_action == "eat":
            if not self.chosen_target or self.chosen_target.ate:
                closest_meal = min(foods, key=lambda food: abs(self.x - food.x) + abs(self.y - food.y))
                best_meal = max(foods, key=lambda food: food.energy)
                cheapest_meal = min(foods, key=lambda food: food.cost)
                
                # TODO: Determine how to 'learn' which way to go to get the best meal based on distance/time, desire, and financial 'mindset'
                #       Maybe we add another row to the Q_TABLE for the individuals options to choose meals
                # determine desire to eat based on energy, fitness, and satisfaction
                if self.lowEndurance():
                    self.chosen_target = closest_meal
                elif self.lowFunds():
                    self.chosen_target = cheapest_meal
                elif self.hungered():
                    self.chosen_target = best_meal
                else:
                    self.chosen_target = random.choice([closest_meal, cheapest_meal, best_meal])
            
            if self.money < self.chosen_target.cost:
                if self.energy < self.chosen_target.energy:
                    self.die()
                else:
                    self.updateQTable()
                    return
            
            if abs(self.x - self.chosen_target.x) < self.size and abs(self.y - self.chosen_target.y) < self.size:
                self.eat(self.chosen_target)
                self.updateQTable()
            else:
                self.moveTo(self.chosen_target)
        
        ### WORK ###
        elif self.chosen_action == "work":
            closest_job = min(jobs, key=lambda job: abs(self.x - job.x) + abs(self.y - job.y))
            easiest_job = min(jobs, key=lambda job: job.energy)
            best_job = max(jobs, key=lambda job: job.pay) # Need to add aspect for experience
            
            # Choose best job if the energy is higher than the wealth
            if self.energy > self.money:
                self.chosen_target = best_job
            # Choose easiest job if the energy is lower than the satisfaction
            elif self.energy < self.satisfaction:
                self.chosen_target = easiest_job
            else:
                self.chosen_target = closest_job
            
            # check if the job is close enough to work
            if abs(self.x - self.chosen_target.x) < self.size and abs(self.y - self.chosen_target.y) < self.size:
                self.work(self.chosen_target)
                self.updateQTable()
                return
            
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
            if abs(self.x - self.partner.x) < self.size and abs(self.y - self.partner.y) < self.size:
                self.mate(self.partner) # TODO: Add Reproductivity and other factors like finance, fitness, busy-ness, etc
                
                # Break up if not attracted (# maybe we add factors like kids, age, etc)
                if not self.attractedTo(self.partner) or not self.partner.attractedTo(self):
                    self.satisfaction -= random.uniform(-20, 5)
                    self.partner.satisfaction -= random.uniform(-20, 5)
                    self.partner.partner = None
                    self.partner = None

                self.updateQTable()
                return
            
            # If we got here, we are not close enough to our partner to mate
            self.moveTo(self.partner)