import pygame
import random
import math


TOTAL_GENERATION = 10
SCREEN_WIDTH = 1200
SCREEN_HEIGHT = 800
INIT_POP_N = 20
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
    return math.floor(math.e ** -(total_population ** 0.475 / 7.25) * total_population)

class Food():
    def __init__(self):
        self.x = random.random() * SCREEN_WIDTH
        self.y = random.random() * SCREEN_HEIGHT
        self.r = FOOD_SIZE
        self.color = (0, random.randint(178, 255), 0)
    
    def draw(self):
        pygame.draw.rect(screen, self.color, (self.x, self.y, self.r, self.r), 1)

class Individual():
    evolution_rate = 1
    actions = {
        "turn_right": (0, 0.01745),
        "turn_left": (0, -0.01745),
        "accelerate": (0.5, 0),
        "decelerate": (-0.5, 0),
        "accel_right": (0.5, 0.01745),
        "accel_left": (0.5, -0.01745),
        "decel_right": (-0.5, 0.01745),
        "decel_left": (-0.5, -0.01745)
    }
    
    action_history = {
        "turn_right": 1,
        "turn_left": 1,
        "accelerate": 1,
        "decelerate": 1,
        "accel_right": 1,
        "accel_left": 1,
        "decel_right": 1,
        "decel_left": 1
    }
    
    action_bias = {
        "turn_right": 1,
        "turn_left": 1,
        "accelerate": 1,
        "decelerate": 1,
        "accel_right": 1,
        "accel_left": 1,
        "decel_right": 1,
        "decel_left": 1
    }
    
    def __init__(self):
        self.alive = True
        self.lifetime = 0
        self.father = None
        self.mother = None
        self.partner = None
        self.children = 0
        self.x = random.random() * SCREEN_WIDTH
        self.y = random.random() * SCREEN_HEIGHT
        self.r = INIT_SIZE
        self.color = (random.randint(178, 255), 0, 0)
        self.direction = math.radians(random.randint(0, 360))  # inherits avg from parents
        self.velocity = 0
        self.max_velocity = 40
        self.acceleration = 0
        self.max_energy = 2500
        self.energy = 1250                                     # inherits average on next generation <- should be used for target
        self.perspective = INIT_SIZE                           # inherits from parent
        self.threshold_acceleration = 0                        # inherits avg_acceleration on next generation <- should be used for target
        self.threshold_velocity = 0                            # inherits avg_vel on next generation <- should be used for max + 1/2 accel
        self.threshold_energy = 1000                           # inherits (max + avg / 2) on next generation <- should be used for target
        self.threshold_energy_conserv = 0                      # TODO: maybe we create a reward to optimize maintaining energy at converservation level?
        self.threshold_perspective = INIT_SIZE                 # inherits from parents and becomes new default perspective
        self.avg_energy = 0                                    # used for threshold and next gen
        self.avg_acceleration = 0                              # used for threshold and next gen
        self.avg_velocity = 0                                  # used for threshold and next gen
        self.avg_direction = 0                                 # used for next gen
        self.avg_perspective = INIT_SIZE                       # used for threshold
        self.avg_energy_conservation = 1                       # used for threshold
        self.change_in_acceleration = 0                        # used for energy deduction
        self.change_in_direction = 0                           # used for energy deduction
        self.threshold_reward = 0                              # used to scale fitness
        self.reward = 0                                        # main fitness factor
        self.fitness = 0                                       # used for genetic evolution
        self.targets_reached = 0                               # fitness scalar
        self.total_actions = 0
    
    def thresholdReward(self, v, t,):
        if v == 0:
            return 0
        
        if t == 0:
            return 1
        
        return (v / t)
    
    def determineThresholdReward(self):
        # acceleration
        atr = self.thresholdReward(self.acceleration, self.threshold_acceleration)
        # velocity
        vtr = self.thresholdReward(self.velocity, self.threshold_velocity)
        # energy
        etr = self.thresholdReward(self.energy, self.threshold_energy)
        # energy conservation
        ectr = self.thresholdReward(self.energyConservation(), self.threshold_energy)
        # perspective
        ptr = self.thresholdReward(self.perspective, self.threshold_perspective)
        
        return (atr + vtr + etr + ectr + ptr)
    
    def energyConservation(self):
        return math.sqrt((self.energy / self.max_energy) ** 2) 

    def distance(a, b):
        return math.sqrt((b.x - a.x) ** 2 + (b.y - a.y) ** 2)
    
    #randomly accelerates/decelerates, and rotates
    def roam(self):
        self.acceleration += random.uniform(-1.0, 1.0)
        self.direction += random.uniform(-0.5, 0.5)
        self.energy -= self.perspective / INIT_SIZE
        self.reward -= self.r
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
            
            if self.total_actions < 1:
                current_fitness = 1 / current_distance
            else:
                current_fitness = (1 / current_distance) * (self.action_history[current_action] / self.total_actions) * self.action_bias

            if max_fitness < current_fitness:
                max_fitness = current_fitness
                best_action = current_action

            self.direction = current_dir
            self.velocity = current_vel
            self.x = current_x
            self.y = current_y

        self.action_bias[best_action] += 1
        self.change_in_acceleration = self.actions[best_action][0]
        self.change_in_direction = self.actions[best_action][1]
        self.acceleration = self.acceleration + self.change_in_acceleration
        self.direction += self.change_in_direction
    
    def updateAverages(self):
        self.avg_energy = (self.avg_energy + self.energy) / 2
        self.avg_velocity = (self.avg_velocity + self.velocity) / 2
        self.avg_acceleration = (self.avg_acceleration + self.acceleration) / 2
        self.avg_direction = (self.avg_direction + self.direction) / 2
        self.avg_perspective = (self.avg_perspective + self.perspective) / 2
        self.avg_energy_conservation = (self.avg_energy_conservation + self.energyConservation()) / 2
        self.threshold_reward += self.determineThresholdReward()
        
    def updateLocation(self):
        self.velocity += self.acceleration
        
        if self.velocity > self.max_velocity:
            self.velocity = self.max_velocity
            
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

        change = math.sqrt((self.change_in_acceleration + self.change_in_direction) ** 2) 
        self.r -= change / 2 if self.r > INIT_SIZE else 0
        self.energy -= math.floor(math.sqrt(distance_traveled ** 2) * change + self.r)
        self.reward += distance_traveled * self.energyConservation()
        
        self.perspective -= distance_traveled + 1 if distance_traveled > 0 else distance_traveled - 1

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
                    self.energy += 750
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
        return (((self.targets_reached * 20 + self.reward) / self.lifetime) + (self.threshold_reward / self.lifetime)) * self.avg_energy_conservation

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
        print("\t - reward: \t\t", self.reward)
        print("\t - threshold reward: \t", self.threshold_reward)
        print("\t - total lifetime: \t", self.lifetime)
        print("\t - targets reached: \t", self.targets_reached)
        print("\t - avg acceleration: \t", self.avg_energy)
        print("\t - avg acceleration: \t", self.avg_acceleration)
        print("\t - avg velocity: \t", self.avg_velocity)
        print("\t - avg direction: \t", self.avg_direction)
        print("\t - avg perspective: \t", self.avg_perspective)
        print("\t - avg energy conserved: ", self.avg_energy_conservation)
        print("#################### \t~~~~~~~~~~~~ \t#################### \n")

    def draw(self):
        pygame.draw.circle(screen, self.color, (self.x, self.y), self.r)

class Society():
    def __init__(self): 
        self.population = [Individual() for _ in range(INIT_POP_N)]
        self.food = [Food() for _ in range(maxFood())]
        self.food_delay = 0

    def maintainFoodSupply(self):
        if len(self.food) < maxFood():
            if self.food_delay >= (maxFood() / total_population):
                self.food.append(Food())
                self.food_delay = 0
                return
            self.food_delay += 1

        elif len(self.food) > maxFood():
            if self.food_delay >= (maxFood() / (total_population / 2)):
                self.food.pop()
                self.food_delay = 0
                return
            self.food_delay += 1

    def maintainHealth(self):
        for individual in self.population:
            if individual.alive:
                if individual.energy <= 0:
                    individual.die()
                    #individual.printStatus(self.population.index(individual))
                    return

                individual.navigate(self.food)

                individual.lifetime += 1

    def sortPopulation(self):
        return sorted(self.population, key=lambda x: x.fitness, reverse=True)



class World():
    mutation_rate = 0.05
    
    def __init__(self):
        self.society = Society();
        self.epoch = 0
    
    def repopulate(self):
        parents = self.society.sortPopulation()
        children = []
        
        # create children, crossover and mutate
        for _ in range(len(parents) / 2):
            parentA = random.choice(parents)
            parentB = random.choice(parents)
            childA = Individual()
            childB = Individual()
            
            childA.fitness = (parentA.fitness + parentB.fitness) / 2
            childA.energy = parentA.avg_energy
            childA.action_bias = parentB.action_bias
            childA.perspective = parentB.perspective
            childA.direction = parentB.avg_direction
            childA.threshold_acceleration = parentA.avg_acceleration
            childA.threshold_energy = parentA.avg_energy
            childA.threshold_energy_conserv = parentA.avg_energy_conservation
            childA.threshold_perspective = parentA.avg_perspective
            childA.threshold_velocity = parentA.avg_velocity

            childB.fitness = (parentA.fitness + parentB.fitness) / 2
            childB.energy = parentB.avg_energy
            childB.action_bias = parentA.action_bias
            childB.perspective = parentA.perspective
            childB.direction = parentA.avg_direction
            childB.threshold_acceleration = parentB.avg_acceleration
            childB.threshold_energy = parentB.avg_energy
            childB.threshold_energy_conserv = parentB.avg_energy_conservation
            childB.threshold_perspective = parentB.avg_perspective
            childB.threshold_velocity = parentB.avg_velocity
            
            children.append(childA)
            children.append(childB)
            
    
    def evolve(self):
        global total_population
        
        if total_population > 0:
            self.society.maintainHealth()
            self.society.maintainFoodSupply()
            return
        


    def draw(self):
        for individual in self.society.population:
            individual.draw()
        
        for edible in self.society.food:
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
        clock.tick(3)


if __name__ == "__main__":
    main()
