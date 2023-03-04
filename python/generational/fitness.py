import pygame
import random
import math


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
    evolution_rate = 1
    actions = {
        "turn_right": (0, 0.25),
        "turn_left": (0, -0.25),
        "accelerate": (0.5, 0),
        "decelerate": (-0.5, 0),
        "accel_right": (0.5, 0.25),
        "accel_left": (0.5, -0.25),
        "decel_right": (-0.5, 0.25),
        "decel_left": (-0.5, -0.25)
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
        self.max_velocity = 100
        self.max_acceleration = 10
        self.max_energy = 2500
        self.velocity = 0
        self.acceleration = 0
        self.energy = 1000                                     # inherits average on next generation <- should be used for target
        self.perspective = INIT_SIZE                           # inherits from parent
        self.threshold_accel = 0                               # inherits avg_vel on next generation <- should be used for target
        self.threshold_velocity = 0                            # inherits avg_vel on next generation <- should be used for max + 1/2 accel
        self.threshold_energy = 1000                           # inherits (max + avg / 2) on next generation <- should be used for target
        self.threshold_energy_conserv = 0                      # TODO: maybe we create a reward to optimize maintaining energy at converservation level?
        self.threshold_perspective = INIT_SIZE                 # inherits from parents and becomes new default perspective
        self.avg_energy = 0                                    # used for threshold and next gen
        self.avg_accel = 0                                     # used for threshold and next gen
        self.avg_vel = 0                                       # used for threshold and next gen
        self.avg_direction = 0                                 # used for next gen
        self.avg_perspective = INIT_SIZE                       # used for threshold
        self.avg_energy_conservation = 1                       # used for threshold
        self.change_in_acceleration = 0                        # used for energy deduction
        self.change_in_direction = 0                           # used for energy deduction
        self.threshold_reward = 0                              # used to scale fitness
        self.reward = 0                                        # main fitness factor
        self.fitness = 0                                       # used for genetic evolution
        self.targets_reached = 0                               # fitness scalar
    
    def thresholdReward(self, v, t, m):
        if v == 0:
            return 0
        
        if t == 0:
            return 1
        
        return (v / t) / (v / m)
    
    def determineThresholdReward(self):
        # acceleration
        atr = self.thresholdReward(self.acceleration, self.threshold_accel, self.max_acceleration)
        # velocity
        vtr = self.thresholdReward(self.velocity, self.threshold_velocity, self.max_velocity)
        # energy
        etr = self.thresholdReward(self.energy, self.threshold_energy, self.max_energy)
        # energy conservation
        ectr = self.thresholdReward(self.energyConservation(), self.threshold_energy, 1)
        # perspective
        ptr = self.thresholdReward(self.perspective, self.threshold_perspective, 1)
        
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
        self.avg_energy = (self.avg_energy + self.energy) / 2
        self.avg_vel = (self.avg_vel + self.velocity) / 2
        self.avg_accel = (self.avg_accel + self.acceleration) / 2
        self.avg_direction = (self.avg_direction + self.direction) / 2
        self.avg_perspective = (self.avg_perspective + self.perspective) / 2
        self.avg_energy_conservation = (self.avg_energy_conservation + self.energyConservation()) / 2
        self.threshold_reward += self.determineThresholdReward()
        
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
            if self.food_delay >= (maxFood() / total_population):
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
        clock.tick(3)


if __name__ == "__main__":
    main()
