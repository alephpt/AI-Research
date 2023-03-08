import pygame
import random
from enum import Enum
import math

SCREEN_WIDTH = 1200
SCREEN_HEIGHT = 800
INIT_EMPLOYERS = 3
INIT_EMPLOYER_SIZE = 16
INIT_POPULATION = 10
INIT_INDIVIDUAL_RADIUS = 5
INIT_FOOD = 5
INIT_FOOD_SIZE = 4

pygame.init()
pygame.font.init()
pygame.display.set_caption("Genetic Decision Making")
text = pygame.font.SysFont('Comic Sans MS', 16)
clock = pygame.time.Clock()
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))

ACTIONS = {
    "maintain": (0, 0),
    "forward": (0.2, 0),
    "reverse": (-0.2, 0),
    "turn_left": (0, -0.0698),
    "turn_right": (0, 0.0698),
    "forward_left": (0.2, -0.0698),
    "forward_right": (0.2, 0.0698),
    "reverse_left": (-0.2, 0.0698),
    "reverse_right": (-0.2, -0.0698),
}

TARGETS = {
    "working": 0,   # need money to get food, and to be more attractive
    "eating": 1,    # need food to have energy, and to not die
    "mating": 2     # need energy to mate, as well as attraction
}

def normalizeDirection(d):
    return (d + math.pi) % (2*math.pi) - math.pi


class Work:
    def __init__(self, identity):
        self.identity = identity
        self.x = random.randint(INIT_EMPLOYER_SIZE, SCREEN_WIDTH - INIT_EMPLOYER_SIZE)
        self.y = random.randint(INIT_EMPLOYER_SIZE, SCREEN_HEIGHT - INIT_EMPLOYER_SIZE)
        self.size = INIT_EMPLOYER_SIZE
        self.color = (255, 125, 0)
        self.work_load = random.randint(50,200)
        self.pay = random.randint(75,150)
    
    def draw(self):
        pygame.draw.rect(screen, self.color, (self.x, self.y, self.size, self.size), 1)


class Food:
    def __init__(self, identity):
        self.identity = identity
        self.x = random.randint(INIT_FOOD_SIZE, SCREEN_WIDTH - INIT_FOOD_SIZE)
        self.y = random.randint(INIT_FOOD_SIZE, SCREEN_HEIGHT - INIT_FOOD_SIZE)
        self.size = INIT_FOOD_SIZE
        self.color = (0, 183, 69)
        self.nutrients = random.randint(300, 750)
        self.cost = random.randint(20,100)
        
    def draw(self):
        pygame.draw.rect(screen, self.color, (self.x, self.y, self.size, self.size), 1)


class Agent:
    def __init__(self, x, y, v, d):
        self.x = x
        self.y = y
        self.velocity = v
        self.direction = d

    def getDistance(self, target):
        return math.sqrt((target.x - self.x) ** 2 + (target.y - self.y) ** 2)
    
    def getProjectedDistance(self, target):
        new_x = self.x + self.velocity * math.cos(self.direction)
        new_y = self.y + self.velocity * math.sin(self.direction)
        return math.sqrt((target.x - new_x) ** 2 + (target.y - new_y) ** 2)

    def getDirection(self, target):
        return math.atan2(target.y - self.y, target.x - self.x)

    def getArrivalTime(self, target):
        return self.getDistance(target) / (self.velocity + 1)

    def getTargetVelocity(self, target):
        return self.getDistance(target) / self.getArrivalTime(target)

    def getReward(self, target, delta_a, delta_r):
        acceleration_reward = -1
        direction_reward = -1
        target_velocity = self.getTargetVelocity(target)
        target_direction = self.getDirection(target)
        velocity_offset = math.tanh(target_velocity - self.velocity)
        direction_offset = math.tanh(normalizeDirection(target_direction - self.direction))
        
        if delta_a == 0 and delta_r == 0 and \
           abs(direction_offset) < 0.03 and abs(velocity_offset) < 0.4:
            direction_reward = 2
            acceleration_reward = 2
        else:
            if delta_a < 0 and self.getProjectedDistance(target) / 2 < self.velocity or \
               delta_a > 0 and self.velocity < target_velocity:
                acceleration_reward = abs(velocity_offset)
            else:
                acceleration_reward = -abs(velocity_offset)

            if delta_r < 0 and self.direction > target_direction or \
               delta_r > 0 and self.direction < target_direction:
                direction_reward = abs(direction_offset)
            else:
                direction_reward = -abs(direction_offset)

        return direction_reward + acceleration_reward


class Individual:
    def __init__(self, identity):
        self.identity = identity
        self.fitness = 0                                       # used for genetic evolution
        self.lifetime = 0
        self.alive = True
        self.father = None
        self.mother = None
        self.partner = None
        self.wants_children = False if random.random() < 0.56 else True
        self.children = 0
        self.sex = identity % 2   # 0 for female, 1 for male
        self.x = random.randint(INIT_INDIVIDUAL_RADIUS, SCREEN_WIDTH - INIT_INDIVIDUAL_RADIUS)
        self.y = random.randint(INIT_INDIVIDUAL_RADIUS, SCREEN_HEIGHT - INIT_INDIVIDUAL_RADIUS)
        self.size = INIT_INDIVIDUAL_RADIUS
        self.color = (255, 125, 125) if self.sex == 0 else (125, 125, 255)
        self.perception = INIT_INDIVIDUAL_RADIUS               # inherits from parent
        self.perception_avg = self.perception                  # becomes nextgen initial perception
        self.target = None
        self.targets_acquired = 0
        self.targets_reached = 0                               # fitness scalar
        self.food = []
        self.food_source = None
        self.partner = None
        self.employer = None
        ## Energy
        self.energy = 1250                                     # inherits average on next generation
        self.energy_avg = self.energy                          # used for threshold and next gen
        self.energy_threshold = 1000                           # inherits average / 2 on next generation
        self.energy_max = 2500                                 # inherits (max + threshold) / 2 on next generation
        ## Energy Conservation
        self.energy_conservation_avg = 1                       # used for threshold in next gen
        self.energy_conserve_threshold = 0                     # used for threshold reward, inherits avg on nextgen
        ## Direction
        self.change_in_direction = 0                            # used for energy deduction
        self.direction = math.radians(random.randint(0, 360))  # inherits avg from parents
        self.direction_avg = 0                                 # used for next gen
        ## Velocity
        self.change_in_velocity = 0                        # used for energy deduction
        self.velocity = 0
        self.velocity_avg = 0                                  # inherits avg on nextgen (used as Velocity + Avg / 2)
        self.velocity_target = 0                               # inherits avg_vel + max / 2 on nextgen (used for threshrold)
        self.velocity_max = 40                                 # needs to be adjustable
        ## Satisfaction
        self.satisfaction = 0
        self.satisfaction_avg = 0                              # passed to children as new target
        self.satisfaction_target = 100                         # inherits parents avg on nextgen
        ## Money
        self.money = 0                                         # used to buy food and determine attraction <- mutate this
        self.money_threshold = 100                             # used to determine score, passed to children
        ## Rewards
        self.attractability = 0                                # used to attracting mates - Needs to incorporate energy conservation
        self.reward = 0                                        # main fitness factor
        self.threshold_reward = 0                              # used to scale fitness
        ## Action Tables
        self.working_tally = 0
        self.eating_tally = 0
        self.mating_tally = 0
        self.target_bias = {
            "working": {"money_bias": 0, "energy_bias": 0, "satisfaction_bias": 0},
            "eating": {"money_bias": 0, "energy_bias": 0, "satisfaction_bias": 0},
            "mating": {"money_bias": 0, "energy_bias": 0, "satisfaction_bias": 0}
        }

    def getDistance(self, target):
        return math.sqrt((target.x - self.x) ** 2 + (target.y - self.y) ** 2)
    
    def energyConservation(self):
        return math.sqrt((self.energy / self.energy_max) ** 2)
    
    def getThresholdReward(self, v, t):
        return 0.0001 if v == 0 else 1 if t == 0 else v / t

    def determineThresholdRewards(self):
        etctr = self.getThresholdReward(self.energy_conservation_avg, self.energy_conserve_threshold)
        entr = self.getThresholdReward(self.energy, self.energy_threshold)
        vetr = self.getThresholdReward(self.velocity, self.velocity_target)
        satr = self.getThresholdReward(self.satisfaction, self.satisfaction_target)
        motr = self.getThresholdReward(self.money, self.money_threshold)

        return (entr + vetr + satr + motr) / 4 * etctr

    def updateAverages(self):
        self.energy_conservation_avg = (self.energy_conservation_avg + self.energyConservation()) / 2
        self.energy_avg = (self.energy_avg + self.energy) / 2
        self.satisfaction_avg = (self.satisfaction_avg + self.satisfaction) / 2
        self.perception_avg = (self.perception_avg + self.perception) / 2
        self.velocity_avg = (self.velocity_avg + self.velocity) / 2
        self.direction_avg = (self.direction_avg + self.direction) / 2
        self.threshold_reward = self.determineThresholdRewards()
    
    def getTargetBias(self, target):
        money_bias = self.target_bias[target]["money_bias"] / self.getThresholdReward(self.money, self.money_threshold)
        energy_bias = self.target_bias[target]["energy_bias"] / self.getThresholdReward(self.energy, self.energy_threshold)
        satisfaction_bias = self.target_bias[target]["satisfaction_bias"] / self.getThresholdReward(self.satisfaction, self.satisfaction_target)
        
        return (money_bias + energy_bias + satisfaction_bias ) + 1 / 3
    
    # TODO: do this if target = None, and backpropagate success
    def chooseTarget(self):
        best_target = None
        best_bias = 1
        
        for target in TARGETS.keys():
            current_target_bias = self.getTargetBias(target)
            
            if best_bias < current_target_bias:
                best_bias = current_target_bias
                best_target = target

        return best_target if best_target is not None else random.choice(list(TARGETS.keys()))
 
    def updateLocation(self):
        self.direction = normalizeDirection(self.direction)
        
        if self.velocity >= self.velocity_max:
            self.velocity_max = self.velocity

        new_x = self.x + (self.velocity * math.cos(self.direction))
        new_y = self.y + (self.velocity * math.sin(self.direction))

        if new_x < self.size or new_x > SCREEN_WIDTH - self.size:
            self.direction = math.pi - self.direction

        if new_y < self.size or new_y > SCREEN_HEIGHT - self.size:
            self.direction = 2 * math.pi - self.direction

        self.x += self.velocity * math.cos(self.direction)
        self.y += self.velocity * math.sin(self.direction)
        self.updateAverages()

    def locateTarget(self, target):
        if self.getDistance(target) < self.perception:
            return True

        return False

    def foundTarget(self, target):
        return self.getDistance(target) < (self.size + target.size)

    def getNextState(self, s, accelerate, rotate):
        new_state = Agent(s.x, s.y, s.velocity, s.direction)
        new_state.velocity += accelerate
        new_state.direction += rotate
        new_state.x += new_state.velocity * math.cos(new_state.direction)
        new_state.y += new_state.velocity * math.sin(new_state.direction)
        return new_state

    def chooseAction(self, target):
        best_reward = -1
        best_action = None
        init_agent = Agent(self.x, self.y, self.velocity, self.direction)

        for A in ACTIONS:
            agent = self.getNextState(init_agent, ACTIONS[A][0], ACTIONS[A][1])
            current_reward = agent.getReward(target, ACTIONS[A][0], ACTIONS[A][1])

            if current_reward > best_reward:
                best_reward = current_reward
                best_action = A

        if best_action is None:
            best_action = random.choice(list(ACTIONS.keys()))

        #print("male" if self.identity == 1 else "female", "best action:", best_action)

        self.change_in_velocity = ACTIONS[best_action][0]
        self.change_in_direction = ACTIONS[best_action][1]
        self.velocity += self.change_in_velocity
        self.direction += self.change_in_direction

    def navigate(self, target):
        d1 = self.getDistance(target)
        self.chooseAction(target)
        self.updateLocation()
        d2 = self.getDistance(target)
        distance_traveled = d1 - d2

        change = math.sqrt(self.change_in_velocity ** 2 + self.change_in_direction ** 2) + 1
        self.size -= change / self.size if self.size > INIT_INDIVIDUAL_RADIUS else 0
        self.energy -= math.floor(self.size / math.sqrt(distance_traveled ** 2) * change)
        self.reward += distance_traveled * self.energyConservation()
        self.perception -= distance_traveled

    def roam(self):
        change_in_velocity = random.uniform(-0.2, 0.2)
        change_in_direction = random.uniform(-0.0698, 0.0698)
        self.velocity += change_in_velocity
        self.direction += change_in_direction
        self.energy -= self.size * math.sqrt(change_in_velocity ** 2 + change_in_direction ** 2)
        self.reward -= self.size
        self.perception += self.perception
        self.updateLocation()

    def work(self, employer):
        self.money += employer.pay
        self.energy -= employer.work_load
        print("You Went To Work!")

    def hungry(self):
        # randomly eat maybe
        gamble = random.uniform(0.01, 0.5)

        if self.energy < self.energy_threshold:
            if self.energy < abs(self.energy - self.energy_threshold) or \
               random.random() < gamble:
                return True

        return False

    def eat(self, food):
        self.money -= food.cost
        self.energy += food.nutrients
        print("You Ate Food!")

    def isCompatible(self, partner):
        return self.sex != partner.sex

    def mate(self, partner):
        chance_of_reproducing = random.uniform(0.15, 0.25)
        chance_of_male = 0.51

        if random.random() < chance_of_reproducing and \
           self.wants_children and partner.wants_children:
            print("You had a baby!")
            sex = "male" if random.random() < chance_of_male else "female"
            print("It's a", sex)

    def die(self):
        self.velocity = 0
        self.color = (75, 75, 75)
        self.threshold_reward = self.determineThresholdRewards()
        self.alive = False
    
    def draw(self):
        pygame.draw.circle(screen, self.color, (self.x, self.y), self.size)

    def printStatus(self):
        print("#################### \tIndividual", self.identity, " \t#################### ")
        print("\t - fitness: \t\t", self.fitness)
        print("\t - reward: \t\t", self.reward)
        print("\t - threshold reward: \t", self.threshold_reward)
        print("\t - total lifetime: \t", self.lifetime)
        print("\t - targets reached: \t", self.targets_reached)
        print("\t - avg energy: \t", self.energy_avg)
        print("\t - avg velocity: \t", self.velocity_avg)
        print("\t - avg direction: \t", self.direction_avg)
        print("\t - avg perspective: \t", self.perception_avg)
        print("\t - avg energy conserved: ", self.energy_conservation_avg)
        print("#################### \t~~~~~~~~~~~~ \t#################### \n")


class Society:
    def __init__(self):
        self.population = [Individual(n) for n in range(INIT_POPULATION)]
        self.employers = [Work(n) for n in range(INIT_EMPLOYERS)]
        self.food_supply = [Food(n) for n in range(INIT_FOOD)]
        self.total_alive = INIT_POPULATION
    
    def mutate(self):
        mutation_rate = 0.02

        for individual in self.population:
            if not individual.alive:
                continue

            if individual.energy <= 0 or individual.satisfaction <0:
                self.total_alive -= 1
                print(individual.identity, " died.")
                individual.die()
                continue

            individual.lifetime += 1

            # not intended to change ratio of bias, added randomness
            # gives option to update new target early
            if individual.target and random.random() < mutation_rate:
                individual.target = individual.chooseTarget()
                print("Individual", individual.identity, ": ", individual.target)

            if individual.target == "working":
                if individual.employer is None:
                    for employer in self.employers:
                        if individual.locateTarget(employer):
                            individual.employer = employer.identity
                            individual.satisfaction += 10
                            break
                    individual.roam()
                else:
                    if individual.locateTarget(self.employers[individual.employer]):
                        if individual.foundTarget(self.employers[individual.employer]):
                            individual.working_tally += 1
                            individual.targets_reached += 1
                            individual.work(self.employers[individual.employer])
                            individual.target = None
                            if random.random() < mutation_rate:
                                individual.employer = None
                            individual.navigate(self.employers[individual.employer])
                        else:
                            individual.navigate(self.employers[individual.employer])
                    else:
                        individual.satisfaction -= 5
                        individual.employer = None
                        individual.roam()
            elif individual.target == "eating":
                if len(individual.food) > 1 and individual.hungry():
                    individual.eat(individual.food.pop())
                    individual.targets_acquired -= 1
                    individual.eating_tally += 1
                    individual.food_source = None
                    individual.target = None
                elif individual.food_source is None:
                    for food in self.food_supply:
                        if individual.locateTarget(food):
                            individual.food_source = food.identity
                            individual.satisfaction += 10
                            break
                    individual.roam()
                elif individual.locateTarget(self.food_supply[individual.food_source]):
                    if individual.foundTarget(self.food_supply[individual.food_source]):
                        individual.eating_tally += 1
                        individual.targets_reached += 1
                        if individual.hungry():
                            individual.eat(self.food_supply[individual.food_source])
                            self.food_supply.remove(self.food_supply[individual.food_source])
                            individual.food_source = None
                        else:
                            individual.food.append(self.food_supply[individual.food_source])
                            self.food_supply.remove(self.food_supply[individual.food_source])
                            individual.food_source = None
                        individual.target = None
                    else:
                        individual.navigate(self.food_supply[individual.food_source])
                else:
                    individual.food_source = None
                    individual.satisfaction -= 1
                    individual.roam()
            elif individual.target == "mating":
                if individual.partner is None:
                    for partner in self.population:
                        if partner.alive and partner != individual and \
                           individual.locateTarget(partner) and \
                           partner.target == "mating" and \
                           partner.identity != individual.mother and \
                           partner.identity != individual.father and \
                           individual.isCompatible(partner) and \
                           partner.isCompatible(individual) and \
                           not partner.partner:
                                individual.partner = partner.identity
                                partner.partner = individual.identity
                        else:
                           continue
                    individual.roam()
                else:
                    if individual.locateTarget(self.population[individual.partner]):
                        if individual.foundTarget(self.population[individual.partner]):
                            partner = self.population[individual.partner]
                            individual.mating_tally += 1
                            individual.targets_reached += 1
                            individual.mate(partner)
                            partner.mating_tally += 1
                            partner.targets_reached += 1
                            partner.mate(individual)
                            if random.random() < mutation_rate:
                                individual.partner = None
                                individual.partner = None
                        else:
                            individual.navigate(self.population[individual.partner])
                    else:
                        individual.partner = None
                        individual.satisfaction -= 1
                        individual.roam()
            else:
                individual.target = "working" #individual.chooseTarget()
                individual.targets_acquired += 1
                individual.satisfaction += 10
                print("Individual", individual.identity, ": ", individual.target)

    def draw(self):
        for individual in self.population:
            individual.draw()
        
        for food in self.food_supply:
            food.draw()
        
        for employer in self.employers:
            employer.draw()


    def printState(self):
        for individual in self.population:
            individual.printStatus()


class World:
    def __init__(self):
        self.society = Society()
        self.running = True

    def evolve(self):
        if self.society.total_alive > 0:
            self.society.mutate()
        elif self.running:
            self.society.printState()
            self.running = False

        return
    
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
        world.evolve()
        
        pygame.display.update()
        clock.tick(5)


if __name__ == "__main__":
    main()
