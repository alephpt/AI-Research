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
    "maintain": (0, 0),
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
        self.work_load = random.randint(50,200)
        self.pay = random.randint(75,150)
    
    def draw(self):
        pygame.draw.rect(screen, self.color, (self.x, self.y, self.size, self.size), 1)


class Food:
    def __init__(self):
        self.x = random.randint(INIT_FOOD_SIZE, SCREEN_WIDTH - INIT_FOOD_SIZE)
        self.y = random.randint(INIT_FOOD_SIZE, SCREEN_HEIGHT - INIT_FOOD_SIZE)
        self.size = INIT_FOOD_SIZE
        self.color = (0, 183, 69)
        self.nutrients = random.randint(300, 750)
        self.cost = random.randint(20,100)
        
    def draw(self):
        pygame.draw.rect(screen, self.color, (self.x, self.y, self.size, self.size), 1)


class Proto:
        def __init__(self, x, y, a, v, r, d):
            self.x = x
            self.y = y
            self.acceleration = a
            self.velocity = v
            self.rotation = r
            self.direction = d

        def getDistance(self, target):
            return math.sqrt((target.x - self.x) ** 2 + (target.y - self.y) ** 2)
        
        def getOrientation(self, target):
            dot = (target.x * self.x) + (target.y * self.y)
            self_orientation = math.sqrt(self.x ** 2 + self.y ** 2)
            target_orientation = math.sqrt(target.x ** 2 + target.y ** 2)

            return math.acos(dot / (self_orientation * target_orientation))


class Individual:
    def __init__(self, identity):
        self.identity = identity
        self.fitness = 0                                       # used for genetic evolution
        self.lifetime = 0
        self.alive = True
        self.father = None
        self.mother = None
        self.partner = None
        self.children = 0
        self.sex = identity % 2   # 0 for female, 1 for male
        self.x = random.randint(INIT_INDIVIDUAL_RADIUS, SCREEN_WIDTH - INIT_INDIVIDUAL_RADIUS)
        self.y = random.randint(INIT_INDIVIDUAL_RADIUS, SCREEN_HEIGHT - INIT_INDIVIDUAL_RADIUS)
        self.size = INIT_INDIVIDUAL_RADIUS
        self.color = (255, 125, 125) if self.sex == 0 else (125, 125, 255)
        self.perception = INIT_INDIVIDUAL_RADIUS               # inherits from parent
        self.perception_avg = self.perception                  # inherits from parents and becomes new default perspective
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
        self.energy_conserv_threshold = 0                      # used for threshold reward, inherits avg on nextgen
        ## Rotation
        self.rotation = 0
        self.change_in_rotation = 0                           # used for energy deduction
        ## Direction
        self.direction = math.radians(random.randint(0, 360))  # inherits avg from parents
        self.direction_avg = 0                                 # used for next gen
        ## Acceleration
        self.acceleration = 0
        self.change_in_acceleration = 0                        # used for energy deduction
        self.acceleration_avg = 0                              # used for threshold and next gen
        self.acceleration_threshold = 0                        # inherits avg_acceleration on next generation
        ## Velocity
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
        etctr = self.getThresholdReward(self.energy_conservation_avg, self.energy_conserv_threshold)
        entr = self.getThresholdReward(self.energy, self.energy_threshold)
        attr = self.getThresholdReward(self.acceleration, self.acceleration_threshold)
        vetr = self.getThresholdReward(self.velocity, self.velocity_target)
        satr = self.getThresholdReward(self.satisfaction, self.satisfaction_target)
        motr = self.getThresholdReward(self.money, self.money_threshold)

        return (entr + attr + vetr + satr + motr) / 5 * etctr

    def updateAverages(self):
        self.energy_conservation_avg = (self.energy_conservation_avg + self.energyConservation()) / 2
        self.energy_avg = (self.energy_avg + self.energy) / 2
        self.satisfaction_avg = (self.satisfaction_avg + self.satisfaction) / 2
        self.perception_avg = (self.perception_avg + self.perception) / 2
        self.acceleration_avg = (self.acceleration_avg + self.acceleration) / 2
        self.velocity_avg = (self.velocity_avg + self.velocity) / 2
        self.direction_avg = (self.direction_avg + self.direction) / 2
        self.threshold_reward = self.determineThresholdRewards()
    
    def getTargetBias(self, target):
        money_bias = self.target_bias[target]["money_bias"] / self.getThresholdReward(self.money, self.money_threshold)
        energy_bias = self.target_bias[target]["energy_bias"] / self.getThresholdReward(self.energy, self.energy_threshold)
        satisfaction_bias = self.target_bias[target]["satisfaction_bias"] / self.getThresholdReward(self.satisfaction, self.satisfaction_target)
        
        return (money_bias + energy_bias + satisfaction_bias ) + 1 / 3
    
    def updateTargetBias(self):
        self.target_bias[self.target]["money_bias"] = self.getThresholdReward(self.working_tally, self.targets_acquired) / \
                                                      self.getThresholdReward(self.money, self.money_threshold) * self.targets_reached

        self.target_bias[self.target]["energy_bias"] = self.getThresholdReward(self.eating_tally, self.targets_acquired) / \
                                                       self.getThresholdReward(self.money, self.money_threshold) * self.targets_reached

        self.target_bias[self.target]["satisfaction_bias"] = self.getThresholdReward(self.mating_tally, self.targets_acquired) /  \
                                                             self.getThresholdReward(self.money, self.money_threshold) * self.targets_reached

    # TODO: do this if target = None, and backpropagate success
    def chooseTarget(self):
        best_target = None
        best_bias = 1
        
        for target in TARGETS.keys():
            current_target_bias = self.getTargetBias(target)
            
            if best_bias < current_target_bias:
                best_bias = current_target_bias
                best_target = target
            
        return best_target if best_target != None else random.choice(list(TARGETS.keys()))
 
    def updateLocation(self):
        self.velocity += self.acceleration
        self.direction += self.rotation
        
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

    def getNextState(self, old_state, action):
        new_state = old_state
        new_state.acceleration += action[0]
        new_state.rotation += action[1]
        new_state.velocity += new_state.acceleration
        new_state.direction += new_state.rotation
        new_state.x += new_state.velocity * math.cos(new_state.direction)
        new_state.y += new_state.velocity * math.sin(new_state.direction)
        return new_state

    def chooseAction(self, target):
        best_reward = SCREEN_WIDTH * SCREEN_HEIGHT
        best_action = None
        init_proto = Proto(self.x, self.y, self.acceleration, self.velocity, self.rotation, self.direction,)

        for A in ACTIONS:
            proto_A = self.getNextState(init_proto, ACTIONS[A])
            for C in ACTIONS:
                proto_C = self.getNextState(proto_A, ACTIONS[C])
                for T in ACTIONS:
                    proto_T = self.getNextState(proto_C, ACTIONS[T])
                    current_reward = proto_T.getDistance(target) * proto_T.getOrientation(target)

                    if current_reward < best_reward:
                        best_reward = current_reward
                        best_action = A

        if best_action == None: 
            best_action = random.choice(list(ACTIONS.keys()))

        print(self.identity, "best action:", best_action)

        self.change_in_acceleration = ACTIONS[best_action][0]
        self.change_in_rotation = ACTIONS[best_action][1]
        self.acceleration += self.change_in_acceleration
        self.rotation += self.change_in_rotation

    def navigate(self, target):
        self.energy -= 1

        d1 = self.getDistance(target)
        self.chooseAction(target)
        self.updateLocation()
        d2 = self.getDistance(target)
        distance = d1 - d2

        change = math.sqrt((self.change_in_acceleration + self.change_in_rotation) ** 2)
        self.size -= change / self.size if self.size > INIT_INDIVIDUAL_RADIUS else 0
        self.energy -= math.floor(self.size + math.sqrt(distance ** 2) * change)
        self.reward += distance * self.energyConservation()
        self.perception -= distance + 1 if distance > 0 else distance - 1

    def roam(self):
        self.acceleration += random.uniform(-0.4, 0.4)
        self.rotation += random.uniform(-0.0698, 0.0698)
        self.energy -= self.perception / INIT_INDIVIDUAL_RADIUS
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

        if random.random() < chance_of_reproducing:
            print("You had a baby!")
            sex = "male" if random.random() < chance_of_male else "female"
            print("It's a", sex)

    def die(self):
        self.velocity = 0
        self.color = (75, 75, 75)
        self.updateTargetBias()
        self.threshold_reward = self.determineThresholdRewards()
        self.alive = False
    
    def draw(self):
        pygame.draw.circle(screen, self.color, (self.x, self.y), self.size)

class Society:
    def __init__(self):
        self.population = [Individual(n) for n in range(INIT_POPULATION)]
        self.employers = [Work() for _ in range(INIT_EMPLOYERS)]
        self.food_supply = [Food() for _ in range(INIT_FOOD)]
        self.total_alive = INIT_POPULATION
    
    def mutate(self):
        mutation_rate = 0.05

        for individual in self.population:
            individual.lifetime += 1
            
            if not individual.alive:
                continue

            if individual.energy <= 0 or individual.satisfaction <0:
                self.total_alive -= 1
                print(individual.identity, " died.")
                individual.die()
                continue

            # not intented to change ratio of bias
            # gives option to update new target early
            #if individual.target != None and random.random() < mutation_rate / 3:
            #   individual.target = individual.chooseTarget()
            #    print("Individual", self.population.index(individual), ": ", individual.target)


            if individual.target == "working":
                if individual.employer == None:
                    for employer in self.employers:
                        if individual.locateTarget(employer):
                            individual.employer = self.employers.index(employer)
                            individual.satisfaction += 10
                            break
                    individual.roam()
                else:
                    if individual.locateTarget(self.employers[individual.employer]):
                        if individual.foundTarget(self.employers[individual.employer]):
                            individual.working_tally += 1
                            individual.targets_reached += 1
                            individual.updateTargetBias()
                            individual.work(self.employers[individual.employer])
                            individual.target = None
                            if random.random() < mutation_rate:
                                individual.employer = None
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
                    individual.food_source = None
                    individual.target = None
                elif individual.food_source == None:
                    for food in self.food_supply:
                        if individual.locateTarget(food):
                            individual.food_source = self.food_supply.index(food)
                            individual.satisfaction += 10
                            break
                    individual.roam()
                elif individual.locateTarget(self.food_supply[individual.food_source]):
                    if individual.foundTarget(self.food_supply[individual.food_source]):
                        individual.eating_tally += 1
                        individual.targets_reached += 1
                        individual.updateTargetBias()
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
                if individual.partner == None:
                    for partner in self.population:
                        if partner.alive and partner != individual and \
                           individual.locateTarget(partner) and \
                           partner.identity != individual.mother and \
                           partner.identity != individual.father and \
                           individual.isCompatible(partner) and \
                           partner.isCompatible(individual) and \
                           partner.partner == None:
                                individual.mate = self.population.index(partner)
                                partner.mate = self.population.index(individual)
                        else:
                           continue
                    individual.roam()
                else:
                    if individual.locateTarget(self.population[individual.partner]):
                        if individual.foundTarget(self.population[individual.partner]):
                            individual.mating_tally += 1
                            individual.targets_reached += 1
                            individual.updateTargetBias()
                            individual.partner(partner)
                            partner.mating_tally += 1
                            partner.targets_reached += 1
                            partner.updateTargetBias()
                            partner.partner(individual)
                            if random.random() < mutation_rate:
                                individual.mate = None
                                individual.mate = None
                        else:
                            individual.navigate(self.population[individual.partner])
                    else:
                        individual.partner = None
                        individual.satisfaction -= 1
                        individual.roam()
            else:
                individual.target = individual.chooseTarget()
                individual.targets_acquired += 1
                individual.satisfaction += 10
                print("Individual", self.population.index(individual), ": ", individual.target)


    
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

    def evolve(self):
        self.society.mutate()
    
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
