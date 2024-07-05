from DGL.micro.utilities.logger import Log, LogLevel
from .agency import State, calculateReward
from .azimuth import Azimuth

from DGL.micro import Settings, Unit
import random

# TODO: Implement a way to 'find the next target'
class Agent(Azimuth):
    def __init__(self, idx):
        super().__init__(idx)

        self.map_size = Settings.GRID_SIZE.value  
        self.age = 0
        self.energy = Settings.INITIAL_E.value # Should we clamp energy
        self.wealth = Settings.INITIAL_W.value # Should we add 'economic' factors? .. If we add a "legal" requirement, will it figure it out? 
        self.state = State.Alive
        self.generation = 0
        self.happiness = 0
        self.sleep_counter = 0
        self.success = False
        self.moving = False
        self.action = (0, 0)

    def takeStep(self):
    # Calculate Rewards
        reward_obj = calculateReward(self.magnitude, self.x, self.y, self.target, self.action)
        self.updateAzimuth(reward_obj)

        # Longer Lives are better
        self.reward += 1 * Settings.LIFETIME_REWARD_SCALAR.value

    # These three functions form the foundations of our future for the Genetic Learning Algorithm
    def work(self):
        if self.energy >= 10:
            self.wealth += Settings.WORK_REWARD.value
            self.energy -= Settings.WORK_COST.value
            self.happiness += Settings.WORK_PLEASURE_FACTOR

    def eat(self):
        if self.wealth >= Settings.COST_OF_GOODS.value:
            self.wealth -= Settings.COST_OF_GOODS.value
            self.energy += Settings.FOOD_REWARD # TODO: HUGE TEST - In Isolation, determine if fixed values or random values are better
            self.happiness += Settings.FOOD_PLEASURE_FACTOR.value
    
    # This function forms as the gradle of our generational genetics
    def sex(self):
        #  TODO: Integrate a large amount of Chance here via Reproduction
        self.energy -= Settings.SEX_COST.value
        self.happiness += Settings.SEX_PLEASURE_FACTOR.value  


    # This update function should way potential opportunities, and pick a course of actions,
    # and then update state, reward, and update the Q Table. # 'Caching' happens on the Epoch level
    def update(self):
        self.age += 1
        self.energy -= 1

        ## Percent of Randomness
        # We have the ability to move in a direction with some randomness
        if self.state == State.Alive or self.target is None or random.uniform(0.0, 1.0) < Settings.IMPULSIVITY.value:
            Log(LogLevel.VERBOSE, f"Agent {self} is choosing a random action")
            self.chooseRandomAction()
            self.moving = True
        else:
            print("Choosing Best Action")
            # Choose the best action
            # TODO: Look ahead at the next square based on the Q Table OR Do a Random Walk
            # This has to be before the move to ensure the target exists

        # All of these get updated when they are doing something, or dead
        if self.state in [State.Dead, State.Sex, State.Working, State.Eating]:
            Log(LogLevel.ERROR, f"Agent {self} caught in state: {self.state}")
            return
        
        # NOTE: They will get stuck here. We need to implement a target obj system
        if self.state == State.Sleeping:
            Log(LogLevel.VERBOSE, f"Sleeping")
            self.energy += Settings.RESTING_VALUE.value
        
            if self.sleep_counter < Settings.MAX_SLEEP.value:
                self.sleep_counter += 1
                #self.sleep()
            else:
                self.state = State.Alive
                self.sleep_counter = 0
            return
        
        if self.moving:
            Log(LogLevel.VERBOSE, f"{self.index} - Moving to {self.target}@({self.target_direction_vector.Vector()})")
            self.energy -= 1
            dx, dy = self.target_direction_vector.Vector()

            if 0 <= self.x + dx <= self.map_size - 1 and 0 <= self.y + dy <= self.map_size - 1:
                self.x += dx
                self.y += dy
        
        if self.energy < 0:
            print(f"Agent {self} has died")
            self.state = State.Dead
