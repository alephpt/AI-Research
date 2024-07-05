from .agency import Status, calculateReward
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
        self.status = Status.Alive
        self.generation = 0
        self.happiness = 0
        self.target_direction_vector = (0, 0)

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

    # TODO: Integrate with Queue Table
    def move(self):
        self.energy -= 1
        dx, dy = self.chosen_direction.Vector()

        if 0 <= self.x + dx <= self.map_size - 1 and 0 <= self.y + dy <= self.map_size - 1:
            self.x += dx
            self.y += dy
  
    # Currently only checks if we are dead or not
    def updateState(self):
        # Update the State Space
        if self.energy < 0:
            print(f"Agent {self} has died")
            self.status = Status.Dead

        # We need to determine how to reward our self for what we are doing

        # Update the Q Table


    # This update function should way potential opportunities, and pick a course of actions,
    # and then update state, reward, and update the Q Table. # 'Caching' happens on the Epoch level
    def update(self, findTarget):
        self.age += 1

        # All of these get updated when they are doing something, or dead
        if self.status in [Status.Dead, Status.Sex, Status.Working, Status.Eating, Status.Alive]:
            return
        
        # NOTE: They will get stuck here. We need to implement a target obj system
        if self.status == Status.Sleeping:
            self.energy += Settings.RESTING_VALUE.value
        
            # If max sleep = 0
                # Determine max sleep
                # set sleep to max sleep
            
            # decrease our sleep

            # If sleep = 0
                # Status.Waking


            return
        
        if self.status == Status.Moving:
            self.move()

            ## Check if we are at the target
                ## If we are at the target we get 100 points


        ## Percent of Randomness
        # We have the ability to move in a direction with some randomness
        if self.target is None or random.uniform(0.0, 1.0) < Settings.IMPULSIVITY.value:
            self.chooseRandomAction(findTarget)
        else:
            print("Choosing Best Action")
            # Choose the best action
            # TODO: Look ahead at the next square based on the Q Table OR Do a Random Walk
            # This has to be before the move to ensure the target exists


        # Calculate Collissionsagent for agent in self.population if agent.status != Status.Dead
        self.takeStep()

        # Update the state space
        self.updateState()