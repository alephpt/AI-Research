from DGL.micro import Log, LogLevel
from .agency import State, calculateReward
from .azimuth import Azimuth

from DGL.micro import Settings
import random

map_size = Settings.GRID_SIZE.value  

# TODO: Implement a way to 'find the next target'
class Agent(Azimuth):
    '''
    params:
    age, energy, wealth, state, generation, happiness, sleep_counter, success, moving, direction
    '''
    def __init__(self, idx):
        super().__init__(idx)
        self.age = 0 # If we add Age in it becomes a gradient

        # State Space
        self.energy = Settings.INITIAL_E.value # Should we clamp energy
        self.wealth = Settings.INITIAL_W.value # Should we add 'economic' factors? .. If we add a "legal" requirement, will it figure it out?
        self.generation = 0
        #self.happiness = 0
        #self.food_counter = 0
        #self.sleep_counter = 0
        self.target = None
        self.success = False
        self.markets = set() 
        self.home = set()
        self.moving = False         # We ONLY use this to update the Draw and 'Moving' Logics
    
    def index(self):
        return self.y * map_size + self.x

    def rewardFactor(self, target):
        return calculateReward(self.magnitude, self.x, self.y, target)

    def takeStep(self):
        # Calculate Rewards
        reward_obj = self.rewardFactor(self.target)
        Log(LogLevel.DEBUG, f"Agent {self} is taking a step towards {self.target.type} {reward_obj}")
        self.updateAzimuth(reward_obj)
        self.moving = True

        # Longer Lives are better
        self.reward += 1 * Settings.LIFETIME_REWARD_SCALAR.value

    # These three functions form the foundations of our future for the Genetic Learning Algorithm
    def work(self):
        Log(LogLevel.VERBOSE, f"Agent {self} is working")
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

    # Option 1: Iterate through all a select group of targets, and choose the one with the lowest magnitude
    # Option 2: Iterate through all possible targets, and choose the one with the highest reward
    def chooseBestTarget(self, targets, value_fn): # We are going to pass a callback in to allow for a dynamic reward function
        Log(LogLevel.DEBUG, f"Agent {self} is choosing the best target:")
        best_value = 0
        best_target = None

        for target in targets:
            Log(LogLevel.DEBUG, f"\tTarget: {target}")
            value = value_fn(target)

            if value > best_value: # or value < 0: # This would account for a negative reward
                best_value = value
                best_target = target

        return best_target


    def chooseBestAction(self):
        Log(LogLevel.DEBUG, f"Agent is looking at target {self.target}")
        
        if self.target is None:
            #
            # Step 1. Let the agent choose a random target ?
            #
            if self.state.needy():
                if self.state in [State.Broke, State.Hungry]:
                    self.target = self.chooseBestTarget(self.markets, lambda market: self.rewardFactor(market)['magnitude'])
                elif self.state == State.Horny:
                    self.target = self.chooseBestTarget(self.home, lambda home: self.rewardFactor(home)['magnitude'])

            #
            # Step 2. Let the agent choose the best target
            # 
            else:

                self.target = random.choice(self.markets.union(self.home))


                # Step 3. Let the agent choose when to change targets
                # Step 4. Let the agent learn to choose the best target
                # Step 5. Always make it do a little bit of randomness
            return

    def updateEnergy(self):
        self.age += 1
        
        if self.state == State.Alive:
            # This is where we would increase happiness and compassion, integrity, and decrease fatigue
            return


    # This update function should way potential opportunities, and pick a course of actions,
    # and then update state, reward, and update the Q Table. # 'Caching' happens on the Epoch level
    def updateValues(self):
        '''
        increases age, decreases energy, and moves the agent in a direction
        '''
        Log(LogLevel.DEBUG, f"Agent {self} is updating values")
        self.updateEnergy()

        ## Percent of Randomness
        # We have the ability to move in a direction with some randomness
        if self.state == State.Alive or random.uniform(0.0, 1.0) < Settings.IMPULSIVITY.value:
            Log(LogLevel.DEBUG, f"Agent {self} is choosing a random action")
            self.chooseRandomState()
            self.moving = True
        else:
            self.chooseBestAction()
            self.takeStep()

            Log(LogLevel.INFO, f"Agent {self} is choosing a calculated action")
            # TODO: Look ahead at the next square based on the choice given from the Q Table

        # I think this is for 'catching' any potential lock states.
        if self.state in [State.Dead]:
            Log(LogLevel.ERROR, f"Agent {self} caught in state: {self.state}")
            return
        
        # NOTE: They will get stuck here. We need to implement a target obj system
        # if self.state == State.Sleeping:
        #     Log(LogLevel.VERBOSE, f"Agent {self} is resting")
        #     self.energy += Settings.RESTING_VALUE.value
        
        #     if self.sleep_counter < Settings.MAX_SLEEP.value:
        #         self.sleep_counter += 1
        #         #self.sleep()
        #     else:
        #         self.state = State.Alive
        #         self.sleep_counter = 0
        #     return
        
        # Update location if we are moving
        if self.moving:
            Log(LogLevel.VERBOSE, f"Agent {self.idx})")
            
            if self.target_direction is not None:
                Log(LogLevel.VERBOSE, f"Agent {self.idx} - Moving in ({self.target_direction})")
                self.energy -= 1
                dx, dy = self.target_direction

                # Make sure we are within bounds
                if 0 <= self.x + dx <= map_size - 1 and 0 <= self.y + dy <= map_size - 1:
                    self.x += dx
                    self.y += dy

            self.moving = False
        
        # Check if we are dead yet
        if self.energy < 0:
            print(f"Agent {self} has died")
            self.state = State.Dead

