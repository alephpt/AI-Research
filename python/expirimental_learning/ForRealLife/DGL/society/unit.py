import random
from .agency.azimuth import Azimuth
from .agency import State, calculateReward
from DGL.cosmos import Log, LogLevel, Settings

map_size = Settings.GRID_SIZE.value  

# TODO: Implement a way to 'find the next target'
class Unit(Azimuth):
    '''
    An Unit should define some 'state of being'.

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
        self.success = False
        self.markets = set() 
        self.home = None
        self.moving = False         # We ONLY use this to update the Draw and 'Moving' Logics
        self.state = State.random()
    
    def index(self):
        return self.y * map_size + self.x

    def rewardFactor(self, target):
        return calculateReward(self.magnitude, self.x, self.y, target)

    def takeStep(self):
        # Calculate Rewards
        reward_obj = self.rewardFactor(self.target)
        Log(LogLevel.DEBUG, "Unit", f"\t{self.state} \
                        \n\t\t\ttarget: {self.target.type if self.target else "None"} \
                        \n\t\t\t[{self}] \
                        \n\t\t\treward:{reward_obj}\n")
        self.updateAzimuth(reward_obj)
        self.moving = True

        # Longer Lives are better
        self.reward += int(Settings.LIFETIME_REWARD_SCALAR.value * 0.01)

    # These three functions form the foundations of our future for the Genetic Learning Algorithm
    def work(self):
        Log(LogLevel.VERBOSE, "Unit", f"{self} is working")
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
        Log(LogLevel.VERBOSE, "Unit", f"{self} is choosing the best target:")
        best_value = 0
        best_target = None

        for target in targets:
            Log(LogLevel.VERBOSE, "Unit", f"\tTarget: {target}")
            value = value_fn(target)

            if value > best_value: # or value < 0: # This would account for a negative reward
                best_value = value
                best_target = target

        return best_target


    def chooseBestAction(self):
        Log(LogLevel.VERBOSE, "Unit", f"is looking at target {self.target.type if self.target else 'undefined'}")
        
        if self.target is None:
            #
            # Step 1. Let the unit choose a random target ?
            #
            if self.state.needy():
                if self.state in [State.Broke, State.Hungry]:
                    self.target = self.chooseBestTarget(self.markets, lambda market: self.rewardFactor(market)['magnitude'])
                elif self.state == State.Horny:
                    self.target = self.chooseBestTarget(self.home, lambda home: self.rewardFactor(home)['magnitude'])

            #
            # Step 2. Let the unit choose the best target
            # 
            else:
                self.target = random.choice([*self.markets, *self.home])


                # Step 3. Let the unit choose when to change targets
                # Step 4. Let the unit learn to choose the best target
                # Step 5. Always make it do a little bit of randomness
            return

    def updateEnergy(self):
        self.age += 1
        
        if self.state == State.Alive:
            # This is where we would increase happiness and compassion, integrity, and decrease fatigue
            return


    # This update function should way potential opportcellies, and pick a course of actions,
    # and then update state, reward, and update the Q Table. # 'Caching' happens on the Epoch level
    def updateValues(self):
        '''
        increases age, decreases energy, and moves the unit in a direction
        '''
        Log(LogLevel.VERBOSE, "Unit", f"{self} is updating values")
        self.updateEnergy()

        ## Percent of Randomness
        # We have the ability to move in a direction with some randomness
        if self.state == State.Alive or random.uniform(0.0, 1.0) < Settings.IMPULSIVITY.value:
            Log(LogLevel.VERBOSE, "Unit", f"{self} is choosing a random action")
            self.chooseRandomState()

        # TODO: Look ahead at the next square based on the choice given from the Q Table
        self.chooseBestAction()
        self.takeStep()

        Log(LogLevel.VERBOSE, "Unit", f"{self} is choosing a calculated action")

        # I think this is for 'catching' any potential lock states.
        if self.state in [State.Dead]:
            Log(LogLevel.ERROR, "Unit:", f"{self} caught in state: {self.state}")
            return
        
        # NOTE: They will get stuck here. We need to implement a target obj system
        # if self.state == State.Sleeping:
        #     Log(LogLevel.VERBOSE, f"Unit {self} is resting")
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
            Log(LogLevel.VERBOSE, "Unit", f"{self.idx})")
            
            if self.target_direction is not None:
                Log(LogLevel.VERBOSE, "Unit", f"{self.idx} - Moving in ({self.target_direction})")
                self.energy -= 1
                dx, dy = self.target_direction

                # Make sure we are within bounds
                if 0 <= self.x + dx <= map_size - 1 and 0 <= self.y + dy <= map_size - 1:
                    self.x += dx
                    self.y += dy

            self.moving = False
        
        # Check if we are dead yet
        if self.energy < 0:
            print(f"Unit {self} has died")
            self.state = State.Dead

