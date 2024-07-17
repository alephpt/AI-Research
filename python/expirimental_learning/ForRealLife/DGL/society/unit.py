import random
from .agency.azimuth import Azimuth
from .agency import State
from DGL.cosmos import Log, LogLevel, Settings
from DGL.cosmos.closet import p

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
        self.happiness = 0
        self.food_counter = 0
        self.sleep_counter = 0
        self.success = False
        self.markets = set() 
        self.home = None
        self.moving = False         # We ONLY use this to update the Draw and 'Moving' Logics
        self.state = State.random()
    
    def index(self):
        return self.y * map_size + self.x

    ## Reward Functions for our Units Movement towards the Target
    def seekTarget(self, prev_d, x, y, target):
        '''
        'findBest' utility function to calculate the reward for the unit

        Returns:
        (distance, cartesian_direction_vector)

        Parameters:
        prev_d: float - The previous distance to the target
        x: int - The x coordinate of the unit
        y: int - The y coordinate of the unit
        target: Cell - The target of the unit
                '''
        # If the previous distance is 0, we are at the target
        if target is None:
            return (None, (0, 0)) 

        if prev_d is 0:
            return (0, (0, 0))

        return p(x, y, target.x, target.y)

    def takeStep(self):
        self.magnitude, self.target_direction = self.seekTarget(self.magnitude, self.x, self.y, self.target)

        # These variables could be mapped to the State Space with a random deviation
        if self.magnitude == 0:
            self.happiness += 100 # These are considered short term rewards
            self.energy += 10 # This should also insentivize the unit to move short distances
            Log(LogLevel.INFO, "Azimuth", f"{self} has reached target")
            self.target_reached = True # We should be able to factor this out by checking state
            self.reward += 1000
            
            # # TODO: We need to trigger a movement to some other state
            self.moving = True


        # Longer Lives are better
        self.reward += int(Settings.LIFETIME_REWARD_SCALAR.value * 0.01)

    # These three functions form the foundations of our future for the Genetic Learning Algorithm
    def work(self):
        #Log(LogLevel.VERBOSE, "Unit", f"{self} is working")
        if self.energy >= 10:
            self.wealth += Settings.WORK_REWARD.value
            self.energy -= Settings.WORK_COST.value
            self.happiness += Settings.WORK_PLEASURE_FACTOR.value

    def eat(self):
        if self.wealth >= Settings.COST_OF_GOODS.value:
            self.wealth -= Settings.COST_OF_GOODS.value
            self.energy += Settings.FOOD_REWARD.value # TODO: HUGE TEST - In Isolation, determine if fixed values or random values are better
            self.happiness += Settings.FOOD_PLEASURE_FACTOR.value
    
    # This function forms as the gradle of our generational genetics
    def sex(self):
        #  TODO: Integrate a large amount of Chance here via Reproduction
        self.energy -= Settings.SEX_COST.value
        self.happiness += Settings.SEX_PLEASURE_FACTOR.value  

    # Option 1: Iterate through all a select group of targets, and choose the one with the lowest magnitude
    # Option 2: Iterate through all possible targets, and choose the one with the highest reward
    def chooseBestTarget(self, targets): # We are going to pass a callback in to allow for a dynamic reward function
        best_value = 0
        best_target = None

        for target in targets:
            distance, _ = self.seekTarget(self.magnitude, self.x, self.y, target)

            if distance < best_value or (distance == 0 and best_value != 0): # or value < 0: # This would account for a negative reward
                Log(LogLevel.WARNING, "Unit", f"\t\tFound New Best Value: {distance} > {best_value}")
                best_value = distance
                best_target = target

        return best_target


    def chooseBestAction(self):
        #Log(LogLevel.VERBOSE, "Unit", f"is looking at target {self.target.type if self.target else 'undefined'}")
        
        if self.target is None:
            #
            # Step 1. Let the unit choose a random target ?
            #
            if self.state.needy():
                Log(LogLevel.INFO, "Unit", f"{self} is in a needy state of {self.state}")
                if self.state in [State.Broke, State.Hungry]:
                    self.target = self.chooseBestTarget(self.markets)
                elif self.state == State.Horny:
                    self.target = self.chooseBestTarget(self.home)

                if self.target is None:
                    Log(LogLevel.WARNING, "Unit", f"{self} is in a needy state, but no target was found")

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


    # This update function should pick a state,
    # and then update state, reward, and update the Q Table. # 'Caching' happens on the Epoch level
    def UpdateState(self):
        '''
        increases age, decreases energy, and moves the unit in a direction
        '''
        self.updateEnergy()

        # We have the ability to move in a direction with some randomness
        if self.state == State.Alive or random.uniform(0.0, 1.0) < Settings.randomImpulse():
            Log(LogLevel.VERBOSE, "Unit", f"{self} is choosing a random action")
            self.chooseRandomState()

        # TODO: Look ahead at the next square based on the choice given from the Q Table
        self.chooseBestAction()
        self.takeStep()

        #Log(LogLevel.VERBOSE, "Unit", f"{self} is choosing a calculated action")

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
            #Log(LogLevel.VERBOSE, "Unit", f"ID:{self.idx}")
            
            if self.target_direction is not None:
                #og(LogLevel.VERBOSE, "Unit", f"Moving towards ({self.target_direction})")
                self.energy -= 1
                dx, dy = self.target_direction

                # Make sure we are within bounds
                if 0 <= self.x + dx <= map_size - 1 and 0 <= self.y + dy <= map_size - 1:
                    self.x += dx
                    self.y += dy
            #else:
                #Log(LogLevel.VERBOSE, "Unit", f"no target direction found.")

            self.moving = False

        # Check if we are dead yet
        if self.energy < 0:
            print(f"Unit {self} has died")
            self.state = State.Dead

