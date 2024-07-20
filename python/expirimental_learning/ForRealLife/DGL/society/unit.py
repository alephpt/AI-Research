import random

from DGL.engine.learning import EncoderState, EthicsMatrix
from .agency import State, Azimuth
from DGL.cosmos import Log, LogLevel, Settings
from DGL.cosmos.closet import p, track

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
        self.fatigue = 10
        self.energy = Settings.INITIAL_E.value # Should we clamp energy
        self.wealth = Settings.INITIAL_W.value # Should we add 'economic' factors? .. If we add a "legal" requirement, will it figure it out?
        self.generation = 0
        self.happiness = 0
        self.food_counter = 0
        self.sleep_counter = 0
        self.success = False
        self.moving = False         # We ONLY use this to update the Draw and 'Moving' Logics
        self.state = State.random()
        print(self)
    
    def __str__(self):
        return f"Unit {self.idx}-[{self.xy()}][{self.ethicsNames()}]"

    def logState(self):
        Log(LogLevel.INFO, "Unit", f"{self} is in state {self.state}")
    
    def index(self):
        return self.y * map_size + self.x

    def move(self):
        if self.target_direction is not None:
            Log(LogLevel.VERBOSE, "Unit", f"Moving towards ({self.target_direction})")
            self.energy -= 1
            dx, dy = self.target_direction

            # Make sure we are within bounds
            if 0 <= self.x + dx <= map_size - 1 and 0 <= self.y + dy <= map_size - 1:
                self.x += dx
                self.y += dy
            #else:
                #Log(LogLevel.VERBOSE, "Unit", f"no target direction found.")

            self.moving = False

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

    def sleep(self):
        #Log(LogLevel.VERBOSE, f"Unit {self} is resting")
        self.energy += Settings.RESTING_VALUE.value
    
        if self.sleep_counter < Settings.MAX_SLEEP.value:
            self.sleep_counter += 1
            #self.sleep()
        else:
            self.state = State.Alive
            self.sleep_counter = 0
        return

    def getOlder(self):
        '''Adds 1 to our age'''
        self.age += 1
        # Longer Lives are better
        self.reward += int(self.age // Settings.LIFETIME_REWARD_SCALAR.value)

    def energyLoss(self):
        '''
        Decreases the energy of the unit by 1
        '''
        if self.wuwei():
            return
        self.energy -= 1
    
    def lifeSuck(self):
        '''
        Decreases the happiness of the unit by 1 if it's greater than 0, and we're not content.
        '''
        if self.content():
            self.happiness += 1
            self.gainCompassion(10)
        elif self.happiness > 0:
            self.happiness -= 1

    def randomTarget(self):
        return random.choice([*self.markets, *self.home])

    def act(self):
        if self.moving:
            Log(LogLevel.WARNING, "Unit", f"{self} is moving to {self.target_direction}")
            self.move()
        #if self.state == State.Sleeping:
        #    self.sleep()

    def orient(self):
        ## TODO: Calculate the entire number of steps needed to reach the target in a list and move this to whenever we 'Pull_*' from the pool
        self.directionVec = track(self.magnitude, self.x, self.y, self.target_selection)
        self.magnitude = self.directionVec.magnitude
        self.target_action = self.directionVec.MoveAction()

        # These variables could be mapped to the State Space with a random deviation
        if self.magnitude == 0:
            self.happiness += 100 # These are considered short term rewards
            self.fatigue -= 10 # This should also insentivize the unit to move short distances
            Log(LogLevel.INFO, "Azimuth", f"{self} has reached target")
            self.target_reached = True # We should be able to factor this out by checking state
            self.reward += 1000
            
            # # TODO: We need to trigger a movement to some other state
            self.moving = True

    # This update function should pick a state,
    # and then update state, reward, and update the Q Table. # 'Caching' happens on the Epoch level
    def updateState(self):
        '''
        increases age, decreases energy, and moves the unit in a direction
        '''
        self.getOlder()
        self.energyLoss()
        self.lifeSuck()

        # I think this is for 'catching' any potential lock states.
        if self.state in [State.Dead]:
            Log(LogLevel.ERROR, "Unit:", f"{self} caught in state: {self.state}")
            return

        # This will randomly spike randomness in the system 'interupting' our Agents decision making process
        if self.state == State.Alive or random.uniform(0.0, 1.0) < Settings.randomImpulse():
            self.chooseRandomState()
            self.chooseRandomEncoder()
        
        # We can have some random action and still not do anything if we have a target_action of Nothing.
        if self.wuwei():
            return

        # TODO: Look ahead at the next square based on the choice given from the Q Table
        self.coincidence()
        self.orient()
        self.act()

        # Check if we are dead yet
        if self.energy < 0:
            print(f"Unit {self} has died")
            self.state = State.Dead






def testEthicsMatrix():
    print("Testing Ethics Matrix")

    Unit1 = Unit(0)
    Unit2 = Unit(1)
    
    print(f"Unit1 Ethics XY: {Unit1.ethics()}")
    print(f"Unit2 Ethics XY: {Unit2.ethics()}")

    Unit1.evaluatePotential(Unit2)
    Unit2.evaluatePotential(Unit1)