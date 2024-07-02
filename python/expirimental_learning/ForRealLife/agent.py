
from enum import Enum
from placement import Placement
import random
import math

# TODO: Add Color Factor or Text Status
# These contribute to our State Space
# But we do not want to map states to variables
class Status(Enum):
    Dead = -1
    Sleeping = 0 #TODO: Implement 'Home' Unit
    Eating = 1
    Working = 2
    Sex = 3 #TODO: Integrate at Home Unit
    Alive = 7
    Horny = 4
    Broke = 5
    Hungry = 6
    # Could potentially add a 'harvest' status which could allow for 'shopping' or 'farming' or 'hunting' or 'gathering'

    __str__ = lambda self: self.name

class Action(Enum):
    Move = 0
    Eat = 1
    Work = 2
    Sleep = 3
    Mate = 4 # Could grow this out into seeking a mate and/or sex and/or 'mate_found'
    Harvest = 5 # Could grow this out into Harvesting Type

    __str__ = lambda self: self.name

class Target(Enum):
    Food = 0
    Work = 1
    Mate = 2
    Home = 3

    __str__ = lambda self: self.name

    @staticmethod
    def random():
        return random.choice([Target.Food, Target.Work, Target.Mate, Target.Home])


                #  HERE   UPLEFT     LEFT   BACKLEFT   DOWN   DOWNRIGHT  RIGHT  UPRIGHT   UP 
directions_map = [(0, 0), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1), (1, 0), (1, 1), (0, 1)]

# Movement Options
class Direction(Enum):
    UpLeft = 0
    Left = 1
    BackLeft = 2
    Down = 3
    DownRight = 4
    Right = 5
    UpRight = 6
    Up = 7

    @staticmethod
    def random():
        return random.choice([Direction.UpLeft, Direction.Left, Direction.BackLeft, Direction.Down, Direction.DownRight, Direction.Right, Direction.UpRight, Direction.Up])

def magnitude(x, y):
    '''
    Returns the magnitude of the vector'''
    return math.sqrt(x ** 2 + y ** 2)


def vector(x, y, magnitude):
    '''
    Returns the normalized vector and magnitude'''
    return (x / magnitude, y / magnitude), magnitude


def calculateDirection(x, y, tX, tY):
    '''
    Calculates the direction vector given the x, y, and target x, y, and returns the vector and magnitude'''
    dX = tX - x
    dY = tY - y

    if dX == 0 and dY == 0:
        return (0, 0), 0

    return vector(x, y, magnitude(dX, dY))


# TODO: Implement a way to 'find the next target'
class Agent(Placement):
    def __init__(self, x, y, size, map_size, unit_type):
        super().__init__(x, y, size, unit_type)
        self.map_size = map_size    
        self.sex = random.choice(['M', 'F'])
        self.energy = 75 # Should we clamp energy
        self.wealth = 25 # Should we add 'economic' factors? .. If we add a "legal" requirement, will it figure it out? 
        self.status = Status.Alive
        self.magnitude = 0
        self.reward = 0
        self.chosen_direction = Direction.random()
        self.target = None
        self.target_direction_vector = (0, 0)


    def calculateTargetReward(self, target):
        '''
        Calculates the reward based on the distance to the target'''
        previous_magnitude = self.magnitude
        target_direction_vector, magnitude = calculateDirection(self.x, self.y, target.x, target.y)
        return (magnitude - previous_magnitude, target_direction_vector)

    # Q Table - State space
        # Directional Vector
        # Energy Level
        # Wealth Level


    # TODO: Integrate with Queue Table
    def move(self):
        self.energy -= 1
        dx, dy = directions_map[self.chosen_direction.value]

        if 0 <= self.x + dx <= self.map_size - 1 and 0 <= self.y + dy <= self.map_size - 1:
            self.x += dx
            self.y += dy
        

    def calculateReward(self, target, status):
        reward = 0

        # If the distance towards the correct target is less
        target_reward, target_direction_vector = self.calculateTargetReward(target)
        reward += target_reward

        # If energy goes up
        if status == Status.Eating:
            reward

        # If wealth goes up
        # TODO: Test if Status Order changes the simulation outcomes
        if status == Status.Working:
            reward += 1
        
        return {
            'reward': reward,
            'target_direction_vector': target_direction_vector
        }

    def updateState(self):
        # Update the State Space
        if self.energy < 0:
            print(f"Agent {self} has died")
            self.status = Status.Dead

        # Update the Q Table

    # Needs to be as random as possible to explore all possible states
    def chooseRandomAction(self, findTarget):
        self.target = findTarget(Target.random()) # TODO: Move this outside of the function

        if self.target is None:
            self.target = self

        # Action
        # Left, Right, Up, Down
        # Change States
        self.chosen_direction = Direction.random()

    # Update
    def update(self, findTarget):
        if self.status == Status.Dead:
            return
        

        if self.target is None:
            self.chooseRandomAction(findTarget)

        self.chosen_direction = Direction.random()


        # Choose the best action
        # TODO: Look ahead at the next square based on the Q Table OR Do a Random Walk
        # This has to be before the move to ensure the target exists
        

            ## Percent of Randomness
            # We have the ability to move in a direction with some randomness
        self.move()

        
        # Calculate Collissions
        # Calculate Rewards
        reward_obj = self.calculateReward(self.target, self.status)
        self.reward += reward_obj['reward']

        # Update the state space
        self.updateState()