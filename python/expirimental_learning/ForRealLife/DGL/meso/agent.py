
from enum import Enum
from DGL.micro import Placement, Status, Target, Action
import random
import math

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

cost_of_service = 5 # TODO: Let every Agent set their own cost of food

# TODO: Implement a way to 'find the next target'
class Agent(Placement):
    def __init__(self, x, y, size, map_size, unit_type):
        super().__init__(x, y, size, unit_type)
        self.map_size = map_size    
        self.age = 0
        self.happiness = 0
        self.sex = random.choice(['M', 'F'])
        self.energy = 75 # Should we clamp energy
        self.wealth = 25 # Should we add 'economic' factors? .. If we add a "legal" requirement, will it figure it out? 
        self.status = Status.Alive
        self.magnitude = 0
        self.reward = 0
        self.chosen_direction = Direction.random()
        self.target = None
        self.target_direction_vector = (0, 0)

    def work(self):
        if self.energy >= 10:
            self.wealth += 10
            self.energy -= 5

    def eat(self):
        if self.wealth >= cost_of_service:
            self.wealth -= cost_of_service
            self.energy += 10 # TODO: HUGE TEST - In Isolation, determine if fixed values or random values are better
            self.happiness += 1
    
    def sex(self):
        self.energy -= 10
        self.happiness += 5 # Integrate a large amount of Chance here

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

        # TODO: Add a randomness for 'happiness factor' to the macro genetic scale 
        #       where some agents care more about different things
    
        # Happier Lives are better
        reward += self.happiness

        # Longer Lives are better
        reward += self.age

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
        
        self.age += 1
        

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