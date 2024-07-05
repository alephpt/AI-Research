import pygame
from .agency import Action, calculateReward
from DGL.micro import Unit, UnitType, Settings
import random


# State Space ? Q
class Azimuth(Unit): 
    def __init__(self, idx):
        x = random.randint(0, Settings.GRID_SIZE.value - 1)
        y = random.randint(0, Settings.GRID_SIZE.value - 1)
        super().__init__(idx, x, y, random.choice([UnitType.Male, UnitType.Female]))
        self.magnitude = 0
        self.target = None
        self.reward = 0 # This will be interesting considering the potential for a Unit State to handle an enumeration of states
    
    def __str__(self):
        return f"({self.x}, {self.y}) - {self.state}"
    
    def updateAzimuth(self, reward_obj):
        self.magnitude = reward_obj.magnitude
        self.action = reward_obj.action
        self.reward += reward_obj.reward

    # Needs to be as random as possible to explore all possible states
    def chooseRandomAction(self, findTarget):
        self.chosen_direction = Action.randomDirection()
        self.target = findTarget(self.target)

        if self.target is None:
            print(f"Agent {self} has no target")
            self.target = self

