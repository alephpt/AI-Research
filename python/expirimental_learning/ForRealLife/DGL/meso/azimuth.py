import pygame
from .agency import MoveAction, State
from DGL.micro import Unit, UnitType, Settings, Log, LogLevel
import random

global gender_count
gender_count = 0

# State Space ? Q
class Azimuth(Unit): 
    def __init__(self, idx):
        global gender_count 
        super().__init__(idx, UnitType.Male if gender_count % 2 == 0 else UnitType.Female)
        gender_count += 1
        self.magnitude = 0
        self.target = None
        self.target_direction = (0, 0)
        self.reward = 0             # This will be interesting considering the potential for a Unit State to handle an enumeration of states
        self.state = State.random()
    
    def __str__(self):
        return f"[{self.idx}]-({self.x},{self.y}) - {self.state} :: '{self.target}' :: "
    
    def updateAzimuth(self, reward_obj):
        self.magnitude = reward_obj['magnitude']
        self.target_direction = reward_obj['target_direction_vector']

        if reward_obj['reward'] == 'here':
            self.reward += 1000
            # Change state from a moving state to an action state
            return

    # Needs to be as random as possible to explore all possible states
    def chooseRandomAction(self):
        self.state = State.random()
        Log(LogLevel.VERBOSE, f"Agent {self} is moving to {self.target}")   


    def findTarget(self, target):
        return self.target_direction

    # TODO: Create a map of updating functions
    def updateState(self, cell):
        if self.state == State.Hungry:
            if cell.type == UnitType.Market:
                self.state = State.Buying_Food
