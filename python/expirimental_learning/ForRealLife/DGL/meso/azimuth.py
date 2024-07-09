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
        self.action_step = MoveAction.random()
        self.target_direction = self.action_step.xy()
        self.reward = 0             # This will be interesting considering the potential for a Unit State to handle an enumeration of states
        self.state = State.random()
    
    def __str__(self):
        return f"AZMTH[{self.idx}]-({self.x},{self.y}) - {self.state} :: moving to '{self.target_direction}' :: "
    
    def updateAzimuth(self, reward_obj):
        self.magnitude = reward_obj['magnitude']
        self.target_direction = reward_obj['target_direction_vector']

        if reward_obj['reward'] == 'here':
            self.target_reached = True # We should be able to factor this out by checking state
            self.reward += 1000
            # Change state from a moving state to an action state
            return

    # Needs to be as random as possible to explore all possible states
    def chooseRandomState(self):
        self.state = State.random()
        self.target_reached = False
        Log(LogLevel.VERBOSE, f"Agent {self} is moving to {self.target}")   


    # TODO: Create a map of updating functions
    def updateState(self, cell):
        Log(LogLevel.INFO, f" ~ Updating State of {self} with {cell}")
        if self.state == State.Hungry:
            Log(LogLevel.INFO, f"Agent {self} is hungry")
            if cell.type == UnitType.Market:

                self.state = State.Buying_Food
