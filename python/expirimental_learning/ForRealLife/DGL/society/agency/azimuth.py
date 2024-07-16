from DGL.society.agency import MoveAction, State
from DGL.engine.cell import Cell
from DGL.cosmos import LogLevel, Log
from ..unittype import UnitType

global gender_count
gender_count = 0

# State Space ? Q
class Azimuth(Cell): 
    '''
    An Azimuth should be used for locational and directional purposes.'''
    def __init__(self, idx):
        global gender_count 
        super().__init__(idx, UnitType.Male if gender_count % 2 == 0 else UnitType.Female)
        gender_count += 1
        self.magnitude = 0
        self.target = None
        self.action_step = MoveAction.random()
        self.target_direction = self.action_step.xy()
        self.reward = 0             # This will be interesting considering the potential for a Cell State to handle an enumeration of states
        self.target_reached = False # This is to control our logic  - has the potential to be refactored
    
    def __str__(self):
        return f"AZMTH[{self.idx}]-({self.x},{self.y}) - {self.state} :: target('{self.target_direction}') :: \]"
    
    # Needs to be as random as possible to explore all possible states
    def chooseRandomState(self):
        self.state = State.random()
        self.target_reached = False
        Log(LogLevel.VERBOSE, "Azimuth", f"{self} chose random state {self.state}")


    # TODO: Create a map of updating functions
    def updateState(self, cell):
        Log(LogLevel.INFO, f" ~ Updating State of {self} with {cell}")
        if self.state == State.Hungry:
            Log(LogLevel.INFO, f"Unit {self} is hungry")
            if cell.type == UnitType.Market:
                Log(LogLevel.INFO, f"Unit {self} is at Market {cell.idx}")
