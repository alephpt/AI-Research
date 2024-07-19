from DGL.society.agency import MoveAction, State
from DGL.engine.cell import Cell
from DGL.engine.learning import TargetingSystem, EthicsMatrix
from DGL.cosmos import LogLevel, Log
from ..unittype import UnitType
from .action import Action

global gender_count
gender_count = 0

# State Space ? Q
class Azimuth(Cell, EthicsMatrix, TargetingSystem): 
    '''
    An Azimuth should be used for locational and directional purposes.'''
    def __init__(self, idx):
        global gender_count 
        super().__init__(idx, UnitType.Male if gender_count % 2 == 0 else UnitType.Female)
        EthicsMatrix.__init__(self)
        TargetingSystem.__init__(self)
        gender_count += 1
        self.magnitude = None
        self.cursor = MoveAction.random() # This value is bound to the mouse hook that allows us to set a selected to a tuple, then unit
        self.target_direction = self.cursor.xy()
        self.reward = 0             # This will be interesting considering the potential for a Cell State to handle an enumeration of states
        self.target_reached = False # This is to control our logic  - has the potential to be refactored
        self.action_taken = Action.Undefined

    def __str__(self):
        return f"[{self.idx}]-({self.x},{self.y}) - {self.state} :: target('{self.target_direction}') :: \]"
    
    def updateAzimuth(self):
        if self.moving:
            self.action = Action.Move

        # This is where we can simply update the Action Taken to help us have 'logging' of OUR logic. 
        # Note this is a projected bias due to my interpretion of the concepts herein            


    # Needs to be as random as possible to explore all possible states
    def chooseRandomState(self):
        self.state = State.random()
        self.target_reached = False
        Log(LogLevel.VERBOSE, "Azimuth", f"{self} chose random state {self.state}")
    
    def chooseRandomAction(self):
        self.action = self.randomAction()

        Log(LogLevel.VERBOSE, "Azimuth", f"{self} chose random action {self.action}")

