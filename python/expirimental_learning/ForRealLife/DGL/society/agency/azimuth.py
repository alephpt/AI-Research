from DGL.society.agency import MoveAction, State
from DGL.engine.cell import Cell
from DGL.engine.learning import TargetingSystem, EthicsMatrix
from DGL.cosmos import LogLevel, Log
from DGL.cosmos.closet import track
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
        self.target_selection = None     # This value is bound to the mouse hook that allows us to set a selected to a tuple, then unit
        self.target_action = MoveAction.random() # This is where we unwrap some target into a MoveAction
        self.target_direction = self.target_action.xy() # This is the actual XY direction of the next step # TODO: Turn into List of steps
        self.reward = 0             # This will be interesting considering the potential for a Cell State to handle an enumeration of states
        self.target_reached = False # This is to control our logic  - has the potential to be refactored
        self.action_taken = Action.Undefined

    def __str__(self):
        return f"[{self.idx}]-({self.x},{self.y}) - {self.state} :: target('{self.target_direction}') :: \]"

    def moveAction(self):
        if self.target_action is None:
            return
        
        self.target_direction = self.target_action.xy()

    def updateAzimuth(self):
        self.moveAction()
        if self.moving:
            self.action = Action.Move

        # This is where we can simply update the Action Taken to help us have 'logging' of OUR logic. 
        # Note this is a projected bias due to my interpretion of the concepts herein            

    # Needs to be as random as possible to explore all possible states
    def chooseRandomState(self):
        self.state = State.random()
        self.target_reached = False
        Log(LogLevel.ALERT, "Azimuth", f"{self} chose random state {self.state}")
    
    def chooseRandomEncoder(self):
        self.encoder_state = self.randomAction()
        Log(LogLevel.ALERT, "Azimuth", f"{self} chose random encoder state {self.encoder_state}")


    # Option 1: Iterate through all a select group of targets, and choose the one with the lowest magnitude
    # Option 2: Iterate through all possible targets, and choose the one with the highest reward
    def chooseBestTarget(self, targets): # We are going to pass a callback in to allow for a dynamic reward function
        best_value = 0
        best_target = None

        for target in targets:
            distance, _ = track(self.magnitude, self.x, self.y, target)

            if distance < best_value or (distance == 0 and best_value != 0): # or value < 0: # This would account for a negative reward
                Log(LogLevel.WARNING, "Unit", f"\t\tFound New Best Value: {distance} > {best_value}")
                best_value = distance
                best_target = target

        return best_target
