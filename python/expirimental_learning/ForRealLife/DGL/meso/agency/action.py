from enum import Enum
import random


## These are the Moves that we can move in
                #  HERE   UPLEFT     LEFT   BACKLEFT   DOWN   DOWNRIGHT  RIGHT  UPRIGHT   UP 
MOVE_MAP = [(0, 0), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1), (1, 0), (1, 1), (0, 1)]

# Movement Option by Name
class MoveAction(Enum):
    '''
    Enumerate the Moves that an Agent can move in. 8-Way Movement.

    Objective: Can we, given a target, calculate the best 
    '''
    NoWhere = -1
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
        return random.choice([
            MoveAction.UpLeft, 
            MoveAction.Left, 
            MoveAction.BackLeft, 
            MoveAction.Down, 
            MoveAction.DownRight, 
            MoveAction.Right, 
            MoveAction.UpRight, 
            MoveAction.Up])
    
    def  __eq__(self, value: object) -> bool:
        return super().__eq__(value)

    def Vector(self):
        return MOVE_MAP[self.value]
    
    def isType(self, other):
        return type(self) == type(other)

    def xy(self):
        return MOVE_MAP[self.value]