from enum import Enum
import random

# Movement Option by Name
class MoveAction(Enum):
    '''
    Enumerate the MoveActions that an Agent can move in. 8-Way Movement.
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
    
    def __eq__(self, other):
        return other in self.__members__.values()

    def Vector(self):
        return MoveActions_map[self.value]
    
    def isType(self, other):
        return type(self) == type(other)


# These help the agent observe what he is doing
class Action(Enum):
    Move = MoveAction
    Eat = 8
    Work = 9
    Sleep = 10 # Requires you to have an avg energy and max level, and needs a sleep slope, AND a home AND 'to do Nothing for 3 turns' (can we get it to 7?)
    Mate = 11 # Could grow this out into seeking a mate and/or sex and/or 'mate_found'

    __str__ = lambda self: self.name

    def __eq__(self, other):
        return other in self.__members__.values()

    @staticmethod
    def random():
        return random.choice([Action.Move.random(), Action.Eat, Action.Work, Action.Sleep, Action.Mate])
    
    @staticmethod
    def randomDirection():
        return MoveAction.random()

    def Vector(self):
        if self == Action.Move:
            return MoveActions_map[self.value.value] # This is probably wrong
        return MoveAction.NoWhere.value
    


## These are the MoveActions that we can move in
                #  HERE   UPLEFT     LEFT   BACKLEFT   DOWN   DOWNRIGHT  RIGHT  UPRIGHT   UP 
MoveActions_map = [(0, 0), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1), (1, 0), (1, 1), (0, 1)]
