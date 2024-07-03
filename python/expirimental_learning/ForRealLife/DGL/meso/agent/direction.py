from enum import Enum
import random


# Movement Option by Name
class Direction(Enum):
    '''
    Enumerate the Directions that an Agent can move in. 8-Way Movement.
    '''
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
    

## These are the directions that we can move in
                #  HERE   UPLEFT     LEFT   BACKLEFT   DOWN   DOWNRIGHT  RIGHT  UPRIGHT   UP 
directions_map = [(0, 0), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1), (1, 0), (1, 1), (0, 1)]


def p(x1, y1, x2, y2):
    '''
    Calculate the Directional Magnitude of a Vector'''
    dx = x2 - x1
    dy = y2 - y1
    magnitude = (dx ** 2 + dy ** 2) ** 0.5
    return (dx / magnitude, dy / magnitude), magnitude
