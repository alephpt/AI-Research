from enum import Enum
import random

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
