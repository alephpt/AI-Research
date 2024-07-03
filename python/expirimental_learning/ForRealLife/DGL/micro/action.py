from enum import Enum

# These help us observe what the agent is doing
class Action(Enum):
    Move = 0
    Eat = 1
    Work = 2
    Sleep = 3
    Mate = 4 # Could grow this out into seeking a mate and/or sex and/or 'mate_found'
    Harvest = 5 # Could grow this out into Harvesting Type

    __str__ = lambda self: self.name
