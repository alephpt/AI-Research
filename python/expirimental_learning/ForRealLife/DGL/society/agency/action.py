from enum import Enum


# These are potential Action states that we could be in based on some given conditions of the AI
class Action(Enum):
    Undefined = -1
    Wuwei = 0           # Doing not doing.
    Moving = 1          # Actually Pursuing a Target
    Trading = 2         # These are more 'conditional' internal states for triggering world events
    Eating = 3
    Sleeping = 4
    Growing = 5
