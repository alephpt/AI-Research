from enum import Enum
import random

# TODO: Add Color Factor or Text Status
# These contribute to our State Space
# But we do not want to map states to variables

# These are Determined by the Action we are doing.
class State(Enum): # (enumerated state, color value, lerp value)
    Static = (-1, (0, 0, 0), 1)
    Dead = (-1, (24, 24, 24), 4)
    Sleeping = (0, (0, 64, 222), 3)
    Eating = (1, (0, 129, 182), 2)
    Working = (2, (128, 155, 24), 2)
    Sex = (3, (182, 122, 222), 2) #TODO: Integrate at Home Unit
    Alive = (4, (111, 55, 111), 2)
    Horny = (5, (91, 61, 111), 2)
    Broke = (6, (64, 77, 12), 2)
    Hungry = (7, (0, 64, 91), 2)
    Content = (8, (128, 200, 191), 2)
    # Could potentially add a 'harvest' status which could allow for 'shopping' or 'farming' or 'hunting' or 'gathering'

    def __str__(self):
        return self.name
    
    def __int__(self):
        return self.value.idx
    
    @staticmethod
    def fromValue(value):
        return State.Dead if value <= 0 else State.Alive

    @staticmethod
    def random():
        return random.choice([State.Alive, State.Horny, State.Broke, State.Hungry])

    # TODO: Integrate this as an optimization once each time we update the state
    def combine(self, other):
        color_value = self.value[1]
        lerp_value = self.value[2]
        r = (color_value[0] + other.value[0]) // lerp_value
        g = (color_value[1] + other.value[1]) // lerp_value
        b = (color_value[2] + other.value[2]) // lerp_value
        return (r, g, b)