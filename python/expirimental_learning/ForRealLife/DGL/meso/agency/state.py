from enum import Enum
import random

# TODO: Add Color Factor or Text Status
# These contribute to our State Space
# But we do not want to map states to variables

# These are Determined by the Action we are doing.
class State(Enum): # (enumerated state, color value, lerp value)
    Dead = (-1, (24, 24, 24), 4)        # End State
    Static = (-1, (0, 0, 0), 1)         # Fixed State
    Alive = (0, (111, 55, 111), 2)      # Default State

    Working = (1, (128, 155, 24), 2)
    Buying_Food = (2, (64, 141, 98), 2)

    Eating = (3, (0, 129, 182), 2)
    Sex = (4, (182, 122, 222), 2) #TODO: Integrate at Home Unit

    Find_Mate = (5, (149, 90, 166), 2)
    Horny = (6, (91, 61, 111), 2)

    Broke = (6, (64, 77, 12), 2)
    Hungry = (7, (0, 64, 91), 2)
    
    Content = (8, (128, 200, 191), 2)
    # Could potentially add a 'harvest' status which could allow for 'shopping' or 'farming' or 'hunting' or 'gathering'
    Sleeping = (0, (0, 64, 222), 3)
    # Harvesting = (9, (128, 200, 191), 2)
    # Building = (10, (128, 200, 191), 2) # Could lead to building Home or Business -> Could even evolve "construction" or "engineer" type

    def idx(self):
        return self.value[0]
    
    def color(self):
        return self.value[1]

    def lerp_factor(self):
        return self.value[2]

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
        lerp_value = self.lerp_factor() + other.lerp_factor() // 2
        r = (color_value[0] + other.value[0]) // lerp_value
        g = (color_value[1] + other.value[1]) // lerp_value
        b = (color_value[2] + other.value[2]) // lerp_value
        return (r, g, b)