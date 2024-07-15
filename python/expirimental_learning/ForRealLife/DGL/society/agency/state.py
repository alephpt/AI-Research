from enum import Enum
import random

# TODO: Add Color Factor or Text Status
# These contribute to our State Space
# But we do not want to map states to variables
#class ActionState(Enum):
    # TradingForWork = (1, (128, 155, 24), 2)
    # TradingForFood = (2, (64, 141, 98), 2)
    # Eating = (3, (0, 129, 182), 2)
    # Farming = (4, (182, 122, 222), 2)
    # Fatigued = (3, (128, 200, 191), 2)
    # Sex = (4, (182, 122, 222), 2) #TODO: Integrate at Home Cell
    # Harvesting = (9, (128, 200, 191), 2) # Creates a 'Token' in society that allows for 'generic' trading of goods
    # Building = (10, (128, 200, 191), 2) # Could lead to building Home or Business -> Could even evolve "construction" or "engineer" type

# These are Determined by the Choice we make from the Neural Network
class State(Enum): # (enumerated state, color value, lerp value)
    Static = (-1, (0, 0, 0), 1)         # Fixed State e.g. Market, Home, etc.
    Dead = (-1, (24, 24, 24), 4)        # End State
    Alive = (0, (111, 55, 111), 2)      # Default State
    #Sleeping = (2, (0, 64, 222), 3)
    
    # These could be 'Chooseable' States
    # Content = (8, (128, 200, 191), 2) // I'd romanticize this to be resulting of compassion and happiness, and a certain level of fatigue
    Broke = (6, (64, 77, 12), 2)            # We really want out Neural Network to be choosing what state we are in
    Hungry = (7, (0, 64, 91), 2)
    Horny = (5, (149, 90, 166), 2)    #Sleeping = (0, (0, 64, 222), 3)


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
    
    def needy(self):
        return self in [State.Broke, State.Hungry, State.Horny]

    @staticmethod
    def fromValue(value):
        return State.Dead if value <= 0 else State.Alive

    @staticmethod
    def random():
        # TODO: Graph the initial curve of the Alive space against the long term trends of the other states
        return random.choice([State.Alive, State.Horny, State.Broke, State.Hungry])

    # TODO: Integrate this as an optimization once each time we update the state
    def combine(self, other):
        color_value = self.value[1]
        lerp_value = self.lerp_factor() + other.lerp_factor() // 2
        r = (color_value[0] + other.value[0]) // lerp_value
        g = (color_value[1] + other.value[1]) // lerp_value
        b = (color_value[2] + other.value[2]) // lerp_value
        return (r, g, b)