from enum import Enum

# TODO: Add Color Factor or Text Status
# These contribute to our State Space
# But we do not want to map states to variables
class Status(Enum):
    Dead = (-1, (24, 24, 24), 4)
    Sleeping = (0, (0, 64, 222), 3)
    Eating = (1, (0, 129, 182), 2)
    Working = (2, (128, 155, 24), 2)
    Sex = (3, (182, 122, 222), 2) #TODO: Integrate at Home Unit
    Alive = (4, (111, 55, 111), 2)
    Horny = (5, (91, 61, 111), 2)
    Broke = (6, (64, 77, 12), 2)
    Hungry = (7, (0, 64, 91), 2)
    # Could potentially add a 'harvest' status which could allow for 'shopping' or 'farming' or 'hunting' or 'gathering'

    def __str__(self):
        return self.name
    
    def __int__(self):
        return self.value.idx
    
    @staticmethod
    def fromValue(value):
        return Status.Dead if value <= 0 else Status.Alive

    # TODO: Integrate this as an optimization once each time we update the state
    def combine(self, other):
        color_value = self.value[1]
        lerp_value = self.value[2]
        r = (color_value[0] + other.value[0]) // lerp_value
        g = (color_value[1] + other.value[1]) // lerp_value
        b = (color_value[2] + other.value[2]) // lerp_value
        return (r, g, b)