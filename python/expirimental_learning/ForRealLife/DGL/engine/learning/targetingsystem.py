import random
from enum import Enum
from DGL.cosmos import Settings

class Target: # This entire class could be a subclass of a 'Unit' class
    def __init__(self, type_of, mag, dxy):
        self.type = type_of
        self.magnitude = mag
        self.x, self.y = dxy
        self.potential = 0
    
    # TODO: Integrate Integrity vs Compassion
    def evaluateTarget(self, target):
        # We have to calculate the way we determine potential

        ## If we are seeking a market, we want to get the ratio of buyers to sellers relative to the state we are seeking
        ## If we are seeking a mate, we want both targets to appeal to each other
        pass

class TargetPool:
    def __init__(self):
        self.size = Settings.POOL_SIZE.value
        self.pool = [Target(None, 0, (0, 0)) for _ in range(self.size)]

# There are our "eyes" in our agent
class TargetingSystem(Enum):
    Pursue = 0                              # This will be the trigger action that will do Nothing until a target is selected
    Pull_First = 1                        # This should map to the first target in the target pool
    Drop_First = 2
    Pull_Segundo = 3                      # This should map to the second target in the target pool
    Drop_Segundo = 4
    Pull_Tre = 5                          # This should map to the third target in the target pool
    Drop_Tre = 6
    Flush_ALL = 7                           # Flush the target pool
    Nothing = 8

    @staticmethod
    def random():
        return TargetingSystem(random.choice([*TargetingSystem]))