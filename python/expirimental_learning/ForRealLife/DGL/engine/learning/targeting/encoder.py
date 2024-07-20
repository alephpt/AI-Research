import random
from DGL.cosmos import BaseType


# There are our "eyes" in our agent
class EncoderState(BaseType):
    Pursue = 0                              # This will be the trigger action that will do Nothing until a target is selected
    Pull_First = 1                        # This should map to the first target in the target pool
    Drop_First = 2
    Pull_Segun = 3                      # This should map to the second target in the target pool
    Drop_Segun = 4
    Pull_Tre = 5                          # This should map to the third target in the target pool
    Drop_Tre = 6
    Flush_ALL = 7                           # Flush the target pool
    Nothing = 8 # If you put this at the first, you have to change the 'if' statement in the System

    def idx(self):
        return self.value

    @staticmethod
    def random():
        return EncoderState(random.choice([*EncoderState]))

