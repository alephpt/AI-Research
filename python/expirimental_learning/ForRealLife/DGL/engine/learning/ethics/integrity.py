import random
from DGL.cosmos import BaseType

# Integrity is our Y-Axis
class Integrity(BaseType):
    Duplicit = -3
    Deceitful = -2
    Deceptive = -1
    Neutral = 0
    Honest = 1
    Trustworthy = 2
    Integral = 3

    @staticmethod
    def random():
        return Integrity(random.choice([*Integrity]))
    
    @staticmethod
    def clamp(value):
        if value < -150:
            return Integrity.Duplicit
        elif value < -100:
            return Integrity.Deceitful
        elif value < -50:
            return Integrity.Deceptive
        elif value < 50:
            return Integrity.Neutral
        elif value < 100:
            return Integrity.Honest
        elif value < 150:
            return Integrity.Trustworthy
        else:
            return Integrity.Integral