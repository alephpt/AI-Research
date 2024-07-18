from enum import Enum
import random

# Integrity is our Y-Axis
class Integrity(Enum):
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

# Compassion is our X-Axis
class Compassion(Enum):
    Heartless = -3
    Cold = -2
    Inconsiderate = -1
    Indifferent = 0
    Content = 1
    Empathetic = 2
    Altruistic = 3

    @staticmethod
    def random():
        return Compassion(random.choice([*Compassion]))
    
class EthicsMatrix:
    def __init__(self):
        self.integrity = Integrity.random()
        self.compassion = Compassion.random()

    # To calculate rewards, we need to determine what happens when someone attempts to 'do something' with a target
    # We need to determine the 'potential' of a target
        # we need to compare the impact of 2 targets on each other based on > or < logics

    # TODO: Move this to the correct class, in the Unit (or inherit Ethics into the Unit, in combination with the TargetingSystem)
    def evaluatePotential(self, target):
        potential = 100

        ## in terms of integrity - 
            ## The more negative the target, the more immediate the rewards should be
            ## The more honest the target, the more stable the outcome should be
            # - High Integrity and Low Integrity avoid each other
        if self.integrity + 3 < target.integrity or self.integrity - 2 > target.integrity:
            potential -= 50

        ## in terms of compassion -
            # either way looks good
        if self.compassion < target.compassion:
            compassion_potential = target.compassion.value - self.compassion.value
            potential += (compassion_potential * 10)
        else: 
            compassion_potential = self.compassion.value - target.compassion.value
            potential += (compassion_potential * 10)

    def xy(self):
        '''Returns the compassion and integrity values, respectively.'''
        return self.compassion.value, self.integrity.value