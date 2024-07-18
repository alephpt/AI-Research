from enum import Enum
import random
from DGL.cosmos import Settings, Log, LogLevel

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
    
    @staticmethod
    def clamp(value):
        if value < -150:
            return Compassion.Heartless
        elif value < -100:
            return Compassion.Cold
        elif value < -50:
            return Compassion.Inconsiderate
        elif value < 50:
            return Compassion.Indifferent
        elif value < 100:
            return Compassion.Content
        elif value < 150:
            return Compassion.Empathetic
        else:
            return Compassion.Altruistic

class EthicsMatrix:
    def __init__(self):
        self.integrity_points = Settings.randFluxInt(300)
        self.integrity = Integrity.clamp(self.integrity_points)
        self.compassion_points = Settings.randFluxInt(300)
        self.compassion = Compassion.clamp(self.compassion_points)

    def __str__(self):
        return f"x: {self.compassion.name}, y: {self.integrity.name}"

    def gainIntegrity(self, value):
        '''Can be used to +/- integrity points'''
        self.integrity_points += value
        self.updateEthics()

    def gainCompassion(self, value):
        '''Can be used to +/- compassion points'''
        self.compassion_points += value
        self.updateEthics()

    def updateEthics(self):
        self.integrity = Integrity.clamp(self.integrity_points)
        self.compassion = Compassion.clamp(self.compassion_points)

    # To calculate rewards, we need to determine what happens when someone attempts to 'do something' with a target
    # We need to determine the 'potential' of a target
        # we need to compare the impact of 2 targets on each other based on > or < logics
    def clampPoints(self, value):
        return max(min(value, 150), -150)

    # TODO: Move this to the correct class, in the Unit (or inherit Ethics into the Unit, in combination with the TargetingSystem)
    def evaluatePotential(self, target):
        Log(LogLevel.INFO, "EthicsMatrix", f"Evaluating Potential between {self} and {target}")
        potential = 100

        ## in terms of integrity - 
            ## The more negative the target, the more immediate the rewards should be
            ## The more honest the target, the more stable the outcome should be
            # - High Integrity and Low Integrity avoid each other
        if self.integrity + 3 < target.integrity or self.integrity - 2 > target.integrity:
            Log(LogLevel.INFO, "EthicsMatrix", f"Integrity is too far apart: -50")
            potential -= 50

        ## in terms of compassion -
            # either way looks good
        if self.compassion < target.compassion:
            compassion_potential = target.compassion.value - self.compassion.value
            potential += (compassion_potential * 10)
            Log(LogLevel.INFO, "EthicsMatrix", f"Lesser compassion observed: {potential}")
        else: 
            compassion_potential = self.compassion.value - target.compassion.value
            potential += (compassion_potential * 10)
            Log(LogLevel.INFO, "EthicsMatrix", f"Greater compassion observed: {potential}")


    def xy(self):
        '''Returns the compassion and integrity values, respectively.'''
        return self.compassion.value, self.integrity.value
    
def testEthicsMatrix():
    print("Testing Ethics Matrix")
    ethics = EthicsMatrix()
    print(f"Ethics: {ethics}")
    print(f"Ethics XY: {ethics.xy()}")
    print("Ethics Matrix Test Complete")
    print()