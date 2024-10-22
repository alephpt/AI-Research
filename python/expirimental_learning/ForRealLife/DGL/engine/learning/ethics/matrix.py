
from DGL.cosmos import Settings, Log, LogLevel
from .integrity import Integrity
from .compassion import Compassion

class EthicsMatrix:
    '''
    The Ethics Matrix is designed in a way that gives more opportunities for a more negative index, and more stable outcomes for a more positive index.
    '''
    def __init__(self):
        Log(LogLevel.INFO, "EthicsMatrix", f"Unit {self.idx} Creating an Ethics Matrix")
        self.integrity_points = Settings.randFluxInt(300)
        self.integrity = Integrity.clamp(self.integrity_points)
        self.compassion_points = Settings.randFluxInt(300)
        self.compassion = Compassion.clamp(self.compassion_points)

    def __str__(self):
        return f"compassion: {self.compassion.name}, integrity: {self.integrity.name}"

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


    def ethicsXY(self):
        '''Returns the compassion and integrity values, respectively.'''
        return self.compassion.value, self.integrity.value
    
    def ethics(self):
        '''Returns the compassion and integrity types, respectively.'''
        return self.compassion, self.integrity
    
    def ethicsNames(self):
        '''Returns the compassion and integrity names, respectively.'''
        return self.compassion.name, self.integrity.name