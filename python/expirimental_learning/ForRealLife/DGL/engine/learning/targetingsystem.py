import random
from enum import Enum
from DGL.cosmos import Settings, Log, LogLevel


# There are our "eyes" in our agent
class TargetAction(Enum):
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
        return TargetAction(random.choice([*TargetAction]))


class Target: # This entire class could be a subclass of a 'Unit' class
    def __init__(self, type_of, dxy, magnitude):
        self.type = type_of
        self.magnitude = magnitude # TODO: TODO: TODO: ## IT IS IMPERITIVE THAT WE CLAMP THE RADIUS IN WHICH WE SCAN FOR POTENTIAL TARGETS # TODO: TODO: TODO:
        self.x, self.y = dxy
        self.potential = 0  # TODO: Determine the Potential of Some Given Target - THIS IS SEPERATE FROM THE ETHICAL MATRIX
    
    @staticmethod
    def new():
        '''Generate a new target'''
        return Target(None, (0, 0), 0)

    def pool(self):
        '''Return the target as a tuple(type, (x, y), magnitude, potential)'''
        return (self.type, (self.x, self.y), self.magnitude, self.potential)

    # TODO: Integrate Integrity vs Compassion
    def evaluateTarget(self, target):
        # We have to calculate the way we determine potential

        ## If we are seeking a market, we want to get the ratio of buyers to sellers relative to the state we are seeking
        ## If we are seeking a mate, we want both targets to appeal to each other
        pass


class TargetingSystem:
    def __init__(self):
        self.size = Settings.POOL_SIZE.value
        self.pool = set(Target.new() for _ in range(self.size))
        self.potential_targets = []
        self.target_pool_size = 0
        self.next_target_idx = 0
        self.action = TargetAction.Nothing
        self.moving = False

    def selectNew(self, idx):
        '''Select a new target from the pool'''
        if self.target_pool_size == 0 or self.target_pool_size == self.size:
            Log(LogLevel.ERROR, "TargetingSystem", "No potential new targets to select from")
            Log(LogLevel.Warning, "TargetingSystem", f"{self.potential_targets} potential targets : {self.size} pool size: Returning None.")
            return

        while self.potential_targets[self.next_target_idx] in self.pool:
            # This has a potential to run forever
            self.next_target_idx += 1 % self.target_pool_size
        
        self.pool[idx] = self.potential_targets[self.next_target_idx]

    # We want to take a given set of potential targets and fill our pool with them
    # While also keeping track of which index we are querying
    def fill(self, target_pool):
        self.target_pool_size = len(target_pool)
        self.potential_targets = target_pool

        # We want to iterate through the potential targets and fill our pool
        # as long as we are under the size of the pool
        for i in range(self.size):
            self.selectNew(i)

    def setAction(self, e):
        '''Set the action of the targeting system'''
        self.action = TargetAction(e)

    def poolValues(self):
        '''Return a list of tuples of the pool of targets'''
        return [t.pool() for t in self.pool]
    
    def doAction(self):
        '''Perform an action on the target pool'''
        if self.action == TargetAction.Pursue:
            self.moving = True
        elif self.action == TargetAction.Pull_First:
            return self.pool[0]
        elif self.action == TargetAction.Drop_First:
            self.selectNew(0)
        elif self.action == TargetAction.Pull_Segundo:
            return self.pool[1]
        elif self.action == TargetAction.Drop_Segundo:
            self.selectNew(1)
        elif self.action == TargetAction.Pull_Tre:
            return self.pool[2]
        elif self.action == TargetAction.Drop_Tre:
            self.selectNew(2)
        elif self.action == TargetAction.Flush_ALL:
            self.pool = [Target.new() for _ in range(self.size)]
        elif self.action == TargetAction.Nothing:
            self.moving = False

def testTargetingSystem():
    print("Testing Targeting System")
    print(f"Random Targeting System: {TargetAction.random().name}")
    print("Targeting System Test Complete")
    print()

