from DGL.cosmos import Settings, Log, LogLevel
from .target import Target
from .action import TargetAction
from DGL.society.agency.state import State

class TargetingSystem:
    def __init__(self):
        Log(LogLevel.INFO, "TargetingSystem", f"Unit {self.idx} Initializing Targeting System")
        self.pool_size = Settings.POOL_SIZE.value
        self.pool = set(Target.new() for _ in range(self.pool_size))
        self.potential_targets = []
        self.target_pool_size = 0
        self.next_target_idx = 0
        self.action = TargetAction.Nothing
        self.state = State.Alive
        self.moving = False

    def content(self):
        return self.action == TargetAction.Nothing and self.state == State.Alive
    
    def wuwei(self):
        Log(LogLevel.WARNING, "TargetingSystem", "Wu Wei")
        return self.action == TargetAction.Nothing

    @staticmethod
    def randomAction():
        '''Return a random TargetAction type'''
        return TargetAction.random()

    def selectNew(self, idx):
        '''Select a new target from the pool'''
        if self.target_pool_size == 0 or self.target_pool_size == self.pool_size:
            Log(LogLevel.ERROR, "TargetingSystem", "No potential new targets to select from")
            Log(LogLevel.Warning, "TargetingSystem", f"{self.potential_targets} potential targets : {self.pool_size} pool size: Returning None.")
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
        for i in range(self.pool_size):
            self.selectNew(i)

    def setAction(self, e):
        '''Set the action of the targeting system'''
        self.action = TargetAction(e)
        return self.action

    def poolValues(self):
        '''Return a list of tuples of the pool of targets'''
        return [t.pool() for t in self.pool]
    
    def doAction(self):
        '''Perform an action on the target pool'''
        Log(LogLevel.ALERT, "TargetingSystem", f"Performing action {self.action.name}")
        if self.action == TargetAction.Pursue:
            Log(LogLevel.ALERT, "TargetingSystem", "Pursuant!")
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
            self.pool = [Target.new() for _ in range(self.pool_size)]
        elif self.action == TargetAction.Nothing:
            Log(LogLevel.ALERT, "TargetingSystem", "No action taken")
            self.moving = False

def testTargetingSystem():
    print("Testing Targeting System")
    print(f"Random Targeting System: {TargetAction.random().name}")
    print("Targeting System Test Complete")
    print()

