from DGL.micro import Unit, UnitType, Settings
from .agency.state import State

class Home(Unit):
    def __init__(self, idx):
        super().__init__(idx, UnitType.Home)
        self.max_sleep = Settings.MAX_SLEEP.value
        self.state = State.Static
        self.home = []

    def getFood(self, agent):
        if self.available_food > len(self.home):
            self.hungered.append(agent)
            return True
        return False

    def update(self):
        for hungry in self.hungered:
            hungry.eat()