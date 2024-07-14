from DGL.micro import Cell, CellType, Settings
from . import State

class Home(Cell):
    def __init__(self, idx):
        super().__init__(idx, CellType.Home)
        self.max_sleep = Settings.MAX_SLEEP.value
        self.state = State.Static
        self.home = []

    def getFood(self, unit):
        if self.available_food > len(self.home):
            self.hungered.append(unit)
            return True
        return False

    def update(self):
        for hungry in self.hungered:
            hungry.eat()