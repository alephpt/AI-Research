from DGL.engine.cell import Cell
from .unittype import UnitType
from DGL.cosmos import Settings
from .agency import State

class Home(Cell):
    def __init__(self, idx):
        super().__init__(idx, UnitType.Home)
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