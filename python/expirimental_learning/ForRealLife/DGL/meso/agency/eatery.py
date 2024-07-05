from DGL.micro import Unit, UnitType, Settings

class Eatery(Unit):
    def __init__(self, idx):
        x = Settings.randomLocation()
        y = Settings.randomLocation()
        super().__init__(idx, x, y, UnitType.Work)
        self.available_food = Settings.AVAILABLE_FOOD.value
        self.hungered = []

    def getFood(self, agent):
        if self.available_food > len(self.hungered):
            self.hungered.append(agent)
            return True
        return False

    def update(self):
        for hungry in self.hungered:
            hungry.eat()