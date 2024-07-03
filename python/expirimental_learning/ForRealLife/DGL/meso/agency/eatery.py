from DGL.micro import Unit, Placement, Settings

class Eatery(Placement):
    def __init__(self):
        super().__init__(Unit.Work)
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