from DGL.micro import Unit, UnitType, Settings
from .agency import State

class Market(Unit):
    def __init__(self, idx):
        x = Settings.randomLocation()
        y = Settings.randomLocation()
        super().__init__(idx, UnitType.Market)
        self.n_exchanges = 0 
        self.exchanges = [] # We can implement this later to allow merchants and buyers to have an ongoing exchange.. maybe even a preference for a certain merchant
        self.max_employees = Settings.MAX_EMPLOYEES.value
        self.status = State.Static

    def exchange(self, buyer, seller):
        buyer.money -= Settings.FOOD_COST.value
        seller.money += Settings.FOOD_COST.value

        buyer.energy += Settings.FOOD_REWARD.value
        seller.energy -= Settings.WORK_COST.value

    def update(self):
        for employee in self.employees:
            employee.work()