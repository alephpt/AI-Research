from DGL.micro import Unit, UnitType, Settings
from .agency.state import State

class Market(Unit):
    def __init__(self, idx):
        super().__init__(idx, UnitType.Market)
        self.n_employees = 0
        self.n_clients = 0
        self.max_employees = Settings.MAX_EMPLOYEES.value
        self.employees = []
        self.clients = []
        self.state = State.Static

    def getJob(self, agent):
        if self.n_employees < self.max_employees:
            self.employees.append(agent)
            self.n_employees += 1
            ## TODO: Determine how 'Work' looks like in the DGL
            return True
        return False

    def getFood(self, agent):
        if self.n_clients < self.max_employees:
            self.clients.append(agent)
            self.n_clients += 1
            return True
        return False

    def update(self):
        for employee in self.employees:
            employee.work()