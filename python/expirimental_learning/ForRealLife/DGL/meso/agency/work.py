from DGL.micro import Unit, UnitType, Settings

class Work(Unit):
    def __init__(self, idx):
        x = Settings.randomLocation()
        y = Settings.randomLocation()
        super().__init__(idx, x, y, UnitType.Work)
        self.n_employees = 0
        self.max_employees = Settings.MAX_EMPLOYEES.value
        self.employees = []

    def getJob(self, agent):
        if self.n_employees < self.max_employees:
            self.employees.append(agent)
            self.n_employees += 1
            ## TODO: Determine how 'Work' looks like in the DGL
            return True
        return False

    def update(self):
        for employee in self.employees:
            employee.work()