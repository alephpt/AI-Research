from enum import Enum

class UnitType(Enum):
    CELL = (0, 125, 55)
    Male = (128, 0, 0)
    Female = (0, 0, 128)
    Market = (128, 128, 0)
    Home = (128, 128, 128)

    def add(self, y): 
        x = self.value
        return (x[0] + y[0], x[1] + y[1], x[2] + y[2])
