from enum import Enum

class UnitType(Enum):
    CELL = (0, 125, 55)
    Male = (128, 0, 0)
    Female = (0, 0, 128)
    Market = (128, 128, 0)
    Home = (128, 128, 128) 
