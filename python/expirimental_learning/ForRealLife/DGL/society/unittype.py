from enum import Enum

class UnitType(Enum):
    Male = (128, 0, 0)
    Female = (0, 0, 128)
    HUMAN = [Male, Female]
    Market = (128, 128, 0)
    Home = (128, 128, 128) 
