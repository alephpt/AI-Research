from enum import Enum

class Unit(Enum):
    Available = (64, 64, 64)
    Male = (128, 0, 0)
    Female = (0, 0, 128)
    Work = (128, 128, 0)
    Food = (0, 128, 0)
    Mating = (128, 0, 128)
    Home = (128, 128, 128) ##=>> We are not implementing this until units can not die a little bit later
    # TODO: Add 'Home'