from enum import Enum

class Unit(Enum):
    Available = (64, 64, 64)
    Male = (128, 0, 0)
    Female = (0, 0, 128)
    Work = (128, 128, 0)
    Food = (0, 128, 0)
    Mating = (128, 0, 128)
    # TODO: Add 'Home'