import random
from enum import Enum

class Color(Enum):
    # Primary Colors for Individuals are Blue, Red and Yellow.. 
    Blue = (0, 0, 255)
    Red = (255, 0, 0)
    Yellow = (255, 255, 0)

    # Seconary Colors are Purple, Green and Orange
    Purple = (255, 0, 255)          # Offspring of Blue and Red
    Green = (0, 255, 0)             # Offspring of Blue and Yellow
    Orange = (255, 165, 0)          # Offspring of Red and Yellow

    # Territiary Colors for 'cross-breeding'
    Olive = (128, 128, 0)            # Offspring of Green and Orange or Yellow and Purple
    Russet = (128, 70, 27)           # Offspring of Orange and Purple
    Brown = (139, 69, 19)            # Offspring of Purple and Green
