from enum import Enum
import random

# TODO: Determine if we can completely refactor this one out.

#TODO: Implement 'Home' Unit
class Target(Enum):
    Food = 0
    Work = 1
    Mate = 2
    Home = 3

    __str__ = lambda self: self.name

    @staticmethod
    def random():
        return random.choice([Target.Food, Target.Work, Target.Mate, Target.Home])