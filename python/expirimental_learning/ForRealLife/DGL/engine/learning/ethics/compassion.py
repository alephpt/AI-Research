import random
from DGL.cosmos import BaseType

# Compassion is our X-Axis
class Compassion(BaseType):
    Heartless = -3
    Cold = -2
    Inconsiderate = -1
    Indifferent = 0
    Content = 1
    Empathetic = 2
    Altruistic = 3

    @staticmethod
    def random():
        return Compassion(random.choice([*Compassion]))
    
    @staticmethod
    def clamp(value):
        if value < -150:
            return Compassion.Heartless
        elif value < -100:
            return Compassion.Cold
        elif value < -50:
            return Compassion.Inconsiderate
        elif value < 50:
            return Compassion.Indifferent
        elif value < 100:
            return Compassion.Content
        elif value < 150:
            return Compassion.Empathetic
        else:
            return Compassion.Altruistic

