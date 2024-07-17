from enum import Enum
from DGL.cosmos.closet.color import Color
from DGL.cosmos import LogLevel, Log


class UnitType(Enum):
    CELL = Color.CELLULAR_GREEN
    Male = Color.UNIT_RED
    Female = Color.UNIT_BLUE
    Market = Color.Olive
    Home = Color.Brown

    def __str__(self):
        return self.name
    
    def color(self):
        return self.value
    
    def add(self, other):
        if isinstance(other, UnitType):
            return self.value.add(other.color())
        
        if isinstance(other, Color):
            return self.value.add(other)
        
        return self.color().add(other)
        