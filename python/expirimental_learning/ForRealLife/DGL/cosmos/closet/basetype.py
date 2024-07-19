from enum import Enum

class BaseType(Enum):
    '''The Base Type of the Unit'''
    def __lt__(self, other):
        if isinstance(other, self.__class__):
            return self.value < other.value
        elif isinstance(other, int):
            return self.value < other
        else:
            return None
        
    def __gt__(self, other):
        if isinstance(other, self.__class__):
            return self.value > other.value
        elif isinstance(other, int):
            return self.value > other
        else:
            return None
        
    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.value == other.value
        elif isinstance(other, int):
            return self.value == other
        else:
            return None

    def __sub__(self, other):
        if isinstance(other, self.__class__):
            return self.value - other.value
        elif isinstance(other, int):
            return self.value - other
        else:
            return None

    def __add__(self, other):
        if isinstance(other, self.__class__):
            return self.value + other.value
        elif isinstance(other, int):
            return self.value + other
        else:
            return None
