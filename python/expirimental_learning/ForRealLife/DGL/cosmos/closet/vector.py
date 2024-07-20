from .direction import MoveAction

    #################
    ## VECTOR MATH ##
    #################

class Vector:
    def __init__(self, dxy, magnitude):
        self.x, self.y = dxy
        self.magnitude = magnitude

    def MoveAction(self):
        return MoveAction.fromValue((self.x, self.y))

    @staticmethod
    def New():
        return Vector((0, 0), 0)
    
    @staticmethod
    def Null():
        return Vector((0, 0), None)
