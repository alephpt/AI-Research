

class Target: # This entire class could be a subclass of a 'Unit' class
    def __init__(self, type_of, dxy, magnitude):
        self.type = type_of
        self.magnitude = magnitude # TODO: TODO: TODO: ## IT IS IMPERITIVE THAT WE CLAMP THE RADIUS IN WHICH WE SCAN FOR POTENTIAL TARGETS # TODO: TODO: TODO:
        self.x, self.y = dxy
        self.potential = 0  # TODO: Determine the Potential of Some Given Target - THIS IS SEPERATE FROM THE ETHICAL MATRIX
    
    @staticmethod
    def new():
        '''Generate a new target'''
        return Target(None, (0, 0), 0)
    
    def load(self, target):
        '''Load a target from a tuple'''
        self.type = target.type
        self.magnitude = target.magnitude
        self.x, self.y = target.dxy
        self.potential = target.potential

    def pool(self):
        '''Return the target as a tuple(type, (x, y), magnitude, potential)'''
        return (self.type, (self.x, self.y), self.magnitude, self.potential)

    # TODO: Integrate Integrity vs Compassion
    def evaluateTarget(self, target):
        # We have to calculate the way we determine potential

        ## If we are seeking a market, we want to get the ratio of buyers to sellers relative to the state we are seeking
        ## If we are seeking a mate, we want both targets to appeal to each other
        pass
