import pygame
from .agency import Action, Target
from DGL.micro import Unit, Settings


# State Space ? Q
class Azimuth: 
    def __init__(self, x, y):
        # Local Calculations
        self.x = x
        self.y = y
        self.size = Settings.CELL_SIZE.value

        # Table Values
        self.state = Unit.Available # This is used to define what state we are actually in.
        self.magnitude = 0
        self.action = Action.random()
        self.target = None
        self.target_direction_vector = (0, 0)
        self.reward = 0 # This will be interesting considering the potential for a Unit State to handle an enumeration of states
    
    def __str__(self):
        return f"({self.x}, {self.y}) - {self.state}"
    
    # We can potentially compare the Unit type to the cell state
    def __eq__(self, other):
        return self.state == other

    # TODO: REALLY REALLY NEED TO TODO THIS
    # TODO: Implement this as a type check with a type update.
    def update(self, occupancy):
        self.state = occupancy

    # Needs to be as random as possible to explore all possible states
    def chooseRandomAction(self, findTarget):
        self.chosen_direction = Action.randomDirection()
        self.target = findTarget(Target.random()) 

        if self.target is None:
            print(f"Agent {self} has no target")
            self.target = self

    def draw(self, screen):
        pygame.draw.rect(screen, self.state.value, (self.x * self.size, self.y * self.size, self.size, self.size), 1)