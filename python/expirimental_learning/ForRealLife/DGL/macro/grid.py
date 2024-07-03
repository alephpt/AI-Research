from DGL.meso import Azimuth
from DGL.micro import Settings

class Grid:
    def __init__(self, screen):
        self.screen = screen
        self.cells = [[Azimuth(x, y) for x in range(Settings.GRID_SIZE.value)] for y in range(Settings.GRID_SIZE.value)]

    ## TODO: IMPLEMENT ALL MAJOR LOCATIONAL BASED UPDATE FUNCTIONS HERE, Can Be Callbacks if Necessary
    def update(self):
        # Here, we will update all of the cells each frame, to determine their new state

        # Iterate through all of the jobs, foods, and agents and determine their new state
        pass

    def draw(self):
        for row in self.cells:
            for col in row:
                col.draw(self.screen)
