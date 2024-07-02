from DGL.meso import Cell

class Grid:
    def __init__(self, count, size, screen):
        self.grid_size = count
        self.cell_size = size
        self.screen = screen
        self.cells = [[Cell(x, y, size) for x in range(count)] for y in range(count)]

    ## TODO: IMPLEMENT ALL MAJOR LOCATIONAL BASED UPDATE FUNCTIONS HERE, Can Be Callbacks if Necessary
    def update(self):
        # Here, we will update all of the cells each frame, to determine their new state

        # Iterate through all of the jobs, foods, and agents and determine their new state
        pass

    def draw(self):
        for row in self.cells:
            for col in row:
                col.draw(self.screen)
