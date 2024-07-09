import pygame

from .genetics import Genome
from DGL.meso import Grid
from DGL.meso.agency import State
from DGL.micro import Settings, Log, LogLevel, UnitType

cell_size = Settings.CELL_SIZE.value
grid_size = Settings.GRID_SIZE.value

  ###############
  ## EVOLUTION ##
  ###############

class World(Genome, Grid):
    def __init__(self, screen):
        print('Creating World with Grid Size:', grid_size)
        self.screen = screen
        super().__init__()
        Grid.__init__(self)

    def update(self):
        Log(LogLevel.VERBOSE, "Updating World")
        population_alive = all([agent.state != State.Dead for agent in self.agents])

        # We should only hit this if all agents are dead
        if not population_alive:
            print('Population is Dead')
            self.repopulate()
            return

        self.updateGrid(self.selected)
        self.updateStatistics()


    def selectUnit(self, mouse_pos):
        '''
        Gives us the ability to select a cell from the UI'''
        x, y = mouse_pos
        x = x // Settings.CELL_SIZE.value
        y = y // Settings.CELL_SIZE.value
        index = y * Settings.GRID_SIZE.value + x

        # This may not work if a unit has moved to this location, but is not at this index
        if self.cells[index].type == UnitType.Available:
            Log(LogLevel.INFO, f"Selected Available at {x}, {y}")
            return

        self.selected = self.cells[index]

        Log(LogLevel.INFO, f"Selected Cell at {x}, {y}")

    # TODO: Move this to the engine
    def gui(self):
        sections = ["Status", "AvgAge", "AvgHealth", "AvgWealth", "AvgHappiness", "AvgReward", 'Selected']
        values = [State.fromValue(self.n_alive), self.avg_age, self.avg_health, self.avg_wealth, self.avg_happiness, self.avg_reward, self.selected.__class__.__name__]
        font = pygame.font.Font(None, 22)

        # Transparent Frame
        new_surface = pygame.Surface((2.5 * cell_size, 7 * 22))
        new_surface.set_alpha(100)
        new_surface.fill((0, 0, 0))
        self.screen.blit(new_surface, ((grid_size - 2.5) * cell_size - 22, 11))

        height_offset = 24
        for i, (section, value) in enumerate(zip(sections, values)):
            section = font.render(f"{section}:", True, (222, 222, 222))
            value = font.render(f"{value}", True, (255, 255, 255))
            width_offset = value.get_width() + 16
            self.screen.blit(section, ((grid_size - 2.5) * cell_size, i * 22 + height_offset))
            self.screen.blit(value, (grid_size * cell_size - width_offset - 22, i * 22 + height_offset))



    def draw(self):
        Log(LogLevel.VERBOSE, "Drawing World")
        self.drawCells()
