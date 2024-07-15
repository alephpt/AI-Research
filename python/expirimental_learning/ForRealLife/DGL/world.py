import pygame

from .society import Genome, Society
from .engine import Engine
from .engine.grid import Grid, CellType
from .society.agency import State
from .cosmos import Settings, LogLevel, Log

screen_size = Settings.SCREEN_SIZE.value
grid_size = Settings.GRID_SIZE.value
cell_size = Settings.CELL_SIZE.value

  ###############
  ## EVOLUTION ##
  ###############

class World(Genome, Grid, Society, Engine):
    def __init__(self):
        print(f'Creating World \tGrid Size: {grid_size},{grid_size}')
        print(f'\t\tTotal Grid Count: {Settings.TOTAL_GRID_COUNT.value}')
        self.screen = pygame.display.set_mode((screen_size, screen_size))
        super().__init__()
        Grid.__init__(self)
        Society.__init__(self)
        Engine.__init__(self)

    def alive(self):
        '''
        Returns all of the alive unites in the grid.'''
        return [unit for unit in self.units if unit.state != State.Dead]

    def update(self):
        Log(LogLevel.VERBOSE, "World", "Updating World")
        population_alive = all([unit.state != State.Dead for unit in self.units])

        # We should only hit this if all units are dead
        if not population_alive:
            Log(LogLevel.DEBUG, "World", "Population is Dead")
            self.repopulate()
            return

        self.updatePopulation(self.selected)
        self.updateStatistics()


    def selectCell(self, mouse_pos):
        '''
        Gives us the ability to select a cell from the UI'''
        x, y = mouse_pos
        x = x // Settings.CELL_SIZE.value
        y = y // Settings.CELL_SIZE.value
        index = y * Settings.GRID_SIZE.value + x

        # This may not work if a cell has moved to this location, but is not at this index
        if self.cells[index].type == CellType.Available:
            Log(LogLevel.INFO, "World", f"Selected Available at {x}, {y}")
            return

        self.selected = self.cells[index]

        Log(LogLevel.INFO, "World", f"Selected Cell at {x}, {y}")

    # TODO: Move this to the engine
    def gui(self):
        sections = ["Status", "AvgAge", "AvgHealth", "AvgWealth", "AvgHappiness", "AvgReward", 'Selected']
        values = [State.fromValue(self.n_alive), self.avg_age, self.avg_health, self.avg_wealth, self.avg_happiness, self.avg_reward, self.selected.__class__.__name__]
        font = pygame.font.Font(None, 22)

        relative_size = grid_size * .1
        relative_offset = grid_size - relative_size

        # Transparent Frame
        new_surface = pygame.Surface((relative_offset * cell_size, 8.2 * 22))
        new_surface.set_alpha(80)
        new_surface.fill((0, 0, 0))
        self.screen.blit(new_surface, (relative_size * cell_size * 7.2, 11))

        height_offset = 24
        for i, (section, value) in enumerate(zip(sections, values)):
            section = font.render(f"{section}:", True, (222, 222, 222, 80))
            value = font.render(f"{value}", True, (255, 255, 255, 80))
            width_offset = value.get_width() + 16
            self.screen.blit(section, ((grid_size - 12.5) * cell_size, i * 22 + height_offset))
            self.screen.blit(value, (grid_size * cell_size - width_offset - 22, i * 22 + height_offset))

    def draw(self):
        Log(LogLevel.VERBOSE, "World", "Drawing..")
        self.drawCells()
