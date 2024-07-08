import pygame

from .genetics import Genome
from DGL.meso import Grid
from DGL.meso.agency import State
from DGL.micro import Settings, Log, LogLevel, UnitType

cell_size = Settings.CELL_SIZE.value
grid_size = Settings.GRID_SIZE.value

# Implement our Dictionary of Placement Types for Society Evolution.

  ###############
  ## EVOLUTION ##
  ###############

# Will eventually host the genetic learning algorithm and epoch loop
class Society(Genome):
    def __init__(self, screen):
        self.screen = screen
        super().__init__()
        print('Creating Society with Grid Size:', grid_size)
        self.grid = Grid()
        self.selected = None

    def update(self):
        Log(LogLevel.VERBOSE, "Updating Society")
        population_alive = all([agent.state != State.Dead for agent in self.grid.agents])

        # We should only hit this if all agents are dead
        if not population_alive:
            print('Population is Dead')
            self.resetGrid()
            return

        self.grid.update()
        self.updateStatistics()

    def resetGrid(self):
        self.grid.repopulate()
        self.selected = None

    def selectT(self, set, x, y):
        for unit in set:
            if unit.x == x and unit.y == y:
                self.selected = unit
                return True
            
        return False

    def selectUnit(self, mouse_pos):
        '''
        Gives us the ability to select a cell from the UI'''
        x, y = mouse_pos
        x = x // Settings.CELL_SIZE.value
        y = y // Settings.CELL_SIZE.value
        index = y * Settings.GRID_SIZE.value + x

        # This may not work if a unit has moved to this location, but is not at this index
        if self.grid.cells[index].type == UnitType.Available:
            Log(LogLevel.INFO, f"Selected Available at {x}, {y}")
            return

        if self.selectT(self.grid.markets, x, y):
            Log(LogLevel.INFO, f"Selected Market at {x}, {y}")
            return
        
        if self.selectT(self.grid.homes, x, y):
            Log(LogLevel.INFO, f"Selected Home at {x}, {y}")
            return
        
        if self.selectT(self.grid.agents, x, y):
            Log(LogLevel.INFO, f"Selected Agent at {x}, {y}")
            return

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
        Log(LogLevel.VERBOSE, "Drawing Society")
        self.grid.draw(self.screen)
