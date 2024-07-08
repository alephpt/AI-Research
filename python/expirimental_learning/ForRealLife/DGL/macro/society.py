import pygame

from .genetics import Genome
from DGL.meso import Grid
from DGL.meso.agency import State
from DGL.micro import Settings, Log, LogLevel

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

    def selectUnit(self, mouse_pos):
        x, y = mouse_pos
        x = x // Settings.CELL_SIZE.value
        y = y // Settings.CELL_SIZE.value

        for market in self.grid.markets:
            if market.x == x and market.y == y:
                Log(LogLevel.INFO, f"Selected Market at {x}, {y}")
                self.selected = market
                return
        
        for house in self.grid.homes:
            if house.x == x and house.y == y:
                Log(LogLevel.INFO, f"Selected House at {x}, {y}")
                self.selected = house
                return
        
        for agent in self.grid.agents:
            if agent.x == x and agent.y == y:
                Log(LogLevel.INFO, f"Selected Agent at {x}, {y}")
                self.selected = agent
                return
            
        for unit in self.grid.cells:
            if unit.x == x and unit.y == y:
                self.selected = unit
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
