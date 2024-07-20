import pygame

from .society import Genome, Society
from .society.agency import State
from .engine import Engine
from .cosmos import Settings, LogLevel, Log


screen_size = Settings.SCREEN_SIZE.value
grid_size = Settings.GRID_SIZE.value
cell_size = Settings.CELL_SIZE.value

  ###############
  ## EVOLUTION ##
  ###############

class World(Genome, Society, Engine):
    def __init__(self):
        Log(LogLevel.INFO, "World", f'Creating Grid Size: {grid_size},{grid_size}')
        Log(LogLevel.INFO, "World", f'Creating Total Grid Count: {Settings.TOTAL_GRID_COUNT.value}')
        Log(LogLevel.INFO, "World", f'Creating Cell Size: {cell_size}')
        Log(LogLevel.INFO, "World", f'Creating Screen Size: {screen_size}')
        self.screen = pygame.display.set_mode((screen_size, screen_size))
        Log(LogLevel.INFO, "World", f'Creating Screen Size: {screen_size},{screen_size}')
        self.running = False        # TODO: Dedup
        self.cells = set()          # TODO: Dedup
        self.units = set()          # TODO: Dedup
        self.selected = None        # TODO: Dedup
        Genome.__init__(self)
        Society.__init__(self)
        Engine.__init__(self)

    def alive(self):
        '''
        Returns all of the alive unites in the grid.'''
        return [unit for unit in self.units if unit.state != State.Dead]

    # This is a callback function that we pass down to the engine to give us a higher order of control
    def update(self):
        #Log(LogLevel.VERBOSE, "World", "Updating World")
        self.updateUnits(self.selected)
        self.drawUnits()

        # We should only hit this if all units are dead
        if len(self.alive()) == 0:
            Log(LogLevel.DEBUG, "World", "Population is Dead")
            self.repopulate()
            return

        self.updateStatistics()

    def run(self):
        Engine.runLoop(self, self.update)