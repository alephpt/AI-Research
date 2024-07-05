import pygame

from DGL.micro import Log, LogLevel
from .genesis import Genesis
from .dna import Genome
from DGL.meso import State
from DGL.micro import Settings, Unit

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
        self.cells = Unit.set()


        # TODO: Fix duplicate placements bug
        self.repopulate()

    def repopulate(self):
        Log(LogLevel.VERBOSE, "Repopulating Society")
        self.population, self.jobs, self.food  = Genesis.creation(self.cells)

    def update(self):
        Log(LogLevel.VERBOSE, "Updating Society")
        population_alive = all([agent.state != State.Dead for agent in self.population])

        # We should only hit this if all agents are dead
        if not population_alive:
            print('Population is Dead')
            #self.jobs, self.food, self.population = self.repopulate()
            return

        # This is where we need to update the state of all of the cells
        for agent in self.population:
            agent.update()
        
        for unit in self.cells:
                unit.update()
    
        self.updateStatistics()

    def drawStatus(self):
        sections = ["Status", "AvgAge", "AvgHealth", "AvgWealth", "AvgHappiness", "AvgReward"]
        values = [State.fromValue(self.n_alive), self.avg_age, self.avg_health, self.avg_wealth, self.avg_happiness, self.avg_reward]
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
        # for job in self.jobs:
        #     job.draw(self.screen)
        # for food in self.food:
        #     food.draw(self.screen)
        for agent in self.population:
            agent.draw(self.screen)
        for unit in self.cells:
            unit.draw(self.screen)
