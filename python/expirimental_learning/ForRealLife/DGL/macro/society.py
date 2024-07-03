import pygame
from .genesis import Genesis
from DGL.meso import Status, Target
from multiprocessing import Pool
import random


# Implement our Dictionary of Placement Types for Society Evolution.

  ###############
  ## EVOLUTION ##
  ###############

# Will eventually host the genetic learning algorithm and epoch loop
class Society:
    def __init__(self, screen):
        self.screen = screen

        # TODO: Fix duplicate placements bug
        self.repopulate()

    def repopulate(self):
        self.population, self.jobs, self.food  = Genesis.creation()

    def findTarget(self, target):
        if target == Target.Food:
            return random.choice(self.food)
        elif target == Target.Work:
            return random.choice(self.jobs)
        # TODO: Implement these later
        # elif target == Target.Mate:
        #     return 
        # elif target == Target.Home:
        #     return random.choice([agent for agent in self.population if agent.status == Status.Alive])
        else:
            return None

    def update(self):
        population_alive = all([agent.status != Status.Dead for agent in self.population])

        # We should only hit this if all agents are dead
        if not population_alive:
            print('Population is Dead')
            #self.jobs, self.food, self.population = self.repopulate()
            return

        for agent in self.population:
            #print(agent)
            agent.update(self.findTarget)
        # for job in self.jobs:
        #     job.update()
        # for food in self.food:
        #     food.update

        self.updateStatistics()

    def drawStatus(self):
        sections = ["Status", "AvgAge", "AvgHealth", "AvgWealth", "AvgHappiness", "AvgReward"]
        values = [Status.fromValue(self.n_alive), self.avg_age, self.avg_health, self.avg_wealth, self.avg_happiness, self.avg_reward]
        font = pygame.font.Font(None, 22)

        # Transparent Frame
        new_surface = pygame.Surface((2.5 * self.cell_size, 7 * 22))
        new_surface.set_alpha(100)
        new_surface.fill((0, 0, 0))
        self.screen.blit(new_surface, ((self.grid_size - 2.5) * self.cell_size - 22, 11))

        height_offset = 24
        for i, (section, value) in enumerate(zip(sections, values)):
            section = font.render(f"{section}:", True, (222, 222, 222))
            value = font.render(f"{value:.5}", True, (255, 255, 255))
            width_offset = value.get_width() + 16
            self.screen.blit(section, ((self.grid_size - 2.5) * self.cell_size, i * 22 + height_offset))
            self.screen.blit(value, (self.grid_size * self.cell_size - width_offset - 22, i * 22 + height_offset))



    def draw(self):
        #print('Drawing Society')
        for job in self.jobs:
            job.draw(self.screen)
        for food in self.food:
            food.draw(self.screen)
        for agent in self.population:
            agent.draw(self.screen)
