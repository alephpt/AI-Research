import pygame
from .grid import Grid
from .society import Society
from DGL.micro import Settings

grid_size = Settings.GRID_SIZE.value
screen_size = Settings.SCREEN_SIZE.value
cell_size = Settings.CELL_SIZE.value
background = Settings.BACKGROUND_COLOR.value
n_jobs = Settings.N_JOBS.value
n_food = Settings.N_FOOD.value
n_population = Settings.N_POPULATION.value

class Engine:
    def __init__(self):
        pygame.init()
        pygame.display.set_caption('For Real Life!?')
        self.running = True
        self.screen = pygame.display.set_mode((screen_size, screen_size)) # Could abstract this out further - but not necessary
        self.grid = Grid(self.screen)
        self.society = Society(self.screen)
        self.clock = pygame.time.Clock()

    def events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                self.running = False

    def genesis(self):
        self.society.populate(n_jobs, n_food, n_population)

    def repopulate(self):
        return self.society.populate(n_jobs, n_food, n_population)

    def update(self):
        self.society.update()
        #self.grid.update()

    def draw(self):
        self.screen.fill(background)
        self.society.draw()
        self.grid.draw()
        self.society.drawStatus()
        pygame.display.flip()

    def run(self):
        self.genesis()

        while self.running:
            self.clock.tick(Settings.FPS.value)
            self.events()
            self.update()
            self.draw()
        pygame.quit()