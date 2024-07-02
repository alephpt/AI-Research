import pygame
from grid import Grid
from society import Society

grid_size = 10
screen_size = 800
background = (24, 24, 24)
n_jobs = 1
n_food = 1
n_population = 2

class Engine:
    def __init__(self):
        pygame.init()
        pygame.display.set_caption('For Real Life!?')
        self.running = True
        self.screen = pygame.display.set_mode((screen_size, screen_size))
        self.grid = Grid(grid_size, screen_size // grid_size, self.screen)
        self.society = Society(grid_size, screen_size // grid_size, self.screen, n_population, n_jobs, n_food)
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
        pygame.display.flip()

    def run(self):
        while self.running:
            self.clock.tick(15) # 60 FPS
            self.events()
            self.update()
            self.draw()
        pygame.quit()