import pygame

from .world import World
from DGL.micro import Settings, LogLevel, Log

screen_size = Settings.SCREEN_SIZE.value
background = Settings.BACKGROUND_COLOR.value

class Engine:
    def __init__(self):
        pygame.init()
        pygame.display.set_caption('For Real Life!?')
        self.running = True
        self.screen = pygame.display.set_mode((screen_size, screen_size)) # Could abstract this out further - but not necessary
        self.world = World(self.screen)
        self.clock = pygame.time.Clock()

    def events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                self.running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                self.world.selectCell(pygame.mouse.get_pos()) # Move this assignment to the World class
            

    def draw(self):
        self.screen.fill(background)
        self.world.draw()
        self.world.gui()
        pygame.display.flip()

    def run(self):
        Log(LogLevel.VERBOSE, "Ingine", "Running DGL Engine")

        while self.running:
            self.events()
            self.world.update()
            self.draw()
            self.clock.tick(Settings.FPS.value)
            
        pygame.quit()