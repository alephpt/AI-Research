import pygame

from DGL.micro import LogLevel, Log
from .society import Society
from DGL.micro import Settings

screen_size = Settings.SCREEN_SIZE.value
background = Settings.BACKGROUND_COLOR.value

class Engine:
    def __init__(self):
        pygame.init()
        pygame.display.set_caption('For Real Life!?')
        self.running = True
        self.screen = pygame.display.set_mode((screen_size, screen_size)) # Could abstract this out further - but not necessary
        self.society = Society(self.screen)
        self.clock = pygame.time.Clock()

    def events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                self.running = False

    def draw(self):
        self.screen.fill(background)
        self.society.draw()
        self.society.drawStatus()
        pygame.display.flip()

    def run(self):
        Log(LogLevel.VERBOSE, "Running DGL Engine")

        while self.running:
            self.clock.tick(Settings.FPS.value)
            self.events()
            self.society.update()
            self.draw()
            
        pygame.quit()