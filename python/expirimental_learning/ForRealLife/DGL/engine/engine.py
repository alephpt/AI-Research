import pygame

from DGL.cosmos import Log, LogLevel, Settings
from DGL import world

background = Settings.BACKGROUND_COLOR.value

class Engine:
    def __init__(self):
        pygame.init()
        pygame.display.set_caption('For Real Life!?')
        self.running = True
        self.clock = pygame.time.Clock()

    def events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                self.running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                self.selectCell(pygame.mouse.get_pos()) # Move this assignment to the World class
            

    def draw(self):
        self.screen.fill(background)
        self.draw()
        self.gui()
        pygame.display.flip()

    def run(self):
        Log(LogLevel.VERBOSE, "Ingine", "Running DGL Engine")

        while self.running:
            self.events()
            self.update()
            self.draw()
            self.clock.tick(Settings.FPS.value)
            
        pygame.quit()