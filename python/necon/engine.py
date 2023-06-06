import pygame

class Engine:
    def __init__(self, title, screen_size):
        self.screen_x, self.screen_y = screen_size
        pygame.init()
        self.screen = pygame.display.set_mode((self.screen_x, self.screen_y))
        pygame.display.set_caption(title)
        self.clock = pygame.time.Clock()
        self.fill_color = (255,255,255)
        self.running = True
        
    def run(self):
        self.clock.tick(30)
        self.screen.fill(self.fill_color)
        pygame.display.flip()