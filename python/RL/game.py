import pygame
import pygame_gui
from player import Player
from enemies import Enemies

class GUI:
    def __init__(self, screen_size, lives, level, score):
        self.screen_size = screen_size
        self.manager = pygame_gui.UIManager(self.screen_size)
        # set pygame gui font to white
        self.lives_label = pygame_gui.elements.UILabel(relative_rect=pygame.Rect((10, 20), (100, 15)), text=f'Lives: {lives}', manager=self.manager)
        self.level_label = pygame_gui.elements.UILabel(relative_rect=pygame.Rect((10, 35), (100, 15)), text=f'Level: {level}', manager=self.manager)
        self.score_label = pygame_gui.elements.UILabel(relative_rect=pygame.Rect((10, 50), (100, 15)), text=f'Score: {score}', manager=self.manager)
        
    def update(self, lives, level, score):
        self.lives_label.set_text(f'Lives: {lives}')
        self.level_label.set_text(f'Level: {level}')
        self.score_label.set_text(f'Score: {score}')
        self.manager.update(1 / 60)
        

class Game:
    def __init__(self):
        pygame.init()
        pygame.display.set_caption("Space Invader Knock Off by P3r5157")    
        icon = pygame.image.load('assets/ufo.png')
        pygame.display.set_icon(icon)
        pygame.event.set_allowed([pygame.QUIT, pygame.KEYDOWN, pygame.KEYUP])
        flags = pygame.DOUBLEBUF | pygame.HWSURFACE
        self.screen_size = (1200, 800)
        self.screen = pygame.display.set_mode(self.screen_size, flags)
        self.screen.set_alpha(None)
        self.clock = pygame.time.Clock()
        self.player = Player(self.screen_size)
        self.enemies = Enemies(4, 2, self.screen_size)
        self.running = True
        self.level = 1
        self.gui = GUI(self.screen_size, self.player.lives, self.level, self.player.score)
    
    def run(self): 
        self.handle_input()
        self.update()
        self.draw()
    
    def handle_input(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                self.running = False
                
            # if keystroke is pressed check whether its right or left
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT or event.key == pygame.K_a:
                    if self.player.x_velocity > -self.player.max_speed:
                        self.player.x_velocity += -self.player.acceleration_rate
                if event.key == pygame.K_RIGHT or event.key == pygame.K_d:
                    if self.player.x_velocity < self.player.max_speed:
                        self.player.x_velocity += self.player.acceleration_rate
            if event.type == pygame.KEYUP or event.type == pygame.K_a or event.type == pygame.K_d:
                if event.key in [pygame.K_UP, pygame.K_DOWN, pygame.K_w, pygame.K_s]:
                    self.player.x_velocity = self.player.x_velocity * 0.1
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    self.player.shoot()
    
    def update(self):
        if len(self.enemies.enemies) == 0:
            self.player.level_up(self.level)
            self.enemies.level_up()
            self.level += 1
            
        
        self.player.update(self.enemies.enemies, self.level)
        self.enemies.update(self.screen_size)
        self.gui.update(self.player.lives, self.level, self.player.score)
        self.gui.manager.update(self.clock.get_time())
    
    def draw(self):
        self.screen.fill((0, 0, 0))
        self.gui.manager.draw_ui(self.screen)
        self.player.render(self.screen, self.player.x, self.player.y)
        self.enemies.draw(self.screen)
        pygame.display.update()
        self.clock.tick(60)