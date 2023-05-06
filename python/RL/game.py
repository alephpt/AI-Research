import pygame
import pygame_gui
from player import Player
from enemies import Enemies
import math

class GUI:
    def __init__(self, screen_size, lives, level, score):
        self.screen_size = screen_size
        self.manager = pygame_gui.UIManager(self.screen_size)
        # set pygame gui font to white
        self.lives_label = pygame_gui.elements.UILabel(relative_rect=pygame.Rect((10, 20), (100, 20)), text=f'Lives: {lives}', manager=self.manager)
        self.level_label = pygame_gui.elements.UILabel(relative_rect=pygame.Rect((10, 35), (100, 20)), text=f'Level: {level}', manager=self.manager)
        self.score_label = pygame_gui.elements.UILabel(relative_rect=pygame.Rect((10, 50), (100, 20)), text=f'Score: {score}', manager=self.manager)
        
    def update(self, lives, level, score):
        self.lives_label.set_text(f'Lives: {lives}')
        self.level_label.set_text(f'Level: {level}')
        self.score_label.set_text(f'Score: {score}')
        self.manager.update(1 / 60)
        

class Game:
    def __init__(self):
        self.running = True
        self.level = 1
        pygame.init()
        pygame.display.set_caption("Space Invader Knock Off by P3r5157")    
        icon = pygame.image.load('assets/ufo.png')
        pygame.display.set_icon(icon)
        pygame.event.set_allowed([pygame.QUIT, pygame.KEYDOWN, pygame.KEYUP])
        flags = pygame.DOUBLEBUF | pygame.HWSURFACE
        self.screen_size = (1200, 800)
        self.player = Player(self.screen_size)
        self.enemies = Enemies(0, self.screen_size)
        self.screen = pygame.display.set_mode(self.screen_size, flags)
        self.screen.set_alpha(None)
        self.timer = 0
        self.clock = pygame.time.Clock()
        self.gui = GUI(self.screen_size, self.player.lives, self.level, self.player.score)
        self.prev_score = 0
    
    def step(self, actions): 
        n_enemies = len(self.enemies.enemies)
        self.handle_input()
        self.process_actions(actions)
        self.update()
        self.draw()
        #find how close the player is to the enemies in the x direction
        dx = math.fabs(self.player.x - self.enemies.enemies[0].x if len(self.enemies.enemies) > 0 else 0)
        adx = math.fabs(self.player.x - self.enemies.attackers[0].x if len(self.enemies.attackers) > 0 else 0)
        ady = math.fabs(self.player.y - self.enemies.attackers[0].y if len(self.enemies.attackers) > 0 else 0)
        enemies_killed = n_enemies - len(self.enemies.enemies)
                 # reward for killing enemies, getting closer to enemies, 
                 # shooting when close, killing * level, getting away from attacking enemies
                 # and negative reward for shooting, time, and dying
        reward = (self.player.score - self.prev_score) + \
                 (self.screen_size[0] - dx) + \
                 ((self.screen_size[0] - dx) * len(self.player.bullets)) + \
                 (enemies_killed * 100 * self.level) + \
                 (self.enemies.attacking * self.level * ((adx / self.screen_size[0]) + (self.screen_size[1] - ady))) \
                 - (len(self.player.bullets) * 10) - (self.timer / 100) - (1000 * self.player.dead)
        self.prev_score = self.player.score
        return reward
    
    def process_actions(self, action):
        if action == 0:
            if self.player.x_velocity > -self.player.max_speed:
                self.player.x_velocity += -self.player.acceleration_rate
        if action == 1:
            if self.player.x_velocity < self.player.max_speed:
                self.player.x_velocity += self.player.acceleration_rate
        if action == 2:
            self.player.x_velocity = self.player.x_velocity * 0.1
        if action == 3:
            self.player.shoot()
    
    def handle_input(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                self.running = False
                
            # # if keystroke is pressed check whether its right or left
            # if event.type == pygame.KEYDOWN :
            #     if event.key == pygame.K_LEFT or event.key == pygame.K_a:
            #         if self.player.x_velocity > -self.player.max_speed:
            #             self.player.x_velocity += -self.player.acceleration_rate
            #     if event.key == pygame.K_RIGHT or event.key == pygame.K_d:
            #         if self.player.x_velocity < self.player.max_speed:
            #             self.player.x_velocity += self.player.acceleration_rate
            # if event.type == pygame.KEYUP or event.type == pygame.K_a or event.type == pygame.K_d:
            #     if event.key in [pygame.K_UP, pygame.K_DOWN, pygame.K_w, pygame.K_s]:
            #         self.player.x_velocity = self.player.x_velocity * 0.1
            # if event.type == pygame.KEYDOWN:
            #     if event.key == pygame.K_SPACE:
            #         self.player.shoot()
    
    def update(self):
        if len(self.enemies.enemies) == 0:
            self.player.level_up(self.level)
            self.enemies.level_up(self.level)
            self.level += 1
            
        if self.player.lives == 0:
            self.running = False
        
        self.player.update(self.enemies, self.level)
        self.enemies.update(self.screen_size)
        self.gui.update(self.player.lives, self.level, self.player.score)
        self.gui.manager.update(self.clock.get_time())
        self.timer += self.clock.get_time()
    
    def draw(self):
        self.screen.fill((0, 0, 0))
        self.gui.manager.draw_ui(self.screen)
        self.player.render(self.screen, self.player.x, self.player.y)
        self.enemies.draw(self.screen)
        pygame.display.update()
        
    def get_states(self):
        states = (self.level, *self.player.get_states(), *self.enemies.get_states())
        return states

        
    def reset(self):
        self.level = 1
        self.player = Player(self.screen_size)
        self.enemies = Enemies(0, self.screen_size)
        self.gui.update(self.player.lives, self.level, self.player.score)
        self.gui.manager.update(self.clock.get_time())
        self.timer = 0
        self.running = True