import pygame
import random

class UFO:
    def __init__(self, x, y, image):
        self.x = x
        self.y = y
        self.image = image
        self.alive = True

    def render(self, screen):
        if self.alive:
            screen.blit(self.image, (self.x, self.y))

class Enemies:
    def __init__(self, n_enemies_per_row, n_rows, screen_size):
        self.image1 = pygame.image.load('enemy1.png')
        self.image1 = pygame.transform.scale(self.image1, (64, 64))
        self.image2 = pygame.image.load('enemy2.png')
        self.image2 = pygame.transform.scale(self.image2, (64, 64))
        self.image3 = pygame.image.load('enemy3.png')
        self.image3 = pygame.transform.scale(self.image3, (64, 64))
        self.image4 = pygame.image.load('enemy4.png')
        self.image4 = pygame.transform.scale(self.image4, (64, 64))
        self.image5 = pygame.image.load('enemy5.png')
        self.image5 = pygame.transform.scale(self.image5, (64, 64))
        self.n_enemies_per_row = n_enemies_per_row
        self.n_rows = n_rows
        self.enemies = [enemy for enemy in self.spawn(screen_size)]
        self.x_velocity = 0.2
        self.y_velocity = 0

    def spawn(self, screen_size):
        x_offset = screen_size[0] // 2 - (self.n_enemies_per_row * 64)
        y_offset = 64
        for rows in range(self.n_rows):
            for enemies in range(self.n_enemies_per_row):
                yield UFO(enemies * 80 + x_offset, rows * 80 + y_offset, random.choice([self.image1, self.image2, self.image3, self.image4, self.image5]))

    def update(self, screen_size):
        self.y_velocity = 0
        
        for enemy in self.enemies:
            if enemy.x < 0:
                self.x_velocity = -self.x_velocity
                self.y_velocity = 32
                break
            elif enemy.x > screen_size[0] - 64:
                self.x_velocity = -self.x_velocity
                self.y_velocity = 32
                break
        
        for enemy in self.enemies:
            enemy.x += self.x_velocity
            enemy.y += self.y_velocity
        
    def draw(self, screen):
        for enemy in self.enemies:
            enemy.render(screen)
        

class Bullet:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def collide(self, enemy):
        if self.x > enemy.x and self.x < enemy.x + 64 and self.y > enemy.y and self.y < enemy.y + 64:
            return True
        return False

    def hit(self, enemies):
        for enemy in enemies:
            if self.collide(enemy):
                enemies.remove(enemy)
                return True

    def render(self, screen, image, x, y):
        screen.blit(image, (x, y))


class Player:
    def __init__(self, screen_size):
        self.x = screen_size[0] / 2 - 32
        self.y = screen_size[1] - 128
        self.x_velocity = 0
        self.acceleration_rate = 0.2
        self.image = pygame.image.load('player.png')
        self.image = pygame.transform.scale(self.image, (64, 64))
        self.bullets = []
        self.bullet_image = pygame.image.load('bullet.png')
        self.bullet_image = pygame.transform.scale(self.bullet_image, (32, 32))
        self.bullet_speed = 0.3
        self.firing_rate = 0.25
        self.last_fired = 0

    def shoot(self):
        if pygame.time.get_ticks() - self.last_fired > self.firing_rate * 1000:
            self.bullets.append(Bullet(self.x + 16, self.y - 32))
            self.last_fired = pygame.time.get_ticks()

    def update(self, enemies):
        self.x += self.x_velocity
        
        for bullet in self.bullets:
            bullet.y -= self.bullet_speed
            if bullet.y < 0:
                self.bullets.remove(bullet)
            if bullet.hit(enemies):
                self.bullets.remove(bullet)

    def render(self, screen, x, y):
        if self.x < 0:
            self.x = 0
            self.x_velocity = 0
            
        if self.x > screen.get_width() - 64:
            self.x = screen.get_width() - 64
            self.x_velocity = 0
            
        screen.blit(self.image, (x, y))
        
        for bullet in self.bullets:
            bullet.render(screen, self.bullet_image, bullet.x, bullet.y)

class Game:
    def __init__(self):
        pygame.init()
        pygame.display.set_caption("Space Invader Knock Off by P3r5157")    
        icon = pygame.image.load('ufo.png')
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
        self.level = 0
    
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
                    self.player.x_velocity += -self.player.acceleration_rate
                if event.key == pygame.K_RIGHT or event.key == pygame.K_d:
                    self.player.x_velocity += self.player.acceleration_rate
            if event.type == pygame.KEYUP or event.type == pygame.K_a or event.type == pygame.K_d:
                if event.key in [pygame.K_UP, pygame.K_DOWN, pygame.K_w, pygame.K_s]:
                    self.player.x_velocity = self.player.x_velocity * 0.1
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    self.player.shoot()
    
    def update(self):
        self.player.update(self.enemies.enemies)
        self.enemies.update(self.screen_size)
    
    def draw(self):
        self.screen.fill((0, 0, 0))
        self.player.render(self.screen, self.player.x, self.player.y)
        self.enemies.draw(self.screen)
        pygame.display.update()