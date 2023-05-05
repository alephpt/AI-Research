import pygame, random

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
        self.image1 = pygame.image.load('assets/enemy1.png')
        self.image1 = pygame.transform.scale(self.image1, (64, 64))
        self.image2 = pygame.image.load('assets/enemy2.png')
        self.image2 = pygame.transform.scale(self.image2, (64, 64))
        self.image3 = pygame.image.load('assets/enemy3.png')
        self.image3 = pygame.transform.scale(self.image3, (64, 64))
        self.image4 = pygame.image.load('assets/enemy4.png')
        self.image4 = pygame.transform.scale(self.image4, (64, 64))
        self.image5 = pygame.image.load('assets/enemy5.png')
        self.image5 = pygame.transform.scale(self.image5, (64, 64))
        self.n_enemies_per_row = n_enemies_per_row
        self.n_rows = n_rows
        self.screen_size = screen_size
        self.enemies = [enemy for enemy in self.spawn(screen_size)]
        self.x_velocity = 1
        self.y_velocity = 0

    def level_up(self):
        self.n_enemies_per_row += 1
        self.n_rows += 1
        self.x_velocity += 0.5
        self.enemies = [enemy for enemy in self.spawn(self.screen_size)]

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