import pygame

class Bullet:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def collide(self, enemy):
        if self.x + 32 > enemy.x and self.x < enemy.x + 64 and self.y > enemy.y and self.y + 32 < enemy.y + 64:
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
        self.acceleration_rate = 0.5
        self.image = pygame.image.load('assets/player.png')
        self.image = pygame.transform.scale(self.image, (64, 64))
        self.bullets = []
        self.bullet_image = pygame.image.load('assets/bullet.png')
        self.bullet_image = pygame.transform.scale(self.bullet_image, (32, 32))
        self.bullet_speed = 10
        self.firing_rate = 1.2
        self.firing_cap = self.firing_rate
        self.last_fired = 0
        self.lives = 3
        self.max_speed = 2
        self.score = 0

    def level_up(self, level):
        self.bullets = []
        self.lives += 1
        self.firing_rate *= 0.8
        self.firing_cap = self.firing_rate
        self.acceleration_rate *= 1.2
        self.max_speed *= 1.2
        self.score += 100 * level

    def shoot(self):
        if pygame.time.get_ticks() - self.last_fired > self.firing_rate * 1000:
            self.firing_rate = self.firing_cap
            self.bullets.append(Bullet(self.x + 16, self.y - 32))
            self.last_fired = pygame.time.get_ticks()

    def update(self, enemies, level):
        self.x += self.x_velocity
        
        for bullet in self.bullets:
            bullet.y -= self.bullet_speed
            if bullet.y < 0:
                self.bullets.remove(bullet)
            if bullet.hit(enemies):
                self.firing_rate *= 0.8
                self.bullets.remove(bullet)
                self.score += 10 * level

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
