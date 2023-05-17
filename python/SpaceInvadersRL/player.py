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
        for enemy in enemies.enemies:
            if self.collide(enemy):
                if enemy in enemies.attackers:
                    enemies.attackers.remove(enemy)
                
                enemies.enemies.remove(enemy)
                return True

    def render(self, screen, image, x, y):
        screen.blit(image, (x, y))
        
    def get_states(self):
        return (
            self.x,
            self.y,
        )

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
        self.dead = False
        self.respawn_timer = 60

    def level_up(self, level):
        self.bullets = []
        self.firing_rate *= 0.9
        self.firing_cap = self.firing_rate
        self.acceleration_rate *= 1.2
        self.max_speed *= 1.2
        self.score += 100 * level
        if level % 3 == 0:
            self.lives += 1

    def die(self):
        self.lives -= 1
        self.x = 600
        self.y = 700
        self.x_velocity = 0
        self.firing_rate = self.firing_cap
        self.score -= 100
        self.dead = True
        # set image to explosion
        #self.image = pygame.image.load('assets/explosion.png')
        #self.image = pygame.transform.scale(self.image, (64, 64))

    def collide(self, enemy):
        if (self.x + 16 > enemy.x and self.x < enemy.x + 48 or \
            self.x < enemy.x + 16 and self.x + 48 > enemy.x) and \
            (self.y + 16 > enemy.y and self.y < enemy.y + 48 or \
            self.y < enemy.y + 16 and self.y + 48 > enemy.y):
            return True
        return False

    def hit(self, enemies):
        for enemy in enemies.enemies:
            if self.collide(enemy):
                enemies.enemies.remove(enemy)
                return True
        return False

    def shoot(self):
        if pygame.time.get_ticks() - self.last_fired > self.firing_rate * 1000 and not self.dead:
            self.bullets.append(Bullet(self.x + 16, self.y - 32))
            self.last_fired = pygame.time.get_ticks()
            self.firing_rate = self.firing_cap

    def update(self, enemies, level):
        self.x += self.x_velocity
        
        if self.dead:
            self.respawn_timer -= 1
            if self.respawn_timer <= 0:
                self.dead = False
                self.respawn_timer = 60
        elif self.hit(enemies):
            self.die()
        
        for bullet in self.bullets:
            bullet.y -= self.bullet_speed
            if bullet.y < 0:
                self.bullets.remove(bullet)
            if bullet.hit(enemies):
                self.firing_rate *= 0.75
                self.bullets.remove(bullet)
                self.score += 10 * level

    def render(self, screen, x, y):
        if self.dead:
            return
        
        if self.x < 0:
            self.x = 0
            self.x_velocity = 0
            
        if self.x > screen.get_width() - 64:
            self.x = screen.get_width() - 64
            self.x_velocity = 0
            
        screen.blit(self.image, (x, y))
        
        for bullet in self.bullets:
            bullet.render(screen, self.bullet_image, bullet.x, bullet.y)

    def get_states(self):
        return (
            (
                self.x, 
                self.y, 
                self.x_velocity,
                self.max_speed,
            ),
            self.dead, 
            self.respawn_timer, 
            (
                self.firing_rate, 
                len(self.bullets)
            ),
        )