import pygame, random

# enemies per row, rows
levels = [(4, 2), (4, 3), (5, 3), (5, 4), (6, 4), (7, 4), (7, 5), (8, 5), (9, 5), (9, 6)]

class UFO:
    def __init__(self, index, x, y, image):
        self.index = index
        self.x = x
        self.y = y
        self.origin_y = y
        self.image = image
        self.attacking = False

    def render(self, screen):
        screen.blit(self.image, (self.x, self.y))
    
    def get_states(self):
        return (
            self.x,
            self.y,
            self.attacking
        )

class Enemies:
    def __init__(self, level, screen_size):
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
        enemies_per_row, rows = levels[level]
        self.n_enemies_per_row = enemies_per_row
        self.n_rows = rows
        self.screen_size = screen_size
        self.enemies = [enemy for enemy in self.spawn(screen_size)]
        self.x_velocity = 1
        self.y_velocity = 0
        self.attackers = []
        self.attacking = False
        self.attack_speed = 2
        self.attack_chance = 0.005

    def level_up(self, level):
        enemies_per_row, rows = levels[level]
        self.n_enemies_per_row = enemies_per_row
        self.n_rows = rows
        self.x_velocity *= 1.1
        self.enemies = [enemy for enemy in self.spawn(self.screen_size)]
        self.attacking = False
        self.attackers = []
        self.attack_speed *= 1.15
        self.attack_chance += 0.005
    
    def spawn(self, screen_size):
        x_offset = screen_size[0] // 2 - (self.n_enemies_per_row * 64)
        y_offset = 64
        for rows in range(self.n_rows):
            for enemies in range(self.n_enemies_per_row):
                index = rows * self.n_enemies_per_row + enemies
                x = enemies * 80 + x_offset
                y = rows * 80 + y_offset
                image = random.choice([self.image1, self.image2, self.image3, self.image4, self.image5])
                yield UFO(index, x, y, image)

    def get_attackers(self):
        col = random.randint(0, self.n_enemies_per_row - 1)
        
        for i in range(self.n_rows):
            index = i * self.n_enemies_per_row + col
            for enemy in self.enemies:
                if enemy.index == index:
                    yield enemy
                    
    def attack(self):
        self.attacking = True
        
        # either we pick a column or we pick a single random enemy
        self.attackers = [enemy for enemy in self.get_attackers()] \
                            if random.choice([True, False]) else \
                            [self.enemies[random.randint(0, len(self.enemies) - 1)]]
                            
        for enemy in self.attackers: 
            enemy.attacking = True

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
            enemy.origin_y += self.y_velocity

            if enemy.attacking:
                if enemy.y > screen_size[1]:
                    enemy.y = 0 - 64
                elif enemy.origin_y > enemy.y and \
                     enemy.origin_y - self.attack_speed <= enemy.y:
                    enemy.attacking = False
                    self.attackers.remove(enemy)
                enemy.y += self.attack_speed
            else:
                enemy.y += self.y_velocity
                
        # check if any enemies are attacking and set attacking to false if not
        self.attacking = False if len(self.attackers) == 0 else True
        
        if not self.attacking and random.random() < self.attack_chance:
            self.attack()
        
    def draw(self, screen):
        for enemy in self.enemies:
            enemy.render(screen)
            
    def get_states(self):
        enemy_states = tuple(enemy.get_states() for enemy in self.enemies)
        return (
            *enemy_states,
        )