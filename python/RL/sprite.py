import pygame

class Explosion:
    def __init__(self):
        self.sheet = pygame.image.load('assets/explosion.png')
        self.sequence = []
        self.frame = 0
        self.frame_counter = 0
        self.frame_rate = 1

        for i in range(5):
            for j in range(5):
                self.sequence.append(self.sheet.subsurface((j * 100, i * 100, 100, 100)))
                
    def get_frame(self):
        image = self.sequence[self.frame]
        image = pygame.transform.scale(image, (64, 64))
        return image
    
    def update(self, dt):
        self.frame_counter += self.frame_rate * dt
        self.frame = int(self.frame_counter)
        
        if self.frame >= len(self.sequence):
            self.frame = None
            self.frame_counter = 0
            return True
        
        return False
    
    def render(self, screen, x, y):
        screen.blit(self.get_frame(), (x, y))
        
def main():
    pygame.init()
    screen = pygame.display.set_mode((800, 600))
    clock = pygame.time.Clock()
    explosion = Explosion()
    running = True
    
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                running = False
        
        screen.fill((0, 0, 0))
        
        if explosion.update(clock.get_time() / 100):
            running = False
        
        if explosion.frame is not None:
            explosion.render(screen, 366, 266)
        
        pygame.display.flip()
        clock.tick(60)
        
if __name__ == '__main__':
    main()