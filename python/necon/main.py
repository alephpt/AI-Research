import pygame
from engine import Engine
from blit import Circle, Triangle

screen_x, screen_y = (1200,800)

def main():
    engine = Engine("Necon", (screen_x, screen_y))
    engine.fill_color = (0,0,0)

    player = Triangle((screen_x/2, screen_y/2), (screen_x/2, screen_y/2), (255, 95, 125))

    while engine.running:
        engine.clock.tick(60)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                engine.running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    engine.running = False
                    # elif event.key == pygame.K_w:
                    #     player.move(0, -5)
                    # elif event.key == pygame.K_s:
                    #     player.move(0, 5)
                    # elif event.key == pygame.K_a:
                    #     player.move(-5, 0)
                    # elif event.key == pygame.K_d:
                    #     player.move(5, 0)

        player.draw(engine.screen)
        engine.run()
        
    pygame.quit()
    
if __name__ == "__main__":
    main()