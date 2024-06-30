import pygame
from engine import Engine
from blit import Shape

screen_x, screen_y = (1200,800)

def main():
    engine = Engine("Necon", (screen_x, screen_y))
    engine.fill_color = (0,0,0)
    player = Shape((screen_x / 2, screen_y / 2), (100,100), (screen_x, screen_y), (255,255,255))

    while engine.running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                engine.running = False

        if pygame.key.get_pressed()[pygame.K_w]:
            player.accelerate(.5)
        elif pygame.key.get_pressed()[pygame.K_s]:
            player.accelerate(-.5)  
        else:
            player.decelerate()    
                      
        if pygame.key.get_pressed()[pygame.K_a]:
            player.turn(-1)
        elif pygame.key.get_pressed()[pygame.K_d]:
            player.turn(1)
        else:
            player.unturn()

        engine.screen.fill(engine.fill_color)
        engine.draw(player)
        
        engine.clock.tick(60)
        pygame.display.flip()

        
    pygame.quit()
    
if __name__ == "__main__":
    main()