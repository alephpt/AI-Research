import pygame
import random
from individual import Individual
from food import Food
from work import Work

EPOCHS = 10
WIDTH = 1200
HEIGHT = 800

def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Generational Learning")
    pygame.font.init()
    text = pygame.font.SysFont('Comic Sans MS', 30)
    clock = pygame.time.Clock()
    running = True

    individuals = [Individual(i, WIDTH, HEIGHT) for i in range(4)]
    foods = [Food(WIDTH, HEIGHT) for _ in range(2)]
    employers = [Work(WIDTH, HEIGHT) for _ in range(2)]

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        screen.fill((0, 0, 0))

        for food in foods:
            food.draw(screen)

        for individual in individuals:
            individual.draw(screen)
            
        for employer in employers:
            employer.draw(screen)

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()
    
if __name__ == "__main__":
    main()