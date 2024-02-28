import pygame
from individual import Individual
from food import Food
from work import Work
import os

EPOCHS = 10
WIDTH = 1200
HEIGHT = 800


def logIndividuals(individuals):
    os.system('cls' if os.name == 'nt' else 'clear')
    
    for individual in individuals:
        ascii_text_color = "\033[1;32;40m" if individual.alive else "\033[1;31;40m"
        
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        print(ascii_text_color)
        print("Individual: ", individual.id)
        print("Sleeping" if individual.sleeping else "Awake")
        print("Target: ", individual.chosen_action)
        print("Satisfaction: ", individual.satisfaction)
        print("Energy: ", individual.energy)
        print("Wealth: ", individual.money)
        print("Age: ", individual.age)
        print("Reward: ", individual.reward)
        print("\033[0;37;40m")
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        print("")
    


# Main Pygame Loop
def main():
    pygame.init()
    main_screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Generational Learning")
    pygame.font.init()
    
    clock = pygame.time.Clock()
    running = True

    individuals = [Individual(i, WIDTH, HEIGHT) for i in range(4)]
    foods = [Food(WIDTH, HEIGHT) for _ in range(2)]
    employers = [Work(WIDTH, HEIGHT) for _ in range(2)]

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        main_screen.fill((0, 0, 0))
                
        for individual in individuals:
            individual.update(foods, employers, individuals)
            individual.draw(main_screen)
            
        logIndividuals(individuals)
            
        for food in foods:
            food.draw(main_screen)
            if food.ate:
                foods.remove(food)
                foods.append(Food(WIDTH, HEIGHT))
            
        for employer in employers:
            employer.draw(main_screen)
            
        pygame.display.flip()
        clock.tick(60)

        if not any([individual.alive for individual in individuals]):
            running = False

    pygame.quit()
    
if __name__ == "__main__":
    main()