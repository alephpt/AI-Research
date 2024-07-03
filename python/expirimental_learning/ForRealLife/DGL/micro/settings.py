from enum import Enum

# This will act as our global accessor for configurations and constants
class Settings(Enum):
    GRID_SIZE = 10  # We started at 10
    SCREEN_SIZE = 1240
    BACKGROUND_COLOR = (24, 24, 24)
    N_JOBS = 1          # We want to make this an average and monitor "work"
    N_FOOD = 1          # We want to make this an average and monitor "food"
    N_POPULATION = 2
    CELL_SIZE = SCREEN_SIZE // GRID_SIZE
    FPS = 2
    INITIAL_E = 25  # What is the ideal energy level? We started at 25. 
    INITIAL_W = 50  # We want to see how far we can take this down. We started at 50.
    COST_OF_GOODS = 5 # TODO: Let every Agent set their own cost of food
    LIFETIME_REWARD_SCALAR = 10     # What incentives work best? Random bonus factors for genetic purposes sounds like a win.