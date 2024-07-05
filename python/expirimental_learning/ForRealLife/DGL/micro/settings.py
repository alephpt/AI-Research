from enum import Enum
import random

    ############
    ## LOGGER ##
    ############

class LogLevel(Enum):
    '''
    Defines the level of logging that will be output to the console.
    '''
    VERBOSE = -1
    DEBUG = 0
    INFO = 1
    WARNING = 2
    ERROR = 3
    CRITICAL = 4

    def __str__(self):
        return self.name


# This will act as our global accessor for configurations and constants
class Settings(Enum):
    DEBUG_LEVEL = LogLevel.VERBOSE

    # MACRO SETTINGS
    GRID_SIZE = 10  # We started at 10
    SCREEN_SIZE = 1240
    BACKGROUND_COLOR = (24, 24, 24)
    CELL_SIZE = SCREEN_SIZE // GRID_SIZE
    FPS = 2

    # MESO-MACRO SETTINGS
    N_JOBS = 1          # We want to make this an average and monitor "work"
    N_FOOD = 1          # We want to make this an average and monitor "food"
    N_POPULATION = 2

    RESTING_VALUE = 5
    RESTING_PLEASURE = 3

    MAX_EMPLOYEES = 2
    MONEY_EARNED = 10
    WORK_COST = 5
    WORK_REWARD = 1
    WORK_PLEASURE_FACTOR = -1

    AVAILABLE_FOOD = 3
    NUTRITIONAL_VALUE = 10  # Can we optimize this to work when the cost outweighs 
    FOOD_COST = 5           # the reward? - Can we factor in peronality_table?
    FOOD_REWARD = 1
    FOOD_PLEASURE_FACTOR = 1

    SEX_COST = 10
    SEX_REWARD = 10
    SEX_PLEASURE_FACTOR = 5

    REPRODUCTION_PLEASURE_FACTOR = 100
    REPRODUCTION_COST = .5                  # This is the cost of reproduction - do we want to make this a random factor, or define our economy?

    # AGENT SETTINGS
    COST_OF_GOODS = 5 # TODO: Let every Agent set their own cost of food
    INITIAL_E = 25                  # Default Energy Level -   Agents should inherit this with their DNA -  What is the ideal energy level? We started at 25. 
    INITIAL_W = 50                  # Default Money Level -    Agents should inherit this with their DNA -   We want to see how far we can take this down. We started at 50.
    LIFETIME_REWARD_SCALAR = 10     # Incentivizes living longer - this should start as 10:1 energy cost - # Maybe we add Random bonus factor for genetic alterations.
    IMPULSIVITY = 0.5               # How likely are you to make a random decision? - We started at 0.5 and need to end with near absolute
    CHANCE_TO_REPRODUCE = 0.5       # How likely are you to reproduce? - We started at 50%, but need to pick randomly, to allow for 'happy accidents'

    @staticmethod
    def randomLocation():
        return (random.randint(0, Settings.GRID_SIZE.value - 1), random.randint(0, Settings.GRID_SIZE.value - 1))