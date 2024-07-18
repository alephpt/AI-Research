from enum import Enum
import math
import random

    ############
    ## LOGGER ##
    ############

class LogLevel(Enum):
    '''
    Defines the level of logging that will be output to the console.
    '''
    VERBOSE = -4
    DEBUG = -3
    INFO = -2
    ALERT = -1
    RELEASE = 0
    WARNING = 1
    ERROR = 2
    FATAL = 3

    def __str__(self):
        return self.name


# This will act as our global accessor for configurations and constants
class Settings(Enum):
    DEBUG_LEVEL = LogLevel.DEBUG.value
    MEMORY_SIZE = 2048 # NOT IMPLEMENTED
    POOL_SIZE = 3

    # RL Learning Rate
    N_STATE_INPUTS = 4
    N_STATE_OUTPUTS = 5

    N_TARGETING_INPUTS = 4
    N_TARGETING_OUTPUTS = 3

    LEARNING_RATE = 0.01
    GAMMA = 0.95
    EPSILON = 0.7


    # SCREEN SETTINGS
    SCREEN_SIZE = 1200
    BACKGROUND_COLOR = (24, 24, 24)
    FPS = 15


    GRID_SIZE = 10  # We started at 10
    CELL_SIZE = SCREEN_SIZE // GRID_SIZE
    GRID_START = GRID_SIZE // 10
    GRID_END = GRID_SIZE - GRID_START

    TOTAL_GRID_BORDER_SIZE = GRID_START ** 2
    TOTAL_GRID_COUNT = GRID_SIZE ** 2
    TOTAL_SPAWN_AREA = int(TOTAL_GRID_COUNT - TOTAL_GRID_BORDER_SIZE)

    UNIT_RADIUS = 5 * CELL_SIZE

    # MESO-MACRO SETTINGS
    N_POPULATION = 2
    N_JOBS = 1                                  # This throttles supply and demand for food and money
    N_HOUSES = 1

    # Unit Defaults
    COST_OF_GOODS = 5                           # TODO: Let every Unit set their own cost of food
    INITIAL_E = 150                             # Default Energy Level -   Units should inherit this with their DNA -  What is the ideal energy level? We started at 25. 
    INITIAL_W = 50                              # Default Money Level -    Units should inherit this with their DNA -   We want to see how far we can take this down. We started at 50.
    LIFETIME_REWARD_SCALAR = 10                 # Incentivizes living longer - this should start as 10:1 energy cost - # Maybe we add Random bonus factor for genetic alterations.
    IMPULSIVITY = 0.1    # 1 would be 100%      # How likely are you to make a random decision? - We started at 0.5 and need to end with near absolute
    CHANCE_TO_REPRODUCE = 0.5                   # How likely are you to reproduce? - We started at 50%, but need to pick randomly, to allow for 'happy accidents'

    # Sleep Values
    MAX_SLEEP = 6                               # Let's allow for this to be traced later
    RESTING_VALUE = 5
    RESTING_COST = 3                            # This could be a bit more dynamic - clamped to a small range
    RESTING_PLEASURE = 3

    # Work Values
    MAX_EMPLOYEES = 2
    MONEY_EARNED = 10
    WORK_COST = 5
    WORK_REWARD = 1
    WORK_PLEASURE_FACTOR = -1

    # Food Values
    NUTRITIONAL_VALUE = 10                      # Can we optimize this to work when the cost outweighs 
    FOOD_COST = 5                               # the reward? - Can we factor in peronality_table?
    FOOD_REWARD = 1
    FOOD_PLEASURE_FACTOR = 1

    # Sex Values
    SEX_COST = 10
    SEX_REWARD = 10
    SEX_PLEASURE_FACTOR = 5
    REPRODUCTION_PLEASURE_FACTOR = 100
    REPRODUCTION_COST = .5                       # This is the cost of reproduction - do we want to make this a random factor, or define our economy?
    Population_Randomness_Factor = .50 # %       # This is the factor that will allow for random population growth
    Population_Deviation = .667                   # This is the deviation that will allow for random population growth

    @staticmethod
    def randomPopulation():
        random.seed()
        percent_population = Settings.N_POPULATION.value * Settings.Population_Randomness_Factor.value
        deviation = percent_population * Settings.Population_Deviation.value
        return 

    @staticmethod
    def randomImpulse():
        impulse_factor = Settings.IMPULSIVITY.value
        impulse_offset = impulse_factor * 1.5
        return impulse_factor + Settings.randFluxFloat(impulse_offset)

    @staticmethod
    def randFluxFloat(deviation):
        '''Returns some value +/- some deviation / 2'''
        random.seed()
        return random.uniform(0.0, deviation) - (deviation / 2)

    @staticmethod
    def randFluxInt(deviation):
        '''Returns some value +/- some deviation / 2'''
        random.seed()
        return random.randint(0, deviation) - (deviation / 2)

    @staticmethod
    def randomLocation():
        return (random.randint(0, Settings.GRID_SIZE.value - 1), random.randint(0, Settings.GRID_SIZE.value - 1))
    
    @staticmethod
    def randomWithinBounds():
        return random.randint(Settings.GRID_START.value, Settings.GRID_END.value - 1), random.randint(Settings.GRID_START.value, Settings.GRID_END.value - 1)
    
    @staticmethod
    def UnitTest():
        print(f"")

        screen = Settings.SCREEN_SIZE.value
        grid = Settings.GRID_SIZE.value
        cell = Settings.CELL_SIZE.value
        grid_count = Settings.TOTAL_GRID_COUNT.value
        #border_count = Settings.TOTAL_GRID_BORDER_SIZE.value
        spawn_start = Settings.GRID_START.value
        spawn_end = Settings.GRID_END.value
        spawn_area = Settings.TOTAL_SPAWN_AREA.value

        print("Creating ")
        print(f"\t({grid}x{grid}) Grid @ {screen}x{screen} Resolution")
        print(f"\t{grid_count}x ({cell}x{cell}) Cells")

        print(f"\t{spawn_area} - ({spawn_start}x{spawn_start}) to ({spawn_end}x{spawn_end}) Spawnable Area")

        print(f"")
