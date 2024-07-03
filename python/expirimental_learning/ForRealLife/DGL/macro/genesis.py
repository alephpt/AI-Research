import random
from DGL.meso import Agent, Work, Eatery
from DGL.micro import Unit, Placement, Settings
from multiprocessing import Pool
from enum import Enum


   ##############
   ## GENESIS ##
   #############

# Macros for Society Genesis
def createAgents():
    return [Agent() for _ in range(Settings.N_POPULATION.value)]

def createJobs():
    return [Work() for _ in range(Settings.N_JOBS.value)]

def createFood():
    return [Eatery() for _ in range(Settings.N_FOOD.value)]

def helper(f):
    return f()

class Genesis(Enum):
    Agency = createAgents
    Workforce = createJobs
    Nutrition = createFood


    # Main Generator
    @staticmethod
    def creation():
        params = [Genesis.Agency, Genesis.Workforce, Genesis.Nutrition]

        with Pool() as pool:
            results = pool.starmap(helper, [(f,) for f in params])
        
        agents, jobs, food = results
        return agents, jobs, food
    
