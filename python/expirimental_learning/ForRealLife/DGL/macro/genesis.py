import random
from DGL.meso import Agent, Market, Home
from DGL.micro import Settings
from multiprocessing import Pool
from enum import Enum


idx_size = Settings.GRID_SIZE.value ** 2

   ##############
   ## GENESIS ##
   #############

def helper(t, n, dedup):
    subscribed = [t(random.randint(0, idx_size)) for _ in range(n)]

    for subscriber in subscribed:
        # If we have a duplicate, we need to generate a new subscriber
        while subscriber.index() in dedup:
            subscriber = t(random.randint(0, idx_size))

        dedup.add(subscriber.index())
        continue

    return subscribed

class Genesis(Enum):
    # Main Generator
    @staticmethod
    def creation(dedup_set):
        tn = [(Agent, Settings.N_POPULATION.value), (Market, Settings.N_JOBS.value), (Home, Settings.N_HOUSES.value)]

        with Pool() as pool:
            results = pool.starmap(helper, [(t, n, dedup_set) for t, n in tn])
        
        agents, jobs, food = results
        return agents, jobs, food
    
