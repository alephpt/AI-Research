from .agency import State
from DGL.cosmos import LogLevel, Log

# TODO: Implement Utility functions that allow for the creation of a new generation of units
# TODO: Integrate the Genome into the Genetic Evolution of the Society
# TODO: Create Batch Functions and Parallelize the Genetic Evolution of the Society
class Genome:
    def __init__(self):
        self.reset()
        self.population_count = 0 # TODO: Graph this over time
        self.population = set()

    # When we Shuffle we want 
    # 15-30% of the Oldest Generations
    # 20-35% Mix of Happiest, Healthiest, Wealthiest and Most Rewarded
    # 50% to be Split
    #    20% is Crossover of the Oldest and Top Performers
    #    30% is Crossover of random Slices of Worst Performers

    def alive():
        pass

    # We are going to start our next generation with the following:
    def reset(self):
        # When we repopulate, we need to graph our outputs of each generation, on an INDIVIDUAL basis - to check for 'overlap'
        self.min_age = 0
        self.avg_age = 0
        self.max_age = 0                # ~1/10th

        self.min_happiness = 0
        self.avg_happiness = 0          
        self.max_happiness = 0          # ~1/10th

        self.min_health = 0
        self.avg_health = 0
        self.max_health = 0             # ~1/10th

        self.min_wealth = 0
        self.avg_wealth = 0
        self.max_wealth = 0             # ~1/10th

        self.min_reward = 0
        self.avg_reward = 0
        self.max_reward = 0             # ~1/10th

        self.n_alive = 0

        self.total_age = 0
        self.total_happiness = 0
        self.total_health = 0
        self.total_wealth = 0
        self.total_reward = 0

        self.n_generations = 0          # ~1/3 - ~1/2

    def updateStatistics(self):
        # If there is none alive, then we don't update
        if 0 == len(self.alive()):
            return
        
        # Otherwise, reset to be clean
        self.reset()

        
        for unit in self.units:
            if unit.state == State.Dead:
                Log.Error(LogLevel.INFO, "Dead Unit Found. - signed, Anamolous.")
                continue
            
            self.n_alive += 1
            self.n_generations = max(self.n_generations, unit.generation)
            
            self.total_age += unit.age
            #self.total_happiness += unit.happiness
            self.total_health += unit.energy
            self.total_wealth += unit.wealth
            self.total_reward += unit.reward

            self.min_age = min(self.min_age, unit.age)
            self.avg_age += unit.age
            self.max_age = max(self.max_age, unit.age)

            #self.min_happiness = min(self.min_happiness, unit.happiness)
            #self.max_happiness = max(self.max_happiness, unit.happiness)

            self.min_health = min(self.min_health, unit.energy)
            self.max_health = max(self.max_health, unit.energy)

            self.min_wealth = min(self.min_wealth, unit.wealth)
            self.max_wealth = max(self.max_wealth, unit.wealth)

            self.min_reward = min(self.min_reward, unit.reward)
            self.max_reward = max(self.max_reward, unit.reward)

        if 0 == self.n_alive:
            Log(LogLevel.INFO, "No Units are Alive")
            return

        self.avg_age = self.total_age // self.n_alive
        self.avg_happiness = 0 #self.total_happiness // self.n_alive
        self.avg_health = self.total_health // self.n_alive
        self.avg_wealth = self.total_wealth // self.n_alive
        self.avg_reward = self.total_reward // self.n_alive