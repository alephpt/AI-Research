import numpy

## THIS ALLOWS FOR AGENTS TO LEARN HOW TO CHOOSE THEIR ACTIONS
##              State->Action Space
## "[QTable] is a table that maps states to actions."
#
#  Keys are possible states, and outputs are optimal weights for determining
#           the best action to take in that state.
#

# State Space Includes
#   - Current Magnitude



class DecisionNetwork:
    def __init__(self):
        # TODO: IT WOULD BE INTERESTING TO SCALE ACROSS AGE
        # These are our input conditions
        self.state_space = {
            "choice": None,                 # Nothing, Working, Eating, Sleeping, Mating, 
            "target_type": None,            # This tells us what we are looking for
            "magnitude": 0,                 ## We need to find the best magnitude regardless
            "succeed": False,               # This tells us the state of our current action - Big Negative if we fail
            #"age": 0,                      ## This is a Gradient Decaying Multiplier
            #"children": 0,                 ## This should be an Exponential Multiplier, but ONLY through the Genetic Algorithm



    # -- In an economy we could have some characteristics of -------------------------------------------------------#
    #       lower integrity having greater influence of higher returns from chances of mating and better trading statistics, 
    #           and in return they work less often and mate more often (increasing expenses long term)              #
    #         low integrity has a negativity happiness impact on self and others, and less attractive               #
    #                                                                                                               #
    #       high compassion means taking more losses in the short term, but having a greater happiness impact on self and others
    #           and in return they work more often (more wealth) and partner with a single mate (less expenses, potential for generation wealth)
    #      high compassion has a positivity happiness impact on self and others                                     #
    #---------------------------------------------------------------------------------------------------------------#
            "wealth": 0,        ## It Takes money to buy food, and work to make money
            "energy": 0,        ## It Takes food to have energy, and takes energh to work
            # "fatigue": 0,          # Increases with Work, and Mating, decreases with sleep, and compassion
            # "happiness": 0,        ## Happiness is a reward multiplier
            # "integrity": 0,        # decreases fatigue, and increases attraction
            # "compassion": 0,       # increases happiness, and attraction - # we add the age to this each time we make a choice
            # "attraction": 0,       # Increases when we make money, and when we are compassionate
        }

        # these are our 'target' conditions
        self.target_space = {
            "food": 0,
            "work": 0,
            "mate": 0,
            "sleep": 0,
            "none": 0
        }

        # These are our output parameters that dictate the 'missing' Status parameters.. 
        # We want it to learn to choose once, and then 'stick' or 'none'
        self.action_space = {
            "find_food": 0,
            "find_work": 0,
            "find_mate": 0,
            "find_sleep": 0,
            "none": 0
        }

        self.n_states = len(self.state_space)
        self.n_choices = len(self.n_choices)
        self.n_actions = len(self.action_space)


    def train(self, x, y):
        self.model.fit(x, y, epochs=1, verbose=0)

    def predict(self, x):
        return self.model.predict(x)

## 

##  THIS ALLOWS FOR AGENTS TO DETERMINE PREFERENCES OVER GENERATIONS
##                  State->Mate Space
## "[MateTable] is a table that maps states to 'choosing a mate'."
#
class MateNetwork:#
    def __init__(self):
        self.mate_table = {}

        self.state_space = {
            "self": 0,              # This is "Who I am
            "partner": 0,           # This is "Who my partner is"
            "agenda": 0,            # This is "What I want"
            "target_agenda": 0,     # This is "What my partner wants"
            "integrity": 0,         # This would steer behaviours regarding 'honesty' or 'deception'
            "compassion": 0,        # This would steer behaviours regarding to commcelly or 'selfishness'
            "restlessness": 0       # This would be a multiplier for the 'impulsivity' , or as an aggregate of the other two
        }

    ## ... ?
    

## Lets Not Even Talk about an Economy..