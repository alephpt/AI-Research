

## THIS ALLOWS FOR AGENTS TO LEARN HOW TO CHOOSE THEIR ACTIONS
##              State->Action Space
## "[QTable] is a table that maps states to actions."
#
#  Keys are possible states, and outputs are optimal weights for determining
#           the best action to take in that state.
#

# State Space Includes
#   - Current Magnitude


class QTable:
    def __init__(self):
        self.q_table = {}
        self.q_table['state'] = 0



##  THIS ALLOWS FOR AGENTS TO DETERMINE PREFERENCES OVER GENERATIONS
##                  State->Mate Space
## "[MateTable] is a table that maps states to 'choosing a mate'."
#
class MateTable:
    def __init__(self):
        self.mate_table = {}
        self.mate_table['state'] = 0