

##              State->Action Space
## "[QTable] is a table that maps states to actions."
class QTable:
    def __init__(self):
        self.q_table = {}
        self.q_table['state'] = 0

##                  State->Mate Space
## "[MateTable] is a table that maps states to 'choosing a mate'."
class MateTable:
    def __init__(self):
        self.mate_table = {}
        self.mate_table['state'] = 0