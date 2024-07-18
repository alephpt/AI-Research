from collections import deque
from enum import Enum
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from DGL.society.agency.state import State

from DGL.cosmos import Settings

class DRL:
    def __init__(self, n_inputs, n_outputs, hidden_layer_size):
        self.learning_rate = Settings.LEARNING_RATE.value 
        self.model = self.buildModel(n_inputs, n_outputs, len(hidden_layer_size), hidden_layer_size)
        self.memory = deque(maxlen=Settings.MEMORY_SIZE.value)
    
    def buildModel(self, n_inputs, n_outputs, n_hidden_layers, hidden_layer_size):
        print(f"Building Model (in: {n_inputs} :: out: {n_outputs}) w/ {n_hidden_layers} hidden layers - {hidden_layer_size}")

        model = Sequential()
        model.add(Dense(n_inputs, activation='relu'))
        for i in range(n_hidden_layers):
            model.add(Dense(hidden_layer_size[i], activation='relu'))
        model.add(Dense(n_outputs, activation='linear'))
        model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=self.learning_rate, use_ema=True))
        return model

    def forward(self, model, state):
        return model.predict(state)

    def train(self, model, state, target):
        model.fit(state, target, epochs=1, verbose=0)
        return model

    def save(self):
        self.model.save(Settings.MODEL_PATH.value)


# There are our "eyes" in our agent
class TargetingSystem(Enum):
    Pursue = 0                              # This will be the trigger action that will do Nothing until a target is selected
    Choose_First = 1                        # This should map to the first target in the target pool
    Replace_First = 2
    Choose_Segundo = 3                      # This should map to the second target in the target pool
    Replace_Segundo = 4
    Choose_Tre = 5                          # This should map to the third target in the target pool
    Replace_Tre = 6
    Flush_ALL = 7                           # Flush the target pool
    Nothing = 8

# These state outputs will determine whether we are able to do some thing or not
state_outputs = [State.Alive, State.Hungry, State.Broke, State.Horny, State.Tired]

def testDRL():
    # State Inputs
    energy = 25         # Comes from 'eating' or 'resting'
    money = 25          # Comes from 'working' or 'trading' - and decays when buying, having kids, and we could create an economy
    happiness = 0       # Comes from rewarding actions like 'mating', 'eating', 'trading' or 'sleeping' - and decays over time
    fatigue = 0         # Comes from doing things, and increases over time, acting as a "decay" on the other stats
    state = 0           # State Conditions are based on 'world' "acheivements" through the AI acheiving certain goals or conditions
    choice = 0          # State Condition from Enum
    tsa = 0             # Target Selected Action from Enum
    targeting_pool = [(0, (0, 0), 0, 0), (0, (0, 0), 0, 0), (0, (0, 0), 0, 0)]      # Target Pool - Tuple of (Type, Direction Vector, Magnitude, Potential)
   
    print("\n\t~~ Testing DRL ~~\n")
    state_inputs = Settings.N_STATE_INPUTS.value
    targeting_outputs = Settings.N_TARGETING_OUTPUTS.value
    d_state_hidden = [7,5,3]
    DRL_T = DRL(state_inputs, targeting_outputs, d_state_hidden)

    # After we query the DRL, we should have a choice, and a target_system_action

    chosen_state = state_outputs[choice]
    chosen_target_action = TargetingSystem(tsa)

    # TODO: Implement "training" and "choice" step to determine which actions we want to take

    print("\n\t~~ DRL Test Complete ~~\n")