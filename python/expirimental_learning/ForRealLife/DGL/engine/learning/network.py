import numpy as np
from enum import Enum

from DGL.cosmos import Log, LogLevel
from .targeting.targetingsystem import TargetingSystem
from DGL.society.agency.state import State
from DGL.cosmos.closet import relu, sigmoid_derivative

from DGL.cosmos import Settings

class Layer:
    def __init__(self, n_inputs, n_neurons, activation, factor=1):
        self.weights = np.random.random((n_inputs, n_neurons)) * factor
        self.biases = np.zeros((1, n_neurons))
        self.activation = activation

### TODO: When we create a new neural network as a result of 'mating' we want to merge 50% randomness with 25% of each parent.
class DRL:
    def __init__(self, n_inputs, n_outputs, hidden_layer_size):
        self.learning_rate = Settings.LEARNING_RATE.value 
        self.model = self.buildModel(n_inputs, n_outputs, len(hidden_layer_size), hidden_layer_size)
    
    def log(self):
        print(f"Learning Rate: {self.learning_rate}")
        print(f"Model: {self.model}")

    def buildModel(self, n_inputs, n_outputs, n_hidden_layers, hidden_layer_size):
        print(f"Building Model (in: {n_inputs} :: out: {n_outputs}) w/ {n_hidden_layers} hidden layers - {hidden_layer_size}")

        model = []
        input_count = n_inputs

        for i in range(n_hidden_layers):
            model.append(Layer(input_count, hidden_layer_size[i]), relu, factor=0.01)
            input_count = hidden_layer_size[i]
            
        return model

    def forward(self, model, state):
        return model.predict(state)

    def train(self, model, state, target):
        model.fit(state, target, epochs=1, verbose=0)
        return model

    def save(self):
        self.model.save(Settings.MODEL_PATH.value)


# These state outputs will determine whether we are able to do some thing or not
state_outputs = [State.Alive, State.Hungry, State.Broke, State.Horny, State.Tired]

def testDRL():
    target_system = TargetingSystem()

    # State Inputs
    integrity_level = 0 
    compassion_level = 0
    energy = 25         # Comes from 'eating' or 'resting'
    money = 25          # Comes from 'working' or 'trading' - and decays when buying, having kids, and we could create an economy
    happiness = 0       # Comes from rewarding actions like 'mating', 'eating', 'trading' or 'sleeping' - and decays over time
    fatigue = 0         # Comes from doing things, and increases over time, acting as a "decay" on the other stats
    state = 0           # State Conditions are based on 'world' "acheivements" through the AI acheiving certain goals or conditions
    choice = 0          # State Condition from Enum             0-4
    tsa = 0             # Target Selected Action from Enum      0-8
    state_dimension = 0 # State Dimension from Enum             0-8
    targeting_pool = target_system.poolValues()
    inputs = [None, energy, money, happiness, fatigue, state, choice, tsa, *targeting_pool]

    # Comes from TargetingSystem
    # Target Pool - Tuple of (Type, Direction Vector, Magnitude, Potential) 

    print("\n\t~~ Testing DRL ~~\n")
    state_inputs = len(inputs)
    targeting_outputs = Settings.N_TARGETING_OUTPUTS.value
    d_state_hidden = [7,5,3]
    DRL_T = DRL(state_inputs, targeting_outputs, d_state_hidden)

    # After we query the DRL, we should have a choice, and a target_system_action
    state_dimension = choice % len(state_outputs)
    tsa = choice // len(state_outputs)

    chosen_state = state_outputs[state_dimension]
    chosen_target_action = target_system.setAction(tsa)
    Log(LogLevel.INFO, "DRL", f"Chose State.{chosen_state}, TargetAction.{chosen_target_action}")
    target_system.doAction()
    # TODO: Implement a Target Action Step, and all potential actions
    # TODO: Implement all stateful occurrences, and the reward/consequence of potential state


    # TODO: Implement "training" and "choice" step to determine which actions we want to take

    print("\n\t~~ DRL Test Complete ~~\n")