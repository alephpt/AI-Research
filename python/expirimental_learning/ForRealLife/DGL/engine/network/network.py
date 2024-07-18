from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

from DGL.cosmos import Settings

class DRL:
    def __init__(self, n_inputs, n_outputs, n_hidden_layers, hidden_layer_size):
        self.learning_rate = Settings.LEARNING_RATE.value 
        self.model = self.buildModel(n_inputs, n_outputs, n_hidden_layers, hidden_layer_size)
    
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
    
def testDRL():
    print("\n\t~~ Testing DRL ~~\n")
    n_state_ins = Settings.N_STATE_INPUTS.value
    n_state_outs = Settings.N_STATE_OUTPUTS.value
    n_state_hidden = 1
    d_state_hidden = [7]
    state_drl = DRL(n_state_ins, n_state_outs, n_state_hidden, d_state_hidden)
    
    n_targeting_ins = Settings.N_TARGETING_INPUTS.value
    n_targeting_outs = Settings.N_TARGETING_OUTPUTS.value
    n_targeting_hidden = 2
    d_targeting_hidden = [5, 3]
    targeting_drl = DRL(n_targeting_ins, n_targeting_outs, n_targeting_hidden, d_targeting_hidden)

    print("\n\t~~ DRL Test Complete ~~\n")