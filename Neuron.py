import numpy as np

class Neuron:
    def __init__ (self, num_inputs):
        "Initialize the neuron with random weights and bias"
        self.weights = np.random.rand(num_inputs)
        self.bias = np.random.randn()
        self.output = None
        self.num_inputs = num_inputs
        self.layer = None # 0 is input layer
        self.layer_index = None
    
    def activate(self, inputs):
        "Activate the neuron with given inputs"

        inputs = np.array(inputs, dtype=float)

        self.output = self.sigmoid(np.dot(self.weights, inputs) + self.bias)

        return self.output
    
    def sigmoid(self, x):
        "Sigmoid activation function"
        return 1 / (1 + np.exp(-x))
