import numpy as np
from Neuron import Neuron
import matplotlib.pyplot as plt

class Network:
    def __init__(self):
        self.weights = [] # list of weight matrices for each layer
        self.biases = [] # list of bias vectors for each layer
        self.nodes = []  # list of numbers of nodes in each layer

    def sigmoid(self, x):
        "Sigmoid activation function"
        return 1 / (1 + np.exp(-x))
        

    def generate(self, num_inputs, num_hidden, num_hidden_layers, num_outputs):
        self.weights = []
        self.biases = []
        self.nodes = [num_inputs] + [num_hidden] * num_hidden_layers + [num_outputs]

        for layer in range(1, len(self.nodes)):  # skip input layer
            inputs = self.nodes[layer - 1]
            outputs = self.nodes[layer]
            
            # Initialize weights and biases for the layer
            weight_matrix = np.random.randn(outputs, inputs) * np.sqrt(1. / inputs)
            bias_vector = np.random.randn(outputs, 1)
            self.weights.append(weight_matrix)
            self.biases.append(bias_vector)

    
    def forward(self, a):
        "Feedforward through the network"
        for b, w in zip(self.biases, self.weights):
            a = self.sigmoid(np.dot(w, a) + b)
        return a


if __name__ == "__main__":
    network = Network()
    network.generate(num_inputs=784, num_hidden=16, num_hidden_layers=2, num_outputs=10)
    print("Network structure: ")
    # Example input
    example_input = np.random.rand(784, 1)  # Example input for MNIST dataset
    output = network.forward(example_input)
    print("Output shape:", output.shape)
    print("Output:", output)
