import numpy as np
from Neuron import Neuron
import matplotlib.pyplot as plt
import struct

# Load MNIST images from .idx3-ubyte file
def load_mnist_images(filename):
    with open(filename, 'rb') as f:
        magic, num_images, rows, cols = struct.unpack('>IIII', f.read(16))
        assert magic == 2051, f"Invalid magic number {magic}"
        image_data = np.frombuffer(f.read(), dtype=np.uint8)
        images = image_data.reshape(num_images, rows * cols)
        images = images.astype(np.float32) / 255.0  # Normalize to [0, 1]
        return images

def load_mnist_labels(filename):
    with open(filename, 'rb') as f:
        magic, num_labels = struct.unpack(">II", f.read(8))
        labels = np.frombuffer(f.read(), dtype=np.uint8)
        return labels
    
def one_hot_encode(labels, num_classes=10):
    return np.eye(num_classes)[labels]

class Network:
    def __init__(self):
        self.weights = [] # list of weight matrices for each layer
        self.biases = [] # list of bias vectors for each layer
        self.nodes = []  # list of numbers of nodes in each layer

    def sigmoid(self, x):
        "Sigmoid activation function"
        return 1 / (1 + np.exp(-x))
    
    def sigmoid_derivative(self, x):
        "Derivative of the sigmoid function"
        return x * (1 - x)

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
        activations = [a]  # Store activations for each layer
        zs = []  # Store weighted inputs for each layer (what we put into the sigmoid)

        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, a) + b  # Weighted input
            zs.append(z)
            a = self.sigmoid(z)
            activations.append(a)
        return activations, zs
    
    def loss(self, output, target):
        "Mean Squared Error loss function"
        return np.mean((output - target) ** 2)  
    
    def backward(self, x, target, learning_rate):
        activations, zs = self.forward(x)
        y_o = activations[-1]
        delta = (y_o - target) * self.sigmoid_derivative(y_o)

        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].T)

        for l in range(2, len(self.nodes)):
            sp = self.sigmoid_derivative(activations[-l])
            delta = np.dot(self.weights[-l + 1].T, delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l - 1].T)

        self.weights = [w - learning_rate * dw for w, dw in zip(self.weights, nabla_w)]
        self.biases = [b - learning_rate * db for b, db in zip(self.biases, nabla_b)]

    def train(self, training_data, epochs, learning_rate):
        for epoch in range(epochs):
            total_loss = 0
            for i, (x, y) in enumerate(training_data, 1):
                a = x.reshape(-1, 1)
                output, _ = self.forward(a)
                loss = self.loss(output[-1], y.reshape(-1,1))
                total_loss += loss
                self.backward(a, y.reshape(-1, 1), learning_rate)
                
                if i % 1000 == 0:
                    print(f"Epoch {epoch+1} Step {i}, Avg Loss: {total_loss / i:.5f}")

                if i >= 10000:
                    break
            learning_rate *= 0.95  # decay per epoch

    def evaluate(self, test_data):
        correct = 0
        for x, y in test_data:
            x = x.reshape(-1, 1)
            activations, _ = self.forward(x)
            pred = np.argmax(activations[-1])
            label = np.argmax(y)
            if pred == label:
                correct += 1
        accuracy = correct / len(test_data)
        print(f"Test Accuracy: {accuracy * 100:.2f}%")
        return accuracy

    



if __name__ == "__main__":
    train_images = load_mnist_images("Data/train-images.idx3-ubyte")
    train_labels = load_mnist_labels("Data/train-labels.idx1-ubyte")
    test_images = load_mnist_images("Data/t10k-images.idx3-ubyte")
    test_labels = load_mnist_labels("Data/t10k-labels.idx1-ubyte")

    train_labels_one_hot = one_hot_encode(train_labels)
    test_labels_one_hot = one_hot_encode(test_labels)

    training_data = list(zip(train_images, train_labels_one_hot))
    test_data = list(zip(test_images, test_labels_one_hot))

    # === Train the model ===
    net = Network()
    net.generate(num_inputs=784, num_hidden=128, num_hidden_layers=2, num_outputs=10)
    net.train(training_data, epochs=1, learning_rate=0.1)
    net.evaluate(test_data)


