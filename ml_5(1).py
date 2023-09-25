import numpy as np

class Perceptron:
    def __init__(self, weights, bias, learning_rate):
        self.weights = weights
        self.bias = bias
        self.learning_rate = learning_rate

    def forward_pass(self, inputs):
        weighted_sum = np.dot(inputs, self.weights) + self.bias
        output = self.step_activation(weighted_sum)
        return output

    def step_activation(self, x):
        if x >= 0:
            return 1
        else:
            return 0

    def train(self, inputs, targets):
        outputs = self.forward_pass(inputs)
        error = targets - outputs
        weight_update = self.learning_rate * error * inputs
        self.weights += weight_update
        self.bias += self.learning_rate * error

# Initialize the perceptron with the given weights and bias
perceptron = Perceptron([0.2, -0.75], 10, 0.05)

# Training data
training_inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
training_targets = np.array([0, 0, 0, 1])

# Train the perceptron
for epoch in range(100):
    perceptron.train(training_inputs, training_targets)

# Test the perceptron
test_inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
test_targets = np.array([0, 0, 0, 1])

test_outputs = perceptron.forward_pass(test_inputs)

# Print the test results
print("Test results:")
for i in range(len(test_inputs)):
    print(f"Input: {test_inputs[i]}, Target: {test_targets[i]}, Output: {test_outputs[i]}")


def calculate_sse(outputs, targets):
    error = outputs - targets
    sse = np.sum(error**2)
    return sse


import matplotlib.pyplot as plt

# Calculate the SSE for each epoch
sse_per_epoch = []
for epoch in range(100):
    perceptron.train(training_inputs, training_targets)
    outputs = perceptron.forward_pass(training_inputs)
    sse = calculate_sse(outputs, training_targets)
    sse_per_epoch.append(sse)

# Plot the epochs against the SSE
plt.plot(range(100), sse_per_epoch)
plt.xlabel("Epochs")
plt.ylabel("Sum-Squared Error")
plt.title("SSE vs. Epochs for AND Gate Perceptron")
plt.show()
