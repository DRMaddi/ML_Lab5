import numpy as np
import matplotlib.pyplot as plt

# Define the training data for the AND gate
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
Y = np.array([0, 0, 0, 1])

learning_rate = 0.05
epochs = 1000

# Define activation functions
def bipolar_step_activation(x):
    return 1 if x >= 0 else -1

def sigmoid_activation(x):
    return 1 / (1 + np.exp(-x))

def relu_activation(x):
    return max(0, x)

# Training function
def train_perceptron(X, Y, activation_function):
    W = np.array([10, 0.2])  # Weights for inputs (W1, W2)
    bias = -0.75
    errors = []
    epoch_numbers = []

    for epoch in range(epochs):
        error = 0
        for i in range(len(X)):
            # Calculate the weighted sum
            weighted_sum = np.dot(X[i], W) + bias

            # Apply the activation function
            prediction = activation_function(weighted_sum)

            # Calculate the error
            error += (Y[i] - prediction) ** 2

            # Update weights
            delta = learning_rate * (Y[i] - prediction)
            W += delta * X[i]
            bias += delta

        # Append error and epoch number for plotting
        errors.append(error)
        epoch_numbers.append(epoch)

        # Check for convergence
        if error <= 0.002:
            print(f"Converged with {activation_function.__name__} after {epoch + 1} epochs")
            break

    return errors, epoch_numbers, W, bias

# Train with different activation functions and get separate weight and bias values
errors_bipolar_step, epoch_numbers_bipolar_step, W_bipolar_step, bias_bipolar_step = train_perceptron(X, Y, bipolar_step_activation)
errors_sigmoid, epoch_numbers_sigmoid, W_sigmoid, bias_sigmoid = train_perceptron(X, Y, sigmoid_activation)
errors_relu, epoch_numbers_relu, W_relu, bias_relu = train_perceptron(X, Y, relu_activation)

# Plotting error vs. epoch for all activation functions
plt.figure(figsize=(12, 6))
plt.subplot(131)
plt.plot(epoch_numbers_bipolar_step, errors_bipolar_step)
plt.xlabel('Epochs')
plt.ylabel('Error')
plt.title('Error vs. Epochs (Bi-Polar Step)')

plt.subplot(132)
plt.plot(epoch_numbers_sigmoid, errors_sigmoid)
plt.xlabel('Epochs')
plt.ylabel('Error')
plt.title('Error vs. Epochs (Sigmoid)')

plt.subplot(133)
plt.plot(epoch_numbers_relu, errors_relu)
plt.xlabel('Epochs')
plt.ylabel('Error')
plt.title('Error vs. Epochs (ReLU)')

plt.tight_layout()
plt.show()

# Print the final weights and biases for each activation function
print(f"Final weights and bias (Bi-Polar Step): W1 = {W_bipolar_step[0]}, W2 = {W_bipolar_step[1]}, Bias = {bias_bipolar_step}")
print(f"Final weights and bias (Sigmoid): W1 = {W_sigmoid[0]}, W2 = {W_sigmoid[1]}, Bias = {bias_sigmoid}")
print(f"Final weights and bias (ReLU): W1 = {W_relu[0]}, W2 = {W_relu[1]}, Bias = {bias_relu}")
