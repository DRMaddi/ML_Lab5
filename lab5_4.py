import numpy as np
import matplotlib.pyplot as plt

# Define the training data for the XOR gate
input_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
target_output = np.array([0, 1, 1, 0])

# Initialize weights and bias
weights = np.array([10, 0.2, -0.75])  # W0, W1, W2
learning_rate = 0.05
epochs = 1000

# Initialize lists to store errors and epoch numbers for each activation function
errors_bipolar_step = []
epoch_numbers_bipolar_step = []
errors_sigmoid = []
epoch_numbers_sigmoid = []
errors_relu = []
epoch_numbers_relu = []

# Define the Bi-Polar Step activation function
def bipolar_step_activation(x):
    return 1 if x >= 0 else -1

# Define the Sigmoid activation function
def sigmoid_activation(x):
    return 1 / (1 + np.exp(-x))

# Define the ReLU activation function
def relu_activation(x):
    return max(0, x)

# Training the perceptron with Bi-Polar Step activation
for epoch in range(epochs):
    error = 0
    for i in range(len(input_data)):
        # Calculate the weighted sum
        weighted_sum = np.dot(input_data[i], weights[1:]) + weights[0]

        # Apply the Bi-Polar Step activation function
        prediction = bipolar_step_activation(weighted_sum)

        # Calculate the error
        error = error + (target_output[i] - prediction) ** 2

        # Update weights
        weights[1:] = weights[1:] + learning_rate * (target_output[i] - prediction) * input_data[i]
        weights[0] = weights[0] + learning_rate * (target_output[i] - prediction)

    # Append error and epoch number for plotting
    errors_bipolar_step.append(error)
    epoch_numbers_bipolar_step.append(epoch)

    # Check for convergence
    if error <= 0.002:
        print(f"Converged with Bi-Polar Step after {epoch + 1} epochs")
        break

# Reinitialize weights for Sigmoid and ReLU activation
weights = np.array([10, 0.2, -0.75])  # Reset weights

# Training the perceptron with Sigmoid activation
for epoch in range(epochs):
    error = 0
    for i in range(len(input_data)):
        # Calculate the weighted sum
        weighted_sum = np.dot(input_data[i], weights[1:]) + weights[0]

        # Apply the Sigmoid activation function
        prediction = sigmoid_activation(weighted_sum)

        # Calculate the error
        error = error + (target_output[i] - prediction) ** 2

        # Update weights
        delta = learning_rate * (target_output[i] - prediction) * prediction * (1 - prediction)
        weights[1:] = weights[1:] + delta * input_data[i]
        weights[0] = weights[0] + delta

    # Append error and epoch number for plotting
    errors_sigmoid.append(error)
    epoch_numbers_sigmoid.append(epoch)

    # Check for convergence
    if error <= 0.002:
        print(f"Converged with Sigmoid after {epoch + 1} epochs")
        break

# Reinitialize weights for ReLU activation
weights = np.array([10, 0.2, -0.75])  # Reset weights

# Training the perceptron with ReLU activation
for epoch in range(epochs):
    error = 0
    for i in range(len(input_data)):
        # Calculate the weighted sum
        weighted_sum = np.dot(input_data[i], weights[1:]) + weights[0]

        # Apply the ReLU activation function
        prediction = relu_activation(weighted_sum)

        # Calculate the error
        error = error + (target_output[i] - prediction) ** 2

        # Update weights
        delta = learning_rate * (target_output[i] - prediction)
        weights[1:] = weights[1:] + delta * input_data[i]
        weights[0] = weights[0] + delta

    # Append error and epoch number for plotting
    errors_relu.append(error)
    epoch_numbers_relu.append(epoch)

    # Check for convergence
    if error <= 0.002:
        print(f"Converged with ReLU after {epoch + 1} epochs")
        break

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
