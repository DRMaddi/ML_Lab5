import numpy as np

# Define the data
transactions = [
    [20, 6, 2, 386, 1],
    [16, 3, 6, 289, 1],
    [27, 6, 2, 393, 1],
    [19, 1, 2, 110, 0],
    [24, 4, 2, 280, 1],
    [22, 1, 5, 167, 0],
    [15, 4, 2, 271, 1],
    [18, 4, 2, 274, 1],
    [21, 1, 4, 148, 0],
    [16, 2, 4, 198, 0],
]

# Normalize the features
transactions = np.array(transactions)
mean = transactions[:, :-1].mean(axis=0)
std = transactions[:, :-1].std(axis=0)
transactions[:, :-1] = (transactions[:, :-1] - mean) / std

# Initialize weights and bias
np.random.seed(0)
weights = np.random.rand(3)
bias = np.random.rand()

# Define the sigmoid activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Set the learning rate
learning_rate = 0.01

# Training the perceptron
epochs = 1000
for epoch in range(epochs):
    total_error = 0
    for row in transactions:
        candies, mangoes, milk_packets, payment, label = row
        z = weights[0] * candies + weights[1] * mangoes + weights[2] * milk_packets + bias
        y_pred = sigmoid(z)
        error = label - y_pred
        weights[0] += learning_rate * error * y_pred * (1 - y_pred) * candies
        weights[1] += learning_rate * error * y_pred * (1 - y_pred) * mangoes
        weights[2] += learning_rate * error * y_pred * (1 - y_pred) * milk_packets
        bias += learning_rate * error * y_pred * (1 - y_pred)
        total_error += error
    if epoch % 100 == 0:
        print(f"Epoch {epoch}: Total Error = {total_error}")

# Classify new transactions
def classify_transaction(candies, mangoes, milk_packets):
    normalized_data = (np.array([candies, mangoes, milk_packets]) - mean[:-1]) / std[:-1]  # Exclude Payment from normalization
    z = np.dot(weights, normalized_data) + bias
    y_pred = sigmoid(z)
    if y_pred >= 0.5:
        return "High Value Tx"
    else:
        return "Low Value Tx"

# Example usage:
new_transaction = (20, 6, 2)
classification = classify_transaction(*new_transaction)
print(f"New Transaction: {new_transaction} => Classification: {classification}")
