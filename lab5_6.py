import numpy as np

# Sigmoid activation and its derivative
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# AND gate data
input_data = np.array([[0,0],[0,1],[1,0],[1,1]])
target_output = np.array([[0],[0],[0],[1]])

# Initialization of weights
v11 = np.random.uniform(size=(2,1))
v12 = np.random.uniform(size=(2,1))
v21 = np.random.uniform(size=(2,1))
v22 = np.random.uniform(size=(2,1))

w1 = np.random.uniform(size=(2,1))
w2 = np.random.uniform(size=(2,1))

learning_rate = 0.05
for iteration in range(1000):
    # Forward propagation
    a_v11 = sigmoid(np.dot(input_data, v11))
    a_v12 = sigmoid(np.dot(input_data, v12))
    a_v21 = sigmoid(np.dot(input_data, v21))
    a_v22 = sigmoid(np.dot(input_data, v22))
    
    h1_input = np.hstack((a_v11, a_v22))
    h2_input = np.hstack((a_v12, a_v21))
    
    h1 = sigmoid(np.dot(h1_input, w1))
    h2 = sigmoid(np.dot(h2_input, w2))
    
    output = sigmoid(h1 + h2)
    
    # Compute error
    error = target_output - output
    if np.mean(np.abs(error)) <= 0.002:
        break
    
    # Backpropagation
    d_output = error * sigmoid_derivative(output)
    d_h1 = (d_output * w1[0, 0] + d_output * w2[0, 0]) * sigmoid_derivative(h1)
    d_h2 = (d_output * w1[1, 0] + d_output * w2[1, 0]) * sigmoid_derivative(h2)
    
    # Update weights
    w1 += learning_rate * h1_input.T.dot(d_output)
    w2 += learning_rate * h2_input.T.dot(d_output)
    v11 += learning_rate * input_data.T.dot(d_h1)
    v12 += learning_rate * input_data.T.dot(d_h2)
    v21 += learning_rate * input_data.T.dot(d_h2)
    v22 += learning_rate * input_data.T.dot(d_h1)

print(output)
