import numpy as np
import matplotlib.pyplot as plt

def step_activation(x):
    return np.where(x > 0, 1, 0)

# AND gate data
input_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
target_output = np.array([[1, 0], [1, 0], [1, 0], [0, 1]])

# Initialization
v11 = np.random.uniform(size=(2, 1))
v12 = np.random.uniform(size=(2, 1))
v21 = np.random.uniform(size=(2, 1))
v22 = np.random.uniform(size=(2, 1))

w11 = np.random.uniform(size=(1, 1))
w12 = np.random.uniform(size=(1, 1))
w21 = np.random.uniform(size=(1, 1))
w22 = np.random.uniform(size=(1, 1))

learning_rate = 0.05
errors = []

for iteration in range(1000):
    total_error = np.zeros(2)
    
    for xi, target in zip(input_data, target_output):
        # Forward Propagation
        av11 = step_activation(np.dot(xi, v11))
        av12 = step_activation(np.dot(xi, v12))
        av21 = step_activation(np.dot(xi, v21))
        av22 = step_activation(np.dot(xi, v22))
        
        h1 = (av11 + av21).reshape(-1, 1)  # Reshape
        h2 = (av12 + av22).reshape(-1, 1)  # Reshape
        
        o1 = step_activation(np.dot(h1.T, w11) + np.dot(h2.T, w21))
        o2 = step_activation(np.dot(h1.T, w12) + np.dot(h2.T, w22))
        
        output = np.array([o1, o2]).reshape(2,)
        
        # Compute Error
        error = target - output
        
        total_error += error**2
        
        # Update weights
        w11 += learning_rate * h1 * error[0]
        w12 += learning_rate * h1 * error[1]
        w21 += learning_rate * h2 * error[0]
        w22 += learning_rate * h2 * error[1]
        
        delta = learning_rate * xi.reshape(2, 1)
        v11 += delta * (w11 * error[0] + w12 * error[1])
        v12 += delta * (w11 * error[0] + w12 * error[1])
        v21 += delta * (w21 * error[0] + w22 * error[1])
        v22 += delta * (w21 * error[0] + w22 * error[1])

    errors.append(np.mean(total_error))
    
    if np.all(np.array(total_error) <= 0.002):
        print(f"Converged at iteration {iteration}")
        break

# Plotting
plt.plot(errors)
plt.xlabel('Iterations')
plt.ylabel('Mean Squared Error')
plt.title('Error Convergence for AND Gate with 2 Output Nodes using Step Activation')
plt.show()
