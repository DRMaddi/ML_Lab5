import numpy as np
from sklearn.neural_network import MLPClassifier

# Training data for AND gate
# AND gate truth table: inputs and corresponding outputs
input_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
target_output = np.array([0, 0, 0, 1])

# Create an MLPClassifier with one hidden layer
mlp = MLPClassifier(hidden_layer_sizes=(2,), activation='logistic', solver='sgd', learning_rate_init=0.05, max_iter=100)

# Train the classifier
mlp.fit(input_data, target_output)

# Print the trained weights and biases
print("Trained Weights (Coefs):")
print(mlp.coefs_)
print("Trained Biases (Intercepts):")
print(mlp.intercepts_)

# Test the trained classifier
def test_classifier(classifier, data, targets):
    predictions = classifier.predict(data)
    accuracy = (sum(predictions == targets) / len(targets)) * 100
    print("Predictions:", predictions)
    print("Accuracy:", accuracy, "%")

# Test the trained classifier
print("\nTesting the Trained Classifier:")
test_classifier(mlp, input_data, target_output)
