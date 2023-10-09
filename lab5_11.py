import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

# Load data from the Excel file
df = pd.read_excel("embeddingsdata.xlsx")

# Separate features (X) and target labels (y)
features = df.drop('Label', axis=1)
labels = df['Label']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Create an MLPClassifier with hidden layers
mlp_classifier = MLPClassifier(hidden_layer_sizes=(100, 50), activation='relu', solver='adam', max_iter=1000)

# Train the classifier on the training data
mlp_classifier.fit(X_train, y_train)

# Make predictions on the test data
y_pred = mlp_classifier.predict(X_test)

# Calculate and print the accuracy on the test data
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy on test data: {accuracy}")
