import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import accuracy_score

# Step 1: Load the Iris dataset
iris = datasets.load_iris()
X = iris.data
y = iris.target.reshape(-1, 1)

# Step 2: Preprocess the data
# One-hot encode the target variable
encoder = OneHotEncoder(sparse=False)
y_onehot = encoder.fit_transform(y)

# Standardize the feature variables
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Step 3: Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_onehot, test_size=0.2, random_state=42)

# Step 4: Define the architecture of the MLP
input_size = X_train.shape[1]
hidden_size = 8
output_size = y_onehot.shape[1]

# Initialize weights and biases
np.random.seed(42)
weights_input_hidden = np.random.rand(input_size, hidden_size)
biases_input_hidden = np.zeros((1, hidden_size))
weights_hidden_output = np.random.rand(hidden_size, output_size)
biases_hidden_output = np.zeros((1, output_size))

# Step 5: Define activation function (e.g., sigmoid) and its derivative
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# Step 6: Set hyperparameters
learning_rate = 0.01
epochs = 10000

# Step 7: Train the MLP using backpropagation
for epoch in range(epochs):
    # Forward pass
    hidden_input = np.dot(X_train, weights_input_hidden) + biases_input_hidden
    hidden_output = sigmoid(hidden_input)

    output_input = np.dot(hidden_output, weights_hidden_output) + biases_hidden_output
    predicted_output = sigmoid(output_input)

    # Backward pass
    error = y_train - predicted_output

    # Output layer
    output_delta = error * sigmoid_derivative(predicted_output)
    weights_hidden_output += np.dot(hidden_output.T, output_delta) * learning_rate
    biases_hidden_output += np.sum(output_delta, axis=0, keepdims=True) * learning_rate

    # Hidden layer
    hidden_error = np.dot(output_delta, weights_hidden_output.T)
    hidden_delta = hidden_error * sigmoid_derivative(hidden_output)
    weights_input_hidden += np.dot(X_train.T, hidden_delta) * learning_rate
    biases_input_hidden += np.sum(hidden_delta, axis=0, keepdims=True) * learning_rate

# Step 8: Make predictions on the test set
hidden_input_test = np.dot(X_test, weights_input_hidden) + biases_input_hidden
hidden_output_test = sigmoid(hidden_input_test)
output_input_test = np.dot(hidden_output_test, weights_hidden_output) + biases_hidden_output
predicted_output_test = sigmoid(output_input_test)

# Step 9: Convert one-hot encoded predictions to class labels
predicted_labels = np.argmax(predicted_output_test, axis=1)

# Step 10: Evaluate the model
accuracy = accuracy_score(np.argmax(y_test, axis=1), predicted_labels)
print(f"Accuracy on the test set: {accuracy}")