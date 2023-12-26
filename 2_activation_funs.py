import numpy as np
import matplotlib.pyplot as plt

# Activation functions
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def tanh(x):
    return np.tanh(x)

def relu(x):
    return np.maximum(0, x)

def sigmoid_derivative(x):
    return x * (1 - x)

def tanh_derivative(x):
    return 1 - np.tanh(x)**2

def relu_derivative(x):
    return np.where(x > 0, 1, 0)

# Initialize weights
def initialize_weights(input_size, hidden_size, output_size):
    np.random.seed(42)
    weights_input_hidden = np.random.randn(input_size, hidden_size)
    weights_hidden_output = np.random.randn(hidden_size, output_size)
    return weights_input_hidden, weights_hidden_output

# Forward propagation
def forward_propagation(inputs, weights_input_hidden, weights_hidden_output, activation_fn):
    hidden_layer_input = np.dot(inputs, weights_input_hidden)
    hidden_layer_output = activation_fn(hidden_layer_input)

    output_layer_input = np.dot(hidden_layer_output, weights_hidden_output)
    output_layer_output = sigmoid(output_layer_input)  # Using sigmoid for the output layer

    return hidden_layer_output, output_layer_output

# Backward propagation
def backward_propagation(inputs, targets, hidden_layer_output, output_layer_output,
                         weights_input_hidden, weights_hidden_output, learning_rate, activation_derivative):
    output_error = targets - output_layer_output
    output_delta = output_error * sigmoid_derivative(output_layer_output)

    hidden_layer_error = output_delta.dot(weights_hidden_output.T)
    hidden_layer_delta = hidden_layer_error * activation_derivative(hidden_layer_output)

    weights_hidden_output += hidden_layer_output.T.dot(output_delta) * learning_rate
    weights_input_hidden += inputs.T.dot(hidden_layer_delta) * learning_rate

# Train neural network
def train_neural_network(inputs, targets, hidden_size, output_size, learning_rate, epochs, activation_fn, activation_derivative):
    input_size = inputs.shape[1]
    weights_input_hidden, weights_hidden_output = initialize_weights(input_size, hidden_size, output_size)

    loss_history = []

    for epoch in range(epochs):
        # Forward Propagation
        hidden_layer_output, output_layer_output = forward_propagation(inputs, weights_input_hidden, weights_hidden_output, activation_fn)

        # Backward Propagation
        backward_propagation(inputs, targets, hidden_layer_output, output_layer_output,
                             weights_input_hidden, weights_hidden_output, learning_rate, activation_derivative)

        # Calculate and store loss
        loss = calculate_loss(targets, output_layer_output)
        loss_history.append(loss)

        if epoch % 1000 == 0:
            print(f"Epoch {epoch}, Loss: {loss}")

    return weights_input_hidden, weights_hidden_output, loss_history

# Calculate loss
def calculate_loss(targets, predictions):
    return np.mean((targets - predictions) ** 2)

# Example usage
inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
targets = np.array([[0], [1], [1], [0]])

hidden_size = 4
output_size = 1
learning_rate = 0.01
epochs = 10000

# Train with sigmoid activation function
sigmoid_weights_input_hidden, sigmoid_weights_hidden_output, sigmoid_loss_history = train_neural_network(
    inputs, targets, hidden_size, output_size, learning_rate, epochs, sigmoid, sigmoid_derivative)

# Train with tanh activation function
tanh_weights_input_hidden, tanh_weights_hidden_output, tanh_loss_history = train_neural_network(
    inputs, targets, hidden_size, output_size, learning_rate, epochs, tanh, tanh_derivative)

# Train with ReLU activation function
relu_weights_input_hidden, relu_weights_hidden_output, relu_loss_history = train_neural_network(
    inputs, targets, hidden_size, output_size, learning_rate, epochs, relu, relu_derivative)

# Plot loss curves
plt.plot(sigmoid_loss_history, label='Sigmoid')
plt.plot(tanh_loss_history, label='Tanh')
plt.plot(relu_loss_history, label='ReLU')
plt.title('Training Loss for Different Activation Functions')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
