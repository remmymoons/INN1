import numpy as np
import matplotlib.pyplot as plt

class Perceptron:
    def __init__(self, input_size, learning_rate=0.01, epochs=100):
        self.weights = np.random.rand(input_size + 1)  # Additional weight for bias
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.errors = []

    def activation(self, x):
        return 1 if x >= 0 else 0

    def predict(self, inputs):
        weighted_sum = np.dot(inputs, self.weights[1:]) + self.weights[0]  # Include bias
        return self.activation(weighted_sum)

    def train(self, training_data, labels):
        for epoch in range(self.epochs):
            total_error = 0
            for inputs, label in zip(training_data, labels):
                prediction = self.predict(inputs)
                error = label - prediction
                total_error += int(error != 0)

                # Update weights
                self.weights[1:] += self.learning_rate * error * inputs
                self.weights[0] += self.learning_rate * error

            self.errors.append(total_error)

            # Print error every 10 epochs
            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Total Error: {total_error}")

            # If there are no errors, terminate early
            if total_error == 0:
                print("Converged early.")
                break

# Example usage
# Training data and labels for OR gate
training_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
labels = np.array([0, 1, 1, 1])

# Create and train a perceptron
perceptron = Perceptron(input_size=2)
perceptron.train(training_data, labels)

# Plot the decision boundary
plt.scatter(training_data[:, 0], training_data[:, 1], c=labels, cmap='viridis', marker='o')
x_vals = np.linspace(-0.5, 1.5, 100)
y_vals = (-perceptron.weights[1] * x_vals - perceptron.weights[0]) / perceptron.weights[2]
plt.plot(x_vals, y_vals, 'r--', label='Decision Boundary')
plt.title('Perceptron Decision Boundary')
plt.xlabel('Input 1')
plt.ylabel('Input 2')
plt.legend()
plt.show()

