"""
XOR Neural Network from scratch — watch weights converge.

Network: 2 inputs → 2 hidden neurons → 1 output
Learns: [0,0]→0, [0,1]→1, [1,0]→1, [1,1]→0
"""
import numpy as np

np.random.seed(42)

# --- Data ---
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

# --- Activation function ---
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# --- Initialize weights randomly (NOT zeros) ---
weights_input_hidden = np.random.uniform(-1, 1, (2, 2))   # 2 inputs → 2 hidden
weights_hidden_output = np.random.uniform(-1, 1, (2, 1))  # 2 hidden → 1 output
bias_hidden = np.random.uniform(-1, 1, (1, 2))
bias_output = np.random.uniform(-1, 1, (1, 1))

learning_rate = 0.5
epochs = 10000

print("=== Initial random weights ===")
print(f"Input→Hidden:\n{weights_input_hidden}")
print(f"Hidden→Output:\n{weights_hidden_output}")
print()

# --- Training loop ---
for epoch in range(epochs):

    # 1. Forward pass
    hidden_input = X @ weights_input_hidden + bias_hidden
    hidden_output = sigmoid(hidden_input)

    final_input = hidden_output @ weights_hidden_output + bias_output
    predicted = sigmoid(final_input)

    # 2. Calculate error
    error = y - predicted

    # 3. Backward pass (gradient calculation)
    d_output = error * sigmoid_derivative(predicted)

    error_hidden = d_output @ weights_hidden_output.T
    d_hidden = error_hidden * sigmoid_derivative(hidden_output)

    # 4. Update weights (purposeful, not random!)
    weights_hidden_output += hidden_output.T @ d_output * learning_rate
    weights_input_hidden += X.T @ d_hidden * learning_rate
    bias_output += np.sum(d_output, axis=0, keepdims=True) * learning_rate
    bias_hidden += np.sum(d_hidden, axis=0, keepdims=True) * learning_rate

    # Print progress
    if epoch % 2000 == 0:
        loss = np.mean(error ** 2)
        print(f"Epoch {epoch:5d} | Loss: {loss:.6f}")
        print(f"  Predictions: {predicted.flatten().round(3)}")
        print(f"  Input→Hidden weights: {weights_input_hidden.flatten().round(3)}")
        print(f"  Hidden→Output weights: {weights_hidden_output.flatten().round(3)}")
        print()

# --- Final result ---
print("=== Final result ===")
print(f"Expected:    {y.flatten()}")
print(f"Predicted:   {predicted.flatten().round(3)}")
print()
print("=== Final weights ===")
print(f"Input→Hidden:\n{weights_input_hidden.round(3)}")
print(f"Hidden→Output:\n{weights_hidden_output.round(3)}")
