"""
XOR Neural Network — Animated ASCII visualization.

Watch the weights and predictions update in real-time.
Press Ctrl+C to stop.
"""
import numpy as np
import os
import time
import sys

np.random.seed(42)

# --- Data ---
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

# --- Activation ---
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# --- Initialize weights randomly ---
w_ih = np.random.uniform(-1, 1, (2, 2))
w_ho = np.random.uniform(-1, 1, (2, 1))
b_h = np.random.uniform(-1, 1, (1, 2))
b_o = np.random.uniform(-1, 1, (1, 1))

learning_rate = 0.5
epochs = 10000

def bar(value, width=20):
    """Create a bar showing value from -10 to +10."""
    normalized = (value + 10) / 20
    normalized = max(0, min(1, normalized))
    pos = int(normalized * width)
    line = list("─" * width)
    line[width // 2] = "│"
    if pos < len(line):
        line[pos] = "●"
    return "".join(line)

def prediction_bar(value):
    """Show prediction as a filled bar 0.0 to 1.0."""
    width = 15
    filled = int(value * width)
    return "█" * filled + "░" * (width - filled)

def render(epoch, loss, predicted, w_ih, w_ho):
    """Render the full ASCII frame."""
    p = predicted.flatten()
    lines = []
    lines.append("╔══════════════════════════════════════════════════════════════╗")
    lines.append("║        XOR NEURAL NETWORK — LEARNING IN REAL TIME           ║")
    lines.append("╠══════════════════════════════════════════════════════════════╣")
    lines.append(f"║  Epoch: {epoch:5d}/10000        Loss: {loss:.6f}               ║")
    lines.append("╠══════════════════════════════════════════════════════════════╣")
    lines.append("║                                                              ║")
    lines.append("║   INPUT        HIDDEN LAYER         OUTPUT                   ║")
    lines.append("║                                                              ║")
    lines.append(f"║   ┌───┐    w1:{w_ih[0,0]:+.2f}    ┌───┐                        ║")
    lines.append(f"║   │ A │────────────────│ H1│──┐                            ║")
    lines.append(f"║   └───┘    w2:{w_ih[1,0]:+.2f}    └───┘  │ w5:{w_ho[0,0]:+.2f}            ║")
    lines.append(f"║        ╲  ╱                    │        ┌───┐              ║")
    lines.append(f"║         ╲╱                     ├────────│ O │              ║")
    lines.append(f"║         ╱╲                     │        └───┘              ║")
    lines.append(f"║        ╱  ╲                    │ w6:{w_ho[1,0]:+.2f}              ║")
    lines.append(f"║   ┌───┐    w3:{w_ih[0,1]:+.2f}    ┌───┐  │                            ║")
    lines.append(f"║   │ B │────────────────│ H2│──┘                            ║")
    lines.append(f"║   └───┘    w4:{w_ih[1,1]:+.2f}    └───┘                        ║")
    lines.append("║                                                              ║")
    lines.append("╠══════════════════════════════════════════════════════════════╣")
    lines.append("║  PREDICTIONS vs EXPECTED                                     ║")
    lines.append("╠══════════════════════════════════════════════════════════════╣")
    lines.append(f"║  [0,0]→ {prediction_bar(p[0])} {p[0]:.3f}  (want 0)  {'✓' if p[0] < 0.1 else '✗'}       ║")
    lines.append(f"║  [0,1]→ {prediction_bar(p[1])} {p[1]:.3f}  (want 1)  {'✓' if p[1] > 0.9 else '✗'}       ║")
    lines.append(f"║  [1,0]→ {prediction_bar(p[2])} {p[2]:.3f}  (want 1)  {'✓' if p[2] > 0.9 else '✗'}       ║")
    lines.append(f"║  [1,1]→ {prediction_bar(p[3])} {p[3]:.3f}  (want 0)  {'✓' if p[3] < 0.1 else '✗'}       ║")
    lines.append("╠══════════════════════════════════════════════════════════════╣")
    lines.append("║  WEIGHT MAGNITUDES        -10         0        +10           ║")
    lines.append(f"║  w1 (A→H1): {bar(w_ih[0,0])}  {w_ih[0,0]:+.3f}              ║")
    lines.append(f"║  w2 (B→H1): {bar(w_ih[1,0])}  {w_ih[1,0]:+.3f}              ║")
    lines.append(f"║  w3 (A→H2): {bar(w_ih[0,1])}  {w_ih[0,1]:+.3f}              ║")
    lines.append(f"║  w4 (B→H2): {bar(w_ih[1,1])}  {w_ih[1,1]:+.3f}              ║")
    lines.append(f"║  w5 (H1→O): {bar(w_ho[0,0])}  {w_ho[0,0]:+.3f}              ║")
    lines.append(f"║  w6 (H2→O): {bar(w_ho[1,0])}  {w_ho[1,0]:+.3f}              ║")
    lines.append("╠══════════════════════════════════════════════════════════════╣")

    if loss < 0.001:
        lines.append("║  ★ CONVERGED! The network learned XOR! ★                    ║")
    else:
        lines.append("║  Training... (Ctrl+C to stop)                                ║")
    lines.append("╚══════════════════════════════════════════════════════════════╝")

    return "\n".join(lines)

# --- Training with animation ---
try:
    for epoch in range(epochs):

        # Forward pass
        hidden_input = X @ w_ih + b_h
        hidden_output = sigmoid(hidden_input)
        final_input = hidden_output @ w_ho + b_o
        predicted = sigmoid(final_input)

        # Error & backprop
        error = y - predicted
        d_output = error * sigmoid_derivative(predicted)
        error_hidden = d_output @ w_ho.T
        d_hidden = error_hidden * sigmoid_derivative(hidden_output)

        # Update weights
        w_ho += hidden_output.T @ d_output * learning_rate
        w_ih += X.T @ d_hidden * learning_rate
        b_o += np.sum(d_output, axis=0, keepdims=True) * learning_rate
        b_h += np.sum(d_hidden, axis=0, keepdims=True) * learning_rate

        # Animate every N epochs
        if epoch % 50 == 0:
            loss = np.mean(error ** 2)
            os.system("clear" if os.name != "nt" else "cls")
            print(render(epoch, loss, predicted, w_ih, w_ho))
            time.sleep(0.03)

    # Final frame
    loss = np.mean(error ** 2)
    os.system("clear" if os.name != "nt" else "cls")
    print(render(epochs, loss, predicted, w_ih, w_ho))

except KeyboardInterrupt:
    print("\n\nStopped by user.")
