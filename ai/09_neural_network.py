"""
Lesson 09: Neural Network Basics
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
from lesson_player import LessonPlayer, cyan, green

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

player = LessonPlayer("Lesson 09: Neural Network Basics")

# ── Step 1 ──────────────────────────────────────────────

player.add_step("Step 1: What is a Neural Network?", f"""\
  Input Layer       Hidden Layer(s)       Output Layer
  (features)        (learned patterns)    (prediction)

    [pixel 1] ---\\
                  \\---> [h1] ---\\
    [pixel 2] -------->  [h2] -------->  [digit 0-9]
                  /---> [h3] ---/
    [pixel 3] ---/

  Each arrow has a "weight" (a number).
  Training = adjusting weights to minimize prediction errors.

  {cyan('Key terms:')}
    Neuron:     one node — multiplies inputs by weights, adds bias
    Layer:      a row of neurons
    Activation: a function that adds non-linearity (ReLU, sigmoid)
    Backprop:   algorithm that computes how to adjust each weight""")

# ── Step 2 ──────────────────────────────────────────────

digits = load_digits()
X = digits.data
y = digits.target

fig, axes = plt.subplots(2, 5, figsize=(10, 4))
for i, ax in enumerate(axes.flat):
    ax.imshow(digits.images[i], cmap='gray')
    ax.set_title(f'Label: {y[i]}')
    ax.axis('off')
plt.suptitle('Sample Handwritten Digits (8x8 pixels)')
plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, "images", "09_digits_samples.png"))
plt.close()

player.add_step("Step 2: The Digits Dataset", f"""\
  8x8 pixel images of handwritten digits (0-9).
  Each pixel is a brightness value (0-16).
  The image is flattened to 64 numbers → that's the input.

  {green('Dataset:')}
    Samples:  {X.shape[0]}
    Features: {X.shape[1]} (8x8 = 64 pixels)
    Classes:  0, 1, 2, 3, 4, 5, 6, 7, 8, 9

  {green('One image as numbers (8x8):')}
{np.array2string(digits.images[0].astype(int), prefix='    ')}

  Sample images saved to images/09_digits_samples.png""")

# ── Step 3 ──────────────────────────────────────────────

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

mlp = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=300, random_state=42)
mlp.fit(X_train_s, y_train)
y_pred = mlp.predict(X_test_s)

player.add_step("Step 3: Training a Neural Network", f"""\
  {cyan('Code:')}
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)   # scale first!

    mlp = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=300)
    mlp.fit(X_train_s, y_train)

  hidden_layer_sizes=(64, 32) means:
    Input (64 pixels) → 64 neurons → 32 neurons → Output (10 digits)

  {green('Result:')}
    Train accuracy: {mlp.score(X_train_s, y_train):.4f}
    Test accuracy:  {mlp.score(X_test_s, y_test):.4f}

    First 10 predictions: {y_pred[:10]}
    First 10 actual:      {y_test[:10]}""")

# ── Step 4 ──────────────────────────────────────────────

cm = confusion_matrix(y_test, y_pred)

fig, ax = plt.subplots(figsize=(8, 6))
im = ax.imshow(cm, cmap='Blues')
ax.set_xticks(range(10))
ax.set_yticks(range(10))
ax.set_xlabel('Predicted')
ax.set_ylabel('Actual')
ax.set_title('Confusion Matrix')
for i in range(10):
    for j in range(10):
        color = 'white' if cm[i, j] > cm.max() / 2 else 'black'
        ax.text(j, i, str(cm[i, j]), ha='center', va='center', color=color)
plt.colorbar(im)
plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, "images", "09_confusion_matrix.png"))
plt.close()

# Find most confused pairs
errors = []
for i in range(10):
    for j in range(10):
        if i != j and cm[i, j] > 0:
            errors.append((cm[i, j], i, j))
errors.sort(reverse=True)
error_text = ""
for count, actual, pred in errors[:5]:
    error_text += f"    {actual} mistaken for {pred}: {count} times\n"

player.add_step("Step 4: Confusion Matrix — What Gets Mixed Up?", f"""\
  {green('Most common errors:')}
{error_text}
  Matrix saved to images/09_confusion_matrix.png
  Diagonal = correct. Off-diagonal = errors.

  Makes sense: 3 and 8, 1 and 8 look similar in low-res images.""")

# ── Step 5 ──────────────────────────────────────────────

arch_results = ""
architectures = [(10,), (64,), (64, 32), (128, 64, 32)]
for arch in architectures:
    m = MLPClassifier(hidden_layer_sizes=arch, max_iter=300, random_state=42)
    m.fit(X_train_s, y_train)
    arch_results += f"    {str(arch):>20s}  train={m.score(X_train_s, y_train):.4f}  test={m.score(X_test_s, y_test):.4f}\n"

player.add_step("Step 5: Comparing Architectures", f"""\
  How do different network shapes affect accuracy?

  {green('Architecture comparison:')}
    {'(neurons per layer)':>20s}  {'train':>8s}  {'test':>8s}
{arch_results}
  More neurons = more capacity to learn complex patterns.
  But diminishing returns — and risk of overfitting.

  {green('Architecture intuition:')}
    (10,)          = tiny brain, misses patterns
    (64,)          = one layer, decent
    (64, 32)       = two layers, can learn hierarchical patterns
    (128, 64, 32)  = three layers, more capacity but slower""")

# ── Step 6 ──────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(mlp.loss_curve_)
ax.set_xlabel('Iteration')
ax.set_ylabel('Loss')
ax.set_title('Training Loss Curve')
ax.set_yscale('log')
plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, "images", "09_loss_curve.png"))
plt.close()

player.add_step("Step 6: Training Loss Curve", f"""\
  The loss curve shows how the network learns over iterations.

  {green('Result:')}
    Starting loss: {mlp.loss_curve_[0]:.4f}
    Final loss:    {mlp.loss_curve_[-1]:.4f}
    Iterations:    {len(mlp.loss_curve_)}

  Loss decreases over time → the network is learning!
  Curve saved to images/09_loss_curve.png

  {green('Summary:')}
    ✓ Neural network = layers of weighted connections
    ✓ MLPClassifier = neural network in scikit-learn
    ✓ Scale your data first (StandardScaler)
    ✓ More layers = more capacity, but diminishing returns
    ✓ Next lesson: build this from scratch with PyTorch""")

# ── Play ────────────────────────────────────────────────

player.play()
