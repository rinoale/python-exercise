"""
Lesson 09: Neural Network Basics

Goal: understand what a neural network IS — layers, weights, activation
      functions, backpropagation — then use sklearn's MLPClassifier.

Type along:
    python3
    >>> import matplotlib.pyplot as plt
    >>> from sklearn.datasets import load_digits
    >>> from sklearn.model_selection import train_test_split
    >>> from sklearn.preprocessing import StandardScaler
    >>> digits = load_digits(); X, y = digits.data, digits.target
    >>> X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    >>> scaler = StandardScaler(); X_train = scaler.fit_transform(X_train); X_test = scaler.transform(X_test)
    >>> from sklearn.neural_network import MLPClassifier
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
  {cyan('A neural network = layers of simple math operations.')}

  {green('Single neuron (what you already know):')}
      output = weight × input + bias       (lesson 03: y = ax + b)

  {green('Neural network = many neurons arranged in layers:')}

      Input Layer       Hidden Layer        Output Layer
      (features)        (learned patterns)  (prediction)

      [pixel 0] ──w₁──╲
      [pixel 1] ──w₂───→ [neuron h₁] ──╲
      [pixel 2] ──w₃──╱                  ╲
                                           → [output: digit 0-9]
      [pixel 0] ──w₄──╲                  ╱
      [pixel 1] ──w₅───→ [neuron h₂] ──╱
      [pixel 2] ──w₆──╱

  {cyan('Each neuron does:')}
      1. Multiply each input by a weight:  Σ(wᵢ × xᵢ) + bias
      2. Apply an activation function:     ReLU(result)
      3. Pass the result to the next layer

  {cyan('Training = finding the right weights for ALL connections.')}
  With 64 inputs → 64 hidden → 10 outputs, that's ~7000 weights.""",

detail=f"""\
  {green('Term: Activation Function')}

      Without activation, stacking layers is pointless:
          layer1: y = w₁x + b₁
          layer2: y = w₂(w₁x + b₁) + b₂ = (w₂w₁)x + (w₂b₁ + b₂)
      That's STILL just y = ax + b — a linear function!
      No matter how many layers, it collapses to one line.

      Activation functions add NON-LINEARITY:
          layer1: y = ReLU(w₁x + b₁)       ← can't simplify away
          Now the network can learn curves, not just lines.

  {cyan('Common activation functions:')}

      ReLU(x) = max(0, x)         ← most popular, simple
          x < 0 → 0
          x ≥ 0 → x
          "Turn off negative values"

      Sigmoid(x) = 1/(1+e⁻ˣ)     ← squashes to (0, 1)
      Tanh(x) = (eˣ-e⁻ˣ)/(eˣ+e⁻ˣ) ← squashes to (-1, 1)

      ReLU:                Sigmoid:
      y│    /              y│         ────
       │   /                │        /
       │──/                 │───────/
       └──────x             └──────────x""")

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

digit_grid = np.array2string(digits.images[0].astype(int), prefix='      ')

player.add_step("Step 2: The Digits Dataset — Images as Numbers", f"""\
  {cyan('Type:')}
      from sklearn.datasets import load_digits
      digits = load_digits()
      X = digits.data      # flattened pixel values
      y = digits.target     # the actual digit (0-9)

  {green('What is it?')}
      8×8 pixel images of handwritten digits.
      Each pixel is a brightness value (0 = white, 16 = black).
      The 8×8 image is FLATTENED into 64 numbers → that's the input.

      X.shape  →  {X.shape}   ({X.shape[0]} images, {X.shape[1]} pixels each)
      y[:10]   →  {y[:10]}

  {green('One image as numbers (digit "0"):')}
{digit_grid}

  {cyan('How the network sees it:')}
      [0, 0, 5, 13, 9, 1, 0, 0, 0, 13, ...]  (64 numbers)
      →  network  →  "this is a 0"

  {green('Images saved to images/09_digits_samples.png')}""")

# ── Step 3 ──────────────────────────────────────────────

player.add_step("Step 3: Forward Pass — How Prediction Works", f"""\
  {cyan('A neural network predicts in 3 steps (the "forward pass"):')}

      Input (64 pixels)  →  Hidden layers  →  Output (10 scores)

  {green('Step by step:')}

      1. {cyan('Input layer:')}  64 pixel values (the image)

      2. {cyan('Hidden layer 1 (64 neurons):')}
         Each neuron:  output = ReLU(w₁×pixel₁ + w₂×pixel₂ + ... + w₆₄×pixel₆₄ + bias)
         64 neurons do this in parallel → 64 hidden values

      3. {cyan('Hidden layer 2 (32 neurons):')}
         Each neuron takes 64 hidden values as input → 32 hidden values

      4. {cyan('Output layer (10 neurons):')}
         One neuron per digit (0-9).
         Highest score = the prediction.

  {green('In matrix form (what actually happens):')}
      h₁ = ReLU(X @ W₁ + b₁)      # (n, 64) × (64, 64) → (n, 64)
      h₂ = ReLU(h₁ @ W₂ + b₂)     # (n, 64) × (64, 32) → (n, 32)
      out = h₂ @ W₃ + b₃           # (n, 32) × (32, 10) → (n, 10)
      prediction = argmax(out)       # pick the highest score

  {cyan('@ is matrix multiplication.')} The whole network is just
  multiply → activate → multiply → activate → multiply → pick max.""")

# ── Step 4 ──────────────────────────────────────────────

player.add_step("Step 4: Backpropagation — How Training Works", f"""\
  {cyan('Training finds the weights that minimize prediction errors.')}

  {green('The training loop (4 steps, repeated many times):')}

      1. {cyan('Forward pass:')}  push data through the network → get prediction
      2. {cyan('Loss:')}          compare prediction to correct answer → compute error
      3. {cyan('Backward pass:')} compute how to adjust each weight to reduce error
      4. {cyan('Update:')}        adjust all weights slightly

      Repeat 100-1000 times.  Each cycle is called an {cyan('epoch')}.

  {green('Term: Loss Function')}
      Measures how wrong the predictions are.  A single number.
      For classification: {cyan('Cross-Entropy Loss')}
          - correct answer has high probability → low loss
          - correct answer has low probability → high loss

  {green('Term: Backpropagation')} (step 3)
      The algorithm that computes: "if I increase weight w₅₇ by 0.01,
      how much does the total loss change?"  This is the GRADIENT.

      Uses the chain rule from calculus:
          dLoss/dW₁ = dLoss/dOut × dOut/dH₂ × dH₂/dH₁ × dH₁/dW₁

      The gradient flows BACKWARD through the network (hence the name).

  {green('Term: Gradient Descent')} (step 4)
      w_new = w_old - learning_rate × gradient
      Move each weight in the direction that reduces loss.""",

detail=f"""\
  {green('Gradient Descent — ASCII Visualization')}

      Loss
      │╲
      │  ╲
      │    ╲
      │      ╲  ← gradient points uphill
      │        ●  ← current weight
      │          ╲
      │            ╲
      │              ⊗  ← minimum (goal)
      └─────────────────── weight

      The gradient tells us which direction is "uphill."
      We move the weight in the OPPOSITE direction (downhill).
      learning_rate controls how big each step is.

      Too small: converges slowly (many iterations needed)
      Too large: overshoots the minimum (loss oscillates)
      Just right: converges quickly and stably

  {cyan('Learning rate analogy:')}
      Finding the bottom of a valley while blindfolded.
      Gradient = which direction is downhill (you can feel the slope).
      Learning rate = how big a step you take each time.
      Too big → you leap past the bottom and oscillate.
      Too small → you'll get there, but it takes forever.""")

# ── Step 5 ──────────────────────────────────────────────

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

mlp = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=300, random_state=42)
mlp.fit(X_train_s, y_train)
y_pred = mlp.predict(X_test_s)

n_params = sum(
    mlp.coefs_[i].size + mlp.intercepts_[i].size
    for i in range(len(mlp.coefs_))
)

player.add_step("Step 5: Train a Neural Network with scikit-learn", f"""\
  {cyan('Type:')}
      from sklearn.datasets import load_digits
      from sklearn.model_selection import train_test_split
      from sklearn.neural_network import MLPClassifier
      from sklearn.preprocessing import StandardScaler

      digits = load_digits()
      X, y = digits.data, digits.target
      X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

      scaler = StandardScaler()
      X_train_s = scaler.fit_transform(X_train)     # ALWAYS scale for NNs
      X_test_s = scaler.transform(X_test)

      mlp = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=300)
      mlp.fit(X_train_s, y_train)
      y_pred = mlp.predict(X_test_s)

      mlp.score(X_train_s, y_train)    →  {mlp.score(X_train_s, y_train):.4f}
      mlp.score(X_test_s, y_test)      →  {mlp.score(X_test_s, y_test):.4f}
      Total parameters: {n_params:,}

  {cyan('Architecture: 64 → 64 → 32 → 10')}
      Input:   64 pixels
      Hidden1: 64 neurons   (64×64 + 64 = {64*64+64:,} params)
      Hidden2: 32 neurons   (64×32 + 32 = {64*32+32:,} params)
      Output:  10 neurons   (32×10 + 10 = {32*10+10:,} params)
      Total:                 {n_params:,} parameters

  {green('MLP = Multi-Layer Perceptron')}
      Perceptron = one neuron.  MLP = many neurons in many layers.
      This is the simplest form of neural network.""")

# ── Step 6 ──────────────────────────────────────────────

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

errors = []
for i in range(10):
    for j in range(10):
        if i != j and cm[i, j] > 0:
            errors.append((cm[i, j], i, j))
errors.sort(reverse=True)
error_text = ""
for count, actual, pred in errors[:5]:
    error_text += f"      {actual} confused with {pred}: {count} times\n"

player.add_step("Step 6: Confusion Matrix — What the Network Gets Wrong", f"""\
  {cyan('Type:')}
      from sklearn.metrics import confusion_matrix
      cm = confusion_matrix(y_test, y_pred)

  {green('Most common mistakes:')}
{error_text}
  {green('Matrix saved to images/09_confusion_matrix.png')}

  {cyan('Makes sense in context:')}
      3 and 8 — both have curved strokes
      1 and 8 — similar structure in 8×8 pixels
      9 and 3 — upper loop can look similar

  {cyan('With only 8×8 pixels, SOME confusion is expected.')}
  Real digit recognizers use 28×28 (MNIST) or larger images
  with convolutional neural networks (CNNs) → 99.5%+ accuracy.""")

# ── Step 7 ──────────────────────────────────────────────

arch_results = ""
architectures = [(10,), (64,), (64, 32), (128, 64, 32)]
arch_params = []
for arch in architectures:
    m = MLPClassifier(hidden_layer_sizes=arch, max_iter=300, random_state=42)
    m.fit(X_train_s, y_train)
    p = sum(m.coefs_[i].size + m.intercepts_[i].size for i in range(len(m.coefs_)))
    arch_params.append(p)
    arch_results += f"      {str(arch):>16s}  params={p:>6,}  train={m.score(X_train_s, y_train):.4f}  test={m.score(X_test_s, y_test):.4f}\n"

player.add_step("Step 7: Architecture — How Many Neurons?", f"""\
  {cyan('How do different network shapes affect accuracy?')}

  {green('Comparison:')}
{arch_results}
  {cyan('Observations:')}
      (10,) = tiny network — not enough capacity to learn patterns.
      (64,) = one hidden layer — decent, but limited.
      (64, 32) = two layers — good balance of capacity and speed.
      (128, 64, 32) = three layers — more params, diminishing returns.

  {green('Term: Capacity')}
      A network's ability to learn complex patterns.
      More neurons = more capacity = can learn harder patterns.
      But too much capacity → overfitting (memorize noise).

  {green('Term: Hyperparameter')}
      A setting YOU choose before training (not learned from data):
      - hidden_layer_sizes = (64, 32)
      - learning_rate = 0.001
      - max_iter = 300
      vs {green('Parameter')} = learned FROM data (weights and biases).""")

# ── Step 8 ──────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(mlp.loss_curve_)
ax.set_xlabel('Iteration')
ax.set_ylabel('Loss')
ax.set_title('Training Loss Curve')
ax.set_yscale('log')
plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, "images", "09_loss_curve.png"))
plt.close()

player.add_step("Step 8: Loss Curve — Watching the Network Learn", f"""\
  {cyan('Type:')}
      import matplotlib.pyplot as plt
      plt.plot(mlp.loss_curve_)
      plt.yscale('log')
      plt.savefig('images/09_loss_curve.png')

  {green('Result:')}
      Starting loss: {mlp.loss_curve_[0]:.4f}
      Final loss:    {mlp.loss_curve_[-1]:.4f}
      Iterations:    {len(mlp.loss_curve_)}

  {green('Saved to images/09_loss_curve.png')}

  {cyan('How to read a loss curve:')}
      Loss going DOWN      → network is learning
      Loss plateau          → converged (or stuck)
      Loss going UP         → problem (learning rate too high, or overfitting)
      Spiky                 → learning rate too high

      Good curve:           Bad curve:
      loss│╲                loss│  /╲  /╲  /╲
          │  ╲                  │ /  \\/  \\/
          │   ╲___              │/
          └──────── iter        └──────── iter""")

# ── Step 9 ──────────────────────────────────────────────

player.add_step("Step 9: Summary — Neural Networks", f"""\
  {green('What you learned:')}

      {cyan('1. Neuron')} = weight × input + bias, then activation.
      {cyan('2. Network')} = layers of neurons: input → hidden → output.
      {cyan('3. Forward pass')} = push data through to get prediction.
      {cyan('4. Loss')} = how wrong the prediction is (cross-entropy).
      {cyan('5. Backpropagation')} = compute gradient of loss w.r.t. each weight.
      {cyan('6. Gradient descent')} = adjust weights to reduce loss.
      {cyan('7. Epoch')} = one pass through all training data.
      {cyan('8. Architecture')} = how many layers, how many neurons per layer.

  {green('Key insights:')}
      - Activation functions are essential (without them, layers collapse)
      - More capacity ≠ always better (overfitting risk)
      - Always scale your data before neural networks
      - The loss curve tells you if training is working

  {green('What\'s missing from sklearn\'s MLP:')}
      - No GPU support (slow for large datasets)
      - Limited architectures (can't do CNNs, RNNs, transformers)
      - No fine-grained control over training

  {green('What\'s next?')}
      Lesson 10: PyTorch — build the SAME network from scratch.
      Full control over every layer, weight, and training step.""")

# ── Play ────────────────────────────────────────────────

player.play()
