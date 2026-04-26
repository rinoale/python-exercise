"""
Lesson 10: Intro to Deep Learning with PyTorch

Goal: rebuild the lesson-09 digit classifier from scratch using PyTorch.
      Understand tensors, nn.Module, the training loop, and autograd.

Type along:
    python3
    >>> from sklearn.datasets import load_digits
    >>> from sklearn.model_selection import train_test_split
    >>> from sklearn.preprocessing import StandardScaler
    >>> digits = load_digits(); X, y = digits.data, digits.target
    >>> X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    >>> scaler = StandardScaler(); X_train = scaler.fit_transform(X_train); X_test = scaler.transform(X_test)
    >>> import torch
    >>> import torch.nn as nn
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from lesson_player import LessonPlayer, cyan, green

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

player = LessonPlayer("Lesson 10: Intro to Deep Learning with PyTorch")

# ── Step 1 ──────────────────────────────────────────────

player.add_step("Step 1: Why PyTorch?", f"""\
  {cyan('scikit-learn (lesson 09):')}
      mlp = MLPClassifier(hidden_layer_sizes=(64, 32))
      mlp.fit(X, y)    # done — but you can't see inside

  {cyan('PyTorch:')}
      You build every layer, define the forward pass,
      and write the training loop yourself.

  {green('Why bother?')}
      - GPU support (train 1000x faster on large data)
      - Custom architectures (CNNs, RNNs, Transformers)
      - Research flexibility (modify anything)
      - Production deployment (TorchScript, ONNX)

  {green('PyTorch vs TensorFlow:')}
      Both are deep learning frameworks.
      PyTorch: more Pythonic, dominant in research.
      TensorFlow: Google's framework, strong in production.
      We use PyTorch because it's the standard for learning.

  {green('PyTorch version:')} {torch.__version__}
  {green('CUDA available:')} {torch.cuda.is_available()}
  (CUDA = GPU support.  We'll use CPU for this lesson.)""")

# ── Step 2 ──────────────────────────────────────────────

a = torch.tensor([1, 2, 3])
b = torch.tensor([4, 5, 6])

x = torch.tensor(3.0, requires_grad=True)
y_val = x ** 2 + 2 * x + 1
y_val.backward()

player.add_step("Step 2: Tensors — PyTorch's Building Block", f"""\
  {cyan('A tensor is a NumPy array with two superpowers:')}
      1. Can run on GPU  (parallel computation)
      2. Tracks gradients automatically  (for backpropagation)

  {cyan('Type:')}
      import torch
      a = torch.tensor([1, 2, 3])
      b = torch.tensor([4, 5, 6])
      a + b    →  {a + b}
      a * b    →  {a * b}        (element-wise)
      a @ b    →  {a @ b}               (dot product)

  {cyan('Autograd — automatic differentiation:')}
      x = torch.tensor(3.0, requires_grad=True)
      y = x**2 + 2*x + 1        # y = 9 + 6 + 1 = {y_val.item()}
      y.backward()               # compute dy/dx
      x.grad                     # dy/dx = 2x + 2 = 2(3) + 2 = {x.grad.item()}

  {green('This is how PyTorch computes gradients for backpropagation.')}
  You define the forward pass, PyTorch computes the backward
  pass automatically.  No manual calculus needed.""",

detail=f"""\
  {green('Tensor vs NumPy array')}

      NumPy:   np.array([1, 2, 3])       →  CPU only, no gradients
      PyTorch: torch.tensor([1, 2, 3])   →  CPU or GPU, gradients

  {cyan('Converting between them:')}
      numpy_array = tensor.numpy()           # tensor → numpy
      tensor = torch.from_numpy(numpy_array) # numpy → tensor

  {cyan('Key tensor operations — try each one:')}
      torch.zeros(3, 4)       # 3×4 matrix of zeros
      torch.ones(2, 3)        # 2×3 matrix of ones
      torch.randn(5)          # 5 random numbers (normal distribution)
      t.shape                 # dimensions
      t.dtype                 # data type (float32, int64, ...)
      t.requires_grad         # is gradient tracking on?

  {green('Term: Computational Graph')}
      When requires_grad=True, PyTorch builds a graph of operations.
      y = x² + 2x + 1 creates:  x → square → multiply → add → y
      .backward() walks this graph in reverse to compute gradients.""")

# ── Step 3 ──────────────────────────────────────────────

digits = load_digits()
X, y = digits.data, digits.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

X_train_t = torch.FloatTensor(X_train)
X_test_t = torch.FloatTensor(X_test)
y_train_t = torch.LongTensor(y_train)
y_test_t = torch.LongTensor(y_test)

player.add_step("Step 3: Prepare Data as Tensors", f"""\
  {cyan('Same digits dataset as lesson 09 — convert to tensors.')}

  {cyan('Type:')}
      from sklearn.datasets import load_digits
      from sklearn.model_selection import train_test_split
      from sklearn.preprocessing import StandardScaler
      digits = load_digits()
      X, y = digits.data, digits.target
      X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

      scaler = StandardScaler()
      X_train = scaler.fit_transform(X_train)
      X_test = scaler.transform(X_test)

      X_train_t = torch.FloatTensor(X_train)    # float for input data
      y_train_t = torch.LongTensor(y_train)     # integer for class labels

  {green('Shapes:')}
      X_train_t: {list(X_train_t.shape)}    ({X_train_t.shape[0]} samples, {X_train_t.shape[1]} features)
      y_train_t: {list(y_train_t.shape)}        ({y_train_t.shape[0]} labels)
      X_test_t:  {list(X_test_t.shape)}
      y_test_t:  {list(y_test_t.shape)}

  {cyan('Why different types?')}
      FloatTensor = 32-bit float  →  for continuous data (pixels)
      LongTensor  = 64-bit integer → for class labels (0, 1, 2, ...)
      CrossEntropyLoss expects LongTensor for labels.""")

# ── Step 4 ──────────────────────────────────────────────


class DigitNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(64, 64)
        self.layer2 = nn.Linear(64, 32)
        self.layer3 = nn.Linear(32, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.layer3(x)
        return x


model = DigitNet()
n_params = sum(p.numel() for p in model.parameters())

player.add_step("Step 4: Build a Neural Network with nn.Module", f"""\
  {cyan('Type:')}
      import torch.nn as nn

      class DigitNet(nn.Module):
          def __init__(self):
              super().__init__()
              self.layer1 = nn.Linear(64, 64)    # 64 inputs → 64 outputs
              self.layer2 = nn.Linear(64, 32)    # 64 → 32
              self.layer3 = nn.Linear(32, 10)    # 32 → 10 (one per digit)
              self.relu = nn.ReLU()

          def forward(self, x):
              x = self.relu(self.layer1(x))      # layer + activation
              x = self.relu(self.layer2(x))
              x = self.layer3(x)                 # no activation on output
              return x                            # raw scores (logits)

      model = DigitNet()

  {green('Architecture:')} 64 → 64 → 32 → 10
  {green('Total parameters:')} {n_params:,}

  {cyan('Key concepts:')}
      nn.Module   = base class for all neural networks in PyTorch
      nn.Linear   = one fully connected layer (weight matrix + bias)
      nn.ReLU     = activation function: max(0, x)
      forward()   = defines how data flows through the network

  {cyan('No activation on the output layer')} — CrossEntropyLoss
  expects raw scores (logits), not probabilities.""",

detail=f"""\
  {green('What nn.Linear(64, 64) contains:')}

      W = weight matrix:  shape (64, 64) = {64*64} weights
      b = bias vector:    shape (64,)    = 64 biases
      Total: {64*64 + 64} parameters

      It computes:  output = input @ W.T + b
      Exactly like lesson 03's  y = ax + b,
      but with matrices instead of scalars.

  {cyan('Parameter count breakdown:')}
      layer1: (64 × 64) + 64 = {64*64 + 64:,}
      layer2: (64 × 32) + 32 = {64*32 + 32:,}
      layer3: (32 × 10) + 10 = {32*10 + 10:,}
      Total:                    {n_params:,}

  {cyan('Try:')}
      list(model.parameters())     # see all weight tensors
      model.layer1.weight.shape    # (64, 64)
      model.layer1.bias.shape      # (64,)""")

# ── Step 5 ──────────────────────────────────────────────

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

losses = []
train_accs = []
log_lines = ""

for epoch in range(100):
    outputs = model(X_train_t)
    loss = criterion(outputs, y_train_t)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    losses.append(loss.item())
    _, predicted = torch.max(outputs, 1)
    acc = (predicted == y_train_t).float().mean().item()
    train_accs.append(acc)

    if (epoch + 1) % 20 == 0:
        log_lines += f"      Epoch {epoch+1:3d}/100  loss={loss.item():.4f}  accuracy={acc:.4f}\n"

player.add_step("Step 5: The Training Loop — Core of Deep Learning", f"""\
  {cyan('This is the most important code in deep learning.')}

  {cyan('Type:')}
      import torch.optim as optim
      criterion = nn.CrossEntropyLoss()
      optimizer = optim.Adam(model.parameters(), lr=0.001)

      for epoch in range(100):
          # 1. FORWARD: push data through the network → get predictions
          outputs = model(X_train_t)
          print(f"outputs.shape = {{outputs.shape}}")   # try on epoch 0
          #   → torch.Size([1257, 10])  — 1257 samples, 10 scores each
          #   Each row is 10 raw scores (logits), one per digit 0-9.
          #   Example: outputs[0] → tensor([-0.12, 0.85, -0.33, ...])
          #   Highest score = the network's guess for that sample.

          # 2. LOSS: how wrong are the predictions?
          loss = criterion(outputs, y_train_t)
          print(f"loss = {{loss.item():.4f}}")
          #   → a single number. High = bad. Low = good.
          #   Epoch 1: ~2.3 (random guessing among 10 classes ≈ ln(10))
          #   Epoch 100: ~0.2 (the network learned the patterns)

          # 3. CLEAR old gradients (PyTorch accumulates them by default)
          optimizer.zero_grad()

          # 4. BACKWARD: compute how each weight should change
          loss.backward()
          # After this, every weight has a .grad attribute:
          print(f"layer1 grad sample = {{model.layer1.weight.grad[0][:5]}}")
          #   → tensor([0.002, -0.001, ...])
          #   These tiny numbers tell the optimizer: "increase this
          #   weight by 0.002, decrease that one by 0.001, ..."

          # 5. UPDATE: adjust all weights to reduce the loss
          optimizer.step()

          # Track progress — print every 20 epochs:
          _, predicted = torch.max(outputs, 1)
          acc = (predicted == y_train_t).float().mean().item()
          if (epoch + 1) % 20 == 0:
              print(f"Epoch {{epoch+1:3d}}/100  loss={{loss.item():.4f}}  accuracy={{acc:.4f}}")

  {green('Training progress:')}
{log_lines}
  {green('Loss goes down, accuracy goes up → the network is learning!')}

  {cyan('Inspect the final state:')}
      outputs = model(X_train_t)
      outputs[0]             # 10 scores for the first sample
      outputs[0].argmax()    # which digit got the highest score?
      y_train_t[0]           # what was the actual label?
      loss.item()            # final loss value (should be < 0.3)

      # See what the model predicts for ALL training data:
      _, predicted = torch.max(outputs, 1)
      predicted[:20]         # first 20 predictions
      y_train_t[:20]         # first 20 actual labels
      (predicted == y_train_t).float().mean()   # overall accuracy""",

detail=f"""\
  {green('Term: Adam Optimizer')}
      Adam = Adaptive Moment Estimation.
      Smarter than basic gradient descent:
      - Adapts learning rate per-parameter (not one rate for all)
      - Uses momentum (keeps moving in the same direction)
      - Handles sparse gradients well

      In practice, Adam is the default choice.  lr=0.001 is good.

  {green('Term: Learning Rate (lr)')}
      Controls how big each weight update is.
      w_new = w_old - lr × gradient

      lr too high → loss oscillates, never converges
      lr too low  → loss decreases very slowly
      lr=0.001    → good default for Adam

  {green('Why zero_grad()?')}
      PyTorch ACCUMULATES gradients by default.
      If you don't zero them, gradients from the previous epoch
      ADD to the current ones → wrong updates.""")

# ── Step 6 ──────────────────────────────────────────────

model.eval()
with torch.no_grad():
    test_outputs = model(X_test_t)
    _, test_predicted = torch.max(test_outputs, 1)
    test_acc = (test_predicted == y_test_t).float().mean().item()

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].plot(losses)
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].set_title('Training Loss')
axes[1].plot(train_accs)
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Accuracy')
axes[1].set_title('Training Accuracy')
axes[1].set_ylim(0, 1.05)
plt.suptitle('PyTorch Training Progress', fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, "images", "10_pytorch_training.png"))
plt.close()

player.add_step("Step 6: Evaluate on Test Data", f"""\
  {cyan('Type:')}
      X_test_t = torch.FloatTensor(X_test)
      y_test_t = torch.LongTensor(y_test)

      model.eval()                         # switch to evaluation mode
      with torch.no_grad():               # don't track gradients
          test_outputs = model(X_test_t)
          _, predicted = torch.max(test_outputs, 1)
          test_acc = (predicted == y_test_t).float().mean()
          print(test_acc.item())           →  {test_acc:.4f}

  {green('Compare with lesson 09 (sklearn MLP):')}
      sklearn MLPClassifier: ~0.97
      PyTorch DigitNet:      {test_acc:.4f}

  {cyan('model.eval()')} — switches off training-specific behaviors
  (dropout, batch normalization).  Always call before testing.

  {cyan('torch.no_grad()')} — disables gradient tracking.
  Faster and uses less memory during inference.

  {green('Plots saved to images/10_pytorch_training.png')}""")

# ── Step 7 ──────────────────────────────────────────────

fig, axes = plt.subplots(2, 5, figsize=(12, 5))
model.eval()
correct = 0
total = 10
pred_text = ""
with torch.no_grad():
    for i, ax in enumerate(axes.flat):
        img = digits.images[len(digits.images) - 10 + i]
        x_input = torch.FloatTensor(scaler.transform(img.reshape(1, -1)))
        output = model(x_input)
        pred = torch.argmax(output).item()
        actual = digits.target[len(digits.target) - 10 + i]
        if pred == actual:
            correct += 1
        ax.imshow(img, cmap='gray')
        color = 'green' if pred == actual else 'red'
        ax.set_title(f'Pred:{pred} True:{actual}', color=color)
        ax.axis('off')
        status = "✓" if pred == actual else "✗"
        pred_text += f"      {status} Image {i+1}: predicted={pred}, actual={actual}\n"

plt.suptitle('Predictions (green=correct, red=wrong)', fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, "images", "10_pytorch_predictions.png"))
plt.close()

player.add_step("Step 7: Predict Individual Digits", f"""\
  {green('Last 10 digit predictions:')}
{pred_text}      Score: {correct}/{total}

  {green('Predictions saved to images/10_pytorch_predictions.png')}

  {cyan('The prediction pipeline for one image:')}
      1. Take 8×8 image → flatten to 64 numbers
      2. Scale with the same scaler used in training
      3. Convert to FloatTensor
      4. model(tensor) → 10 scores (one per digit)
      5. argmax → pick the digit with highest score""")

# ── Step 8 ──────────────────────────────────────────────

player.add_step("Step 8: Summary — sklearn vs PyTorch", f"""\
  {green('What you learned:')}

      {cyan('1. Tensors')}     = arrays with GPU + autograd support
      {cyan('2. nn.Module')}   = base class for building networks
      {cyan('3. nn.Linear')}   = one fully connected layer
      {cyan('4. Training loop')} = forward → loss → backward → update
      {cyan('5. eval/no_grad')} = switch to inference mode

  {green('sklearn vs PyTorch:')}
      ┌─────────────┬──────────────────┬──────────────────────┐
      │             │ sklearn          │ PyTorch              │
      ├─────────────┼──────────────────┼──────────────────────┤
      │ Code        │ 3 lines          │ 20+ lines            │
      │ Control     │ Limited          │ Full                 │
      │ GPU         │ No               │ Yes                  │
      │ Custom arch │ MLP only         │ Anything             │
      │ Use case    │ Prototyping      │ Research/Production  │
      └─────────────┴──────────────────┴──────────────────────┘

  {green('What\'s next?')}
      Lesson 11: build a TRANSFORMER from scratch — the architecture
      behind GPT and Claude.  Same PyTorch building blocks,
      arranged to process text instead of images.""")

# ── Play ────────────────────────────────────────────────

player.play()
