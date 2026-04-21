"""
Lesson 10: Intro to Deep Learning with PyTorch
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

a = torch.tensor([1, 2, 3])
b = torch.tensor([4, 5, 6])

x = torch.tensor(3.0, requires_grad=True)
y_val = x ** 2 + 2 * x + 1
y_val.backward()

player.add_step("Step 1: Tensors — PyTorch's Core", f"""\
  Tensors are like numpy arrays but with two superpowers:
    1. Can run on GPU (fast parallel computation)
    2. Track gradients automatically (for backpropagation)

  {cyan('Basic operations:')}
    a = torch.tensor([1, 2, 3])
    b = torch.tensor([4, 5, 6])
    a + b = {a + b}
    a * b = {a * b}    (element-wise)
    a @ b = {a @ b}           (dot product)

  {cyan('Autograd (automatic differentiation):')}
    x = torch.tensor(3.0, requires_grad=True)
    y = x² + 2x + 1       # y = {y_val.item()}
    y.backward()           # compute dy/dx
    x.grad = {x.grad.item()}            # dy/dx = 2x + 2 = 2(3) + 2 = 8

  {green('PyTorch version:')} {torch.__version__}
  {green('CUDA available:')} {torch.cuda.is_available()}""")

# ── Step 2 ──────────────────────────────────────────────

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

player.add_step("Step 2: Prepare Data", f"""\
  Same digits dataset as lesson 09 — now with PyTorch tensors.

  {cyan('Code:')}
    X_train_t = torch.FloatTensor(X_train)    # numpy → tensor
    y_train_t = torch.LongTensor(y_train)     # labels as integers

  {green('Shapes:')}
    Training: X={list(X_train_t.shape)}, y={list(y_train_t.shape)}
    Test:     X={list(X_test_t.shape)}, y={list(y_test_t.shape)}

  FloatTensor for input data (continuous values).
  LongTensor for labels (integer class IDs).""")

# ── Step 3 ──────────────────────────────────────────────


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

player.add_step("Step 3: Build a Neural Network", f"""\
  In scikit-learn: MLPClassifier(hidden_layer_sizes=(64, 32))
  In PyTorch: we define the architecture ourselves.

  {cyan('Code:')}
    class DigitNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer1 = nn.Linear(64, 64)   # 64 → 64
            self.layer2 = nn.Linear(64, 32)   # 64 → 32
            self.layer3 = nn.Linear(32, 10)   # 32 → 10
            self.relu = nn.ReLU()

        def forward(self, x):
            x = self.relu(self.layer1(x))     # layer + activation
            x = self.relu(self.layer2(x))
            x = self.layer3(x)                # raw scores
            return x

  {green('Model:')}
    Architecture: 64 → 64 → 32 → 10
    Total parameters: {n_params:,}
    ReLU activation: max(0, x) — adds non-linearity""")

# ── Step 4 ──────────────────────────────────────────────

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
        log_lines += f"    Epoch {epoch+1:3d}/100  loss={loss.item():.4f}  accuracy={acc:.4f}\n"

player.add_step("Step 4: The Training Loop", f"""\
  This is the core of deep learning — 4 steps repeated:

  {cyan('The loop:')}
    for epoch in range(100):
        outputs = model(X_train_t)        # 1. Forward pass
        loss = criterion(outputs, y_train) # 2. Compute loss
        optimizer.zero_grad()              # 3. Clear old gradients
        loss.backward()                    #    Compute new gradients
        optimizer.step()                   #    Update weights

  {green('Training progress:')}
{log_lines}
  Loss goes down, accuracy goes up → the network is learning!

  {cyan('Key components:')}
    CrossEntropyLoss: measures how wrong the predictions are
    Adam optimizer:   adjusts weights intelligently (adaptive learning rate)
    lr=0.001:         learning rate — how big each adjustment is""")

# ── Step 5 ──────────────────────────────────────────────

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

player.add_step("Step 5: Evaluate on Test Data", f"""\
  {cyan('Code:')}
    model.eval()                           # switch to evaluation mode
    with torch.no_grad():                  # don't track gradients
        test_outputs = model(X_test_t)
        _, predicted = torch.max(test_outputs, 1)

  {green('Result:')}
    Test accuracy: {test_acc:.4f}

  {green('Compare with lesson 09 (sklearn MLP):')}
    sklearn MLPClassifier: ~0.97
    PyTorch DigitNet:      {test_acc:.4f}

  Training plots saved to images/10_pytorch_training.png""")

# ── Step 6 ──────────────────────────────────────────────

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
        pred_text += f"    {status} Image {i+1}: predicted={pred}, actual={actual}\n"

plt.suptitle('Predictions (green=correct, red=wrong)', fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, "images", "10_pytorch_predictions.png"))
plt.close()

player.add_step("Step 6: Predictions & Course Summary", f"""\
  {green('Last 10 digit predictions:')}
{pred_text}    Score: {correct}/{total}

  Predictions saved to images/10_pytorch_predictions.png

  {green('Course Summary — What You Learned:')}
    01. NumPy:         Arrays and math operations
    02. Pandas:        Load and explore data
    03. Regression:    y = ax + b, R² score, train/test split
    04. Polynomial:    Curves, overfitting, cross-validation
    05. Classification: Categories, confusion matrix
    06. Preprocessing: Scaling, encoding, pipelines
    07. Trees/Forests: Ensembles, feature importance
    08. Clustering:    Unsupervised learning, K-Means
    09. Neural Nets:   MLP with scikit-learn
    10. PyTorch:       Build and train from scratch

  Next steps: CNNs (images), RNNs (sequences), Transformers (language)""")

# ── Play ────────────────────────────────────────────────

player.play()
