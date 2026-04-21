"""
Lesson 05: Logistic Regression & Classification
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from lesson_player import LessonPlayer, cyan, green

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

player = LessonPlayer("Lesson 05: Logistic Regression & Classification")

# ── Step 1 ──────────────────────────────────────────────

player.add_step("Step 1: Regression vs Classification", f"""\
  {cyan('Regression')} (lessons 01-04):
    Input → predict a NUMBER
    Example: area → price (137500, 93000, ...)

  {cyan('Classification')} (this lesson):
    Input → predict a CATEGORY
    Example: flower measurements → species (setosa, versicolor, virginica)

  Despite the name, "logistic regression" is a classification algorithm.
  It outputs probabilities (0.0 to 1.0) for each category.""")

# ── Step 2 ──────────────────────────────────────────────

iris = load_iris()
X = iris.data
y = iris.target

samples = ""
for i in range(5):
    samples += f"    {X[i]} → {iris.target_names[y[i]]}\n"

player.add_step("Step 2: The Iris Dataset", f"""\
  A classic ML dataset: 150 flowers, 4 measurements, 3 species.

  {cyan('Code:')}
    from sklearn.datasets import load_iris
    iris = load_iris()

  {green('Result:')}
    Samples:  {X.shape[0]}
    Features: {X.shape[1]} — {iris.feature_names}
    Classes:  {list(iris.target_names)}

  First 5 samples:
{samples}
  Each row has 4 numbers → model predicts the species.""")

# ── Step 3 ──────────────────────────────────────────────

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

player.add_step("Step 3: Train a Classifier", f"""\
  {cyan('Code:')}
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3
    )
    model = LogisticRegression(max_iter=200)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

  {green('Result:')}
    First 10 predictions: {y_pred[:10]}
    First 10 actual:      {y_test[:10]}

    Accuracy: {acc:.4f} ({int(acc * len(y_test))}/{len(y_test)} correct)""")

# ── Step 4 ──────────────────────────────────────────────

cm = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=iris.target_names)

cm_text = ""
for i, name in enumerate(iris.target_names):
    cm_text += f"    {name:>12s}  {cm[i]}\n"

player.add_step("Step 4: Confusion Matrix & Report", f"""\
  Accuracy alone can be misleading. The confusion matrix shows
  which categories get mixed up.

  {green('Confusion Matrix:')}
    Rows = actual, Columns = predicted
                  [setosa versi virgin]
{cm_text}
  Diagonal = correct predictions. Off-diagonal = errors.

  {green('Classification Report:')}
{report}
  precision: of all predicted X, how many were actually X?
  recall:    of all actual X, how many did we find?
  f1-score:  harmonic mean of precision and recall""")

# ── Step 5 ──────────────────────────────────────────────

X_2d = X[:, 2:4]
X_train_2d, X_test_2d, y_train_2d, y_test_2d = train_test_split(
    X_2d, y, test_size=0.3, random_state=42
)
model_2d = LogisticRegression(max_iter=200)
model_2d.fit(X_train_2d, y_train_2d)
score_2d = model_2d.score(X_test_2d, y_test_2d)

x_min, x_max = X_2d[:, 0].min() - 0.5, X_2d[:, 0].max() + 0.5
y_min, y_max = X_2d[:, 1].min() - 0.5, X_2d[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                      np.linspace(y_min, y_max, 200))
Z = model_2d.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

fig, ax = plt.subplots(figsize=(8, 6))
ax.contourf(xx, yy, Z, alpha=0.3, cmap='viridis')
scatter = ax.scatter(X_2d[:, 0], X_2d[:, 1], c=y, cmap='viridis',
                     edgecolors='black', s=50)
ax.set_xlabel('Petal Length (cm)')
ax.set_ylabel('Petal Width (cm)')
ax.set_title('Logistic Regression Decision Boundaries')
plt.colorbar(scatter, label='Species (0=setosa, 1=versicolor, 2=virginica)')
plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, "images", "05_classification.png"))
plt.close()

player.add_step("Step 5: Visualizing Decision Boundaries", f"""\
  Using only 2 features (petal length & width) we can see how
  the model divides the space into regions.

  {cyan('Code:')}
    X_2d = X[:, 2:4]    # just petal length & width
    model_2d.fit(X_train_2d, y_train_2d)

  {green('Result:')}
    2D model accuracy: {score_2d:.4f}
    (Lower than 4-feature model because we dropped 2 features)

  Plot saved to images/05_classification.png
    - Each colored region = model predicts that species
    - Dots = actual data points
    - Dots matching their region's color = correct

  {green('Key lesson:')} classification predicts categories, not numbers.
  The model learns boundaries that separate different classes.""")

# ── Play ────────────────────────────────────────────────

player.play()
