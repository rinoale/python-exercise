"""
Lesson 05: Classification — Predicting Categories

Goal: switch from regression (predict a number) to classification
      (predict a category). Learn logistic regression, accuracy,
      confusion matrix, precision/recall.

Type along:
    python3
    >>> from sklearn.datasets import load_iris
    >>> from sklearn.model_selection import train_test_split
    >>> iris = load_iris(); X, y = iris.data, iris.target
    >>> X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
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

player = LessonPlayer("Lesson 05: Classification — Predicting Categories")

# ── Step 1 ──────────────────────────────────────────────

player.add_step("Step 1: Regression vs Classification", f"""\
  {cyan('Regression')} (lessons 01-04):
      Input → predict a NUMBER
      "Given house size, predict the price"
      Output: 137500, 93000, 210000, ...

  {cyan('Classification')} (this lesson):
      Input → predict a CATEGORY
      "Given flower measurements, predict the species"
      Output: setosa, versicolor, or virginica

  {green('The key difference:')}
      Regression:      y ∈ ℝ  (any real number)
      Classification:  y ∈ {{0, 1, 2, ...}}  (discrete categories)

  {cyan('Real-world classification examples:')}
      Email         → spam / not spam        (2 classes)
      Image         → cat / dog / bird       (3 classes)
      Digit image   → 0, 1, 2, ..., 9       (10 classes)
      Medical scan  → benign / malignant     (2 classes)

  {green('Term: Binary vs Multi-class')}
      2 categories  = binary classification
      3+ categories = multi-class classification""")

# ── Step 2 ──────────────────────────────────────────────

iris = load_iris()
X = iris.data
y = iris.target

samples = ""
for i in range(3):
    samples += f"      {X[i]} → {iris.target_names[y[i]]}\n"
for i in [50, 100]:
    samples += f"      {X[i]} → {iris.target_names[y[i]]}\n"

player.add_step("Step 2: The Iris Dataset", f"""\
  {cyan('Type:')}
      from sklearn.datasets import load_iris
      iris = load_iris()
      X = iris.data        # measurements (features)
      y = iris.target       # species (labels)

  {green('What is it?')}
      150 flowers.  For each flower, 4 measurements:
      {iris.feature_names}

      3 species (our categories to predict):
      {list(iris.target_names)}

  {green('Sample data:')}
      [sepal_len, sepal_wid, petal_len, petal_wid] → species
{samples}
  {cyan('Then try:')}
      X.shape        →  {X.shape}     (150 samples, 4 features)
      y.shape        →  {y.shape}       (150 labels)
      np.unique(y)   →  {np.unique(y)}  (3 classes: 0, 1, 2)

  {green('Term: Feature')}
      Each column of X is a feature (sepal length, petal width, etc).
      The model uses ALL features together to make its prediction.""")

# ── Step 3 ──────────────────────────────────────────────

player.add_step("Step 3: How Logistic Regression Works", f"""\
  {cyan('Despite the name, logistic regression is a CLASSIFIER.')}

  {cyan('Idea:')}
      1. Compute a score for each class:
         score = w₁×sepal_len + w₂×sepal_wid + w₃×petal_len + w₄×petal_wid + bias

      2. Convert scores to probabilities (using the sigmoid/softmax function):
         probabilities must sum to 1.0:
         P(setosa)=0.95, P(versicolor)=0.04, P(virginica)=0.01

      3. Pick the class with highest probability:
         prediction = setosa

  {green('Sigmoid function (for 2 classes):')}
      Converts any number to a value between 0 and 1.

      P(class=1) = 1 / (1 + e^(-score))

      score = -5  →  P ≈ 0.007   (almost certainly class 0)
      score =  0  →  P = 0.5     (uncertain — coin flip)
      score = +5  →  P ≈ 0.993   (almost certainly class 1)

              1.0 │         ──────────
                  │        ╱
              0.5 │───────●───────────
                  │      ╱
              0.0 │─────────
                  └───────────────────
                  -5    0    +5   score

  {green('Term: Softmax')} (for 3+ classes)
      Generalizes sigmoid to multiple classes.
      Turns a vector of scores into probabilities that sum to 1.""")

# ── Step 4 ──────────────────────────────────────────────

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

proba = model.predict_proba(X_test[:3])
proba_text = ""
for i in range(3):
    proba_text += f"      Sample {i+1}: [{proba[i][0]:.3f}, {proba[i][1]:.3f}, {proba[i][2]:.3f}]"
    proba_text += f"  → predict {iris.target_names[y_pred[i]]}"
    proba_text += f"  (actual: {iris.target_names[y_test[i]]})\n"

player.add_step("Step 4: Train and Predict", f"""\
  {cyan('Type:')}
      from sklearn.linear_model import LogisticRegression
      from sklearn.model_selection import train_test_split
      from sklearn.metrics import accuracy_score

      X_train, X_test, y_train, y_test = train_test_split(
          X, y, test_size=0.3, random_state=42
      )
      model = LogisticRegression(max_iter=200)
      model.fit(X_train, y_train)

      y_pred = model.predict(X_test)
      accuracy_score(y_test, y_pred)  →  {acc:.4f}

  {green('Accuracy: {acc:.1%}')}  ({int(acc * len(y_test))}/{len(y_test)} correct)

  {cyan('Peek at probabilities (model.predict_proba):')}
      Columns: [setosa,   versicolor, virginica]
{proba_text}
  {green('The model doesn\'t just say "setosa".')}
  It says "95% setosa, 4% versicolor, 1% virginica" and picks
  the highest.  This confidence score is useful in practice.""")

# ── Step 5 ──────────────────────────────────────────────

cm = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=iris.target_names)

cm_text = ""
for i, name in enumerate(iris.target_names):
    cm_text += f"      {name:>12s}  {cm[i]}\n"

player.add_step("Step 5: Confusion Matrix — Where Errors Happen", f"""\
  {cyan('Accuracy alone can be misleading.')}
  Example: 99% of emails are not spam.  A model that always says
  "not spam" gets 99% accuracy but catches ZERO spam.

  {cyan('Type:')}
      from sklearn.metrics import confusion_matrix
      cm = confusion_matrix(y_test, y_pred)

  {green('Confusion Matrix:')}
      Rows = ACTUAL class,  Columns = PREDICTED class
                    [setosa versi  virgin]
{cm_text}
  {cyan('How to read it:')}
      Diagonal = correct predictions (the model got it right)
      Off-diagonal = errors (confused one species for another)

      cm[0][0]={cm[0][0]}: actual setosa, predicted setosa → correct
      cm[1][2]={cm[1][2]}: actual versicolor, predicted virginica → error

  {green('ASCII matrix:')}
                  Predicted
                  set  ver  vir
      Actual set [ {cm[0][0]:2d}   {cm[0][1]:2d}   {cm[0][2]:2d} ]
           ver   [ {cm[1][0]:2d}   {cm[1][1]:2d}   {cm[1][2]:2d} ]
           vir   [ {cm[2][0]:2d}   {cm[2][1]:2d}   {cm[2][2]:2d} ]
                    ↑ diagonal = correct""")

# ── Step 6 ──────────────────────────────────────────────

player.add_step("Step 6: Precision, Recall, F1", f"""\
  {cyan('Type:')}
      from sklearn.metrics import classification_report
      print(classification_report(y_test, y_pred,
                                  target_names=iris.target_names))

  {green('Result:')}
{report}
  {cyan('What these mean:')}

      {green('Precision:')} "Of everything I predicted as X, how many were actually X?"
          If I predicted 20 emails as spam, and 18 were really spam:
          precision = 18/20 = 0.90

      {green('Recall:')} "Of all actual X, how many did I find?"
          If there were 25 spam emails, and I found 18:
          recall = 18/25 = 0.72

      {green('F1-score:')} harmonic mean of precision and recall.
          Balances both.  Useful when classes are imbalanced.

  {cyan('When to care about which:')}
      Medical diagnosis → high recall (don't miss cancer)
      Spam filter       → high precision (don't block real email)""")

# ── Step 7 ──────────────────────────────────────────────

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

player.add_step("Step 7: Decision Boundaries — What the Model Sees", f"""\
  {cyan('Type:')}
      X_2d = X[:, 2:4]    # petal length and petal width only
      X_train_2d, X_test_2d, y_train_2d, y_test_2d = train_test_split(
          X_2d, y, test_size=0.3, random_state=42)
      model_2d = LogisticRegression(max_iter=200)
      model_2d.fit(X_train_2d, y_train_2d)
      model_2d.score(X_test_2d, y_test_2d)  →  {score_2d:.4f}

  {green('Plot saved to images/05_classification.png')}

  {cyan('What the plot shows:')}
      The model divides the 2D space into 3 colored REGIONS.
      Any new flower that falls in the yellow region → predict setosa.
      The BOUNDARIES between regions are the decision boundaries.

  {green('ASCII version:')}
      petal_width
      2.5 │          ████████████
          │       ███ virginica ███
      1.5 │    ███████████████████
          │ ▒▒▒ versicolor ▒▒▒▒▒▒
      0.5 │▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒
          │░░░ setosa ░░░░░
      0.0 └────────────────────────
          0    2    4    6   petal_length

  {green('Key insight:')} classification = drawing boundaries in feature space.""")

# ── Step 8 ──────────────────────────────────────────────

player.add_step("Step 8: Summary", f"""\
  {green('What you learned:')}

      {cyan('1. Classification')} predicts categories, not numbers.
         Output = one of {{setosa, versicolor, virginica}}.

      {cyan('2. Logistic Regression')} computes score → probability → class.
         Uses sigmoid/softmax to convert scores to probabilities.

      {cyan('3. Accuracy')} = correct / total.  Simple but can be misleading.

      {cyan('4. Confusion Matrix')} shows which classes get mixed up.
         Diagonal = correct.  Off-diagonal = errors.

      {cyan('5. Precision')} = "of my predictions for X, how many were right?"
         {cyan('Recall')}    = "of all actual X, how many did I find?"
         {cyan('F1')}        = balance of both.

      {cyan('6. Decision Boundaries')} = lines/curves that separate classes.

  {green('ML vocabulary added:')}
      Classification, binary, multi-class, sigmoid, softmax,
      probability, decision boundary, precision, recall, F1.

  {green('What\'s next?')}
      Lesson 06: real data is messy — missing values, different scales,
      text categories.  Preprocessing fixes these before training.""")

# ── Play ────────────────────────────────────────────────

player.play()
