"""
Lesson 04: Polynomial Regression & Overfitting
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score
from lesson_player import LessonPlayer, cyan, green

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

player = LessonPlayer("Lesson 04: Polynomial Regression & Overfitting")

# ── Step 1 ──────────────────────────────────────────────

np.random.seed(42)
X = np.linspace(-3, 3, 30).reshape(-1, 1)
y_true = 0.5 * X.ravel() ** 2 + X.ravel() + 2
y = y_true + np.random.randn(30) * 1.5

player.add_step("Step 1: When a Line Isn't Enough", f"""\
  Lesson 03 used a straight line: y = ax + b
  But what if the data follows a curve?

  {cyan('Code:')}
    # True pattern: y = 0.5x² + x + 2
    X = np.linspace(-3, 3, 30)
    y_true = 0.5 * X² + X + 2
    y = y_true + noise

  {green('Data:')}
    30 points following a quadratic curve + random noise
    X range: [{X.min():.1f}, {X.max():.1f}]
    y range: [{y.min():.1f}, {y.max():.1f}]

  A straight line will miss the curve → low R².
  Solution: fit a polynomial (x², x³, ...) instead.""")

# ── Step 2 ──────────────────────────────────────────────

degrees = [1, 2, 5, 15]
X_plot = np.linspace(-3, 3, 100).reshape(-1, 1)
scores = {}

fig, axes = plt.subplots(1, 4, figsize=(20, 4))
for ax, degree in zip(axes, degrees):
    model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
    model.fit(X, y)
    y_plot = model.predict(X_plot)
    scores[degree] = model.score(X, y)
    ax.scatter(X, y, alpha=0.6, s=20)
    ax.plot(X_plot, y_plot, color='red')
    ax.plot(X_plot, 0.5 * X_plot.ravel() ** 2 + X_plot.ravel() + 2,
            color='green', linestyle='--', alpha=0.5, label='true curve')
    ax.set_title(f'Degree {degree}  R²={scores[degree]:.3f}')
    ax.set_ylim(-5, 20)
    ax.legend(fontsize=8)
plt.suptitle('Underfitting vs Overfitting', fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, "images", "04_polynomial_regression.png"))
plt.close()

lines = ""
for d in degrees:
    lines += f"    Degree {d:2d}: R² = {scores[d]:.4f}\n"

player.add_step("Step 2: Fitting Curves of Different Degrees", f"""\
  PolynomialFeatures turns [x] into [1, x, x², x³, ...].
  Then LinearRegression fits a curve instead of a line.

  {cyan('Code:')}
    model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
    model.fit(X, y)

  {green('Train R² for each degree:')}
{lines}
  degree=1: straight line → underfitting (misses the curve)
  degree=2: quadratic → matches the true pattern
  degree=15: wiggly → memorizes noise (overfitting!)

  Plot saved to images/04_polynomial_regression.png""")

# ── Step 3 ──────────────────────────────────────────────

player.add_step("Step 3: Underfitting vs Overfitting", f"""\
  {cyan('Underfitting')} (degree 1):
    - Model is too simple to capture the pattern
    - Bad on training data AND new data
    - Like studying only chapter 1 for a 10-chapter exam

  {green('Good fit')} (degree 2):
    - Model complexity matches the true pattern
    - Good on training data AND new data
    - Like studying the right material

  {cyan('Overfitting')} (degree 15):
    - Model is too complex, memorizes noise
    - Perfect on training data, terrible on new data
    - Like memorizing answers instead of understanding

  The goal of ML: find the sweet spot between under and overfitting.""")

# ── Step 4 ──────────────────────────────────────────────

cv_results = ""
for degree in [1, 2, 3, 5, 10, 15]:
    model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
    cv = cross_val_score(model, X, y, cv=5, scoring='r2')
    model.fit(X, y)
    train_r2 = model.score(X, y)
    marker = "  ← best" if degree == 2 else ""
    cv_results += f"    Degree {degree:2d}: train={train_r2:.4f}  CV={cv.mean():.4f}{marker}\n"

player.add_step("Step 4: Cross-Validation — Detect Overfitting", f"""\
  Instead of one train/test split, cross-validation does 5 splits:
    Fold 1: [TEST][train][train][train][train]
    Fold 2: [train][TEST][train][train][train]
    ...and averages the scores. More reliable.

  {cyan('Code:')}
    from sklearn.model_selection import cross_val_score
    cv_scores = cross_val_score(model, X, y, cv=5, scoring='r2')

  {green('Result (train R² vs cross-validated R²):')}
{cv_results}
  Train R² goes up with complexity, but CV R² peaks at degree 2.
  Higher degrees: train looks great, CV collapses → overfitting.

  {green('Key lesson:')} always evaluate with cross-validation, not train score.""")

# ── Play ────────────────────────────────────────────────

player.play()
