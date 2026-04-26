"""
Lesson 04: Polynomial Regression & Overfitting

Goal: understand why straight lines aren't always enough,
      how to fit curves, and the critical concept of overfitting.

Type along:
    cd ai/
    python3
    >>> import numpy as np
    >>> from sklearn.linear_model import LinearRegression
    >>> from sklearn.preprocessing import PolynomialFeatures
    >>> from sklearn.pipeline import make_pipeline
    >>> from sklearn.model_selection import cross_val_score
    >>> np.random.seed(42)
    >>> X = np.linspace(-3, 3, 30).reshape(-1, 1)
    >>> y = 0.5 * X.ravel() ** 2 + X.ravel() + 2 + np.random.randn(30) * 1.5
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

# ── Generate data ──────────────────────────────────────

np.random.seed(42)
X = np.linspace(-3, 3, 30).reshape(-1, 1)
y_true = 0.5 * X.ravel() ** 2 + X.ravel() + 2
y = y_true + np.random.randn(30) * 1.5

# ── Step 1 ──────────────────────────────────────────────

lin = LinearRegression()
lin.fit(X, y)
lin_r2 = lin.score(X, y)

player.add_step("Step 1: When a Straight Line Fails", f"""\
  {cyan('The data:')}
      True pattern: y = 0.5x² + x + 2  (a CURVE, not a line)
      We added random noise to simulate real-world data.

  {cyan('Try fitting a straight line:')}
      from sklearn.linear_model import LinearRegression
      model = LinearRegression()
      model.fit(X, y)
      model.score(X, y)  →  {lin_r2:.4f}

  {green('R² = {lin_r2:.4f} — that\'s low.')}  The line misses the curve:

      y
      │                  · ·
      │              · ·         ← data curves up
      │          · ·
      │      · · ──────────────  ← straight line misses it
      │  · ·
      │· ·
      └──────────────────── x

  {cyan('The problem:')} a straight line (degree 1) can't bend.
  {cyan('The solution:')} give the model higher-degree terms: x², x³, ...""")

# ── Step 2 ──────────────────────────────────────────────

pf = PolynomialFeatures(degree=2)
X_poly_example = pf.fit_transform(np.array([[2], [3]]))

player.add_step("Step 2: PolynomialFeatures — Creating Curves", f"""\
  {cyan('What PolynomialFeatures does:')}
      Turns [x] into [1, x, x², x³, ...] depending on degree.

  {cyan('Type:')}
      from sklearn.preprocessing import PolynomialFeatures
      pf = PolynomialFeatures(degree=2)
      pf.fit_transform([[2], [3]])

  {green('Result:')}
      Input:  [[2],     →  Output:  [[1, 2, 4],       ← [1, x, x²]
               [3]]                  [1, 3, 9]]       ← [1, x, x²]

  {cyan('Why does this help?')}
      Linear regression finds: y = a₀ + a₁x + a₂x²
      Even though the ALGORITHM is still "linear regression",
      the MODEL is now a polynomial (a curve).

      The trick: treat x² as a NEW feature.
      LinearRegression sees 2 features: x and x².
      It doesn't know x² came from x — it just finds
      the best weights for both.

  {green('Term: Feature Engineering')}
      Creating new features from existing ones (like x → x²).
      This is one of the most important skills in ML.""")

# ── Step 3 ──────────────────────────────────────────────

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
    lines += f"      Degree {d:2d}: R² = {scores[d]:.4f}\n"

player.add_step("Step 3: Fit Curves of Different Degrees", f"""\
  {cyan('Type:')}
      from sklearn.pipeline import make_pipeline
      for degree in [1, 2, 5, 15]:
          model = make_pipeline(PolynomialFeatures(degree),
                                LinearRegression())
          model.fit(X, y)
          print(degree, model.score(X, y))

  {green('Training R² for each degree:')}
{lines}
  {green('Plot saved to images/04_polynomial_regression.png')}

  {cyan('What happened:')}
      Degree 1:  straight line — misses the curve entirely
      Degree 2:  parabola — matches the true shape
      Degree 5:  starts wiggling to hit individual points
      Degree 15: wild oscillations — passes through every point

  Notice R² goes UP with complexity.  Degree 15 looks perfect!
  But is it actually better?  No — it's overfitting.""")

# ── Step 4 ──────────────────────────────────────────────

player.add_step("Step 4: Overfitting — The Central Problem of ML", f"""\
  {green('This is the MOST IMPORTANT concept in machine learning.')}

  {cyan('Underfitting')} (degree 1):
      Model too simple → can't capture the real pattern.
      Bad on training data, bad on new data.
      Like studying only chapter 1 for a 10-chapter exam.

  {cyan('Good fit')} (degree 2):
      Model complexity matches the true pattern.
      Good on training data, good on new data.
      Like studying the right material.

  {cyan('Overfitting')} (degree 15):
      Model too complex → memorizes noise instead of learning.
      PERFECT on training data, TERRIBLE on new data.
      Like memorizing exact exam answers instead of understanding.

      Training data (seen):     New data (unseen):
      R² = 1.000 (perfect!)    R² = -5.3 (catastrophic!)

  {green('The goal of ML:')}
      Find the sweet spot between underfitting and overfitting.
      You want the model to learn the PATTERN, not the NOISE.

  {green('How do you detect overfitting?')}
      Compare training score vs test score.
      Big gap = overfitting.  That's why train/test split exists.""")

# ── Step 5 ──────────────────────────────────────────────

cv_results = ""
for degree in [1, 2, 3, 5, 10, 15]:
    model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
    cv = cross_val_score(model, X, y, cv=5, scoring='r2')
    model.fit(X, y)
    train_r2 = model.score(X, y)
    marker = "  ← best" if degree == 2 else ""
    cv_results += f"      Degree {degree:2d}: train={train_r2:.4f}  CV={cv.mean():+.4f}{marker}\n"

player.add_step("Step 5: Cross-Validation — Smarter Than Train/Test", f"""\
  {cyan('Problem with train/test split:')}
      One random split might be lucky or unlucky.
      Different splits → different scores.

  {cyan('Cross-validation:')} try MULTIPLE splits and average.
      Fold 1: [TEST][train][train][train][train]
      Fold 2: [train][TEST][train][train][train]
      Fold 3: [train][train][TEST][train][train]
      Fold 4: [train][train][train][TEST][train]
      Fold 5: [train][train][train][train][TEST]
      Final score = average of all 5 test scores.

  {cyan('Type:')}
      from sklearn.model_selection import cross_val_score
      model = make_pipeline(PolynomialFeatures(2), LinearRegression())
      scores = cross_val_score(model, X, y, cv=5, scoring='r2')
      scores.mean()

  {green('Train R² vs Cross-Validated R²:')}
{cv_results}
  {green('Key observation:')}
      Train R² goes UP with degree (always).
      CV R² peaks at degree 2, then COLLAPSES.
      Degree 15: train=1.0 but CV is negative → severe overfitting.

  {green('Rule:')} always evaluate with CV or a held-out test set,
  NEVER with training score alone.""",

detail=f"""\
  {green('Term: Bias-Variance Tradeoff')}

      {cyan('Bias')} = error from wrong assumptions (model too simple).
          High bias → underfitting → misses the real pattern.

      {cyan('Variance')} = error from sensitivity to training data (model too complex).
          High variance → overfitting → fits noise, not signal.

      Total error = bias² + variance + irreducible noise

      As model complexity increases:
          bias ↓ (fits better)
          variance ↑ (more sensitive to specific data)

      Error    │
               │╲  bias
               │  ╲          ╱ variance
               │    ╲      ╱
               │      ╲  ╱
               │        ⊗  ← sweet spot
               └─────────────────
                simple ──→ complex

      The sweet spot minimizes TOTAL error.
      Cross-validation finds this sweet spot empirically.""")

# ── Step 6 ──────────────────────────────────────────────

player.add_step("Step 6: Summary — Overfitting is the Enemy", f"""\
  {green('What you learned:')}

      {cyan('1. PolynomialFeatures')}
         Turns x into [1, x, x², ...] — lets linear regression fit curves.

      {cyan('2. Underfitting vs Overfitting')}
         Too simple → misses patterns.  Too complex → memorizes noise.

      {cyan('3. Training score lies')}
         Higher training score doesn't mean a better model.
         Always use test score or cross-validation.

      {cyan('4. Cross-validation')}
         Split data K ways, average the scores.
         More reliable than a single train/test split.

  {green('Rules of thumb:')}
      - Simple model that works > complex model that overfits
      - If train >> test score, you're overfitting
      - More data reduces overfitting (harder to memorize)
      - Regularization also helps (lesson 06)

  {green('What\'s next?')}
      Lesson 05: switch from predicting NUMBERS to predicting
      CATEGORIES (classification). Same ideas, different output.""")

# ── Play ────────────────────────────────────────────────

player.play()
