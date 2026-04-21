"""
Lesson 03: Linear Regression
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from lesson_player import LessonPlayer, cyan, green

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

player = LessonPlayer("Lesson 03: Linear Regression — from manual to scikit-learn")

# ── Load data ───────────────────────────────────────────

df = pd.read_csv(os.path.join(BASE_DIR, 'sample.csv'))
x = df['x']
y = df['y']
x_mean = x.mean()
y_mean = y.mean()

# ── Step 1 ──────────────────────────────────────────────

a = ((x - x_mean) * (y - y_mean)).sum() / ((x - x_mean) ** 2).sum()
b = y_mean - a * x_mean

player.add_step("Step 1: Full Linear Regression (y = ax + b)", f"""\
  In lessons 01-02, we only computed slope 'a' on centered data.
  Now we add intercept 'b' so predictions work on real data.

  {cyan('Formula:')}
    a = sum((xi - x̄)(yi - ȳ)) / sum((xi - x̄)²)
    b = ȳ - a * x̄

  {cyan('Code:')}
    a = ((x - x_mean) * (y - y_mean)).sum() / ((x - x_mean) ** 2).sum()
    b = y_mean - a * x_mean

  {green('Result:')}
    slope (a)     = {a:.2f}
    intercept (b) = {b:.2f}
    formula: y = {a:.2f} * x + ({b:.2f})""")

# ── Step 2 ──────────────────────────────────────────────

y_pred = a * x + b
x_new = 40
y_new = a * x_new + b

player.add_step("Step 2: Making Predictions", f"""\
  Now we can predict y for any x we've never seen before.

  {cyan('Code:')}
    y_pred = a * x + b            # predictions for all data
    y_new = a * 40 + b            # predict for x=40

  {green('Result:')}
    Prediction for x=40: y = {a:.2f} * 40 + ({b:.2f}) = {y_new:.0f}

  Examples:
    x=35 → y = {a * 35 + b:.0f}
    x=40 → y = {a * 40 + b:.0f}
    x=45 → y = {a * 45 + b:.0f}""")

# ── Step 3 ──────────────────────────────────────────────

ss_res = ((y - y_pred) ** 2).sum()
ss_tot = ((y - y_mean) ** 2).sum()
r_squared = 1 - (ss_res / ss_tot)

player.add_step("Step 3: R² Score — How Good Is the Line?", f"""\
  R² = 1 - (sum of squared residuals) / (sum of squared total)
    1.0 = perfect fit
    0.0 = no better than just guessing the mean

  {cyan('Code:')}
    ss_res = ((y - y_pred) ** 2).sum()    # error from our line
    ss_tot = ((y - y_mean) ** 2).sum()    # error from the mean
    r_squared = 1 - (ss_res / ss_tot)

  {green('Result:')}
    R² = {r_squared:.4f}

  Our line explains {r_squared*100:.1f}% of the variation in the data.
  The remaining {(1-r_squared)*100:.1f}% is noise or patterns a line can't capture.
  (That's why lesson 04 tries a curve instead.)""")

# ── Step 4 ──────────────────────────────────────────────

X = x.values.reshape(-1, 1)
Y = y.values
model = LinearRegression()
model.fit(X, Y)

player.add_step("Step 4: scikit-learn — Same Thing in 3 Lines", f"""\
  scikit-learn does all the math for us.

  {cyan('Code:')}
    from sklearn.linear_model import LinearRegression

    X = x.values.reshape(-1, 1)   # scikit-learn needs 2D input
    Y = y.values
    model = LinearRegression()
    model.fit(X, Y)               # compute a and b

  {green('Result:')}
    slope (coef)  = {model.coef_[0]:.2f}       (same as our a!)
    intercept     = {model.intercept_:.2f}  (same as our b!)
    R² score      = {model.score(X, Y):.4f}     (same R²!)
    Predict(x=40) = {model.predict([[40]])[0]:.0f}

  Manual math and scikit-learn give the exact same answer.""")

# ── Step 5 ──────────────────────────────────────────────

X_train, X_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42
)
model2 = LinearRegression()
model2.fit(X_train, y_train)
train_score = model2.score(X_train, y_train)
test_score = model2.score(X_test, y_test)

player.add_step("Step 5: Train/Test Split", f"""\
  Problem: we tested on the SAME data we trained on.
  That's like grading a test with the exact same questions you studied.

  Solution: split data into training set (80%) and test set (20%).

  {cyan('Code:')}
    X_train, X_test, y_train, y_test = train_test_split(
        X, Y, test_size=0.2, random_state=42
    )
    model2 = LinearRegression()
    model2.fit(X_train, y_train)

  {green('Result:')}
    Training samples: {len(X_train)}
    Test samples:     {len(X_test)}

    R² on training data: {train_score:.4f}
    R² on test data:     {test_score:.4f}

  Test R² is lower — the model is slightly overfit.
  If the gap is huge, the model memorized instead of learning.""")

# ── Step 6 ──────────────────────────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].scatter(x, y, alpha=0.6, label='Actual data')
axes[0].plot(x.sort_values(), a * x.sort_values() + b, color='red',
             label=f'y = {a:.1f}x + {b:.0f}')
axes[0].set_xlabel('x')
axes[0].set_ylabel('y')
axes[0].set_title('Linear Regression')
axes[0].legend()
residuals = y - y_pred
axes[1].scatter(y_pred, residuals, alpha=0.6)
axes[1].axhline(y=0, color='red', linestyle='--')
axes[1].set_xlabel('Predicted y')
axes[1].set_ylabel('Residual (actual - predicted)')
axes[1].set_title('Residual Plot')
plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, "images", "03_linear_regression.png"))
plt.close()

player.add_step("Step 6: Visualization", f"""\
  Two plots saved to images/03_linear_regression.png:

  Left: Scatter plot + regression line
    - Blue dots = actual data
    - Red line  = our prediction line y = {a:.1f}x + {b:.0f}

  Right: Residual plot (prediction errors)
    - Each dot = (predicted value, actual - predicted)
    - Points near the red line (y=0) = good predictions
    - Random scatter = model is reasonable
    - Patterns in residuals = model is missing something

  {green('Summary of Lesson 03:')}
    ✓ Full formula: y = {a:.2f}x + {b:.2f}
    ✓ R² = {r_squared:.4f} ({r_squared*100:.1f}% of variation explained)
    ✓ scikit-learn confirms our manual math
    ✓ Train/test split validates on unseen data""")

# ── Play ────────────────────────────────────────────────

player.play()
