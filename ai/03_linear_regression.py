"""
Lesson 03: Linear Regression — The Complete Model

Goal: go from  y = a*x  (centered) to  y = a*x + b  (real-world),
      learn R² score, train/test split, and scikit-learn.

Type along:
    cd ai/
    python3
    >>> import numpy as np, pandas as pd
    >>> from sklearn.linear_model import LinearRegression
    >>> from sklearn.model_selection import train_test_split
    >>> df = pd.read_csv('sample.csv')
    >>> x, y = df['x'], df['y']
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

player = LessonPlayer("Lesson 03: Linear Regression — y = ax + b")

# ── Load data ───────────────────────────────────────────

df = pd.read_csv(os.path.join(BASE_DIR, 'sample.csv'))
x = df['x']
y = df['y']
x_mean = x.mean()
y_mean = y.mean()

# ── Step 1 ──────────────────────────────────────────────

player.add_step("Step 1: Why We Need an Intercept", f"""\
  {cyan('Lesson 01-02:')} y = a * x  (line passes through origin)
  {cyan('Problem:')}  real data rarely passes through (0, 0).

      y
      │         ·  · ·
      │      · ·/            ← y = ax + b  (with intercept)
      │    · · /
      │  · · /
    b │---/                  ← b = where line crosses y-axis
      │  /
      │/
      └──────────────── x
     (0,0)

  {cyan('The intercept (b)')} shifts the line up or down.
      y = ax       →  line through origin
      y = ax + b   →  line through (0, b)

  {green('In ML terms:')}
      a = weight  (how much x influences y)
      b = bias    (the baseline value when x = 0)

  Every neural network layer is this same formula:
      output = weight × input + bias""")

# ── Step 2 ──────────────────────────────────────────────

a = ((x - x_mean) * (y - y_mean)).sum() / ((x - x_mean) ** 2).sum()
b = y_mean - a * x_mean

player.add_step("Step 2: Compute Slope and Intercept", f"""\
  {cyan('Type:')}
      x_mean = x.mean()              →  {x_mean:.4f}
      y_mean = y.mean()              →  {y_mean:.4f}

      a = ((x - x_mean) * (y - y_mean)).sum() / ((x - x_mean)**2).sum()
      b = y_mean - a * x_mean

  {green('Result:')}
      a (slope)     = {a:.4f}
      b (intercept) = {b:.4f}

  {cyan('The formula for b:')}
      The best-fit line ALWAYS passes through (x_mean, y_mean).
      So:  y_mean = a * x_mean + b
      Therefore:  b = y_mean - a * x_mean
                    = {y_mean:.2f} - {a:.2f} × {x_mean:.2f}
                    = {b:.2f}

  {green('The complete model:')}
      y = {a:.2f} × x + ({b:.2f})""",

detail=f"""\
  {green('Term: Slope and Intercept')}

      Slope (a) = rise / run = Δy / Δx
          "For each 1-unit increase in x, y changes by {a:.2f}"

      Intercept (b) = the y value when x = 0
          "When x is zero, the predicted y is {b:.2f}"

  {green('Term: Weight and Bias (ML vocabulary)')}

      In machine learning, we say:
          weight = a = {a:.2f}     (same as slope)
          bias   = b = {b:.2f}    (same as intercept)

      These are the PARAMETERS of the model — the numbers
      the algorithm learned from data.

      A model with 1 weight + 1 bias = 2 parameters total.
      GPT-3 has 175 billion parameters (same concept, bigger).""")

# ── Step 3 ──────────────────────────────────────────────

y_pred = a * x + b
x_new = 40
y_new = a * x_new + b

player.add_step("Step 3: Make Predictions", f"""\
  {cyan('Type:')}
      y_pred = a * x + b              # predict for all training data
      y_pred[:5]  →  {(a * x + b)[:5].values.round(1)}

  {cyan('Predict for values we\'ve never seen:')}
      x_new = 40
      y_new = a * 40 + b
      y_new  →  {y_new:.1f}

  {green('More predictions — try these:')}
      a * 0 + b    →  {a * 0 + b:.1f}     (x=0: just the intercept)
      a * 10 + b   →  {a * 10 + b:.1f}    (x=10)
      a * 20 + b   →  {a * 20 + b:.1f}   (x=20)
      a * 50 + b   →  {a * 50 + b:.1f}   (x=50)
      a * 100 + b  →  {a * 100 + b:.1f}  (x=100)

  {green('Key point:')}
      The model never saw x=50 or x=100 in training.
      But it can predict because it learned the PATTERN,
      not the individual data points.""")

# ── Step 4 ──────────────────────────────────────────────

ss_res = ((y - y_pred) ** 2).sum()
ss_tot = ((y - y_mean) ** 2).sum()
r_squared = 1 - (ss_res / ss_tot)

player.add_step("Step 4: R² Score — How Good Is Our Model?", f"""\
  {cyan('The question:')} how much of y\'s variation does our model explain?

  {cyan('Type:')}
      ss_res = ((y - y_pred) ** 2).sum()      # error from our line
      ss_tot = ((y - y_mean) ** 2).sum()       # error from just using mean
      r_squared = 1 - (ss_res / ss_tot)
      r_squared  →  {r_squared:.4f}

  {green('What is R²?')}
      R² compares two strategies:
          Dumb strategy:  always predict y_mean for everything
          Our model:      predict y = ax + b

      R² = 1 - (our model's error / dumb strategy's error)

      R² = 1.0  →  our model is perfect (zero error)
      R² = 0.0  →  our model is no better than guessing the mean
      R² < 0.0  →  our model is WORSE than guessing (something's wrong)

  {green('Our result:')}
      R² = {r_squared:.4f}
      Our line explains {r_squared*100:.1f}% of y's variation.
      The remaining {(1-r_squared)*100:.1f}% is noise or non-linear patterns.""",

detail=f"""\
  {green('Breaking down R²')}

      ss_res = {ss_res:.2f}    (Sum of Squared Residuals — our model's error)
      ss_tot = {ss_tot:.2f}    (Total Sum of Squares — dumb strategy's error)

      R² = 1 - ({ss_res:.2f} / {ss_tot:.2f})
         = 1 - {ss_res/ss_tot:.4f}
         = {r_squared:.4f}

  {cyan('ASCII intuition:')}

      Using mean only:               Using our line:
      y                              y
      │   ·  ← far from mean        │   ·  ← close to line
      │───────── mean                │  /  · ← small error
      │ ·    ← far from mean        │ / ·
      │   ·  ← far from mean        │/·
      └──────── x                    └──────── x
      Large total error              Small residual error

  {green('Term: Residual')}
      residual = actual_y - predicted_y
      It's the vertical distance from a point to the line.
      Positive residual = point is ABOVE the line.
      Negative residual = point is BELOW the line.""")

# ── Step 5 ──────────────────────────────────────────────

X = x.values.reshape(-1, 1)
Y = y.values
model = LinearRegression()
model.fit(X, Y)

player.add_step("Step 5: scikit-learn — Same Thing in 3 Lines", f"""\
  Everything we did by hand, scikit-learn does in 3 lines.

  {cyan('Type:')}
      from sklearn.linear_model import LinearRegression
      X = x.values.reshape(-1, 1)     # sklearn needs 2D input: (n, 1)
      Y = y.values
      model = LinearRegression()
      model.fit(X, Y)

  {cyan('Then try:')}
      model.coef_[0]         →  {model.coef_[0]:.4f}    (our a = {a:.4f})
      model.intercept_       →  {model.intercept_:.4f}  (our b = {b:.4f})
      model.score(X, Y)      →  {model.score(X, Y):.4f}   (our R² = {r_squared:.4f})
      model.predict([[40]])  →  {model.predict([[40]])[0]:.1f}   (our prediction = {y_new:.1f})

  {green('Why use scikit-learn?')}
      - Less code (3 lines vs 10)
      - Handles edge cases and numerical stability
      - Same API for ALL models (swap LinearRegression for
        any other model and the code barely changes)
      - train/test split, cross-validation, pipelines built in""",

detail=f"""\
  {cyan('Why reshape(-1, 1)?')}

      Our x is a 1D array:   shape = (100,)
      scikit-learn expects 2D: shape = (100, 1)

      Because in general you have MULTIPLE features:
          1 feature:   X.shape = (100, 1)    ← this lesson
          4 features:  X.shape = (100, 4)    ← lesson 05 (iris)
          64 features: X.shape = (100, 64)   ← lesson 09 (digits)

      reshape(-1, 1) means "keep all rows, make 1 column"
          -1 = "figure out the row count automatically"
           1 = "exactly 1 column"

  {cyan('The scikit-learn API pattern (same for every model):')}
      model = SomeModel()         # create
      model.fit(X_train, y_train) # train
      model.predict(X_new)        # predict
      model.score(X_test, y_test) # evaluate

      This pattern works for LinearRegression, DecisionTree,
      RandomForest, NeuralNetwork — everything in scikit-learn.""")

# ── Step 6 ──────────────────────────────────────────────

X_train, X_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42
)
model2 = LinearRegression()
model2.fit(X_train, y_train)
train_score = model2.score(X_train, y_train)
test_score = model2.score(X_test, y_test)

player.add_step("Step 6: Train/Test Split — Honest Evaluation", f"""\
  {cyan('Problem:')} we trained and tested on the SAME data.
  That's like grading an exam with the exact questions you studied.

  {cyan('Solution:')} split data into training (80%) and test (20%).

  {cyan('Type:')}
      from sklearn.model_selection import train_test_split
      X_train, X_test, y_train, y_test = train_test_split(
          X, Y, test_size=0.2, random_state=42
      )
      model2 = LinearRegression()
      model2.fit(X_train, y_train)     # train on 80%

  {green('Result:')}
      Training samples: {len(X_train)}
      Test samples:     {len(X_test)}

      R² on training data: {train_score:.4f}   (how well it fits training data)
      R² on test data:     {test_score:.4f}   (how well it predicts NEW data)

  {green('What the gap tells you:')}
      Small gap  →  model generalizes well
      Large gap  →  model is overfitting (memorizing, not learning)""",

detail=f"""\
  {green('Term: Overfitting vs Underfitting')}

      {cyan('Underfitting:')} model is too simple to capture the pattern.
          - Bad on training data AND test data.
          - Fix: use a more complex model.

      {cyan('Good fit:')} model captures the real pattern.
          - Good on training data AND test data.
          - This is what we want.

      {cyan('Overfitting:')} model memorizes noise in the training data.
          - Great on training data, bad on test data.
          - Fix: simpler model, more data, or regularization.

      Accuracy     │   ╱ train ────────────
                   │  ╱       ╲
                   │ ╱   test  ╲────────
                   │╱
                   └─────────────────────
                    simple ──→ complex
                         (model complexity)

  {green('Term: Generalization')}
      The whole point of ML is to predict on data the model
      has NEVER seen.  This ability is called generalization.

      Train/test split measures generalization.
      If test score ≈ train score, the model generalizes well.""")

# ── Step 7 ──────────────────────────────────────────────

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

player.add_step("Step 7: Visualization — Regression + Residuals", f"""\
  Saved two plots to images/03_linear_regression.png

  {cyan('Left: Scatter plot + regression line')}
      Blue dots = actual data
      Red line  = y = {a:.1f}x + {b:.0f}

  {cyan('Right: Residual plot')}
      x-axis = predicted y
      y-axis = actual - predicted (the error)
      Red dashed line = zero error

  {green('How to read the residual plot:')}
      Random scatter around 0  →  model is appropriate
      A pattern (curve, fan)   →  model is missing something

      Good residual plot:       Bad residual plot:
       ·  ·    ·                     ·       ·
      ──·──·──·──·──            ──·──────·─────
        ·   ·   ·                  · · · ·
      (random = good)           (curve = need polynomial)

  The residual plot is the FIRST thing to check after fitting.
  Lesson 04 shows what happens when a line isn't enough.""")

# ── Step 8 ──────────────────────────────────────────────

player.add_step("Step 8: Summary — The Complete Linear Regression", f"""\
  {green('What you learned:')}

      {cyan('1. Model:')}      y = ax + b  (2 parameters: slope + intercept)
      {cyan('2. Training:')}   find a and b that minimize squared errors
      {cyan('3. R² score:')}   how much variation the model explains (0 to 1)
      {cyan('4. sklearn:')}    model.fit() / model.predict() / model.score()
      {cyan('5. Split:')}      train on 80%, test on 20% → honest evaluation

  {green('ML vocabulary recap:')}
      Feature   = input (x)          Weight    = slope (a)
      Label     = output (y)         Bias      = intercept (b)
      Training  = finding a, b       Inference = predicting new y
      R² score  = model quality      Overfit   = memorizing noise

  {green('What\'s next?')}
      Lesson 04: what if the data follows a CURVE, not a line?
      We'll fit polynomials (x², x³, ...) and learn about
      overfitting — the most important concept in ML.""")

# ── Play ────────────────────────────────────────────────

player.play()
