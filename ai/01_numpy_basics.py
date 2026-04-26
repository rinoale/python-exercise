"""
Lesson 01: NumPy Basics — Your First Machine Learning Model

Goal: teach a computer to find a pattern in data using nothing but
      basic math operations in NumPy.

Open a Python REPL alongside this lesson and type every code block yourself.
    python3
    >>> import numpy as np
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from lesson_player import LessonPlayer, bold, cyan, green

player = LessonPlayer("Lesson 01: NumPy Basics — Your First ML Model")

# ── Step 1 ──────────────────────────────────────────────

player.add_step("Step 1: What is Machine Learning?", f"""\
  {cyan('Machine learning = finding patterns in data automatically.')}

  Traditional programming:
      human writes rules  →  program follows rules  →  output

  Machine learning:
      human gives data    →  algorithm finds rules   →  output

  {green('Example:')}
      Data:    x=1 → y=2,   x=2 → y=3.9,   x=3 → y=6.1
      Pattern: y ≈ 2 * x
      Predict: x=4 → y ≈ ?

  The algorithm doesn't know y = 2x.  It has to DISCOVER that
  relationship by looking at the numbers.

  {cyan('This lesson:')} we will build this algorithm step by step,
  from raw data to a prediction — using only basic arithmetic.""")

# ── Step 2 ──────────────────────────────────────────────

x = np.array([1, 2, 3])
y = np.array([2, 3.9, 6.1])

player.add_step("Step 2: NumPy Arrays — Why Not Plain Python Lists?", f"""\
  {cyan('Type this in your Python REPL:')}
      import numpy as np
      x = np.array([1, 2, 3])
      y = np.array([2, 3.9, 6.1])

  {green('Check the results:')}
      x          →  {x}
      y          →  {y}
      type(x)    →  {type(x).__name__}

  {cyan('Why NumPy instead of Python lists?')}
      # Python list — this FAILS:
      [1, 2, 3] * [4, 5, 6]           →  TypeError!

      # NumPy array — element-wise math works:
      np.array([1,2,3]) * np.array([4,5,6])  →  {np.array([1,2,3]) * np.array([4,5,6])}

  NumPy lets you do math on entire arrays at once.
  This is called {cyan('vectorization')} — it's also 10-100x faster than
  looping through elements one by one (the loop happens in C).""",

detail=f"""\
  {cyan('Key NumPy operations — try each one:')}
      x + y       →  {x + y}           (add matching elements)
      x * y       →  {x * y}         (multiply matching elements)
      x ** 2      →  {x ** 2}           (square each element)
      x.sum()     →  {x.sum()}                 (sum all elements)
      x.mean()    →  {x.mean()}               (average)
      x.shape     →  {x.shape}              (dimensions: 3 elements)
      len(x)      →  {len(x)}                 (same as shape[0])

  {cyan('Scalar operations broadcast to every element:')}
      x * 2       →  {x * 2}           (multiply each by 2)
      x + 10      →  {x + 10}          (add 10 to each)

  {green('Term: Vectorization')}
      Instead of:  total = 0; for i in x: total += i
      Write:       x.sum()
      Same result, but NumPy does the loop in compiled C code.""")

# ── Step 3 ──────────────────────────────────────────────

player.add_step("Step 3: Visualize the Data (in Your Head)", f"""\
  Before any math, look at what we have:

      x:  1      2      3          (input — what we know)
      y:  2.0    3.9    6.1        (output — what we want to predict)

  Plot it mentally:

      y
      7 │
      6 │              · (3, 6.1)
      5 │
      4 │       · (2, 3.9)
      3 │
      2 │ · (1, 2)
      1 │
      0 └──────────────── x
        0  1   2   3

  {green('Observation:')} the points roughly follow a straight line.
  As x goes up by 1, y goes up by about 2.

  {cyan('Our goal:')} find the number 'a' (the slope) such that
      y ≈ a * x
  gives the best prediction for any x.""")

# ── Step 4 ──────────────────────────────────────────────

x_mean = x.mean()
y_mean = y.mean()

player.add_step("Step 4: The Mean — Center of Gravity", f"""\
  {cyan('Type:')}
      x.mean()    →  {x_mean}
      y.mean()    →  {y_mean}

  {cyan('What is the mean?')}
      mean = sum of all values / count
      x.mean() = (1 + 2 + 3) / 3 = {x_mean}
      y.mean() = (2 + 3.9 + 6.1) / 3 = {y_mean}

  {green('Why do we need it?')}
      The mean is the "center of gravity" of the data.
      If you balanced the data points on a seesaw, the mean
      is where the pivot point would be.

      y
      6 │              ·
      4 │- - - -●- - - - - -  ← y mean = {y_mean}
        │       · ↑
      2 │ ·       center
      0 └──────────────── x
        0  1 ↑ 2   3
             x mean = {x_mean}

  We'll use the mean to "center" the data in the next step.""")

# ── Step 5 ──────────────────────────────────────────────

xc = x - x_mean
yc = y - y_mean

player.add_step("Step 5: Centering — Shift Data to the Origin", f"""\
  {cyan('Type:')}
      xc = x - x.mean()
      yc = y - y.mean()
      xc    →  {xc}
      yc    →  {yc}

  {cyan('What happened?')}
      Original:  x = [1,   2,   3  ]    y = [2,    3.9,  6.1 ]
      Centered:  xc= [{xc[0]:.0f},  {xc[1]:.0f},  {xc[2]:.0f} ]    yc= [{yc[0]:.1f}, {yc[1]:.1f}, {yc[2]:.1f}]

      We subtracted the mean from every element.
      Now the center of the data sits at (0, 0).

  {green('Why center?')}
      After centering, we only need to find slope 'a' in:
          y = a * x           (line through the origin)

      Without centering, we'd need BOTH slope and intercept:
          y = a * x + b       (line that may not pass through origin)

      Centering removes 'b' from the equation — one less thing
      to figure out.  We'll add 'b' back in lesson 03.""",

detail=f"""\
  {cyan('Verify the centering worked:')}
      xc.mean()  →  {xc.mean()}    (zero, or very close to zero)
      yc.mean()  →  {yc.mean():.1e}    (essentially zero — floating point)

  {green('Visual: before vs after centering')}

      BEFORE                        AFTER (centered)
      y                             yc
      6 │           ·                2 │           ·
      4 │      ·                     0 │- - · - - - - -
      2 │ ·                         -2 │ ·
      0 └────────── x               0 └────────── xc
        0  1  2  3                   -1  0  1

      Same shape, just shifted so the center is at (0,0).""")

# ── Step 6 ──────────────────────────────────────────────

xy = xc * yc
xx = xc * xc

player.add_step("Step 6: How Do X and Y Move Together?", f"""\
  {cyan('Type:')}
      xy = xc * yc        # multiply matching elements
      xx = xc * xc        # square each x deviation
      xy    →  {xy}
      xx    →  {xx}

  {cyan('What does xc * yc tell us?')}
      Point 1: xc=-1, yc=-2.0  →  xc*yc = {xy[0]:.1f}  (both negative → positive product)
      Point 2: xc= 0, yc=-0.1  →  xc*yc = {xy[1]:.1f}  (zero × anything = ~0)
      Point 3: xc= 1, yc= 2.1  →  xc*yc = {xy[2]:.1f}  (both positive → positive product)

  {green('Key insight:')}
      When x goes UP and y also goes UP → product is POSITIVE
      When x goes UP but y goes DOWN   → product is NEGATIVE

      All products positive → x and y move in the SAME direction.
      This is called {cyan('positive correlation')}.

  {cyan('Type:')}
      xy.sum()   →  {xy.sum()}    (total: how strongly they co-move)
      xx.sum()   →  {xx.sum()}     (total: how spread out x is)""",

detail=f"""\
  {green('Term: Covariance')}
      xy.sum() / len(x) is the {cyan('covariance')} of x and y.
      Covariance = {xy.sum() / len(x):.4f}

      Covariance > 0  →  x↑ then y↑  (positive relationship)
      Covariance < 0  →  x↑ then y↓  (negative relationship)
      Covariance ≈ 0  →  no linear relationship

  {green('Term: Variance')}
      xx.sum() / len(x) is the {cyan('variance')} of x.
      Variance = {xx.sum() / len(x):.4f}

      Variance measures how spread out x is from its mean.
      High variance = data is widely spread.
      Low variance  = data is clustered near the mean.

      Standard deviation = sqrt(variance) = {np.sqrt(xx.sum() / len(x)):.4f}
      (Same idea, but in original units instead of squared units.)""")

# ── Step 7 ──────────────────────────────────────────────

a = xy.sum() / xx.sum()

player.add_step("Step 7: Find the Slope — The 'Learning' Step", f"""\
  {cyan('Type:')}
      a = xy.sum() / xx.sum()
      a    →  {a}

  {cyan('The formula:')}
      a = sum(xc * yc) / sum(xc * xc)
      a = {xy.sum()} / {xx.sum()}
      a = {a}

  {green('What does this mean?')}
      a = {a} is the SLOPE of the best-fit line.
      "For every 1-unit increase in x, y increases by {a}."

  {green('THIS IS THE MACHINE LEARNING MOMENT.')}
      The computer just "learned" the relationship: y ≈ {a} * x
      It didn't memorize the 3 data points.
      It extracted ONE number ({a}) that captures the pattern.

      Training data:  6 values  (x=1,2,3 and y=2,3.9,6.1)
      Learned model:  1 value   (a={a})

  {cyan('The model is SMALLER than the data.')}
      This is true for all ML: compress data into patterns.""",

detail=f"""\
  {green('Why this formula? — The Least Squares Idea')}

      We want the line y = a*x that makes the smallest total error.

      For each point, the error (called {cyan('residual')}) is:
          error_i = actual_y_i - predicted_y_i
                  = yc_i - a * xc_i

      We want to minimize the sum of SQUARED errors:
          total_error = Σ (yc_i - a * xc_i)²

      {cyan('Why squared?')}
          - Positive and negative errors don't cancel out
          - Larger errors get penalized more heavily
          - The math works out to a clean formula (calculus)

      Taking the derivative and setting it to 0:
          d/da Σ(yc - a*xc)² = 0
          Σ 2(yc - a*xc)(-xc) = 0
          Σ xc*yc = a * Σ xc*xc
          a = Σ(xc*yc) / Σ(xc*xc)     ← the formula we used

      This is called {cyan('Ordinary Least Squares (OLS)')}.
      Every linear regression in the world uses this idea.""")

# ── Step 8 ──────────────────────────────────────────────

predictions = a * x
errors = y - predictions

player.add_step("Step 8: Make Predictions and Check Errors", f"""\
  {cyan('Type:')}
      predictions = a * x           # predict y for each x
      predictions  →  {predictions}

      errors = y - predictions      # how far off are we?
      errors       →  {errors}

  {green('Compare actual vs predicted:')}
      x=1:  actual y=2.0,   predicted={a*1:.1f},  error={2.0 - a*1:+.1f}
      x=2:  actual y=3.9,   predicted={a*2:.1f},  error={3.9 - a*2:+.1f}
      x=3:  actual y=6.1,   predicted={a*3:.1f},  error={6.1 - a*3:+.1f}

  {green('Predict new values we haven\'t seen:')}
  {cyan('Type:')}
      a * 4      →  {a * 4:.1f}     (prediction for x=4)
      a * 5      →  {a * 5:.1f}    (prediction for x=5)
      a * 10     →  {a * 10:.1f}    (prediction for x=10)

  The model generalizes — it can predict for ANY x,
  not just the 3 points it was trained on.""",

detail=f"""\
  {green('How good is our model? — Measuring Error')}

  {cyan('Mean Absolute Error (MAE):')}
      Average of |error| — easy to understand.
      MAE = {np.abs(errors).mean():.4f}
      "On average, our prediction is off by {np.abs(errors).mean():.2f}"

  {cyan('Mean Squared Error (MSE):')}
      Average of error² — penalizes large errors more.
      MSE = {(errors**2).mean():.4f}

  {cyan('Root Mean Squared Error (RMSE):')}
      Square root of MSE — back in original units.
      RMSE = {np.sqrt((errors**2).mean()):.4f}

  {cyan('Type these yourself:')}
      np.abs(errors).mean()              →  {np.abs(errors).mean():.4f}  (MAE)
      (errors**2).mean()                 →  {(errors**2).mean():.4f}  (MSE)
      np.sqrt((errors**2).mean())        →  {np.sqrt((errors**2).mean()):.4f}  (RMSE)

  In ML, MSE is the most common because its derivative is clean
  (important for training neural networks later).""")

# ── Step 9 ──────────────────────────────────────────────

player.add_step("Step 9: What Just Happened — The ML Pipeline", f"""\
  You just completed the entire machine learning pipeline:

      ┌─────────────┐     ┌──────────────┐     ┌────────────┐
      │  1. DATA     │ ──→ │  2. TRAIN    │ ──→ │ 3. PREDICT │
      │  x=[1,2,3]  │     │  find a={a}  │     │ x=4 → {a*4:.1f} │
      │  y=[2,3.9,  │     │  (learning)  │     │ x=5 → {a*5:.1f} │
      │     6.1]    │     │              │     │            │
      └─────────────┘     └──────────────┘     └────────────┘

  {green('ML vocabulary for what we did:')}
      {cyan('Feature')}      = input variable (x) — what you know
      {cyan('Label/Target')} = output variable (y) — what you want to predict
      {cyan('Model')}        = the learned formula: y = {a} * x
      {cyan('Weight')}       = the number the model learned: a = {a}
      {cyan('Training')}     = finding the best weight from data
      {cyan('Inference')}    = using the trained model to predict new values
      {cyan('Loss')}         = how wrong the predictions are (MSE = {(errors**2).mean():.4f})

  {green('The core idea of ALL machine learning:')}
      1. Start with data (features + labels)
      2. Pick a model shape (y = a*x)
      3. Find the weights that minimize the loss
      4. Use the weights to predict on new data

  This NEVER changes — from this 1-weight model to GPT's
  175 billion weights.  Only the scale is different.""",

detail=f"""\
  {green('Scaling up: from 1 weight to billions')}

      This lesson:       y = a * x              →  1 weight
      Lesson 03:         y = a*x + b            →  2 weights
      Lesson 09:         64→64→32→10 network    →  ~7,000 weights
      Lesson 10:         PyTorch digit classifier→  ~7,000 weights
      Lesson 11:         Mini transformer       →  ~110,000 weights
      GPT-3 (2020):      transformer            →  175,000,000,000 weights

      The jump from 1 to 175 billion seems impossible.
      But the principle is identical:
          1. Data in
          2. Multiply by weights
          3. Measure error
          4. Adjust weights to reduce error
          5. Repeat

  {cyan('What changes at scale:')}
      - More layers of multiplication (depth)
      - More weights per layer (width)
      - More data to learn from (trillions of words)
      - More compute to find good weights (thousands of GPUs)
      - Smarter optimization (Adam, not just the formula we used)

  {cyan('What stays the same:')}
      - weights × input = prediction
      - compare prediction to reality = loss
      - adjust weights to reduce loss = training""")

# ── Play ────────────────────────────────────────────────

player.play()
