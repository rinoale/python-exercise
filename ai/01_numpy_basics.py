"""
Lesson 01: NumPy Basics
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from lesson_player import LessonPlayer, bold, cyan, green

player = LessonPlayer("Lesson 01: NumPy Basics")

# ── Step 1 ──────────────────────────────────────────────

x = np.array([1, 2, 3])
y = np.array([2, 3.9, 6.1])

player.add_step("Step 1: Creating Arrays", f"""\
  np.array() converts a Python list into a numpy array.
  Numpy arrays support math operations that regular lists don't.

  {cyan('Code:')}
    import numpy as np
    x = np.array([1, 2, 3])
    y = np.array([2, 3.9, 6.1])

  {green('Result:')}
    x = {x}
    y = {y}
    type(x) = {type(x).__name__}""")

# ── Step 2 ──────────────────────────────────────────────

x_mean = x.mean()
y_mean = y.mean()

player.add_step("Step 2: Mean (Average)", f"""\
  .mean() calculates the average of all elements.

  {cyan('Code:')}
    x.mean()    # (1 + 2 + 3) / 3
    y.mean()    # (2 + 3.9 + 6.1) / 3

  {green('Result:')}
    x.mean() = {x_mean}
    y.mean() = {y_mean}""")

# ── Step 3 ──────────────────────────────────────────────

xc = x - x.mean()
yc = y - y.mean()

player.add_step("Step 3: Centering Data", f"""\
  Centering = subtract the mean from each element.
  This shifts the data so the new mean is 0.
  Why? It simplifies the math for computing the slope.

  {cyan('Code:')}
    xc = x - x.mean()    # [1-2, 2-2, 3-2]
    yc = y - y.mean()    # [2-4, 3.9-4, 6.1-4]

  {green('Result:')}
    xc = {xc}        (mean is now 0)
    yc = {yc}    (mean is now 0)""")

# ── Step 4 ──────────────────────────────────────────────

xx = xc * xc
yy = yc * yc
xy = xc * yc

player.add_step("Step 4: Element-wise Operations", f"""\
  Multiply matching elements:
    [a, b, c] * [d, e, f] = [a*d, b*e, c*f]

  {cyan('Code:')}
    xx = xc * xc    # squared deviations of x
    yy = yc * yc    # squared deviations of y
    xy = xc * yc    # cross-product: how x and y move together

  {green('Result:')}
    xx = {xx}
    yy = {yy}
    xy = {xy}

    xy.sum() = {xy.sum()}   (positive = x and y move in same direction)""")

# ── Step 5 ──────────────────────────────────────────────

a = xy.sum() / xx.sum()

player.add_step("Step 5: Computing the Slope", f"""\
  Least squares formula:
    a = sum(xc * yc) / sum(xc * xc)

  'a' is the slope: for each 1-unit increase in x, y increases by 'a'.

  {cyan('Code:')}
    a = xy.sum() / xx.sum()    # {xy.sum()} / {xx.sum()}

  {green('Result:')}
    a = {a}

  So the line is:  y = {a} * x
  Meaning: each unit of x adds ~{a} to y.

  Example predictions:
    x=4 → y ≈ {a * 4 - (a * x_mean - y_mean):.1f}
    x=5 → y ≈ {a * 5 - (a * x_mean - y_mean):.1f}""")

# ── Play ────────────────────────────────────────────────

player.play()
