"""
Lesson 02: Pandas Usage
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from lesson_player import LessonPlayer, cyan, green

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

player = LessonPlayer("Lesson 02: Pandas Usage")

# ── Step 1 ──────────────────────────────────────────────

df = pd.read_csv(os.path.join(BASE_DIR, 'sample.csv'))

player.add_step("Step 1: Loading CSV", f"""\
  pd.read_csv() reads a CSV file into a DataFrame.
  A DataFrame is a 2D table with labeled columns (like a spreadsheet).

  {cyan('Code:')}
    import pandas as pd
    df = pd.read_csv('sample.csv')

  {green('Result:')}
    Shape: {df.shape[0]} rows x {df.shape[1]} columns
    Columns: {list(df.columns)}

    First 5 rows:
{df.head().to_string(index=True)}""")

# ── Step 2 ──────────────────────────────────────────────

x = df['x']
y = df['y']

player.add_step("Step 2: Exploring Data", f"""\
  Access a column by name: df['column_name']
  Returns a Series (1D labeled array — one column of the spreadsheet).

  {cyan('Code:')}
    x = df['x']
    y = df['y']
    df.head(3)      # first 3 rows

  {green('Result:')}
    df.head(3):
{df.head(3).to_string(index=True)}

    x[:5] = {x[:5].values}
    y[:5] = {y[:5].values}""")

# ── Step 3 ──────────────────────────────────────────────

desc = df.describe()
x_mean = desc['x']['mean']
x_std = desc['x']['std']

player.add_step("Step 3: Summary Statistics", f"""\
  .describe() shows count, mean, std, min, 25%, 50%, 75%, max
  for each numeric column.

  {cyan('Code:')}
    df.describe()

  {green('Result:')}
{desc.to_string()}""",

detail=f"""\
  {cyan('What each row means:')}
    count  = number of values (100)
    mean   = average
    std    = standard deviation — how spread out the values are
    min    = smallest value
    25%    = 25% of values are below this (Q1)
    50%    = the middle value, aka median (Q2)
    75%    = 75% of values are below this (Q3)
    max    = largest value

  {cyan('std and the 68-95-99.7 rule (normal distribution):')}
    ~68% of values fall within  mean ± 1 std  →  {x_mean - x_std:.1f} ~ {x_mean + x_std:.1f}
    ~95% of values fall within  mean ± 2 std  →  {x_mean - 2*x_std:.1f} ~ {x_mean + 2*x_std:.1f}
    ~99% of values fall within  mean ± 3 std  →  {x_mean - 3*x_std:.1f} ~ {x_mean + 3*x_std:.1f}

                        ██
                       ████
                      ██████
                   ████████████
               ████████████████████
          ▁▂▃▅████████████████████████▅▃▂▁
          |---------|---mean---|---------|
         -3σ       -1σ       +1σ       +3σ

  {cyan('std vs range:')}
    range (max-min) = {desc['x']['max'] - desc['x']['min']:.1f}  ← stretched by outliers
    std             = {desc['x']['std']:.1f}   ← measures the typical spread
    IQR (75%-25%)   = {desc['x']['75%'] - desc['x']['25%']:.1f}   ← similar to std, ignores outliers""")

# ── Step 4 ──────────────────────────────────────────────

df_c = df - df.mean()
desc_c = df_c.describe()

player.add_step("Step 4: Centering the DataFrame", f"""\
  Subtract the mean of each column from every value.
  Same as lesson 01, but pandas does it for all columns at once.

  {cyan('Code:')}
    df_c = df - df.mean()
    df_c.describe()

  {green('Result (notice mean ≈ 0):')}
{desc_c.to_string()}""")

# ── Step 5 ──────────────────────────────────────────────

xx = df_c['x'] * df_c['x']
xy = df_c['x'] * df_c['y']
a = xy.sum() / xx.sum()

player.add_step("Step 5: Computing Slope", f"""\
  Same formula as lesson 01, now on 100 real data points.
    a = sum(xc * yc) / sum(xc * xc)

  {cyan('Code:')}
    xx = df_c['x'] * df_c['x']
    xy = df_c['x'] * df_c['y']
    a = xy.sum() / xx.sum()

  {green('Result:')}
    a = {a:.2f}

  For each 1-unit increase in x, y increases by ~{a:.0f}.""")

# ── Step 6 ──────────────────────────────────────────────

plt.figure()
plt.scatter(df_c['x'], df_c['y'])
plt.plot(df_c['x'], a * df_c['x'], color='red')
plt.savefig(os.path.join(BASE_DIR, "images", "02_regression.png"))
plt.close()

player.add_step("Step 6: Plotting", f"""\
  matplotlib draws scatter plots and line plots.

  {cyan('Code:')}
    plt.scatter(df_c['x'], df_c['y'])          # dots
    plt.plot(df_c['x'], a * df_c['x'])         # regression line
    plt.savefig('images/02_regression.png')

  {green('Result:')}
    Plot saved to images/02_regression.png

  The scatter shows data points, the red line is y = {a:.0f} * x.
  Points close to the line → good fit. Far away → noise.""")

# ── Play ────────────────────────────────────────────────

player.play()
