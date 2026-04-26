"""
Lesson 02: Pandas — Working with Real Data

Goal: load a CSV file, explore it, understand its shape,
      and compute the slope on 100 real data points.

Open a Python REPL:
    cd ai/
    python3
    >>> import pandas as pd
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from lesson_player import LessonPlayer, cyan, green

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

player = LessonPlayer("Lesson 02: Pandas — Working with Real Data")

# ── Step 1 ──────────────────────────────────────────────

player.add_step("Step 1: Why Pandas?", f"""\
  {cyan('Lesson 01:')} we typed 3 data points by hand.
  {cyan('Real ML:')}   data comes from files — CSV, databases, APIs.

  {green('Pandas')} is the standard tool for loading and manipulating
  tabular data (rows and columns, like a spreadsheet).

      NumPy array  = a grid of numbers (math operations)
      Pandas DataFrame = a table with column NAMES + mixed types

  {cyan('Analogy for Ruby developers:')}
      NumPy  ≈  Array of numbers with math methods
      Pandas ≈  ActiveRecord result set / CSV::Table

  {cyan('Key Pandas objects:')}
      DataFrame  = 2D table (the whole spreadsheet)
      Series     = 1D column (one column of the spreadsheet)""")

# ── Step 2 ──────────────────────────────────────────────

df = pd.read_csv(os.path.join(BASE_DIR, 'sample.csv'))

player.add_step("Step 2: Load a CSV File", f"""\
  {cyan('Type:')}
      import pandas as pd
      df = pd.read_csv('sample.csv')

  {cyan('Then try:')}
      type(df)        →  pandas.core.frame.DataFrame
      df.shape        →  {df.shape}       (100 rows, 2 columns)
      df.columns      →  {list(df.columns)}
      df.head()       →  first 5 rows
      df.tail(3)      →  last 3 rows
      df.dtypes       →  data types per column

  {green('df.head() output:')}
{df.head().to_string(index=True)}""")

# ── Step 3 ──────────────────────────────────────────────

x = df['x']
y = df['y']

player.add_step("Step 3: Access Columns — Series", f"""\
  {cyan('Type:')}
      x = df['x']
      y = df['y']

  {green('What is x?')}
      type(x)    →  {type(x).__name__}
      x.shape    →  {x.shape}
      x[:5]      →  {x[:5].values}

  {cyan('A Series is one column of the DataFrame.')}
      DataFrame = the whole table
      Series    = one column

  {cyan('Other ways to access columns:')}
      df['x']        →  Series  (bracket notation — always works)
      df.x           →  Series  (dot notation — only if name is valid Python)
      df[['x','y']]  →  DataFrame (double brackets = multiple columns)

  {cyan('Try:')}
      x.values       →  the underlying NumPy array
      type(x.values) →  {type(x.values).__name__}

  So Pandas wraps NumPy arrays, adding labels and extra methods.""")

# ── Step 4 ──────────────────────────────────────────────

desc = df.describe()
x_mean = desc['x']['mean']
x_std = desc['x']['std']

player.add_step("Step 4: Summary Statistics — describe()", f"""\
  {cyan('Type:')}
      df.describe()

  {green('Result:')}
{desc.to_string()}

  {cyan('What each row means:')}
      count = number of non-missing values
      mean  = average (center of the data)
      std   = standard deviation (spread of the data)
      min   = smallest value
      25%   = first quartile (25% of values are below this)
      50%   = median (middle value)
      75%   = third quartile
      max   = largest value""",

detail=f"""\
  {green('Term: Standard Deviation (std)')}

      std measures how far values typically are from the mean.
      Small std = data is clustered tightly around the mean.
      Large std = data is spread out widely.

  {cyan('The 68-95-99.7 Rule (for bell-shaped data):')}
      ~68% of data falls within  mean ± 1 std  →  {x_mean - x_std:.1f} to {x_mean + x_std:.1f}
      ~95% of data falls within  mean ± 2 std  →  {x_mean - 2*x_std:.1f} to {x_mean + 2*x_std:.1f}
      ~99% of data falls within  mean ± 3 std  →  {x_mean - 3*x_std:.1f} to {x_mean + 3*x_std:.1f}

                          ██
                         ████
                        ██████
                     ████████████
                 ████████████████████
            ▁▂▃▅████████████████████████▅▃▂▁
            |---------|---mean---|---------|
           -3σ       -1σ       +1σ       +3σ

  {cyan('Try:')}
      x.std()          →  {x.std():.4f}
      x.median()       →  {x.median():.4f}   (same as 50%)
      x.max() - x.min()→  {x.max() - x.min():.4f}  (range)""")

# ── Step 5 ──────────────────────────────────────────────

df_c = df - df.mean()
desc_c = df_c.describe()

player.add_step("Step 5: Center the Entire DataFrame", f"""\
  {cyan('Type:')}
      df_c = df - df.mean()
      df_c.describe()

  {green('Result (notice mean ≈ 0 for both columns):')}
{desc_c.to_string()}

  {cyan('What happened?')}
      Pandas subtracted each column's mean from every value
      in that column — the same centering we did in lesson 01,
      but on ALL columns at once.

  {cyan('Verify:')}
      df_c.mean()

      x    {df_c['x'].mean():.2e}     (essentially zero)
      y    {df_c['y'].mean():.2e}     (essentially zero)

  {green('Why center?')}
      Same reason as lesson 01: after centering, we can find
      the slope with  a = sum(xc*yc) / sum(xc*xc)
      without worrying about the intercept.""")

# ── Step 6 ──────────────────────────────────────────────

xx = df_c['x'] * df_c['x']
xy = df_c['x'] * df_c['y']
a = xy.sum() / xx.sum()

player.add_step("Step 6: Compute the Slope on 100 Points", f"""\
  {cyan('Type:')}
      xx = df_c['x'] * df_c['x']     # variance components
      xy = df_c['x'] * df_c['y']     # covariance components
      a = xy.sum() / xx.sum()
      a    →  {a:.4f}

  {green('Same formula as lesson 01:')}
      a = Σ(xc * yc) / Σ(xc * xc)

      Numerator:   xy.sum() = {xy.sum():.2f}  (covariance direction)
      Denominator: xx.sum() = {xx.sum():.2f}   (variance of x)

  {green('Interpretation:')}
      For every 1-unit increase in x, y increases by ~{a:.0f}.

  {cyan('Compare with lesson 01:')}
      Lesson 01 (3 points):   a = 2.05
      Lesson 02 (100 points): a = {a:.2f}

  More data → more reliable estimate of the true pattern.""",

detail=f"""\
  {green('Term: Covariance and Correlation')}

  {cyan('Covariance = xy.sum() / len(x)')}
      = {xy.sum() / len(x):.4f}
      Tells direction: positive (x↑ y↑) or negative (x↑ y↓).
      Problem: the number depends on the units/scale of x and y.

  {cyan('Correlation = covariance / (std_x * std_y)')}
      = {(xy.sum() / len(x)) / (df_c['x'].std() * df_c['y'].std()):.4f}
      Always between -1 and +1.  Unit-free.
          +1 = perfect positive linear relationship
           0 = no linear relationship
          -1 = perfect negative linear relationship

  {cyan('Try:')}
      df.corr()    →  correlation matrix

{df.corr().to_string()}

      The correlation between x and y is strong and positive.""")

# ── Step 7 ──────────────────────────────────────────────

plt.figure(figsize=(10, 6))
plt.scatter(df['x'], df['y'], alpha=0.6, label='Actual data')
x_line = np.linspace(df['x'].min(), df['x'].max(), 100)
y_line = a * (x_line - x.mean()) + y.mean()
plt.plot(x_line, y_line, color='red', linewidth=2, label=f'y = {a:.1f}x (centered)')
plt.xlabel('x')
plt.ylabel('y')
plt.title('100 Data Points with Best-Fit Line')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, "images", "02_regression.png"))
plt.close()

player.add_step("Step 7: Visualize — Scatter Plot + Line", f"""\
  {cyan('The code (we save to a file since this is a terminal):')}
      import matplotlib.pyplot as plt
      plt.scatter(df['x'], df['y'], alpha=0.6)
      plt.plot(x_sorted, a * x_sorted, color='red')
      plt.savefig('images/02_regression.png')

  {green('Plot saved to images/02_regression.png')}
      Open it to see 100 blue dots with a red regression line.

  {cyan('What the plot tells you:')}
      - Blue dots = actual data (100 points)
      - Red line  = our model's prediction: y = {a:.1f} * x
      - Dots close to line = good prediction
      - Dots far from line = noise our model can't capture

  {green('ASCII approximation:')}
      y
      │                    · · ·
      │               · ·/· ·
      │           · · /· ·
      │        · · /· ·
      │      · ·/ · ·
      │   · ·/ · ·
      │  ·/· ·
      │ /·
      └──────────────────── x
             red line / through the data""")

# ── Step 8 ──────────────────────────────────────────────

player.add_step("Step 8: Summary — Pandas for ML", f"""\
  {green('What you learned:')}

      {cyan('1. Load data')}
         df = pd.read_csv('file.csv')

      {cyan('2. Explore data')}
         df.head(), df.describe(), df.shape, df.dtypes

      {cyan('3. Access columns')}
         x = df['column_name']  →  Series (1D)

      {cyan('4. Math on columns')}
         df - df.mean()         →  center all columns
         df['x'] * df['y']     →  element-wise operations
         df.corr()              →  correlation matrix

      {cyan('5. Compute slope')}
         Same formula as lesson 01, now on 100 real points.

  {green('What\'s missing?')}
      We only have slope 'a'.  We need intercept 'b' too,
      so predictions work on UN-centered data.
      That's lesson 03: y = a*x + b  (full linear regression).""")

# ── Play ────────────────────────────────────────────────

player.play()
