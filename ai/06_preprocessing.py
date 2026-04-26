"""
Lesson 06: Data Preprocessing — Cleaning Before Training

Goal: handle missing values, scale features, encode categories,
      and chain it all into a pipeline.

Real data is messy.  Models need clean numbers.
This lesson teaches you to bridge that gap.

Type along:
    python3
    >>> import pandas as pd, numpy as np
    >>> from sklearn.datasets import load_iris
    >>> from sklearn.model_selection import train_test_split
    >>> iris = load_iris(); X, y = iris.data, iris.target
    >>> X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from lesson_player import LessonPlayer, cyan, green

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

player = LessonPlayer("Lesson 06: Data Preprocessing")

# ── Step 1 ──────────────────────────────────────────────

data = pd.DataFrame({
    'age':    [25, 30, np.nan, 35, 40, np.nan, 28, 33],
    'salary': [50000, 60000, 55000, np.nan, 80000, 70000, np.nan, 65000],
    'city':   ['Seoul', 'Busan', 'Seoul', 'Daegu', np.nan, 'Busan', 'Seoul', 'Daegu']
})

player.add_step("Step 1: Why Preprocessing Matters", f"""\
  {cyan('Models need clean numbers.  Real data is never clean.')}

  {cyan('Type:')}
      import pandas as pd, numpy as np
      data = pd.DataFrame({{
          'age':    [25, 30, np.nan, 35, 40, np.nan, 28, 33],
          'salary': [50000, 60000, 55000, np.nan, 80000, 70000, np.nan, 65000],
          'city':   ['Seoul', 'Busan', 'Seoul', 'Daegu', np.nan, 'Busan', 'Seoul', 'Daegu']
      }})
      data

  {green('Result:')}
{data.to_string()}

  {cyan('Three problems to fix:')}
      1. {green('Missing values')} (NaN) — age, salary, and city have gaps
         data.isnull().sum()
{data.isnull().sum().to_string()}

      2. {green('Different scales')} — age is 25-40, salary is 50000-80000
         A model using distances would think salary matters 1000x more.

      3. {green('Text data')} — "Seoul", "Busan" aren't numbers.
         Models can only multiply numbers, not strings.

  {green('Term: NaN')} = "Not a Number". Pandas uses it for missing data.""")

# ── Step 2 ──────────────────────────────────────────────

dropped = data.dropna()
imputer = SimpleImputer(strategy='mean')
data_filled = data.copy()
data_filled[['age', 'salary']] = imputer.fit_transform(data[['age', 'salary']])
data_filled['city'] = data['city'].fillna('Unknown')

player.add_step("Step 2: Handle Missing Values", f"""\
  {cyan('Strategy 1: Drop rows with missing values')}
      data.dropna()  →  {len(dropped)} rows left (lost {len(data) - len(dropped)} of {len(data)}!)
      Simple, but wastes data.  Only use if very few rows are missing.

  {cyan('Strategy 2: Fill (impute) missing values')}
      from sklearn.impute import SimpleImputer
      imputer = SimpleImputer(strategy='mean')
      data[['age','salary']] = imputer.fit_transform(data[['age','salary']])
      data['city'] = data['city'].fillna('Unknown')

  {green('Result after filling:')}
{data_filled.to_string()}

  {cyan('Imputation strategies:')}
      strategy='mean'           →  fill with column average (numeric)
      strategy='median'         →  fill with middle value (robust to outliers)
      strategy='most_frequent'  →  fill with most common value (categorical)
      strategy='constant'       →  fill with a fixed value

  {green('Term: Imputation')}
      Replacing missing values with estimated values.
      Better than dropping rows — preserves data size.
      But the filled values are GUESSES, not truth.""",

detail=f"""\
  {green('When to use which strategy:')}

      {cyan('mean')} — good for normally distributed numeric data.
          age mean = {data['age'].mean():.1f}

      {cyan('median')} — good when outliers exist.
          If salaries were [50k, 60k, 60k, 1000k]:
          mean = 292k (pulled up by outlier)
          median = 60k (resistant to outlier)

      {cyan('most_frequent')} — good for categorical data.
          city most frequent = {data['city'].mode().values[0]}

      {cyan('Advanced:')} KNN imputation (fill based on similar rows),
      or train a model to predict the missing values.
      But SimpleImputer covers 90% of cases.""")

# ── Step 3 ──────────────────────────────────────────────

ages = np.array([25, 30, 35, 40, 28]).reshape(-1, 1)
salaries = np.array([50000, 60000, 70000, 80000, 55000]).reshape(-1, 1)

scaler_std = StandardScaler()
ages_std = scaler_std.fit_transform(ages)

scaler_mm = MinMaxScaler()
ages_mm = scaler_mm.fit_transform(ages)
salaries_mm = MinMaxScaler().fit_transform(salaries)

player.add_step("Step 3: Feature Scaling", f"""\
  {cyan('Problem:')}
      age:    25, 30, 35, 40     (range: 15)
      salary: 50000, 60000, ...   (range: 30000)

      Many ML algorithms use DISTANCE between points.
      Without scaling, salary dominates just because its
      numbers are bigger — not because it matters more.

  {cyan('StandardScaler')} — transforms to mean=0, std=1:
      from sklearn.preprocessing import StandardScaler
      scaler = StandardScaler()
      ages_scaled = scaler.fit_transform(ages)

      ages:     {ages.ravel()} →  {ages_std.ravel().round(2)}

  {cyan('MinMaxScaler')} — transforms to range [0, 1]:
      ages:     {ages.ravel()} →  {ages_mm.ravel().round(2)}
      salaries: {salaries.ravel()} →  {salaries_mm.ravel().round(2)}

  {green('Now both features are on the same scale!')}

  {green('Rule:')} always scale before training distance-based models
  (KNN, SVM, neural networks, K-Means).
  Tree-based models (decision trees, random forests) don't need it.""",

detail=f"""\
  {green('StandardScaler formula:')}
      x_scaled = (x - mean) / std

      For age=25:  (25 - {ages.mean():.1f}) / {ages.std():.2f} = {ages_std[0][0]:.2f}

      After scaling: mean ≈ 0, std ≈ 1 for every feature.
      "How many standard deviations from the mean?"

  {green('MinMaxScaler formula:')}
      x_scaled = (x - min) / (max - min)

      For age=25: (25 - 25) / (40 - 25) = 0.00
      For age=40: (40 - 25) / (40 - 25) = 1.00

      After scaling: all values between 0 and 1.

  {cyan('Which to use?')}
      StandardScaler — when data is roughly normally distributed.
      MinMaxScaler   — when you need a fixed range [0, 1].
      Neither changes the SHAPE of the data or the relationships.""")

# ── Step 4 ──────────────────────────────────────────────

cities = ['Seoul', 'Busan', 'Daegu', 'Seoul', 'Busan']
le = LabelEncoder()
cities_encoded = le.fit_transform(cities)

ohe = OneHotEncoder(sparse_output=False)
cities_onehot = ohe.fit_transform(np.array(cities).reshape(-1, 1))

ohe_text = ""
for city, encoded in zip(cities, cities_onehot):
    ohe_text += f"      {city:>6s} → {encoded.astype(int)}\n"

player.add_step("Step 4: Encoding Categories — Text to Numbers", f"""\
  {cyan('Models can\'t process text.  Convert strings to numbers.')}

  {cyan('LabelEncoder')} — text → integer:
      from sklearn.preprocessing import LabelEncoder
      le = LabelEncoder()
      le.fit_transform(['Seoul','Busan','Daegu','Seoul','Busan'])
      →  {cities_encoded}

      {green('Problem:')} implies Busan(0) < Daegu(1) < Seoul(2).
      But cities have NO natural order!  The model might think
      Seoul is "bigger" than Busan.

  {cyan('OneHotEncoder')} — text → binary columns (no false ordering):
      from sklearn.preprocessing import OneHotEncoder
      ohe = OneHotEncoder(sparse_output=False)
      ohe.fit_transform(cities_array)

      Categories: {list(ohe.categories_[0])}
                  [Busan, Daegu, Seoul]
{ohe_text}
      Each city gets its own column.  1 = yes, 0 = no.
      No ordering imposed.

  {green('Rule of thumb:')}
      LabelEncoder → ordinal data (small < medium < large)
      OneHotEncoder → nominal data (Seoul, Busan — no order)""")

# ── Step 5 ──────────────────────────────────────────────

iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.3, random_state=42
)

model_raw = LogisticRegression(max_iter=200)
model_raw.fit(X_train, y_train)
score_raw = model_raw.score(X_test, y_test)

pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=200))
pipe.fit(X_train, y_train)
score_pipe = pipe.score(X_test, y_test)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].scatter(X_train[:, 0], X_train[:, 2], c=y_train, cmap='viridis', alpha=0.6)
axes[0].set_xlabel('Sepal Length (cm)')
axes[0].set_ylabel('Petal Length (cm)')
axes[0].set_title('Before Scaling')
X_train_scaled = StandardScaler().fit_transform(X_train)
axes[1].scatter(X_train_scaled[:, 0], X_train_scaled[:, 2], c=y_train, cmap='viridis', alpha=0.6)
axes[1].set_xlabel('Sepal Length (scaled)')
axes[1].set_ylabel('Petal Length (scaled)')
axes[1].set_title('After StandardScaler')
plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, "images", "06_preprocessing.png"))
plt.close()

player.add_step("Step 5: Pipeline — Chain Preprocessing + Model", f"""\
  {cyan('Problem:')} you must remember to scale test data the SAME WAY
  as training data.  Easy to forget → bugs.

  {cyan('Solution: Pipeline chains steps together.')}
      from sklearn.pipeline import make_pipeline
      from sklearn.linear_model import LogisticRegression
      from sklearn.datasets import load_iris
      iris = load_iris()
      X_train, X_test, y_train, y_test = train_test_split(
          iris.data, iris.target, test_size=0.3, random_state=42)

      model_raw = LogisticRegression(max_iter=200)
      model_raw.fit(X_train, y_train)
      model_raw.score(X_test, y_test)  →  {score_raw:.4f}

      pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=200))
      pipe.fit(X_train, y_train)
      pipe.score(X_test, y_test)       →  {score_pipe:.4f}

  {green('Compare:')}
      Without scaling: {score_raw:.4f}
      With pipeline:   {score_pipe:.4f}

  {green('Plot saved to images/06_preprocessing.png')}
      Left:  before scaling (different axis ranges)
      Right: after StandardScaler (centered at 0, same scale)

  {cyan('Pipeline steps execute in order:')}
      pipe.fit():     StandardScaler.fit_transform → LogisticRegression.fit
      pipe.predict(): StandardScaler.transform → LogisticRegression.predict

  {green('Why pipelines matter:')}
      - No data leakage (scaler only sees training data)
      - Clean code (one object to save/load for production)
      - Works with cross_val_score, GridSearchCV, etc.""")

# ── Step 6 ──────────────────────────────────────────────

player.add_step("Step 6: Summary — The Preprocessing Checklist", f"""\
  {green('Before training ANY model, check these:')}

      {cyan('1. Missing values')}
         df.isnull().sum()
         Fix: SimpleImputer(strategy='mean') or drop rows

      {cyan('2. Feature scaling')}
         StandardScaler (mean=0, std=1) or MinMaxScaler (0 to 1)
         Required for: neural networks, SVM, KNN, K-Means
         Not needed for: decision trees, random forests

      {cyan('3. Categorical encoding')}
         LabelEncoder for ordinal (small < medium < large)
         OneHotEncoder for nominal (red, blue, green)

      {cyan('4. Pipeline')}
         Chain preprocessing + model into one object.
         make_pipeline(StandardScaler(), LogisticRegression())

  {green('Preprocessing workflow:')}
      Raw data → Handle NaN → Scale numbers → Encode text → Train

  {green('What\'s next?')}
      Lesson 07: decision trees and random forests — models that
      DON'T need scaling and can show you which features matter.""")

# ── Play ────────────────────────────────────────────────

player.play()
