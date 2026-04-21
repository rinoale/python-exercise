"""
Lesson 06: Data Preprocessing
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

player.add_step("Step 1: Why Preprocessing?", f"""\
  Real-world data is messy. ML models need clean numbers.

  {cyan('Common problems:')}
    1. Missing values (NaN) — sensors fail, users skip fields
    2. Different scales — age is 25-40, salary is 50000-80000
    3. Text categories — "Seoul", "Busan" aren't numbers

  {green('Example messy data:')}
{data.to_string()}

  Missing values:
{data.isnull().sum().to_string()}

  We need to fix ALL of these before training a model.""")

# ── Step 2 ──────────────────────────────────────────────

dropped = data.dropna()
imputer = SimpleImputer(strategy='mean')
data_filled = data.copy()
data_filled[['age', 'salary']] = imputer.fit_transform(data[['age', 'salary']])
data_filled['city'] = data['city'].fillna('Unknown')

player.add_step("Step 2: Handling Missing Values", f"""\
  {cyan('Strategy 1: Drop rows')}
    data.dropna()  →  {len(dropped)} rows (lost {len(data) - len(dropped)} of {len(data)}!)

  {cyan('Strategy 2: Fill with mean')}
    imputer = SimpleImputer(strategy='mean')
    data[['age','salary']] = imputer.fit_transform(data[['age','salary']])

  {green('Result after filling:')}
{data_filled.to_string()}

  Dropping loses too much data. Filling with mean is usually better.
  Other strategies: median, most_frequent, or a constant value.""")

# ── Step 3 ──────────────────────────────────────────────

ages = np.array([25, 30, 35, 40, 28]).reshape(-1, 1)
salaries = np.array([50000, 60000, 70000, 80000, 55000]).reshape(-1, 1)

scaler_std = StandardScaler()
ages_std = scaler_std.fit_transform(ages)

scaler_mm = MinMaxScaler()
ages_mm = scaler_mm.fit_transform(ages)
salaries_mm = MinMaxScaler().fit_transform(salaries)

player.add_step("Step 3: Feature Scaling", f"""\
  Problem: age is 25-40, salary is 50000-80000.
  Many ML algorithms use distances — salary would dominate
  just because its numbers are bigger.

  {cyan('StandardScaler')} (mean=0, std=1):
    ages     {ages.ravel()} → {ages_std.ravel().round(2)}

  {cyan('MinMaxScaler')} (range 0-1):
    ages     {ages.ravel()} → {ages_mm.ravel().round(2)}
    salaries {salaries.ravel()} → {salaries_mm.ravel().round(2)}

  Now both features are on the same scale!
  Rule of thumb: always scale before training.""")

# ── Step 4 ──────────────────────────────────────────────

cities = ['Seoul', 'Busan', 'Daegu', 'Seoul', 'Busan']
le = LabelEncoder()
cities_encoded = le.fit_transform(cities)

ohe = OneHotEncoder(sparse_output=False)
cities_onehot = ohe.fit_transform(np.array(cities).reshape(-1, 1))

ohe_text = ""
for city, encoded in zip(cities, cities_onehot):
    ohe_text += f"    {city:>6s} → {encoded.astype(int)}\n"

player.add_step("Step 4: Encoding Categories", f"""\
  ML models need numbers, not text. Two approaches:

  {cyan('LabelEncoder:')} text → integer
    {cities} → {cities_encoded}
    Problem: implies order (Busan=0 < Daegu=1 < Seoul=2)

  {cyan('OneHotEncoder:')} text → binary columns (no false ordering)
    Categories: {list(ohe.categories_[0])}
               [Busan, Daegu, Seoul]
{ohe_text}
  Use LabelEncoder for ordinal data (small/medium/large).
  Use OneHotEncoder for nominal data (cities, colors).""")

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

player.add_step("Step 5: Pipeline — Chain Everything Together", f"""\
  Pipeline chains preprocessing + model into one object.
  No risk of forgetting to scale test data.

  {cyan('Code:')}
    from sklearn.pipeline import make_pipeline
    pipe = make_pipeline(StandardScaler(), LogisticRegression())
    pipe.fit(X_train, y_train)      # scales + trains
    pipe.score(X_test, y_test)      # scales + predicts

  {green('Result:')}
    Without scaling: {score_raw:.4f}
    With pipeline:   {score_pipe:.4f}

  Plot saved to images/06_preprocessing.png

  {green('Summary:')}
    ✓ Fill missing values (don't drop rows)
    ✓ Scale features to same range
    ✓ Encode text categories as numbers
    ✓ Use pipelines to chain it all together""")

# ── Play ────────────────────────────────────────────────

player.play()
