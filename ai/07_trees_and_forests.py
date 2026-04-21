"""
Lesson 07: Decision Trees & Random Forests
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from lesson_player import LessonPlayer, cyan, green

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

player = LessonPlayer("Lesson 07: Decision Trees & Random Forests")

# ── Load data ───────────────────────────────────────────

wine = load_wine()
X = wine.data
y = wine.target
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# ── Step 1 ──────────────────────────────────────────────

player.add_step("Step 1: What is a Decision Tree?", f"""\
  A decision tree is a flowchart of if/else questions:

    "Is alcohol > 13.0?"
       ├── Yes → "Is color_intensity > 3.5?"
       │           ├── Yes → Class 0
       │           └── No  → Class 2
       └── No  → "Is proline > 700?"
                   ├── Yes → Class 1
                   └── No  → Class 2

  It splits data at each node by the feature that best separates classes.

  {green('Wine Dataset:')}
    Samples:  {X.shape[0]} wines
    Features: {X.shape[1]} measurements (alcohol, color, etc.)
    Classes:  {list(wine.target_names)} (3 wine types)""")

# ── Step 2 ──────────────────────────────────────────────

tree = DecisionTreeClassifier(max_depth=3, random_state=42)
tree.fit(X_train, y_train)

fig, ax = plt.subplots(figsize=(20, 8))
plot_tree(tree, feature_names=wine.feature_names,
          class_names=wine.target_names, filled=True, rounded=True, ax=ax)
plt.title('Decision Tree (max_depth=3)')
plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, "images", "07_decision_tree.png"), dpi=100)
plt.close()

player.add_step("Step 2: Training a Decision Tree", f"""\
  {cyan('Code:')}
    tree = DecisionTreeClassifier(max_depth=3)
    tree.fit(X_train, y_train)

  {green('Result:')}
    Train accuracy: {tree.score(X_train, y_train):.4f}
    Test accuracy:  {tree.score(X_test, y_test):.4f}

  Tree visualization saved to images/07_decision_tree.png
  Each box shows: the split rule, how many samples, the predicted class.

  max_depth=3 means at most 3 levels of questions.
  Limiting depth prevents the tree from getting too specific.""")

# ── Step 3 ──────────────────────────────────────────────

depth_results = ""
for depth in [2, 3, 5, 10, None]:
    t = DecisionTreeClassifier(max_depth=depth, random_state=42)
    t.fit(X_train, y_train)
    tr = t.score(X_train, y_train)
    te = t.score(X_test, y_test)
    d = str(depth) if depth else "None"
    note = "← good balance" if depth == 3 else ("← overfitting" if depth is None else "")
    depth_results += f"    {d:>5}: train={tr:.4f}  test={te:.4f}  {note}\n"

player.add_step("Step 3: Depth vs Overfitting", f"""\
  Deeper tree = more questions = more specific rules.
  Too deep → memorizes training data (overfitting).

  {green('Accuracy by tree depth:')}
{depth_results}
  With unlimited depth, train accuracy = 1.0 (memorized everything)
  but test accuracy doesn't improve — it overfit.

  Same problem as lesson 04 (polynomial degree too high).
  Solution 1: limit depth.
  Solution 2: use many trees together (next step).""")

# ── Step 4 ──────────────────────────────────────────────

forest = RandomForestClassifier(n_estimators=100, random_state=42)
forest.fit(X_train, y_train)

tree_cv = cross_val_score(
    DecisionTreeClassifier(max_depth=3, random_state=42), X, y, cv=5
)
forest_cv = cross_val_score(
    RandomForestClassifier(n_estimators=100, random_state=42), X, y, cv=5
)

player.add_step("Step 4: Random Forest — Many Trees Vote Together", f"""\
  Problem: a single tree is unstable. Small changes → different tree.
  Solution: train 100 trees on random data subsets, then vote.

  {cyan('Code:')}
    forest = RandomForestClassifier(n_estimators=100)
    forest.fit(X_train, y_train)

  {green('Result:')}
    Forest train accuracy: {forest.score(X_train, y_train):.4f}
    Forest test accuracy:  {forest.score(X_test, y_test):.4f}

  {green('Cross-Validation Comparison:')}
    Decision Tree: {tree_cv.mean():.4f} (+/- {tree_cv.std():.4f})
    Random Forest: {forest_cv.mean():.4f} (+/- {forest_cv.std():.4f})

  Random Forest: higher accuracy, lower variance.
  This is the power of "ensemble" methods — crowds are smarter.""")

# ── Step 5 ──────────────────────────────────────────────

importances = forest.feature_importances_
indices = np.argsort(importances)[::-1]

top5 = ""
for i in range(5):
    bar = "#" * int(importances[indices[i]] * 40)
    top5 += f"    {i+1}. {wine.feature_names[indices[i]]:>30s}: {importances[indices[i]]:.4f}  {bar}\n"

fig, ax = plt.subplots(figsize=(10, 6))
ax.barh(range(len(importances)), importances[indices], align='center')
ax.set_yticks(range(len(importances)))
ax.set_yticklabels([wine.feature_names[i] for i in indices])
ax.set_xlabel('Importance')
ax.set_title('Random Forest Feature Importance')
ax.invert_yaxis()
plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, "images", "07_feature_importance.png"))
plt.close()

player.add_step("Step 5: Feature Importance", f"""\
  Random Forest tells us which features matter most for predictions.

  {green('Top 5 features:')}
{top5}
  Plot saved to images/07_feature_importance.png

  {green('Why this matters:')}
    - Drop unimportant features → simpler, faster model
    - Understand your data → which measurements actually matter?
    - Debug problems → unexpected important features = data leak?

  {green('Summary:')}
    ✓ Decision tree = flowchart of if/else rules
    ✓ Limit depth to prevent overfitting
    ✓ Random forest = 100 trees voting → more accurate, more stable
    ✓ Feature importance = which inputs matter most""")

# ── Play ────────────────────────────────────────────────

player.play()
