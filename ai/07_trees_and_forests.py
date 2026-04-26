"""
Lesson 07: Decision Trees & Random Forests

Goal: learn tree-based models — interpretable, powerful, and
      the foundation for gradient boosting (XGBoost, LightGBM).

Type along:
    python3
    >>> from sklearn.datasets import load_wine
    >>> from sklearn.model_selection import train_test_split
    >>> wine = load_wine(); X, y = wine.data, wine.target
    >>> X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    >>> from sklearn.tree import DecisionTreeClassifier
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
  {cyan('A decision tree is a flowchart of yes/no questions.')}

      "Is alcohol > 13.0?"
         ├── Yes → "Is color_intensity > 3.5?"
         │           ├── Yes → Class 0 (type A wine)
         │           └── No  → Class 2 (type C wine)
         └── No  → "Is proline > 700?"
                     ├── Yes → Class 1 (type B wine)
                     └── No  → Class 2 (type C wine)

  {green('How it\'s built (training):')}
      At each step, the algorithm tries EVERY feature and EVERY
      threshold, and picks the split that best separates the classes.

      All data
      ├── alcohol > 13.0?  (best split found by trying all features)
      │   ├── Yes: 50 samples (mostly class 0)
      │   └── No:  74 samples (mostly class 1 and 2)
      └── then split each group again... until pure or max depth

  {green('Wine Dataset:')}
      Samples:  {X.shape[0]} wines
      Features: {X.shape[1]} chemical measurements
      Classes:  {list(wine.target_names)} (3 wine types)

  {cyan('Try:')}
      from sklearn.datasets import load_wine
      wine = load_wine()
      wine.feature_names    →  {wine.feature_names[:5]}  ...""",

detail=f"""\
  {green('Term: Gini Impurity')}
      How does the tree decide which split is "best"?
      It uses Gini impurity — a measure of how mixed a group is.

      Gini = 1 - Σ(pᵢ²)    where pᵢ = fraction of class i

      Pure group (all class A):  Gini = 1 - 1.0² = 0.0   (perfect)
      Mixed (50/50):             Gini = 1 - 0.5² - 0.5² = 0.5 (worst for 2 classes)

      The tree picks splits that REDUCE Gini the most.
      Like a game of 20 Questions — pick the most informative question.

  {green('Term: Information Gain')}
      = parent Gini - weighted average of children's Gini
      Higher information gain = better split.

  {green('Alternative: Entropy')}
      Another impurity measure: Entropy = -Σ(pᵢ log₂ pᵢ)
      Gini and entropy give very similar trees in practice.""")

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

player.add_step("Step 2: Train and Visualize a Tree", f"""\
  {cyan('Type:')}
      from sklearn.model_selection import train_test_split
      X, y = wine.data, wine.target
      X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

      from sklearn.tree import DecisionTreeClassifier, plot_tree
      tree = DecisionTreeClassifier(max_depth=3, random_state=42)
      tree.fit(X_train, y_train)

  {green('Result:')}
      Train accuracy: {tree.score(X_train, y_train):.4f}
      Test accuracy:  {tree.score(X_test, y_test):.4f}

  {cyan('Visualize the tree:')}
      import matplotlib.pyplot as plt
      fig, ax = plt.subplots(figsize=(20, 8))
      plot_tree(tree, feature_names=wine.feature_names,
                class_names=wine.target_names, filled=True, ax=ax)
      plt.savefig('images/07_decision_tree.png')

  {green('Open images/07_decision_tree.png to see the tree!')}
      Each box shows:
          - The split rule (e.g., "proline <= 755.0")
          - Gini impurity (how mixed the group is)
          - Number of samples in this group
          - Predicted class (color-coded)

  {cyan('max_depth=3')} limits the tree to 3 levels of questions.
  This prevents overfitting (too specific rules).""")

# ── Step 3 ──────────────────────────────────────────────

depth_results = ""
for depth in [2, 3, 5, 10, None]:
    t = DecisionTreeClassifier(max_depth=depth, random_state=42)
    t.fit(X_train, y_train)
    tr = t.score(X_train, y_train)
    te = t.score(X_test, y_test)
    d = str(depth) if depth else "None(unlimited)"
    note = " ← good" if depth == 3 else (" ← overfit" if depth is None else "")
    depth_results += f"      {d:>16}: train={tr:.4f}  test={te:.4f}{note}\n"

player.add_step("Step 3: Depth vs Overfitting", f"""\
  {cyan('Deeper tree = more questions = more specific rules.')}
  Too deep → memorizes training data.

  {cyan('Type:')}
      for depth in [2, 3, 5, 10, None]:
          t = DecisionTreeClassifier(max_depth=depth)
          t.fit(X_train, y_train)
          print(depth, t.score(X_train, y_train), t.score(X_test, y_test))

  {green('Result:')}
{depth_results}
  {green('Observation:')}
      depth=None: train=1.000 (memorized everything!)
      but test doesn't improve — the extra specificity is noise.

      Same pattern as lesson 04 (polynomial degree too high).
      Trees overfit easily.  That's why we need FORESTS.

  {cyan('Analogy:')}
      Shallow tree: "If alcohol > 13 → type A"  (general rule)
      Deep tree:    "If alcohol=13.12 and hue=1.04 and ..."  (memorized)""")

# ── Step 4 ──────────────────────────────────────────────

forest = RandomForestClassifier(n_estimators=100, random_state=42)
forest.fit(X_train, y_train)

tree_cv = cross_val_score(
    DecisionTreeClassifier(max_depth=3, random_state=42), X, y, cv=5
)
forest_cv = cross_val_score(
    RandomForestClassifier(n_estimators=100, random_state=42), X, y, cv=5
)

player.add_step("Step 4: Random Forest — Wisdom of the Crowd", f"""\
  {cyan('Problem:')} a single tree is unreliable.  Slightly different data
  → completely different tree.  Prone to overfitting.

  {cyan('Solution: Random Forest')} — train 100 trees, each on:
      - A random SUBSET of the training data (bagging)
      - A random SUBSET of features at each split

      Then let all 100 trees VOTE on the prediction.

  {cyan('Type:')}
      from sklearn.ensemble import RandomForestClassifier
      forest = RandomForestClassifier(n_estimators=100)
      forest.fit(X_train, y_train)
      forest.score(X_train, y_train)
      forest.score(X_test, y_test)

  {green('Result:')}
      Forest train: {forest.score(X_train, y_train):.4f}
      Forest test:  {forest.score(X_test, y_test):.4f}

  {green('Cross-Validation comparison:')}
      Single tree:   {tree_cv.mean():.4f} (±{tree_cv.std():.4f})
      Random Forest: {forest_cv.mean():.4f} (±{forest_cv.std():.4f})

  {green('Random Forest wins:')} higher accuracy, lower variance.""",

detail=f"""\
  {green('Term: Ensemble Learning')}
      Combining multiple weak models into one strong model.
      Random Forest is an ensemble of decision trees.

      Types of ensembles:
      {cyan('Bagging')} (what Random Forest does):
          Train many models on random subsets → average/vote.
          Reduces VARIANCE (overfitting).

      {cyan('Boosting')} (XGBoost, LightGBM, CatBoost):
          Train models sequentially, each fixing the previous one's errors.
          Reduces BIAS (underfitting).

      {cyan('Stacking:')}
          Train different model TYPES, then train a meta-model on their outputs.

  {green('Why randomness helps:')}
      If every tree sees the same data and features,
      they'd all make the same mistakes.
      Randomness → diversity → errors cancel out.
      Same reason polling 100 random people is more reliable
      than asking 1 expert.""")

# ── Step 5 ──────────────────────────────────────────────

importances = forest.feature_importances_
indices = np.argsort(importances)[::-1]

top5 = ""
for i in range(5):
    bar = "█" * int(importances[indices[i]] * 40)
    top5 += f"      {i+1}. {wine.feature_names[indices[i]]:>22s}: {importances[indices[i]]:.4f}  {bar}\n"

bottom3 = ""
for i in range(-1, -4, -1):
    bottom3 += f"         {wine.feature_names[indices[i]]:>22s}: {importances[indices[i]]:.4f}\n"

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

player.add_step("Step 5: Feature Importance — Which Inputs Matter?", f"""\
  {cyan('Random Forest tells you which features matter most.')}

  {cyan('Type:')}
      forest.feature_importances_
      # then sort and display

  {green('Top 5 most important features:')}
{top5}
  {green('Least important:')}
{bottom3}
  {green('Plot saved to images/07_feature_importance.png')}

  {cyan('Why this matters in practice:')}
      - {green('Feature selection:')} drop unimportant features → simpler model
      - {green('Understanding:')} which measurements actually distinguish wines?
      - {green('Debugging:')} if an irrelevant feature ranks high → data leak?
      - {green('Cost:')} if the most important feature is expensive to measure,
        can you achieve similar accuracy without it?

  {green('Term: Feature Importance')}
      Measured by how much each feature reduces Gini impurity
      across all trees.  Higher = more useful for predictions.""")

# ── Step 6 ──────────────────────────────────────────────

player.add_step("Step 6: Summary", f"""\
  {green('What you learned:')}

      {cyan('1. Decision Tree')} = flowchart of yes/no questions.
         Splits data by the feature/threshold that best separates classes.
         Uses Gini impurity to measure "best."

      {cyan('2. Overfitting in trees:')} unlimited depth → memorization.
         Fix: limit max_depth, or use a forest.

      {cyan('3. Random Forest')} = 100 trees voting together.
         Each tree trained on random data subset + random features.
         More accurate AND more stable than a single tree.

      {cyan('4. Feature Importance')} = which inputs matter most.
         Use it for feature selection and understanding your data.

  {green('ML vocabulary added:')}
      Gini impurity, information gain, bagging, ensemble,
      feature importance, random forest.

  {green('Decision tree vs other models:')}
      ✓ No scaling needed (trees don't use distances)
      ✓ Handles mixed types (numbers + categories)
      ✓ Interpretable (you can read the tree)
      ✗ Single trees overfit easily
      ✗ Not great for very high-dimensional data

  {green('What\'s next?')}
      Lesson 08: unsupervised learning — what if you have NO labels?""")

# ── Play ────────────────────────────────────────────────

player.play()
