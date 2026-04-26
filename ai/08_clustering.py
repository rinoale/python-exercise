"""
Lesson 08: K-Means Clustering — Unsupervised Learning

Goal: learn what to do when you have NO labels.
      K-Means discovers groups in data automatically.

Type along:
    python3
    >>> import matplotlib.pyplot as plt
    >>> from sklearn.datasets import make_blobs
    >>> X, _ = make_blobs(n_samples=300, centers=4, cluster_std=0.8, random_state=42)
    >>> from sklearn.cluster import KMeans
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from lesson_player import LessonPlayer, cyan, green

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

player = LessonPlayer("Lesson 08: K-Means Clustering")

# ── Step 1 ──────────────────────────────────────────────

player.add_step("Step 1: Supervised vs Unsupervised Learning", f"""\
  {cyan('Everything so far was SUPERVISED learning:')}
      We had LABELED data:  (input, correct_answer)
      "Here's a flower measurement → it's setosa"
      The model learns to predict the correct label.

  {cyan('UNSUPERVISED learning:')}
      We have UNLABELED data:  just inputs, no answers.
      "Here are 300 data points → find structure yourself"

  {green('Analogy:')}
      Supervised:   teacher gives you the answer key
      Unsupervised: you sort a pile of objects into groups
                    without being told what the groups are

  {green('Real-world unsupervised examples:')}
      Customer segmentation — group customers by behavior
        (nobody told you what the groups are)
      Anomaly detection    — find unusual transactions
      Topic modeling       — discover themes in documents
      Image compression    — group similar colors together

  {green('Term: Clustering')}
      Grouping similar data points together.
      K-Means is the most common clustering algorithm.""")

# ── Step 2 ──────────────────────────────────────────────

X, y_true = make_blobs(n_samples=300, centers=4, cluster_std=0.8, random_state=42)

player.add_step("Step 2: How K-Means Works — The Algorithm", f"""\
  {cyan('K-Means groups data into K clusters in 4 steps:')}

      Step 1: Place K random points (centroids) in the space
      Step 2: Assign each data point to its NEAREST centroid
      Step 3: Move each centroid to the MEAN of its assigned points
      Step 4: Repeat steps 2-3 until centroids stop moving

  {green('Visual (K=2, 1D):')}

      Iteration 1:  Place random centroids (C)
        data:  · · ·        · · · ·
        C₁▼                    ▼C₂

      Iteration 2:  Assign to nearest, move centroids
        [· · ·]      [· · · ·]
           ▼C₁          ▼C₂

      Iteration 3:  Converged — centroids stopped moving
        [· · ·]      [· · · ·]
           ▼C₁          ▼C₂    (same as before → done!)

  {cyan('Generate test data:')}
      from sklearn.datasets import make_blobs
      X, _ = make_blobs(n_samples=300, centers=4, cluster_std=0.8)
      X.shape  →  {X.shape}  (300 points in 2D)

  {green('Term: Centroid')}
      The center point of a cluster.
      It's the mean (average) of all points in that cluster.""",

detail=f"""\
  {green('Term: Distance (how "nearest" is measured)')}

      K-Means uses {cyan('Euclidean distance')} by default:
          d = √((x₁-x₂)² + (y₁-y₂)²)

      In 2D, this is the Pythagorean theorem.
      In higher dimensions, same formula with more terms.

  {green('Term: Convergence')}
      When the algorithm stops changing — centroids don't move
      between iterations.  This is guaranteed to happen (though
      it may find a LOCAL optimum, not the global best).

  {cyan('Why does K-Means work?')}
      Each iteration reduces the total within-cluster distance.
      Assigning to nearest centroid → reduces distance.
      Moving centroid to mean → reduces distance further.
      So total error can only go DOWN, never UP → must converge.""")

# ── Step 3 ──────────────────────────────────────────────

kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
kmeans.fit(X)
labels = kmeans.labels_
centers = kmeans.cluster_centers_

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].scatter(X[:, 0], X[:, 1], alpha=0.5, s=30)
axes[0].set_title('Raw Data (no labels)')
axes[1].scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', alpha=0.5, s=30)
axes[1].scatter(centers[:, 0], centers[:, 1],
                c='red', marker='X', s=200, edgecolors='black', linewidths=2,
                label='Centroids')
axes[1].set_title('K-Means Clusters (K=4)')
axes[1].legend()
plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, "images", "08_clustering.png"))
plt.close()

centroid_text = ""
for i, center in enumerate(centers):
    centroid_text += f"      Cluster {i}: center=({center[0]:.2f}, {center[1]:.2f})  size={np.sum(labels == i)}\n"

player.add_step("Step 3: Run K-Means", f"""\
  {cyan('Type:')}
      from sklearn.cluster import KMeans
      kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
      kmeans.fit(X)

      kmeans.labels_            # which cluster each point belongs to
      kmeans.cluster_centers_   # centroid coordinates
      kmeans.inertia_           # total within-cluster distance

  {green('Result:')}
      labels[:10]  →  {labels[:10]}
      inertia      →  {kmeans.inertia_:.1f}

  {green('Clusters found:')}
{centroid_text}
  {green('Plot saved to images/08_clustering.png')}
      Left:  raw data — we can eyeball 4 groups
      Right: K-Means found them automatically (colored + red X centroids)

  {cyan('Key attributes:')}
      kmeans.labels_           →  cluster ID for each point (0-3)
      kmeans.cluster_centers_  →  centroid coordinates
      kmeans.inertia_          →  total squared distance to centroids""",

detail=f"""\
  {green('Term: Inertia (within-cluster sum of squares)')}
      inertia = Σ (distance from each point to its centroid)²

      Lower inertia = tighter clusters = better.
      But inertia ALWAYS decreases with more clusters:
      K=300 would give inertia≈0 (every point is its own cluster).
      So inertia alone can't pick the right K.

  {green('What is n_init=10?')}
      K-Means starts with RANDOM centroids.
      Different starting positions → different results.
      n_init=10 means: run 10 times, keep the best result.
      This helps avoid bad random starts.

  {green('Why random_state=42?')}
      K-Means uses randomness in two places:
          1. Choosing initial centroid positions
          2. Breaking ties when points are equidistant

      Without random_state, you get DIFFERENT results each run:
          Run 1:  labels = [0, 3, 2, 0, ...]   inertia = 358.8
          Run 2:  labels = [3, 3, 0, 1, ...]   inertia = 362.0
          Run 3:  labels = [2, 1, 3, 2, ...]   inertia = 358.8

      {cyan('Two things change:')}
          {cyan('Label numbers:')} the IDs (0,1,2,3) are arbitrary — cluster
          "0" in one run might be cluster "3" in another. The GROUPING
          is the same, just the names differ. Like calling the same
          team "Team A" vs "Team 1."

          {cyan('Inertia:')} slightly different starting centroids can land
          in different local optima. K-Means finds A good solution,
          not THE best solution. That's why n_init runs it multiple
          times — but with different random seeds, even n_init=10
          can give slightly different final results.

      random_state=42 fixes the random seed → same result every time.
      The number 42 is arbitrary (any integer works). It's standard
      in tutorials so everyone gets the same output.

      {cyan('In production:')} you usually DON'T set random_state.
      You want the algorithm to explore freely. Fixed seeds are
      for reproducibility in learning, testing, and debugging.""")

# ── Step 4 ──────────────────────────────────────────────

inertias = []
K_range = range(1, 11)
for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X)
    inertias.append(km.inertia_)

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(K_range, inertias, 'bo-')
ax.set_xlabel('K (number of clusters)')
ax.set_ylabel('Inertia (lower = tighter clusters)')
ax.set_title('Elbow Method')
ax.axvline(x=4, color='red', linestyle='--', label='Elbow at K=4')
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, "images", "08_elbow.png"))
plt.close()

elbow_text = ""
for k, inertia in zip(K_range, inertias):
    bar = "█" * int(inertia / max(inertias) * 30)
    marker = " ← elbow" if k == 4 else ""
    elbow_text += f"      K={k:2d}: {inertia:8.0f}  {bar}{marker}\n"

player.add_step("Step 4: Choosing K — The Elbow Method", f"""\
  {cyan('We used K=4 because we knew there are 4 groups.')}
  {cyan('In real life, you don\'t know K.  How to find it?')}

  {green('Elbow Method:')} try K=1 to 10, plot inertia vs K.

  {cyan('Type:')}
      import matplotlib.pyplot as plt
      inertias = []
      for k in range(1, 11):
          km = KMeans(n_clusters=k, n_init=10)
          km.fit(X)
          inertias.append(km.inertia_)
      plt.plot(range(1, 11), inertias, 'bo-')

  {green('Inertia by K:')}
{elbow_text}
  {green('How to read it:')}
      Inertia drops sharply K=1 → K=4.
      After K=4, the improvement is tiny.
      The "elbow" (bend in the curve) = best K.

  {green('Plot saved to images/08_elbow.png')}

  {cyan('Intuition:')} adding cluster 5 barely helps because
  the real data only has 4 groups.  Extra clusters just
  split existing groups arbitrarily.""")

# ── Step 5 ──────────────────────────────────────────────

sil_text = ""
best_k = 2
best_score = -1
for k in range(2, 8):
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    lbl = km.fit_predict(X)
    score = silhouette_score(X, lbl)
    if score > best_score:
        best_score = score
        best_k = k
    best = " ← best" if k == best_k else ""
    sil_text += f"      K={k}: silhouette = {score:.4f}{best}\n"

player.add_step("Step 5: Silhouette Score — Cluster Quality", f"""\
  {cyan('Another way to pick K and measure cluster quality.')}

  {green('Silhouette score for each point:')}
      s = (b - a) / max(a, b)

      a = average distance to OTHER points in SAME cluster
      b = average distance to points in NEAREST OTHER cluster

      s = +1  →  point is far from other clusters (great)
      s =  0  →  point is on the boundary (ambiguous)
      s = -1  →  point is in the wrong cluster (bad)

  {cyan('Type:')}
      from sklearn.metrics import silhouette_score
      for k in range(2, 8):
          km = KMeans(n_clusters=k, n_init=10)
          labels = km.fit_predict(X)
          print(k, silhouette_score(X, labels))

  {green('Result:')}
{sil_text}
  {green('Both Elbow and Silhouette agree: K={best_k} is best.')}

  {cyan('When they disagree:')}
      Silhouette is more reliable.  Elbow can be ambiguous
      (sometimes there's no clear bend).""")

# ── Step 6 ──────────────────────────────────────────────

player.add_step("Step 6: Summary", f"""\
  {green('What you learned:')}

      {cyan('1. Unsupervised learning')} = no labels, discover structure.
         vs Supervised = has labels, learn to predict.

      {cyan('2. K-Means algorithm:')}
         Place centroids → assign nearest → move to mean → repeat.

      {cyan('3. Choosing K:')}
         Elbow method = plot inertia, find the bend.
         Silhouette score = measure cluster quality (-1 to +1).

      {cyan('4. K-Means limitations:')}
         - You must choose K in advance
         - Assumes spherical clusters (bad for elongated shapes)
         - Sensitive to random initialization (use n_init)
         - Sensitive to outliers (centroid gets pulled)

  {green('Alternatives to K-Means:')}
      DBSCAN    — finds clusters of any shape, auto-detects K
      Hierarchical — builds a tree of clusters, you choose where to cut
      GMM       — like K-Means but with soft assignments (probabilities)

  {green('What\'s next?')}
      Lesson 09: neural networks — models with LAYERS of weights.
      Everything so far was one layer.  Multiple layers = deep learning.""")

# ── Play ────────────────────────────────────────────────

player.play()
