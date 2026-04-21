"""
Lesson 08: K-Means Clustering
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

player.add_step("Step 1: Supervised vs Unsupervised", f"""\
  {cyan('Supervised learning')} (lessons 03-07):
    Training data has labels: (input, correct_answer)
    "Here are 150 flowers with their species → learn the pattern"

  {cyan('Unsupervised learning')} (this lesson):
    Training data has NO labels: just inputs
    "Here are 300 data points → find the groups yourself"

  {green('Real-world examples:')}
    - Customer segmentation (no predefined groups)
    - Anomaly detection (what's "normal"?)
    - Image compression (group similar colors)
    - Topic discovery in documents""")

# ── Step 2 ──────────────────────────────────────────────

X, y_true = make_blobs(n_samples=300, centers=4, cluster_std=0.8, random_state=42)

player.add_step("Step 2: How K-Means Works", f"""\
  K-Means groups data into K clusters. Algorithm:

    1. Place K random centroids (cluster centers)
    2. Assign each point to its nearest centroid
    3. Move each centroid to the mean of its assigned points
    4. Repeat steps 2-3 until centroids stop moving

  {cyan('Visualize:')}
    Step 1:  x   x   o o o   x    (x=centroid, o=data)
    Step 2:  [x  o o] [o o  x]    (assign to nearest)
    Step 3:  [  x o o] [o x  ]    (move centroid to mean)
    Step 4:  repeat until stable

  We have 300 points in 2D with 4 hidden clusters.
  Let's see if K-Means can find them.""")

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
    centroid_text += f"    Cluster {i}: center=({center[0]:.2f}, {center[1]:.2f})  size={np.sum(labels == i)}\n"

player.add_step("Step 3: K-Means Result", f"""\
  {cyan('Code:')}
    kmeans = KMeans(n_clusters=4)
    kmeans.fit(X)
    labels = kmeans.labels_        # which cluster each point belongs to
    centers = kmeans.cluster_centers_

  {green('Result:')}
{centroid_text}
  Plot saved to images/08_clustering.png
    Left:  raw data (we can see groups, but the computer doesn't)
    Right: K-Means found the 4 clusters (colored) with centroids (red X)""")

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
    bar = "#" * int(inertia / max(inertias) * 30)
    marker = " ← elbow" if k == 4 else ""
    elbow_text += f"    K={k:2d}: {inertia:8.0f}  {bar}{marker}\n"

player.add_step("Step 4: Choosing K — Elbow Method", f"""\
  We used K=4 because we knew the answer. In real life, we don't.
  The Elbow Method: try different K, plot inertia (total distance).

  {green('Inertia by K:')}
{elbow_text}
  Inertia drops sharply until K=4, then barely improves.
  The "elbow" in the curve = best K.

  Plot saved to images/08_elbow.png""")

# ── Step 5 ──────────────────────────────────────────────

sil_text = ""
for k in range(2, 8):
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    lbl = km.fit_predict(X)
    score = silhouette_score(X, lbl)
    best = " ← best" if k == 4 else ""
    sil_text += f"    K={k}: silhouette = {score:.4f}{best}\n"

player.add_step("Step 5: Silhouette Score", f"""\
  Another way to pick K. Silhouette score measures cluster quality:
    +1 = points are far from other clusters (great)
     0 = points are on the border (ambiguous)
    -1 = points are in the wrong cluster (bad)

  {green('Silhouette by K:')}
{sil_text}
  Both Elbow and Silhouette agree: K=4 is best.

  {green('Summary:')}
    ✓ Unsupervised = no labels, discover structure
    ✓ K-Means = assign points to nearest centroid, repeat
    ✓ Elbow method = find the bend in the inertia curve
    ✓ Silhouette score = measure cluster quality (-1 to +1)""")

# ── Play ────────────────────────────────────────────────

player.play()
