"""
Embedding Training — Animated ASCII visualization.

Watch how sentence vectors move during training.
Similar sentences get pulled CLOSER, dissimilar ones get pushed APART.

This simulates how embedding models (like OpenAI's text-embedding, Sentence-BERT)
learn to produce meaningful vectors.

Press Ctrl+C to stop.

=== HOW REAL EMBEDDING MODELS ARE TRAINED ===

Data required:
  - Pairs of similar sentences (positive pairs)
  - Pairs of dissimilar sentences (negative pairs)
  - NO manual vector labels — the vectors are LEARNED

Where this data comes from:
  - Question + answer from the same page (similar)
  - Two random sentences from different pages (dissimilar)
  - Paraphrase datasets
  - Search query + clicked result (similar)

The training objective (contrastive loss):
  - If pair is similar:    make their vectors closer   (reduce distance)
  - If pair is dissimilar: make their vectors farther   (increase distance)
"""
import numpy as np
import os
import time
import math

np.random.seed(42)

# === 10 SENTENCES grouped by topic ===
# The model must LEARN which are similar — we provide pair labels
sentences = [
    "The cat sat on the mat",           # 0 - animals
    "A dog played in the park",         # 1 - animals
    "The kitten slept on the sofa",     # 2 - animals
    "Python is a programming language", # 3 - tech
    "JavaScript runs in the browser",   # 4 - tech
    "Code review improves quality",     # 5 - tech
    "The sun rises in the east",        # 6 - nature
    "Rain falls from the clouds",       # 7 - nature
    "Stars shine in the night sky",     # 8 - nature
    "The stock market crashed today",   # 9 - finance
]

# Topic labels (for display only — the model doesn't see these)
topics = ["animal", "animal", "animal", "tech", "tech", "tech",
          "nature", "nature", "nature", "finance"]

topic_colors = {
    "animal":  "🐱",
    "tech":    "💻",
    "nature":  "🌿",
    "finance": "💰",
}

# === TRAINING PAIRS ===
# This is the data we curate: which sentences are similar?
# Format: (idx_a, idx_b, similar?)
similar_pairs = [
    (0, 1, True),   # cat & dog → similar (animals)
    (0, 2, True),   # cat & kitten → similar
    (1, 2, True),   # dog & kitten → similar
    (3, 4, True),   # python & javascript → similar (tech)
    (3, 5, True),   # python & code review → similar
    (4, 5, True),   # javascript & code review → similar
    (6, 7, True),   # sun & rain → similar (nature)
    (6, 8, True),   # sun & stars → similar
    (7, 8, True),   # rain & stars → similar
    (0, 3, False),  # cat & python → dissimilar
    (0, 6, False),  # cat & sun → dissimilar
    (0, 9, False),  # cat & stock → dissimilar
    (3, 6, False),  # python & sun → dissimilar
    (3, 9, False),  # python & stock → dissimilar
    (6, 9, False),  # sun & stock → dissimilar
    (1, 4, False),  # dog & javascript → dissimilar
    (2, 7, False),  # kitten & rain → dissimilar
    (5, 8, False),  # code review & stars → dissimilar
]

# === INITIALIZE random embeddings (2D so we can visualize) ===
# In reality: 384-3072 dimensions. We use 2D for ASCII plotting.
embeddings = np.random.randn(10, 2) * 0.5

learning_rate = 0.05
epochs = 500


def cosine_sim(a, b):
    dot = np.dot(a, b)
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    if norm < 1e-10:
        return 0.0
    return dot / norm


def contrastive_step(embeddings, pairs, lr):
    """One training step: push similar pairs closer, dissimilar pairs apart."""
    total_loss = 0.0
    grad = np.zeros_like(embeddings)

    for i, j, is_similar in pairs:
        diff = embeddings[i] - embeddings[j]
        dist = np.linalg.norm(diff)

        if is_similar:
            # Pull together: minimize distance, but keep a small gap
            target_dist = 0.3
            if dist > target_dist:
                loss = (dist - target_dist) ** 2
                direction = diff / (dist + 1e-10)
                grad[i] -= lr * 2 * (dist - target_dist) * direction
                grad[j] += lr * 2 * (dist - target_dist) * direction
            else:
                loss = 0.0
        else:
            # Push apart: maximize distance (up to margin)
            margin = 3.0
            if dist < margin:
                loss = (margin - dist) ** 2
                direction = diff / (dist + 1e-10)
                grad[i] += lr * 2 * (margin - dist) * direction
                grad[j] -= lr * 2 * (margin - dist) * direction
            else:
                loss = 0.0

        total_loss += loss

    embeddings += grad
    return total_loss / len(pairs)


def plot_embeddings(embeddings, width=60, height=25):
    """Plot 2D embeddings as ASCII."""
    grid = [[' ' for _ in range(width)] for _ in range(height)]

    # Find bounds with padding
    x_min = embeddings[:, 0].min() - 0.5
    x_max = embeddings[:, 0].max() + 0.5
    y_min = embeddings[:, 1].min() - 0.5
    y_max = embeddings[:, 1].max() + 0.5

    # Keep aspect ratio reasonable
    x_range = max(x_max - x_min, 1.0)
    y_range = max(y_max - y_min, 1.0)

    # Draw axes if visible
    zero_col = int((0 - x_min) / x_range * (width - 1))
    zero_row = int((y_max - 0) / y_range * (height - 1))
    if 0 <= zero_col < width:
        for r in range(height):
            if grid[r][zero_col] == ' ':
                grid[r][zero_col] = '·'
    if 0 <= zero_row < height:
        for c in range(width):
            if grid[zero_row][c] == ' ':
                grid[zero_row][c] = '·'

    # Plot each sentence as its index number
    for idx in range(len(embeddings)):
        col = int((embeddings[idx, 0] - x_min) / x_range * (width - 1))
        row = int((y_max - embeddings[idx, 1]) / y_range * (height - 1))
        col = max(0, min(width - 1, col))
        row = max(0, min(height - 1, row))
        grid[row][col] = str(idx)

    return '\n'.join('  ' + ''.join(row) for row in grid)


def similarity_matrix(embeddings):
    """Show cosine similarities between all pairs."""
    n = len(embeddings)
    lines = []
    lines.append(f"  {'':>4}" + "".join(f"{i:>5}" for i in range(n)))
    lines.append("  " + "-" * (4 + 5 * n))
    for i in range(n):
        row = f"  {i:>3}|"
        for j in range(n):
            sim = cosine_sim(embeddings[i], embeddings[j])
            if i == j:
                row += "    -"
            elif sim > 0.8:
                row += f" {sim:>.2f}"
            elif sim > 0.5:
                row += f" {sim:>.2f}"
            else:
                row += f" {sim:>.2f}"
        lines.append(row)
    return '\n'.join(lines)


def render(epoch, loss, embeddings):
    """Render the full ASCII frame."""
    lines = []
    lines.append("╔═══════════════════════════════════════════════════════════════════╗")
    lines.append("║       EMBEDDING TRAINING — CONTRASTIVE LEARNING                  ║")
    lines.append("║       Watch vectors cluster by topic during training              ║")
    lines.append("╠═══════════════════════════════════════════════════════════════════╣")
    lines.append(f"║  Epoch: {epoch:>4}/{epochs}        Loss: {loss:.4f}                          ║")
    lines.append("╠═══════════════════════════════════════════════════════════════════╣")
    lines.append("║  SENTENCES (each number = a sentence)                            ║")
    lines.append("║                                                                   ║")

    for idx, (sent, topic) in enumerate(zip(sentences, topics)):
        icon = topic_colors[topic]
        vec = embeddings[idx]
        line = f"║  {idx} {icon} {sent:<40} [{vec[0]:+.2f}, {vec[1]:+.2f}]  ║"
        lines.append(line)

    lines.append("║                                                                   ║")
    lines.append("╠═══════════════════════════════════════════════════════════════════╣")
    lines.append("║  2D EMBEDDING SPACE  (similar sentences should cluster)           ║")
    lines.append("║" + " " * 67 + "║")

    plot = plot_embeddings(embeddings)
    for plot_line in plot.split('\n'):
        padded = f"║{plot_line:<67}║"
        lines.append(padded)

    lines.append("║" + " " * 67 + "║")
    lines.append("╠═══════════════════════════════════════════════════════════════════╣")
    lines.append("║  COSINE SIMILARITY (within-topic should be high)                  ║")
    lines.append("║" + " " * 67 + "║")

    sim = similarity_matrix(embeddings)
    for sim_line in sim.split('\n'):
        padded = f"║{sim_line:<67}║"
        lines.append(padded)

    lines.append("║" + " " * 67 + "║")

    # Show cluster quality
    within_sims = []
    across_sims = []
    for i in range(10):
        for j in range(i + 1, 10):
            s = cosine_sim(embeddings[i], embeddings[j])
            if topics[i] == topics[j]:
                within_sims.append(s)
            else:
                across_sims.append(s)
    avg_within = sum(within_sims) / len(within_sims)
    avg_across = sum(across_sims) / len(across_sims)

    lines.append("╠═══════════════════════════════════════════════════════════════════╣")
    within_bar = "█" * int(max(0, avg_within) * 20)
    across_bar = "█" * int(max(0, avg_across) * 20)
    lines.append(f"║  Avg similarity WITHIN topic:  {avg_within:+.3f}  {within_bar:<20}    ║")
    lines.append(f"║  Avg similarity ACROSS topics: {avg_across:+.3f}  {across_bar:<20}    ║")
    gap = avg_within - avg_across
    if gap > 0.8:
        quality = "★ Excellent clustering!"
    elif gap > 0.5:
        quality = "◆ Good separation"
    elif gap > 0.2:
        quality = "○ Learning..."
    else:
        quality = "· Not yet separated"
    lines.append(f"║  Gap (higher = better):        {gap:+.3f}  {quality:<20}      ║")

    lines.append("╠═══════════════════════════════════════════════════════════════════╣")
    if loss < 0.1:
        lines.append("║  ★ Topics are clustered! Embedding model trained!                ║")
    else:
        lines.append("║  Training... (Ctrl+C to stop)                                    ║")
    lines.append("╚═══════════════════════════════════════════════════════════════════╝")

    return "\n".join(lines)


# === TRAINING LOOP ===
try:
    for epoch in range(epochs + 1):
        loss = contrastive_step(embeddings, similar_pairs, learning_rate)

        # Decrease learning rate over time
        if epoch % 100 == 0 and epoch > 0:
            learning_rate *= 0.8

        if epoch % 5 == 0:
            os.system("clear" if os.name != "nt" else "cls")
            print(render(epoch, loss, embeddings))
            time.sleep(0.05)

    # Final frame
    os.system("clear" if os.name != "nt" else "cls")
    print(render(epochs, loss, embeddings))

    print("\n=== HOW TO CURATE DATA FOR EMBEDDING TRAINING ===")
    print()
    print("  What you need:")
    print("    1. POSITIVE pairs: sentences that SHOULD be similar")
    print("       - Same topic, paraphrases, question+answer")
    print("    2. NEGATIVE pairs: sentences that should NOT be similar")
    print("       - Different topics, random pairs")
    print()
    print("  Where to get pairs:")
    print("    - FAQ: question + its answer = positive pair")
    print("    - Wiki: sentences from same article = positive pair")
    print("    - Search logs: query + clicked result = positive pair")
    print("    - Random sampling: any two unrelated sentences = negative pair")
    print()
    print("  For RAG specifically:")
    print("    - Use your actual user queries + the docs that answer them")
    print("    - Fine-tune an embedding model on YOUR domain data")
    print("    - This makes retrieval much more accurate for your use case")

except KeyboardInterrupt:
    print("\n\nStopped by user.")
