"""
Attention Training — Animated ASCII visualization.

Watch a tiny transformer LEARN attention from scratch.
W_Q, W_K, W_V start random. Over epochs, the model learns
which words to attend to for next-token prediction.

Task: predict the last word in simple patterns.
  "the cat sat"    -> "down"
  "the dog ran"    -> "fast"
  "the bird flew"  -> "high"

You'll see:
  - Loss decreasing epoch by epoch
  - Attention weights shifting from random to meaningful
  - W_Q, W_K, W_V values converging
  - Predictions improving from wrong to correct

Controls:
  SPACE  = toggle auto-play / pause
  Arrows = browse slides manually
  q      = quit

Press Ctrl+C to stop.
"""

import os
import sys
import math
import select
import termios
import tty


# -- Vocabulary and data --

VOCAB = ["the", "cat", "dog", "bird", "sat", "ran", "flew", "down", "fast", "high"]
WORD2ID = {w: i for i, w in enumerate(VOCAB)}
VOCAB_SIZE = len(VOCAB)

TRAIN_DATA = [
    (["the", "cat", "sat"], "down"),
    (["the", "dog", "ran"], "fast"),
    (["the", "bird", "flew"], "high"),
]

# -- Hyperparameters --

D_MODEL = 4
D_K = 3
LR = 0.05
EPOCHS = 800
SEED = 42


# -- Math utilities --

def softmax(vals):
    mx = max(vals)
    exps = [math.exp(v - mx) for v in vals]
    s = sum(exps)
    return [e / s for e in exps]


def mat_mul(A, B):
    rows_a, cols_a = len(A), len(A[0])
    cols_b = len(B[0])
    C = [[0.0] * cols_b for _ in range(rows_a)]
    for i in range(rows_a):
        for j in range(cols_b):
            for k in range(cols_a):
                C[i][j] += A[i][k] * B[k][j]
    return C


def mat_vec(M, v):
    return [sum(M[i][j] * v[j] for j in range(len(v))) for i in range(len(M))]


def transpose(M):
    rows, cols = len(M), len(M[0])
    return [[M[i][j] for i in range(rows)] for j in range(cols)]


def dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def randn():
    # Box-Muller from a simple LCG
    global _seed
    _seed = (_seed * 1103515245 + 12345) & 0x7fffffff
    u1 = (_seed / 0x7fffffff) * 0.9998 + 0.0001
    _seed = (_seed * 1103515245 + 12345) & 0x7fffffff
    u2 = (_seed / 0x7fffffff)
    return math.sqrt(-2 * math.log(u1)) * math.cos(2 * math.pi * u2)


def init_matrix(rows, cols, scale=0.3):
    return [[randn() * scale for _ in range(cols)] for _ in range(rows)]


_seed = SEED


# -- Model parameters --

EMBEDDINGS = init_matrix(VOCAB_SIZE, D_MODEL)
W_Q = init_matrix(D_MODEL, D_K)
W_K = init_matrix(D_MODEL, D_K)
W_V = init_matrix(D_MODEL, D_K)
W_OUT = init_matrix(D_K, VOCAB_SIZE)

scale = math.sqrt(D_K)


# -- Forward pass --

def forward(input_words):
    seq_len = len(input_words)
    ids = [WORD2ID[w] for w in input_words]

    # Embedding lookup
    X = [list(EMBEDDINGS[i]) for i in ids]

    # Q, K, V for each token
    Qs = []
    Ks = []
    Vs = []
    for i in range(seq_len):
        Qs.append([sum(X[i][d] * W_Q[d][k] for d in range(D_MODEL)) for k in range(D_K)])
        Ks.append([sum(X[i][d] * W_K[d][k] for d in range(D_MODEL)) for k in range(D_K)])
        Vs.append([sum(X[i][d] * W_V[d][k] for d in range(D_MODEL)) for k in range(D_K)])

    # Attention for last token (causal: sees all tokens)
    last = seq_len - 1
    scores = [dot(Qs[last], Ks[j]) / scale for j in range(seq_len)]
    attn_weights = softmax(scores)

    # Weighted sum of values
    context = [0.0] * D_K
    for j in range(seq_len):
        for d in range(D_K):
            context[d] += attn_weights[j] * Vs[j][d]

    # Output projection to vocabulary logits
    logits = [sum(context[d] * W_OUT[d][v] for d in range(D_K)) for v in range(VOCAB_SIZE)]
    probs = softmax(logits)

    return X, Qs, Ks, Vs, attn_weights, context, logits, probs


# -- Loss and backward pass (manual gradient descent) --

def train_step():
    total_loss = 0.0
    all_attn = []
    all_probs = []

    for input_words, target in TRAIN_DATA:
        target_id = WORD2ID[target]
        X, Qs, Ks, Vs, attn_w, context, logits, probs = forward(input_words)
        all_attn.append((input_words, attn_w))
        all_probs.append((input_words, target, probs))

        # Cross-entropy loss
        total_loss += -math.log(probs[target_id] + 1e-10)

        # Gradient of loss w.r.t. logits (softmax + cross-entropy)
        d_logits = list(probs)
        d_logits[target_id] -= 1.0

        # Gradient through W_OUT: context^T @ d_logits
        d_context = [0.0] * D_K
        for d in range(D_K):
            for v in range(VOCAB_SIZE):
                W_OUT[d][v] -= LR * context[d] * d_logits[v]
                d_context[d] += W_OUT[d][v] * d_logits[v]

        # Gradient through attention weighted sum
        seq_len = len(input_words)
        last = seq_len - 1
        ids = [WORD2ID[w] for w in input_words]

        d_attn_weights = [0.0] * seq_len
        d_Vs = [[0.0] * D_K for _ in range(seq_len)]
        for j in range(seq_len):
            for d in range(D_K):
                d_attn_weights[j] += d_context[d] * Vs[j][d]
                d_Vs[j][d] += d_context[d] * attn_w[j]

        # Gradient through softmax
        d_scores = [0.0] * seq_len
        for i in range(seq_len):
            for j in range(seq_len):
                if i == j:
                    d_scores[i] += d_attn_weights[j] * attn_w[i] * (1 - attn_w[i])
                else:
                    d_scores[i] += d_attn_weights[j] * (-attn_w[i] * attn_w[j])

        # Scale gradient
        d_scores = [ds / scale for ds in d_scores]

        # Gradient through Q @ K^T for last token
        d_Q_last = [0.0] * D_K
        d_Ks = [[0.0] * D_K for _ in range(seq_len)]
        for j in range(seq_len):
            for d in range(D_K):
                d_Q_last[d] += d_scores[j] * Ks[j][d]
                d_Ks[j][d] += d_scores[j] * Qs[last][d]

        # Update W_Q (from last token only)
        for d in range(D_MODEL):
            for k in range(D_K):
                W_Q[d][k] -= LR * X[last][d] * d_Q_last[k]

        # Update W_K
        for j in range(seq_len):
            for d in range(D_MODEL):
                for k in range(D_K):
                    W_K[d][k] -= LR * X[j][d] * d_Ks[j][k]

        # Update W_V
        for j in range(seq_len):
            for d in range(D_MODEL):
                for k in range(D_K):
                    W_V[d][k] -= LR * X[j][d] * d_Vs[j][k]

        # Update embeddings
        for j in range(seq_len):
            eid = ids[j]
            for d in range(D_MODEL):
                grad = 0.0
                if j == last:
                    for k in range(D_K):
                        grad += d_Q_last[k] * W_Q[d][k]
                for k in range(D_K):
                    grad += d_Ks[j][k] * W_K[d][k]
                    grad += d_Vs[j][k] * W_V[d][k]
                EMBEDDINGS[eid][d] -= LR * grad

    return total_loss / len(TRAIN_DATA), all_attn, all_probs


# -- Pre-compute all epochs into slides --

WIDTH = 74


def title_line(text):
    pad = WIDTH - 4 - len(text)
    return "  " + text + " " * max(pad, 0)


def separator():
    return "  " + "-" * (WIDTH - 4)


def blank():
    return ""


def weight_bar(value, width=20):
    normalized = (value + 2) / 4
    normalized = max(0.0, min(1.0, normalized))
    pos = int(normalized * (width - 1))
    mid = width // 2
    line = list("-" * width)
    line[mid] = "|"
    pos = min(pos, width - 1)
    line[pos] = "#"
    return "".join(line)


def loss_bar(loss, max_loss=3.0):
    normalized = min(1.0, loss / max_loss)
    width = 30
    filled = int(normalized * width)
    return "#" * filled + "-" * (width - filled)


def build_slides():
    slides = []

    # Save initial state for the title slide
    w_q_init = [row[:] for row in W_Q]
    w_k_init = [row[:] for row in W_K]

    # -- Title slide --
    s = []
    s.append(blank())
    s.append(title_line("ATTENTION TRAINING -- WATCH IT LEARN"))
    s.append(title_line("A tiny transformer learns next-token prediction"))
    s.append(separator())
    s.append(blank())
    s.append(title_line("Task: predict the last word"))
    s.append(title_line('  "the cat sat"  -> "down"'))
    s.append(title_line('  "the dog ran"  -> "fast"'))
    s.append(title_line('  "the bird flew" -> "high"'))
    s.append(blank())
    s.append(title_line("The model starts with RANDOM weights."))
    s.append(title_line("It must learn:"))
    s.append(title_line("  - W_Q, W_K: which words to attend to"))
    s.append(title_line("  - W_V: what information to extract"))
    s.append(title_line("  - W_OUT: how to map context to predictions"))
    s.append(blank())
    s.append(separator())
    s.append(title_line("Watch attention weights shift from random to meaningful."))
    s.append(blank())
    slides.append(("Title", 0, s))

    # -- Training epochs --
    losses = []
    snapshot_epochs = []

    # Decide which epochs to snapshot
    # Dense at start (changes fast), sparse later
    snap_set = set()
    for e in range(0, 50, 2):
        snap_set.add(e)
    for e in range(50, 200, 5):
        snap_set.add(e)
    for e in range(200, EPOCHS + 1, 20):
        snap_set.add(e)
    snap_set.add(EPOCHS - 1)

    for epoch in range(EPOCHS):
        loss, all_attn, all_probs = train_step()
        losses.append(loss)

        if epoch not in snap_set:
            continue

        snapshot_epochs.append(epoch)

        s = []
        s.append(blank())
        s.append(title_line(f"EPOCH {epoch:>4d}/{EPOCHS}"))
        s.append(separator())

        # Loss with bar
        s.append(title_line(f"  Loss: {loss:.4f}  {loss_bar(loss)}"))
        s.append(blank())

        # Attention weights for each training example
        s.append(title_line("  ATTENTION WEIGHTS (last token looks at):"))
        s.append(blank())
        for input_words, attn_w in all_attn:
            label = " ".join(input_words)
            parts = []
            for j, w in enumerate(input_words):
                pct = int(attn_w[j] * 100)
                parts.append(f"{w}={pct:2d}%")
            attn_str = "  ".join(parts)
            s.append(title_line(f"    {label:>16}:  {attn_str}"))

        s.append(blank())

        # Predictions
        s.append(title_line("  PREDICTIONS:"))
        for input_words, target, probs in all_probs:
            label = " ".join(input_words)
            pred_id = probs.index(max(probs))
            pred_word = VOCAB[pred_id]
            pred_conf = probs[pred_id]
            target_prob = probs[WORD2ID[target]]
            ok = "ok" if pred_word == target else "X "
            s.append(title_line(
                f'    {label:>16} -> "{pred_word}" ({pred_conf:.0%})'
                f'   want "{target}" ({target_prob:.0%})  {ok}'
            ))

        s.append(blank())

        # W_Q weights (compact: show as flat row per row)
        s.append(title_line("  W_Q (query weights):"))
        for d in range(D_MODEL):
            vals = "  ".join(f"{W_Q[d][k]:+.2f}" for k in range(D_K))
            bars = "  ".join(weight_bar(W_Q[d][k], 10) for k in range(D_K))
            s.append(title_line(f"    [{vals}]  {bars}"))

        s.append(blank())

        s.append(title_line("  W_K (key weights):"))
        for d in range(D_MODEL):
            vals = "  ".join(f"{W_K[d][k]:+.2f}" for k in range(D_K))
            bars = "  ".join(weight_bar(W_K[d][k], 10) for k in range(D_K))
            s.append(title_line(f"    [{vals}]  {bars}"))

        s.append(blank())

        step = min(epoch + 1, 8)
        slides.append((f"Epoch {epoch}", step, s))

    # -- Final summary --
    s = []
    s.append(blank())
    s.append(title_line("TRAINING COMPLETE"))
    s.append(separator())
    s.append(blank())

    s.append(title_line(f"  Final loss: {losses[-1]:.4f}"))
    s.append(blank())

    # Show final predictions
    s.append(title_line("  Final predictions:"))
    for input_words, target in TRAIN_DATA:
        _, _, _, _, attn_w, _, _, probs = forward(input_words)
        pred_id = probs.index(max(probs))
        pred_word = VOCAB[pred_id]
        pred_conf = probs[pred_id]
        ok = "ok" if pred_word == target else "X"
        s.append(title_line(
            f'    "{" ".join(input_words)}" -> "{pred_word}" ({pred_conf:.0%})  {ok}'
        ))
    s.append(blank())

    # Show final attention patterns
    s.append(title_line("  Final attention patterns:"))
    for input_words, target in TRAIN_DATA:
        _, _, _, _, attn_w, _, _, _ = forward(input_words)
        parts = []
        for j, w in enumerate(input_words):
            pct = int(attn_w[j] * 100)
            parts.append(f"{w}={pct}%")
        s.append(title_line(f'    "{" ".join(input_words)}":  {", ".join(parts)}'))
    s.append(blank())

    s.append(separator())
    s.append(title_line("  What the model learned:"))
    s.append(title_line("    - W_Q/W_K learned to make the verb the highest-"))
    s.append(title_line("      scoring key for predicting the next word"))
    s.append(title_line("    - W_V learned to extract verb identity"))
    s.append(title_line("    - W_OUT learned to map verb context to target"))
    s.append(blank())
    s.append(title_line("  sat->down, ran->fast, flew->high"))
    s.append(title_line("  The attention mechanism figured out WHICH word"))
    s.append(title_line("  matters for the prediction, just from training data."))
    s.append(blank())
    s.append(title_line("  End of animation. Press q to quit, arrows to review."))
    s.append(blank())
    slides.append(("Summary", 9, s))

    return slides


# -- Terminal I/O --

def get_key(fd, timeout=None):
    if timeout is not None:
        ready, _, _ = select.select([fd], [], [], timeout)
        if not ready:
            return None
    ch = os.read(fd, 1)
    if ch == b'\x1b':
        ready, _, _ = select.select([fd], [], [], 0.1)
        if ready:
            ch2 = os.read(fd, 1)
            if ch2 == b'[':
                ch3 = os.read(fd, 1)
                if ch3 == b'A':
                    return 'UP'
                elif ch3 == b'B':
                    return 'DOWN'
                elif ch3 == b'C':
                    return 'RIGHT'
                elif ch3 == b'D':
                    return 'LEFT'
        return 'ESC'
    if ch == b' ':
        return 'SPACE'
    if ch in (b'q', b'Q'):
        return 'QUIT'
    return ch.decode('utf-8', errors='ignore')


def render_slide(slide, idx, total, auto_play, speed):
    name, step, lines = slide

    output = []
    output.append("\033[2J\033[H")
    output.append("")
    output.append(f"  ATTENTION TRAINING            Slide {idx + 1}/{total}    Part {step}/9")
    output.append("  " + "=" * (WIDTH - 4))

    for line in lines:
        output.append(line)

    output.append("  " + "=" * (WIDTH - 4))
    if auto_play:
        output.append(f"  >> AUTO-PLAY (SPACE=pause, arrows=browse, q=quit)  speed={speed:.1f}s")
    else:
        output.append("  == PAUSED (SPACE=play, arrows=browse, q=quit)")

    return "\r\n".join(output) + "\r\n"


def main():
    slides = build_slides()
    total = len(slides)
    idx = 0
    auto_play = True

    # Start slow (title), then fast through early epochs, slow down for later
    base_speed = 2.5

    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)

    try:
        tty.setraw(fd)

        while True:
            if idx == 0:
                speed = 3.0
            elif idx == total - 1:
                speed = 3.0
            elif idx < 15:
                speed = 0.4
            elif idx < 40:
                speed = 0.15
            elif idx < 80:
                speed = 0.06
            else:
                speed = 0.03

            sys.stdout.write(render_slide(slides[idx], idx, total, auto_play, speed))
            sys.stdout.flush()

            if auto_play:
                key = get_key(fd, timeout=speed)
                if key is None:
                    if idx < total - 1:
                        idx += 1
                    else:
                        auto_play = False
                    continue
            else:
                key = get_key(fd, timeout=None)

            if key == 'QUIT':
                break
            elif key == 'SPACE':
                auto_play = not auto_play
            elif key in ('RIGHT', 'DOWN'):
                auto_play = False
                if idx < total - 1:
                    idx += 1
            elif key in ('LEFT', 'UP'):
                auto_play = False
                if idx > 0:
                    idx -= 1

    except KeyboardInterrupt:
        pass
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        sys.stdout.write("\033[2J\033[H")
        sys.stdout.flush()
        print("Animation ended.")


if __name__ == "__main__":
    main()
