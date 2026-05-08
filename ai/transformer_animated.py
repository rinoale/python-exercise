"""
Full Transformer — Animated ASCII visualization.

Watch an input sentence flow through every component of a Transformer:
  1. Tokenizer (text -> IDs)
  2. Embedding lookup (IDs -> vectors)
  3. Positional encoding (add position info)
  4. Transformer Block (x2 layers):
     a. Layer Norm
     b. Multi-Head Attention (2 heads)
     c. Residual connection
     d. Layer Norm
     e. Feed-Forward Network (expand, activate, compress)
     f. Residual connection
  5. Final Layer Norm
  6. Output projection (vector -> vocabulary logits)
  7. Softmax + prediction

Sentence: "The hungry cat sat ___"
Uses simplified dimensions (4-dim) so every number fits on screen.

Controls:
  SPACE  = toggle auto-play / pause
  Arrows = browse slides manually
  q      = quit
"""

import os
import sys
import math
import select
import termios
import tty
import random

random.seed(42)

# ============================================================
# Model "weights" — small enough to show every number
# d_model=4, d_head=2, n_heads=2, d_ff=8, vocab_size=8
# ============================================================

VOCAB = {
    "The": 0, "hungry": 1, "cat": 2, "sat": 3,
    "mat": 4, "on": 5, "warm": 6, "quietly": 7,
}
ID_TO_TOKEN = {v: k for k, v in VOCAB.items()}
VOCAB_SIZE = len(VOCAB)

TOKENS = ["The", "hungry", "cat", "sat"]
TOKEN_IDS = [VOCAB[t] for t in TOKENS]

# Embedding table (vocab_size x d_model)
EMB_TABLE = [
    [ 0.10,  0.30,  0.80,  0.20],   # 0: The
    [ 0.70,  0.20,  0.10,  0.50],   # 1: hungry
    [ 0.80,  0.10,  0.30,  0.90],   # 2: cat
    [ 0.20,  0.80,  0.70,  0.10],   # 3: sat
    [ 0.70,  0.20,  0.50,  0.80],   # 4: mat
    [ 0.10,  0.40,  0.90,  0.10],   # 5: on
    [ 0.60,  0.30,  0.20,  0.40],   # 6: warm
    [ 0.30,  0.60,  0.50,  0.20],   # 7: quietly
]

# Positional encoding (precomputed for positions 0-3, dim=4)
POS_ENC = [
    [ 0.00,  1.00,  0.00,  1.00],   # pos 0
    [ 0.84,  0.54,  0.01,  1.00],   # pos 1
    [ 0.91, -0.42,  0.02,  1.00],   # pos 2
    [ 0.14, -0.99,  0.03,  1.00],   # pos 3
]

# --- Layer 1 weights ---

# Head 1: W_Q, W_K, W_V  (d_model x d_head = 4x2)
L1_H1_WQ = [[ 0.3,  0.1], [ 0.5,  0.2], [ 0.1,  0.4], [ 0.2,  0.6]]
L1_H1_WK = [[ 0.4,  0.3], [ 0.1,  0.5], [ 0.6,  0.1], [ 0.2,  0.4]]
L1_H1_WV = [[ 0.2,  0.5], [ 0.4,  0.1], [ 0.3,  0.3], [ 0.1,  0.6]]

# Head 2: W_Q, W_K, W_V
L1_H2_WQ = [[ 0.1,  0.4], [ 0.3,  0.1], [ 0.5,  0.3], [ 0.4,  0.2]]
L1_H2_WK = [[ 0.2,  0.1], [ 0.6,  0.3], [ 0.1,  0.5], [ 0.3,  0.2]]
L1_H2_WV = [[ 0.5,  0.2], [ 0.1,  0.4], [ 0.2,  0.1], [ 0.3,  0.5]]

# W_O: projects concatenated heads back (4x4)
L1_WO = [
    [ 0.3,  0.1,  0.2,  0.4],
    [ 0.2,  0.5,  0.1,  0.3],
    [ 0.4,  0.2,  0.5,  0.1],
    [ 0.1,  0.3,  0.4,  0.2],
]

# FFN: W1 (4x8), W2 (8x4)
L1_W1 = [
    [ 0.3,  0.1, -0.2,  0.5,  0.2, -0.1,  0.4,  0.3],
    [-0.1,  0.4,  0.3, -0.2,  0.5,  0.1,  0.2, -0.3],
    [ 0.2, -0.3,  0.5,  0.1, -0.1,  0.4,  0.3,  0.2],
    [ 0.4,  0.2,  0.1,  0.3, -0.2,  0.5, -0.1,  0.4],
]
L1_W2 = [
    [ 0.2, -0.1,  0.3,  0.1],
    [ 0.1,  0.4, -0.2,  0.3],
    [-0.3,  0.2,  0.1,  0.4],
    [ 0.4,  0.1,  0.2, -0.1],
    [ 0.1, -0.3,  0.4,  0.2],
    [ 0.3,  0.2, -0.1,  0.5],
    [-0.2,  0.3,  0.5,  0.1],
    [ 0.2,  0.1,  0.3, -0.2],
]

# LayerNorm params (gamma, beta) for layer 1
L1_LN1_G = [1.0, 1.0, 1.0, 1.0]
L1_LN1_B = [0.0, 0.0, 0.0, 0.0]
L1_LN2_G = [1.0, 1.0, 1.0, 1.0]
L1_LN2_B = [0.0, 0.0, 0.0, 0.0]

# --- Layer 2 weights (different from layer 1) ---

L2_H1_WQ = [[ 0.2,  0.3], [ 0.4,  0.1], [ 0.1,  0.5], [ 0.3,  0.2]]
L2_H1_WK = [[ 0.3,  0.2], [ 0.2,  0.4], [ 0.5,  0.1], [ 0.1,  0.3]]
L2_H1_WV = [[ 0.1,  0.3], [ 0.3,  0.2], [ 0.4,  0.5], [ 0.2,  0.1]]

L2_H2_WQ = [[ 0.4,  0.2], [ 0.1,  0.3], [ 0.3,  0.1], [ 0.2,  0.5]]
L2_H2_WK = [[ 0.1,  0.4], [ 0.3,  0.2], [ 0.2,  0.3], [ 0.5,  0.1]]
L2_H2_WV = [[ 0.3,  0.1], [ 0.2,  0.5], [ 0.1,  0.3], [ 0.4,  0.2]]

L2_WO = [
    [ 0.2,  0.3,  0.1,  0.4],
    [ 0.4,  0.1,  0.3,  0.2],
    [ 0.1,  0.4,  0.2,  0.3],
    [ 0.3,  0.2,  0.4,  0.1],
]

L2_W1 = [
    [ 0.2,  0.3, -0.1,  0.4,  0.1, -0.2,  0.3,  0.2],
    [-0.2,  0.1,  0.4, -0.1,  0.3,  0.2,  0.1, -0.3],
    [ 0.3, -0.2,  0.2,  0.3, -0.1,  0.5,  0.2,  0.1],
    [ 0.1,  0.4, -0.3,  0.2,  0.4,  0.1, -0.2,  0.3],
]
L2_W2 = [
    [ 0.3, -0.2,  0.1,  0.2],
    [ 0.1,  0.3, -0.1,  0.4],
    [-0.2,  0.1,  0.3,  0.2],
    [ 0.2,  0.2,  0.4, -0.1],
    [ 0.4, -0.1,  0.2,  0.3],
    [ 0.1,  0.4, -0.2,  0.1],
    [-0.1,  0.2,  0.3,  0.4],
    [ 0.3,  0.1,  0.1, -0.3],
]

L2_LN1_G = [1.0, 1.0, 1.0, 1.0]
L2_LN1_B = [0.0, 0.0, 0.0, 0.0]
L2_LN2_G = [1.0, 1.0, 1.0, 1.0]
L2_LN2_B = [0.0, 0.0, 0.0, 0.0]

# Final layer norm
FINAL_LN_G = [1.0, 1.0, 1.0, 1.0]
FINAL_LN_B = [0.0, 0.0, 0.0, 0.0]

# Output projection W_out: d_model x vocab_size (4x8)
W_OUT = [
    [ 0.5, -0.3,  0.1,  0.2,  0.8,  0.0,  0.3, -0.1],
    [-0.2,  0.6,  0.3, -0.1, -0.1,  0.7,  0.2,  0.4],
    [ 0.1,  0.2,  0.7,  0.5,  0.3, -0.2,  0.6,  0.1],
    [ 0.3, -0.1, -0.2,  0.8,  0.4,  0.1, -0.3,  0.5],
]


# ============================================================
# Math helpers
# ============================================================

def mat_vec(mat, vec):
    """matrix (rows x cols stored as list-of-rows transposed) times vector."""
    cols = len(mat[0])
    return [round(sum(vec[r] * mat[r][c] for r in range(len(vec))), 4)
            for c in range(cols)]


def mat_mat(A_rows, B_rows):
    """A (m x k) @ B (k x n), both stored row-major."""
    m = len(A_rows)
    k = len(A_rows[0])
    n = len(B_rows[0])
    result = []
    for i in range(m):
        row = []
        for j in range(n):
            s = sum(A_rows[i][r] * B_rows[r][j] for r in range(k))
            row.append(round(s, 4))
        result.append(row)
    return result


def vec_add(a, b):
    return [round(x + y, 4) for x, y in zip(a, b)]


def vec_scale(a, s):
    return [round(x * s, 4) for x in a]


def dot(a, b):
    return round(sum(x * y for x, y in zip(a, b)), 4)


def softmax(vals):
    mx = max(vals)
    exps = [math.exp(v - mx) for v in vals]
    s = sum(exps)
    return [round(e / s, 4) for e in exps]


def relu(vals):
    return [round(max(0, v), 4) for v in vals]


def layer_norm(vec, gamma, beta, eps=1e-5):
    n = len(vec)
    mean = sum(vec) / n
    var = sum((v - mean) ** 2 for v in vec) / n
    std = math.sqrt(var + eps)
    normed = [(v - mean) / std for v in vec]
    out = [round(g * x + b, 4) for g, x, b in zip(gamma, normed, beta)]
    return out, round(mean, 4), round(std, 4)


def fmt(v, w=6):
    return "[" + ", ".join(f"{x:+.2f}" for x in v) + "]"


def fmts(v):
    return "[" + ", ".join(f"{x:.2f}" for x in v) + "]"


def pct_bar(val, width=20):
    filled = int(val * width)
    return "#" * filled + "." * (width - filled)


# ============================================================
# Run the full forward pass, storing every intermediate result
# ============================================================

# Step 1: Embedding lookup
embeddings = [list(EMB_TABLE[tid]) for tid in TOKEN_IDS]

# Step 2: Add positional encoding
pos_added = [vec_add(embeddings[i], POS_ENC[i]) for i in range(4)]


def run_attention_layer(x_in, h1_wq, h1_wk, h1_wv, h2_wq, h2_wk, h2_wv,
                        wo, w1, w2, ln1_g, ln1_b, ln2_g, ln2_b):
    """Run one transformer block, return all intermediates."""
    info = {}
    n_tok = len(x_in)
    d_k = 2

    # --- Sub-layer 1: Multi-head attention ---
    # LayerNorm before attention
    normed1 = []
    for i in range(n_tok):
        ln_out, mn, sd = layer_norm(x_in[i], ln1_g, ln1_b)
        normed1.append(ln_out)
    info['ln1'] = [list(v) for v in normed1]

    # Head 1
    h1_qs = [mat_vec(h1_wq, normed1[i]) for i in range(n_tok)]
    h1_ks = [mat_vec(h1_wk, normed1[i]) for i in range(n_tok)]
    h1_vs = [mat_vec(h1_wv, normed1[i]) for i in range(n_tok)]
    info['h1_q'] = h1_qs
    info['h1_k'] = h1_ks
    info['h1_v'] = h1_vs

    # Head 1 attention
    h1_attn = []
    h1_out = []
    for i in range(n_tok):
        scores = [dot(h1_qs[i], h1_ks[j]) / math.sqrt(d_k) for j in range(i + 1)]
        weights = softmax(scores)
        h1_attn.append(weights)
        out = [0.0] * d_k
        for j in range(i + 1):
            for d in range(d_k):
                out[d] += weights[j] * h1_vs[j][d]
        h1_out.append([round(v, 4) for v in out])
    info['h1_attn'] = h1_attn
    info['h1_out'] = h1_out

    # Head 2
    h2_qs = [mat_vec(h2_wq, normed1[i]) for i in range(n_tok)]
    h2_ks = [mat_vec(h2_wk, normed1[i]) for i in range(n_tok)]
    h2_vs = [mat_vec(h2_wv, normed1[i]) for i in range(n_tok)]
    info['h2_q'] = h2_qs
    info['h2_k'] = h2_ks
    info['h2_v'] = h2_vs

    h2_attn = []
    h2_out = []
    for i in range(n_tok):
        scores = [dot(h2_qs[i], h2_ks[j]) / math.sqrt(d_k) for j in range(i + 1)]
        weights = softmax(scores)
        h2_attn.append(weights)
        out = [0.0] * d_k
        for j in range(i + 1):
            for d in range(d_k):
                out[d] += weights[j] * h2_vs[j][d]
        h2_out.append([round(v, 4) for v in out])
    info['h2_attn'] = h2_attn
    info['h2_out'] = h2_out

    # Concatenate heads and project
    concat = [h1_out[i] + h2_out[i] for i in range(n_tok)]
    info['concat'] = concat

    projected = [mat_vec(wo, concat[i]) for i in range(n_tok)]
    info['projected'] = projected

    # Residual connection
    after_attn = [vec_add(x_in[i], projected[i]) for i in range(n_tok)]
    info['after_attn_residual'] = after_attn

    # --- Sub-layer 2: FFN ---
    normed2 = []
    for i in range(n_tok):
        ln_out, mn, sd = layer_norm(after_attn[i], ln2_g, ln2_b)
        normed2.append(ln_out)
    info['ln2'] = normed2

    # FFN: expand, ReLU, compress (for each token independently)
    ffn_outs = []
    ffn_hiddens = []
    ffn_relus = []
    for i in range(n_tok):
        hidden = mat_vec(w1, normed2[i])
        ffn_hiddens.append(hidden)
        activated = relu(hidden)
        ffn_relus.append(activated)
        out = mat_vec(w2, activated)
        ffn_outs.append(out)
    info['ffn_hidden'] = ffn_hiddens
    info['ffn_relu'] = ffn_relus
    info['ffn_out'] = ffn_outs

    # Residual connection
    after_ffn = [vec_add(after_attn[i], ffn_outs[i]) for i in range(n_tok)]
    info['after_ffn_residual'] = after_ffn

    return after_ffn, info


# Run Layer 1
layer1_out, layer1_info = run_attention_layer(
    pos_added,
    L1_H1_WQ, L1_H1_WK, L1_H1_WV,
    L1_H2_WQ, L1_H2_WK, L1_H2_WV,
    L1_WO, L1_W1, L1_W2,
    L1_LN1_G, L1_LN1_B, L1_LN2_G, L1_LN2_B,
)

# Run Layer 2
layer2_out, layer2_info = run_attention_layer(
    layer1_out,
    L2_H1_WQ, L2_H1_WK, L2_H1_WV,
    L2_H2_WQ, L2_H2_WK, L2_H2_WV,
    L2_WO, L2_W1, L2_W2,
    L2_LN1_G, L2_LN1_B, L2_LN2_G, L2_LN2_B,
)

# Final layer norm on last token
final_vec = layer2_out[3]
final_normed, final_mn, final_sd = layer_norm(final_vec, FINAL_LN_G, FINAL_LN_B)

# Output projection: last token -> logits
logits = mat_vec(W_OUT, final_normed)

# Softmax -> probabilities
probs = softmax(logits)

# Prediction
pred_id = probs.index(max(probs))
pred_token = ID_TO_TOKEN[pred_id]


# ============================================================
# Build slides
# ============================================================

WIDTH = 76


def T(text):
    pad = WIDTH - 4 - len(text)
    return "  " + text + " " * max(pad, 0)


def sep():
    return "  " + "-" * (WIDTH - 4)


def blank():
    return ""


def build_slides():
    slides = []

    # === TITLE ===
    s = []
    s.append(blank())
    s.append(T("FULL TRANSFORMER -- STEP BY STEP"))
    s.append(T("Every operation, from input text to predicted next token"))
    s.append(sep())
    s.append(blank())
    s.append(T('  Input:  "The hungry cat sat ___"'))
    s.append(T("  Goal:   predict what comes after 'sat'"))
    s.append(blank())
    s.append(T("  Model:  2 layers, 2 attention heads, dim=4"))
    s.append(T("          (real models: 96 layers, 96 heads, dim=4096)"))
    s.append(T("          (same operations, just bigger numbers)"))
    s.append(blank())
    s.append(sep())
    s.append(T("  The pipeline:"))
    s.append(T("    text -> tokens -> embeddings -> +position"))
    s.append(T("    -> [LayerNorm -> Attention -> +Residual"))
    s.append(T("       -> LayerNorm -> FFN -> +Residual] x2 layers"))
    s.append(T("    -> LayerNorm -> output projection -> softmax"))
    s.append(T("    -> predicted next token"))
    s.append(blank())
    slides.append(("Title", 0, s))

    # === ARCHITECTURE DIAGRAM ===
    s = []
    s.append(blank())
    s.append(T("THE TRANSFORMER ARCHITECTURE"))
    s.append(sep())
    s.append(blank())
    s.append(T('  "The hungry cat sat"'))
    s.append(T("         |"))
    s.append(T("    [ Tokenizer ]           text -> integer IDs"))
    s.append(T("         |"))
    s.append(T("    [ Embedding ]           ID -> 4-dim vector"))
    s.append(T("         |"))
    s.append(T("    [ + Position ]          add position info"))
    s.append(T("         |"))
    s.append(T("    +=================+"))
    s.append(T("    | Layer 1         |"))
    s.append(T("    |  LayerNorm      |     stabilize values"))
    s.append(T("    |  Attention(2h)  |     tokens share info"))
    s.append(T("    |  +Residual      |     preserve original"))
    s.append(T("    |  LayerNorm      |     stabilize again"))
    s.append(T("    |  FFN            |     apply knowledge"))
    s.append(T("    |  +Residual      |     preserve again"))
    s.append(T("    +=================+"))
    s.append(T("         |"))
    s.append(T("    +=================+"))
    s.append(T("    | Layer 2         |     (same ops, different weights)"))
    s.append(T("    +=================+"))
    s.append(T("         |"))
    s.append(T("    [ Final LayerNorm ]"))
    s.append(T("    [ Output W_out    ]     4-dim -> 8 vocab scores"))
    s.append(T("    [ Softmax         ]     scores -> probabilities"))
    s.append(T("         |"))
    s.append(T("    predicted token"))
    s.append(blank())
    slides.append(("Architecture", 0, s))

    # === STEP 1: TOKENIZER ===
    s = []
    s.append(blank())
    s.append(T("STEP 1: TOKENIZER"))
    s.append(T("Convert text to integer IDs using a vocabulary lookup"))
    s.append(sep())
    s.append(blank())
    s.append(T("  Vocabulary (8 words for this demo):"))
    s.append(blank())
    for tok, tid in sorted(VOCAB.items(), key=lambda x: x[1]):
        marker = " <--" if tok in TOKENS else ""
        s.append(T(f'    "{tok:8s}" -> ID {tid}{marker}'))
    s.append(blank())
    s.append(sep())
    s.append(T('  "The hungry cat sat" -> [0, 1, 2, 3]'))
    s.append(blank())
    s.append(T("  Real models: 50K-128K vocabulary"))
    s.append(T("  GPT uses BPE (byte pair encoding) -- splits words"))
    s.append(T('  into subwords: "unhappiness" -> ["un", "happiness"]'))
    s.append(blank())
    slides.append(("Tokenizer", 1, s))

    # === STEP 2: EMBEDDING LOOKUP ===
    s = []
    s.append(blank())
    s.append(T("STEP 2: EMBEDDING LOOKUP"))
    s.append(T("Each token ID indexes a row in the embedding table"))
    s.append(sep())
    s.append(blank())
    s.append(T("  Embedding table (learned during training):"))
    s.append(blank())
    for tid in range(VOCAB_SIZE):
        tok = ID_TO_TOKEN[tid]
        marker = " <--" if tok in TOKENS else ""
        s.append(T(f"    ID {tid} -> {fmt(EMB_TABLE[tid])}{marker}"))
    s.append(blank())
    s.append(sep())
    s.append(T("  Our tokens after lookup:"))
    for i, tok in enumerate(TOKENS):
        s.append(T(f'    "{tok:7s}" (ID {TOKEN_IDS[i]}) -> {fmt(embeddings[i])}'))
    s.append(blank())
    s.append(T("  Each word is now a 4-dim vector. But no position info yet."))
    s.append(blank())
    slides.append(("Embedding", 2, s))

    # === STEP 3: POSITIONAL ENCODING ===
    s = []
    s.append(blank())
    s.append(T("STEP 3: POSITIONAL ENCODING"))
    s.append(T("Add position information so the model knows word order"))
    s.append(sep())
    s.append(blank())
    s.append(T("  Position vectors (from sine/cosine waves):"))
    for i in range(4):
        s.append(T(f"    pos {i}: {fmt(POS_ENC[i])}"))
    s.append(blank())
    s.append(T("  Embedding + Position = Input to transformer:"))
    s.append(blank())
    for i, tok in enumerate(TOKENS):
        s.append(T(f'    "{tok:7s}" {fmt(embeddings[i])}'))
        s.append(T(f'             + {fmt(POS_ENC[i])}  (pos {i})'))
        s.append(T(f'             = {fmt(pos_added[i])}'))
        if i < 3:
            s.append(blank())
    s.append(blank())
    s.append(sep())
    s.append(T("  Now each vector encodes BOTH meaning AND position."))
    s.append(blank())
    slides.append(("Positional", 3, s))

    # === STEP 4: LAYER 1 — LAYER NORM ===
    s = []
    s.append(blank())
    s.append(T("LAYER 1, STEP 4a: LAYER NORM (before attention)"))
    s.append(T("Normalize each vector: mean=0, variance=1"))
    s.append(sep())
    s.append(blank())
    ln1 = layer1_info['ln1']
    for i, tok in enumerate(TOKENS):
        vec = pos_added[i]
        mn = sum(vec) / 4
        var = sum((v - mn) ** 2 for v in vec) / 4
        sd = math.sqrt(var + 1e-5)
        s.append(T(f'  "{tok:7s}" input:  {fmt(vec)}'))
        s.append(T(f'            mean={mn:+.2f}  std={sd:.2f}'))
        s.append(T(f'            normed: {fmt(ln1[i])}'))
        if i < 3:
            s.append(blank())
    s.append(blank())
    s.append(sep())
    s.append(T("  Every vector now has mean~0, std~1."))
    s.append(T("  This prevents values from exploding across layers."))
    s.append(blank())
    slides.append(("L1 LayerNorm", 4, s))

    # === STEP 4b: LAYER 1 — MULTI-HEAD ATTENTION (Head 1) ===
    s = []
    s.append(blank())
    s.append(T("LAYER 1, STEP 4b: MULTI-HEAD ATTENTION"))
    s.append(T("2 heads run in parallel, each asking different questions"))
    s.append(sep())
    s.append(blank())
    s.append(T("  HEAD 1 -- Q, K, V for each token (normed input @ W):"))
    s.append(blank())
    s.append(T("           Q            K            V"))
    for i, tok in enumerate(TOKENS):
        q = fmts(layer1_info['h1_q'][i])
        k = fmts(layer1_info['h1_k'][i])
        v = fmts(layer1_info['h1_v'][i])
        s.append(T(f'  "{tok:7s}" {q:14s} {k:14s} {v:14s}'))
    s.append(blank())
    s.append(T("  HEAD 1 attention weights (Q.K / sqrt(2), softmax):"))
    s.append(blank())
    s.append(T("               The    hungry  cat    sat"))
    for i, tok in enumerate(TOKENS):
        row = f'  "{tok:7s}"  '
        for j in range(4):
            if j > i:
                row += "   .   "
            else:
                w = layer1_info['h1_attn'][i][j]
                pct = int(w * 100)
                row += f"  {pct:3d}% "
        s.append(T(row))
    s.append(blank())
    s.append(sep())
    best_j = layer1_info['h1_attn'][3].index(max(layer1_info['h1_attn'][3]))
    s.append(T(f'  Head 1: "sat" attends most to "{TOKENS[best_j]}" ({int(layer1_info["h1_attn"][3][best_j]*100)}%)'))
    s.append(blank())
    slides.append(("L1 Head 1", 4, s))

    # === Head 2 ===
    s = []
    s.append(blank())
    s.append(T("LAYER 1, STEP 4b (continued): HEAD 2"))
    s.append(T("Different W matrices -> different attention pattern"))
    s.append(sep())
    s.append(blank())
    s.append(T("  HEAD 2 -- Q, K, V:"))
    s.append(blank())
    s.append(T("           Q            K            V"))
    for i, tok in enumerate(TOKENS):
        q = fmts(layer1_info['h2_q'][i])
        k = fmts(layer1_info['h2_k'][i])
        v = fmts(layer1_info['h2_v'][i])
        s.append(T(f'  "{tok:7s}" {q:14s} {k:14s} {v:14s}'))
    s.append(blank())
    s.append(T("  HEAD 2 attention weights:"))
    s.append(blank())
    s.append(T("               The    hungry  cat    sat"))
    for i, tok in enumerate(TOKENS):
        row = f'  "{tok:7s}"  '
        for j in range(4):
            if j > i:
                row += "   .   "
            else:
                w = layer1_info['h2_attn'][i][j]
                pct = int(w * 100)
                row += f"  {pct:3d}% "
        s.append(T(row))
    s.append(blank())
    s.append(sep())
    best_j2 = layer1_info['h2_attn'][3].index(max(layer1_info['h2_attn'][3]))
    s.append(T(f'  Head 2: "sat" attends most to "{TOKENS[best_j2]}" ({int(layer1_info["h2_attn"][3][best_j2]*100)}%)'))
    s.append(T(f"  Different head, different pattern -- that's the point!"))
    s.append(blank())
    slides.append(("L1 Head 2", 4, s))

    # === Concatenate and project ===
    s = []
    s.append(blank())
    s.append(T("LAYER 1, STEP 4c: CONCATENATE HEADS + PROJECT"))
    s.append(T("Combine both heads' outputs into one vector per token"))
    s.append(sep())
    s.append(blank())
    s.append(T("  For each token: concat(head1_out, head2_out) @ W_O"))
    s.append(blank())
    s.append(T("           Head1 out   Head2 out   Concat       Projected"))
    for i, tok in enumerate(TOKENS):
        h1 = fmts(layer1_info['h1_out'][i])
        h2 = fmts(layer1_info['h2_out'][i])
        cc = fmts(layer1_info['concat'][i])
        pr = fmts(layer1_info['projected'][i])
        s.append(T(f'  "{tok:3s}" {h1:14s} {h2:14s} {cc} {pr}'))
    s.append(blank())
    s.append(sep())
    s.append(T("  W_O (4x4 matrix) learns how to combine the two heads'"))
    s.append(T("  different perspectives into a single representation."))
    s.append(blank())
    slides.append(("L1 Concat+Project", 4, s))

    # === Residual connection #1 ===
    s = []
    s.append(blank())
    s.append(T("LAYER 1, STEP 4d: RESIDUAL CONNECTION #1"))
    s.append(T("Add the attention output to the ORIGINAL input (skip)"))
    s.append(sep())
    s.append(blank())
    s.append(T("  output = input + attention(input)"))
    s.append(blank())
    for i, tok in enumerate(TOKENS):
        orig = pos_added[i] if True else layer1_out[i]
        proj = layer1_info['projected'][i]
        res = layer1_info['after_attn_residual'][i]
        s.append(T(f'  "{tok:7s}" {fmt(orig)}  (original input)'))
        s.append(T(f'           + {fmt(proj)}  (attention output)'))
        s.append(T(f'           = {fmt(res)}'))
        if i < 3:
            s.append(blank())
    s.append(blank())
    s.append(sep())
    s.append(T("  The original signal is preserved! Attention only ADDS"))
    s.append(T("  new information on top. Nothing from the input is lost."))
    s.append(blank())
    slides.append(("L1 Residual 1", 4, s))

    # === FFN ===
    s = []
    s.append(blank())
    s.append(T("LAYER 1, STEP 4e: FEED-FORWARD NETWORK"))
    s.append(T("Each token independently: expand(4->8), ReLU, compress(8->4)"))
    s.append(sep())
    s.append(blank())
    s.append(T('  Showing "sat" (token 3) through FFN:'))
    s.append(blank())
    sat_ln2 = layer1_info['ln2'][3]
    s.append(T(f"  After LayerNorm:  {fmt(sat_ln2)}"))
    s.append(blank())
    sat_hid = layer1_info['ffn_hidden'][3]
    s.append(T(f"  Expand (x @ W1):  {fmts(sat_hid)}"))
    s.append(T( "                    (4-dim -> 8-dim, wider space for knowledge)"))
    s.append(blank())
    sat_relu = layer1_info['ffn_relu'][3]
    s.append(T(f"  ReLU:             {fmts(sat_relu)}"))
    s.append(T( "                    (negative -> 0, decides what to keep)"))
    n_active = sum(1 for v in sat_relu if v > 0)
    n_dead = 8 - n_active
    s.append(T(f"                    {n_active} neurons active, {n_dead} dead"))
    s.append(blank())
    sat_ffn = layer1_info['ffn_out'][3]
    s.append(T(f"  Compress (x @ W2): {fmt(sat_ffn)}"))
    s.append(T( "                    (8-dim -> 4-dim, back to model dimension)"))
    s.append(blank())
    s.append(sep())
    s.append(T("  Attention = tokens talk to each other"))
    s.append(T("  FFN = each token consults its own knowledge, alone"))
    s.append(T("  ~67% of model parameters live in FFN layers"))
    s.append(blank())
    slides.append(("L1 FFN", 4, s))

    # === Residual connection #2 ===
    s = []
    s.append(blank())
    s.append(T("LAYER 1, STEP 4f: RESIDUAL CONNECTION #2"))
    s.append(T("Add FFN output to the post-attention result"))
    s.append(sep())
    s.append(blank())
    s.append(T("  output = post_attention + FFN(post_attention)"))
    s.append(blank())
    for i, tok in enumerate(TOKENS):
        attn_res = layer1_info['after_attn_residual'][i]
        ffn_o = layer1_info['ffn_out'][i]
        final = layer1_out[i]
        s.append(T(f'  "{tok:7s}" {fmt(attn_res)}  (after attention+residual)'))
        s.append(T(f'           + {fmt(ffn_o)}  (FFN output)'))
        s.append(T(f'           = {fmt(final)}'))
        if i < 3:
            s.append(blank())
    s.append(blank())
    s.append(sep())
    s.append(T("  Layer 1 complete! Each token now carries:"))
    s.append(T("    - its original meaning (from embedding)"))
    s.append(T("    - context from other tokens (from attention)"))
    s.append(T("    - processed knowledge (from FFN)"))
    s.append(blank())
    slides.append(("L1 Residual 2", 4, s))

    # === Layer 1 summary ===
    s = []
    s.append(blank())
    s.append(T("LAYER 1 COMPLETE -- BEFORE vs AFTER"))
    s.append(sep())
    s.append(blank())
    s.append(T("  Input to Layer 1 (just embedding + position):"))
    for i, tok in enumerate(TOKENS):
        s.append(T(f'    "{tok:7s}" {fmt(pos_added[i])}'))
    s.append(blank())
    s.append(T("  Output of Layer 1 (enriched with context + knowledge):"))
    for i, tok in enumerate(TOKENS):
        s.append(T(f'    "{tok:7s}" {fmt(layer1_out[i])}'))
    s.append(blank())
    s.append(sep())
    s.append(T("  Every number changed. The vectors now encode"))
    s.append(T("  relationships between words, not just individual words."))
    s.append(blank())
    s.append(T('  "sat" no longer means just "sat" -- it carries'))
    s.append(T('  information about WHO sat and HOW.'))
    s.append(blank())
    s.append(T("  But one layer of refinement isn't enough."))
    s.append(T("  Layer 2 will refine further with different weights..."))
    s.append(blank())
    slides.append(("L1 Summary", 4, s))

    # === LAYER 2 (condensed) ===
    s = []
    s.append(blank())
    s.append(T("LAYER 2: SAME OPERATIONS, DIFFERENT WEIGHTS"))
    s.append(T("Deeper patterns emerge from richer input"))
    s.append(sep())
    s.append(blank())
    s.append(T("  Layer 2 Head 1 attention (sat's row):"))
    row = "    "
    for j in range(4):
        if j <= 3:
            w = layer2_info['h1_attn'][3][j]
            row += f'{TOKENS[j]}:{int(w*100)}%  '
    s.append(T(row))
    s.append(blank())
    s.append(T("  Layer 2 Head 2 attention (sat's row):"))
    row = "    "
    for j in range(4):
        if j <= 3:
            w = layer2_info['h2_attn'][3][j]
            row += f'{TOKENS[j]}:{int(w*100)}%  '
    s.append(T(row))
    s.append(blank())
    s.append(sep())
    s.append(blank())
    s.append(T("  Layer 2 sees RICHER input (Layer 1's output)."))
    s.append(T('  "sat" already carries info about "cat" from Layer 1.'))
    s.append(T("  Layer 2 can build on that -- finding deeper patterns"))
    s.append(T("  that weren't visible in the raw embeddings."))
    s.append(blank())
    s.append(T("  Output of Layer 2:"))
    for i, tok in enumerate(TOKENS):
        s.append(T(f'    "{tok:7s}" {fmt(layer2_out[i])}'))
    s.append(blank())
    slides.append(("Layer 2", 5, s))

    # === STEP 5: FINAL LAYER NORM ===
    s = []
    s.append(blank())
    s.append(T("STEP 5: FINAL LAYER NORM"))
    s.append(T("Normalize the last token's vector before output"))
    s.append(sep())
    s.append(blank())
    s.append(T("  Only the LAST token matters for next-token prediction."))
    s.append(T("  (Other tokens were needed for attention, but we only"))
    s.append(T("  predict from the final position.)"))
    s.append(blank())
    s.append(T(f'  "sat" (last token) after Layer 2:'))
    s.append(T(f"    {fmt(final_vec)}"))
    s.append(blank())
    s.append(T(f"  mean = {final_mn:+.4f}    std = {final_sd:.4f}"))
    s.append(blank())
    s.append(T(f"  After LayerNorm:"))
    s.append(T(f"    {fmt(final_normed)}"))
    s.append(blank())
    s.append(sep())
    s.append(T("  This normalized vector is the model's final"))
    s.append(T('  understanding of "what comes after sat?"'))
    s.append(T("  It encodes the entire sentence's meaning."))
    s.append(blank())
    slides.append(("Final LN", 6, s))

    # === STEP 6: OUTPUT PROJECTION ===
    s = []
    s.append(blank())
    s.append(T("STEP 6: OUTPUT PROJECTION"))
    s.append(T("Convert 4-dim vector to scores for each word in vocabulary"))
    s.append(sep())
    s.append(blank())
    s.append(T(f"  logits = final_vector @ W_out"))
    s.append(T(f"           (4-dim)        (4x8)  ->  (8 scores)"))
    s.append(blank())
    s.append(T(f"  final_vector = {fmt(final_normed)}"))
    s.append(blank())
    s.append(T("  W_out has one column per vocabulary word."))
    s.append(T("  Dot product with each column = score for that word."))
    s.append(blank())
    s.append(T("  Logits (raw scores):"))
    for tid in range(VOCAB_SIZE):
        tok = ID_TO_TOKEN[tid]
        score = logits[tid]
        bar_len = max(0, int((score + 1) * 8))
        bar = "#" * bar_len
        marker = " <-- highest" if tid == pred_id else ""
        s.append(T(f'    "{tok:8s}": {score:+.3f}  {bar}{marker}'))
    s.append(blank())
    s.append(sep())
    s.append(T("  Higher logit = model thinks this word is more likely."))
    s.append(T("  But these are raw numbers, not probabilities yet."))
    s.append(blank())
    slides.append(("Output proj", 7, s))

    # === STEP 7: SOFTMAX + PREDICTION ===
    s = []
    s.append(blank())
    s.append(T("STEP 7: SOFTMAX -> PREDICTION"))
    s.append(T("Convert logits to probabilities, pick the winner"))
    s.append(sep())
    s.append(blank())
    s.append(T("  softmax(logits) -> probabilities (sum to 1.0):"))
    s.append(blank())

    sorted_probs = sorted(range(VOCAB_SIZE), key=lambda i: probs[i], reverse=True)
    for rank, tid in enumerate(sorted_probs):
        tok = ID_TO_TOKEN[tid]
        p = probs[tid]
        pct = p * 100
        bar = pct_bar(p, 30)
        marker = " <<< PREDICTED" if rank == 0 else ""
        s.append(T(f'    "{tok:8s}": {pct:5.1f}%  {bar}{marker}'))
    s.append(blank())
    s.append(T(f"  Sum of all probabilities: {sum(probs):.2f}"))
    s.append(blank())
    s.append(sep())
    s.append(T(f'  The model predicts: "{pred_token}"'))
    s.append(blank())
    s.append(T(f'  "The hungry cat sat" -> next word: "{pred_token}"'))
    s.append(blank())
    s.append(T("  In a real model, this prediction would be much better"))
    s.append(T("  because: 4096 dims, 96 layers, trained on trillions"))
    s.append(T("  of tokens. But the PROCESS is identical."))
    s.append(blank())
    slides.append(("Prediction", 8, s))

    # === FULL JOURNEY RECAP ===
    s = []
    s.append(blank())
    s.append(T("THE COMPLETE JOURNEY OF A TOKEN"))
    s.append(T('Tracking "sat" from input to prediction'))
    s.append(sep())
    s.append(blank())
    s.append(T(f'  1. Tokenizer:    "sat" -> ID 3'))
    s.append(T(f"  2. Embedding:    ID 3 -> {fmt(embeddings[3])}"))
    s.append(T(f"  3. + Position:   {fmt(pos_added[3])}"))
    s.append(T(f"  4. L1 LayerNorm: {fmt(layer1_info['ln1'][3])}"))
    s.append(T(f"  5. L1 Attn out:  {fmt(layer1_info['projected'][3])}"))
    s.append(T(f"  6. L1 +Residual: {fmt(layer1_info['after_attn_residual'][3])}"))
    s.append(T(f"  7. L1 FFN out:   {fmt(layer1_info['ffn_out'][3])}"))
    s.append(T(f"  8. L1 +Residual: {fmt(layer1_out[3])}"))
    s.append(T(f"  9. L2 (full):    {fmt(layer2_out[3])}"))
    s.append(T(f"  10. Final LN:    {fmt(final_normed)}"))
    s.append(T(f'  11. Prediction:  "{pred_token}" ({max(probs)*100:.1f}%)'))
    s.append(blank())
    s.append(sep())
    s.append(T("  11 transformations. Each one small and simple."))
    s.append(T("  Together: from raw text to meaningful prediction."))
    s.append(blank())
    slides.append(("Token journey", 9, s))

    # === PARAMETER COUNT ===
    s = []
    s.append(blank())
    s.append(T("WHAT'S INSIDE THE MODEL FILE?"))
    s.append(T("Every number we used was a learned parameter"))
    s.append(sep())
    s.append(blank())
    s.append(T("  Our toy model:                  Real model (Llama 8B):"))
    s.append(T("  Embedding:  8 x 4 = 32         128K x 4096 = 524M"))
    s.append(T("  Per layer:                      Per layer:"))
    s.append(T("    W_Q x2h:  2x(4x2) = 16         4096x4096 = 16.7M"))
    s.append(T("    W_K x2h:  2x(4x2) = 16         4096x4096 = 16.7M"))
    s.append(T("    W_V x2h:  2x(4x2) = 16         4096x4096 = 16.7M"))
    s.append(T("    W_O:      4x4     = 16         4096x4096 = 16.7M"))
    s.append(T("    W1(FFN):  4x8     = 32         4096x11008= 45.1M"))
    s.append(T("    W2(FFN):  8x4     = 32         11008x4096= 45.1M"))
    s.append(T("    LayerNorm: 2x4x2  = 16         2x4096x2  = 16K"))
    s.append(T("  x2 layers  = 288               x32 layers  = 6,464M"))
    s.append(T("  Output:     4x8     = 32         4096x128K = 524M"))
    s.append(T("  ----------------------          ----------------------"))
    s.append(T("  Total:      ~352 params          ~8,000,000,000 params"))
    s.append(blank())
    s.append(sep())
    s.append(T("  Same architecture. Same operations. Same math."))
    s.append(T("  The only difference: scale."))
    s.append(blank())
    slides.append(("Parameters", 10, s))

    # === SUMMARY ===
    s = []
    s.append(blank())
    s.append(T("SUMMARY: THE COMPLETE TRANSFORMER"))
    s.append(sep())
    s.append(blank())
    s.append(T("  1. TOKENIZE:    text -> integer IDs"))
    s.append(T("  2. EMBED:       IDs -> vectors (what each word means)"))
    s.append(T("  3. + POSITION:  add where each word is"))
    s.append(T("  4. For each layer:"))
    s.append(T("     a. LAYER NORM:   stabilize values"))
    s.append(T("     b. ATTENTION:    tokens share info (Q.K -> weights -> blend V)"))
    s.append(T("        - multiple heads = multiple questions at once"))
    s.append(T("     c. + RESIDUAL:   preserve original signal"))
    s.append(T("     d. LAYER NORM:   stabilize again"))
    s.append(T("     e. FFN:          apply stored knowledge"))
    s.append(T("        - expand, activate (ReLU), compress"))
    s.append(T("     f. + RESIDUAL:   preserve again"))
    s.append(T("  5. FINAL NORM:  normalize last token"))
    s.append(T("  6. OUTPUT:      vector -> vocabulary scores"))
    s.append(T("  7. SOFTMAX:     scores -> probabilities"))
    s.append(T("  8. PREDICT:     pick the highest probability token"))
    s.append(blank())
    s.append(sep())
    s.append(T("  Every operation is simple math: multiply, add, divide."))
    s.append(T("  The intelligence is in the WEIGHTS, not the operations."))
    s.append(T("  Training adjusts billions of weights over trillions of"))
    s.append(T("  examples until these simple operations produce language."))
    s.append(blank())
    s.append(T("  End of animation. Press q to quit, arrows to review."))
    s.append(blank())
    slides.append(("Summary", 11, s))

    return slides


# ============================================================
# Terminal I/O
# ============================================================

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
    max_step = 11

    output = []
    output.append("\033[2J\033[H")
    output.append("")
    output.append(f"  FULL TRANSFORMER ANIMATION        Slide {idx + 1}/{total}    Step {step}/{max_step}")
    output.append("  " + "=" * (WIDTH - 4))

    for line in lines:
        output.append(line)

    output.append("  " + "=" * (WIDTH - 4))
    if auto_play:
        output.append(f"  >> AUTO-PLAY (SPACE=pause, arrows=browse, q=quit)   speed={speed:.1f}s")
    else:
        output.append("  == PAUSED (SPACE=play, arrows=browse, q=quit)")

    return "\r\n".join(output) + "\r\n"


def main():
    slides = build_slides()
    total = len(slides)
    idx = 0
    auto_play = True

    speeds = [
        3.5, 3.5,           # title, architecture
        2.5, 2.5, 2.5,      # tokenizer, embedding, position
        2.5, 3.0, 3.0,      # L1: layernorm, head1, head2
        2.5, 2.5,            # L1: concat, residual1
        3.0, 2.5,            # L1: FFN, residual2
        2.5,                 # L1 summary
        3.0,                 # Layer 2
        2.5, 2.5,            # final LN, output proj
        3.0,                 # prediction
        3.0, 2.5,            # token journey, parameters
        3.5,                 # summary
    ]
    default_speed = 2.5

    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)

    try:
        tty.setraw(fd)

        while True:
            speed = speeds[idx] if idx < len(speeds) else default_speed
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
        print("Transformer animation ended.")


if __name__ == "__main__":
    main()
