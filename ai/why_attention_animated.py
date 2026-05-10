"""
Why Attention Is the Most Important Feature — Animated ASCII visualization.

Shows WHY attention changed everything, not just HOW it works.
Complements attention_animated.py (which covers Q/K/V mechanics).

Flow:
  1. The RNN bottleneck: signal decays over distance
  2. Attention's direct access: every token sees every token
  3. Causal masking: preventing the model from cheating
  4. Scaling by sqrt(d_k): why it matters
  5. Multi-head attention: multiple perspectives at once
  6. O(n^2) tradeoff: the cost of seeing everything

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


# -- Data --

TOKENS = ["The", "cat", "that", "I", "saw", "yesterday", "sat", "down"]

EMBEDDINGS = [
    [0.1, 0.3, 0.8, 0.2],   # The
    [0.8, 0.1, 0.3, 0.9],   # cat
    [0.3, 0.6, 0.2, 0.4],   # that
    [0.5, 0.2, 0.7, 0.1],   # I
    [0.2, 0.9, 0.1, 0.5],   # saw
    [0.4, 0.3, 0.6, 0.8],   # yesterday
    [0.7, 0.2, 0.5, 0.8],   # sat  (similar to cat)
    [0.1, 0.5, 0.4, 0.3],   # down
]

W_Q = [
    [0.3, 0.1, -0.2],
    [0.1, 0.4, 0.3],
    [-0.1, 0.2, 0.5],
    [0.4, -0.1, 0.1],
]

W_K = [
    [0.2, 0.3, 0.1],
    [-0.1, 0.1, 0.4],
    [0.3, -0.2, 0.2],
    [0.1, 0.5, -0.1],
]

W_V = [
    [0.4, -0.1, 0.2],
    [0.1, 0.3, -0.2],
    [-0.2, 0.2, 0.4],
    [0.3, 0.1, 0.3],
]

# Second head has different weights
W_Q2 = [
    [-0.1, 0.3, 0.2],
    [0.5, 0.1, -0.3],
    [0.2, -0.1, 0.4],
    [0.1, 0.4, 0.1],
]

W_K2 = [
    [0.3, -0.2, 0.1],
    [0.1, 0.4, 0.2],
    [-0.3, 0.1, 0.3],
    [0.2, 0.1, 0.5],
]


def mat_vec(matrix, vec):
    cols = len(matrix[0])
    result = []
    for c in range(cols):
        s = 0.0
        for r in range(len(vec)):
            s += vec[r] * matrix[r][c]
        result.append(round(s, 3))
    return result


def dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def softmax(vals):
    mx = max(vals)
    exps = [math.exp(v - mx) for v in vals]
    s = sum(exps)
    return [e / s for e in exps]


def fmt_vec(v):
    return "[" + ", ".join(f"{x:+.2f}" for x in v) + "]"


# -- Pre-compute --

Qs = [mat_vec(W_Q, e) for e in EMBEDDINGS]
Ks = [mat_vec(W_K, e) for e in EMBEDDINGS]
Vs = [mat_vec(W_V, e) for e in EMBEDDINGS]

Qs2 = [mat_vec(W_Q2, e) for e in EMBEDDINGS]
Ks2 = [mat_vec(W_K2, e) for e in EMBEDDINGS]

d_k = 3
scale = math.sqrt(d_k)

# "sat" = index 6, compute its attention over tokens 0..6 (causal)
SAT = 6
raw_scores_sat = [dot(Qs[SAT], Ks[j]) for j in range(SAT + 1)]
scaled_scores_sat = [s / scale for s in raw_scores_sat]
attn_weights_sat = softmax(scaled_scores_sat)

# Head 2 scores for "sat"
raw_scores_sat_h2 = [dot(Qs2[SAT], Ks2[j]) for j in range(SAT + 1)]
scaled_scores_sat_h2 = [s / scale for s in raw_scores_sat_h2]
attn_weights_sat_h2 = softmax(scaled_scores_sat_h2)

# RNN simulation
W_rnn_h = [[0.3, -0.1, 0.2, 0.1],
            [0.1, 0.4, -0.2, 0.3],
            [-0.2, 0.1, 0.3, 0.1],
            [0.2, -0.1, 0.1, 0.4]]
W_rnn_x = [[0.5, 0.1, -0.1, 0.2],
            [0.2, 0.3, 0.1, -0.1],
            [-0.1, 0.2, 0.4, 0.1],
            [0.1, -0.1, 0.2, 0.5]]


def rnn_step(h, x):
    new_h = []
    for i in range(4):
        val = 0.0
        for j in range(4):
            val += W_rnn_h[i][j] * h[j]
            val += W_rnn_x[i][j] * x[j]
        new_h.append(math.tanh(val))
    return new_h


def cosine_sim(a, b):
    dot_ab = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a < 1e-8 or norm_b < 1e-8:
        return 0.0
    return dot_ab / (norm_a * norm_b)


# Run RNN and track similarity to "cat" state
rnn_h = [0.0, 0.0, 0.0, 0.0]
rnn_states = []
cat_state = None
for i, emb in enumerate(EMBEDDINGS):
    rnn_h = rnn_step(rnn_h, emb)
    rnn_states.append(list(rnn_h))
    if i == 1:  # "cat"
        cat_state = list(rnn_h)

rnn_sims = [cosine_sim(st, cat_state) for st in rnn_states]


# -- Build slides --

WIDTH = 76


def title_line(text):
    pad = WIDTH - 4 - len(text)
    return "  " + text + " " * max(pad, 0)


def separator():
    return "  " + "-" * (WIDTH - 4)


def blank():
    return ""


def bar_char(val, max_val=1.0):
    normalized = max(0.0, val) / max_val
    n = int(normalized * 30)
    return "#" * n


def build_slides():
    slides = []

    # ======== Title ========
    s = []
    s.append(blank())
    s.append(title_line("WHY ATTENTION IS THE MOST IMPORTANT FEATURE"))
    s.append(title_line("The single idea that made Large Language Models possible"))
    s.append(separator())
    s.append(blank())
    s.append(title_line('Sentence: "The cat that I saw yesterday sat down"'))
    s.append(blank())
    s.append(title_line("Question: when the model reaches 'sat',"))
    s.append(title_line("how does it know WHO sat? (the cat, 5 words back)"))
    s.append(blank())
    s.append(separator())
    s.append(title_line("We'll compare two approaches:"))
    s.append(title_line("  1. RNN:       pass information through a chain"))
    s.append(title_line("  2. Attention: let every word look at every other word"))
    s.append(blank())
    s.append(title_line("Spoiler: approach 2 changed everything."))
    s.append(blank())
    slides.append(("Title", 0, s))

    # ======== Part 1: The RNN Bottleneck ========
    s = []
    s.append(blank())
    s.append(title_line("PART 1: THE RNN BOTTLENECK"))
    s.append(title_line("An RNN processes words one by one, left to right"))
    s.append(separator())
    s.append(blank())
    s.append(title_line("  The --> cat --> that --> I --> saw --> yesterday --> sat"))
    s.append(title_line("   h1     h2      h3     h4    h5       h6          h7"))
    s.append(blank())
    s.append(title_line("  Each h = tanh(W_h * prev_h + W_x * current_word)"))
    s.append(blank())
    s.append(title_line("  The hidden state h is a FIXED-SIZE vector (here: 4 dims)."))
    s.append(title_line("  EVERYTHING the model knows must fit in those 4 numbers."))
    s.append(blank())
    s.append(separator())
    s.append(title_line("  Problem: by the time we reach 'sat' (h7),"))
    s.append(title_line("  the information about 'cat' (h2) has been overwritten"))
    s.append(title_line("  by 5 subsequent words passing through the same bottleneck."))
    s.append(blank())
    slides.append(("RNN chain", 1, s))

    # ======== RNN signal decay ========
    s = []
    s.append(blank())
    s.append(title_line("THE BOTTLENECK IN NUMBERS"))
    s.append(title_line("How much of 'cat' survives in later hidden states?"))
    s.append(separator())
    s.append(blank())
    s.append(title_line("  Cosine similarity between h_cat and each later state:"))
    s.append(blank())

    for i, tok in enumerate(TOKENS):
        sim = rnn_sims[i]
        bar = bar_char(sim)
        marker = " <-- cat" if i == 1 else ""
        s.append(title_line(f'  h{i+1} "{tok:9s}"  sim={sim:+.3f}  {bar}{marker}'))

    s.append(blank())
    s.append(separator())
    s.append(title_line("  The signal from 'cat' DECAYS as more words flow through."))
    s.append(title_line("  By 'sat' (5 words later), the cat signal is weakened."))
    s.append(title_line("  This is the INFORMATION BOTTLENECK."))
    s.append(blank())
    slides.append(("RNN decay", 1, s))

    # ======== RNN vs Attention path ========
    s = []
    s.append(blank())
    s.append(title_line("RNN vs ATTENTION: PATH LENGTH"))
    s.append(title_line("How many hops to connect 'cat' and 'sat'?"))
    s.append(separator())
    s.append(blank())
    s.append(title_line("  RNN path (5 hops, signal decays at each):"))
    s.append(blank())
    s.append(title_line("  cat -> h3 -> h4 -> h5 -> h6 -> sat"))
    s.append(title_line("    (that)  (I)  (saw) (yest.)"))
    s.append(title_line("    100%   ~80%  ~65%   ~50%  ~40%  of cat signal remains"))
    s.append(blank())
    s.append(separator())
    s.append(blank())
    s.append(title_line("  Attention path (1 hop, no decay):"))
    s.append(blank())
    s.append(title_line("  cat <---------------------------> sat"))
    s.append(title_line("              DIRECT CONNECTION"))
    s.append(title_line("    100% of cat signal available to sat"))
    s.append(blank())
    s.append(separator())
    s.append(title_line("  This is the core insight: attention removes the bottleneck"))
    s.append(title_line("  by letting every token talk DIRECTLY to every other token."))
    s.append(blank())
    slides.append(("Path length", 1, s))

    # ======== Part 2: Attention direct access ========
    s = []
    s.append(blank())
    s.append(title_line("PART 2: ATTENTION GIVES DIRECT ACCESS"))
    s.append(title_line("'sat' computes attention over ALL previous words"))
    s.append(separator())
    s.append(blank())
    s.append(title_line(f'  Q_sat = {fmt_vec(Qs[SAT])}'))
    s.append(blank())
    s.append(title_line("  sat asks each token: 'how relevant are you to me?'"))
    s.append(blank())

    for j in range(SAT + 1):
        tok = TOKENS[j]
        w = attn_weights_sat[j]
        pct = int(w * 100)
        bar = "#" * int(w * 40)
        marker = " <-- subject!" if j == 1 else ""
        s.append(title_line(f'  sat -> "{tok:9s}"  {pct:2d}%  {bar}{marker}'))

    s.append(blank())
    s.append(separator())
    s.append(title_line("  No matter how far apart 'cat' and 'sat' are,"))
    s.append(title_line("  attention computes their relevance DIRECTLY."))
    s.append(title_line("  Distance = 5 words, but attention doesn't care."))
    s.append(blank())
    slides.append(("Direct access", 2, s))

    # ======== Attention vs RNN comparison ========
    s = []
    s.append(blank())
    s.append(title_line("SIDE BY SIDE: HOW MUCH 'CAT' REACHES 'SAT'?"))
    s.append(separator())
    s.append(blank())
    s.append(title_line("  RNN:"))
    rnn_sim = rnn_sims[SAT]
    bar_rnn = bar_char(max(0, rnn_sim))
    s.append(title_line(f'    cat signal at sat: {rnn_sim:+.3f}  {bar_rnn}'))
    s.append(title_line("    (passed through 5 hidden states, decayed)"))
    s.append(blank())
    s.append(title_line("  Attention:"))
    cat_weight = attn_weights_sat[1]
    bar_attn = bar_char(cat_weight)
    s.append(title_line(f'    cat weight at sat:  {cat_weight:.3f}  {bar_attn}'))
    s.append(title_line("    (direct Q.K dot product, no intermediary)"))
    s.append(blank())
    s.append(separator())
    s.append(title_line("  The attention mechanism preserves connections"))
    s.append(title_line("  regardless of distance. This is why:"))
    s.append(blank())
    s.append(title_line('  "The cat that I saw at the park near the river sat"'))
    s.append(title_line("         ^                                      ^"))
    s.append(title_line("         10 words apart -- no problem for attention"))
    s.append(blank())
    slides.append(("Comparison", 2, s))

    # ======== Part 3: Causal Masking ========
    s = []
    s.append(blank())
    s.append(title_line("PART 3: CAUSAL MASKING"))
    s.append(title_line("When generating text, don't peek at the future"))
    s.append(separator())
    s.append(blank())
    s.append(title_line("  Generating: 'The cat that I saw yesterday ___ ___'"))
    s.append(title_line("  To predict the next word, the model must NOT see it."))
    s.append(blank())
    s.append(title_line("  Without masking, 'The' could see 'sat' (cheating!)."))
    s.append(title_line("  The mask blocks future positions:"))
    s.append(blank())

    header = "           "
    for tok in TOKENS:
        header += f" {tok:>5}"
    s.append(title_line(header))
    s.append(title_line("           " + "-" * (6 * len(TOKENS))))

    for i, tok in enumerate(TOKENS):
        row = f'  {tok:>9} '
        for j in range(len(TOKENS)):
            if j <= i:
                row += "   ok"
            else:
                row += " MASK"
        s.append(title_line(row))

    s.append(blank())
    s.append(separator())
    s.append(title_line("  'The' can only see itself."))
    s.append(title_line("  'sat' can see The, cat, that, I, saw, yesterday, sat."))
    s.append(title_line("  Nobody can see 'down' until it's their turn."))
    s.append(blank())
    slides.append(("Causal mask", 3, s))

    # ======== Masking in numbers ========
    s = []
    s.append(blank())
    s.append(title_line("CAUSAL MASK IN THE SCORE MATRIX"))
    s.append(title_line("Set future scores to -inf, then softmax gives them 0"))
    s.append(separator())
    s.append(blank())
    s.append(title_line("  Focus on 'that' (word 3) as an example:"))
    s.append(blank())

    # Compute scores for "that" (index 2)
    that_scores_all = [dot(Qs[2], Ks[j]) / scale for j in range(len(TOKENS))]
    s.append(title_line("  Raw scores (before masking):"))
    for j, tok in enumerate(TOKENS):
        score = that_scores_all[j]
        s.append(title_line(f'    that -> "{tok:9s}":  {score:+.3f}'))
    s.append(blank())

    # After masking
    s.append(title_line("  After masking (future = -inf):"))
    masked = []
    for j, tok in enumerate(TOKENS):
        if j <= 2:
            score = that_scores_all[j]
            masked.append(score)
            s.append(title_line(f'    that -> "{tok:9s}":  {score:+.3f}'))
        else:
            s.append(title_line(f'    that -> "{tok:9s}":  -inf   (MASKED)'))

    weights_that = softmax(masked)
    s.append(blank())
    s.append(title_line("  After softmax:"))
    masked_j = 0
    for j, tok in enumerate(TOKENS):
        if j <= 2:
            w = weights_that[masked_j]
            pct = int(w * 100)
            s.append(title_line(f'    that -> "{tok:9s}":  {pct:3d}%'))
            masked_j += 1
        else:
            s.append(title_line(f'    that -> "{tok:9s}":    0%  (softmax(-inf) = 0)'))

    s.append(blank())
    slides.append(("Mask numbers", 3, s))

    # ======== Part 4: Scaling ========
    s = []
    s.append(blank())
    s.append(title_line("PART 4: WHY SCALE BY sqrt(d_k)?"))
    s.append(title_line("Without scaling, attention becomes too sharp"))
    s.append(separator())
    s.append(blank())
    s.append(title_line("  Dot products grow with dimension size."))
    s.append(title_line("  If d_k is large, scores become huge."))
    s.append(title_line("  Huge inputs to softmax -> nearly one-hot output."))
    s.append(blank())
    s.append(title_line("  Example: 5 random scores at different scales:"))
    s.append(blank())

    # Demo with fixed values
    base = [0.8, 1.2, 0.5, 0.9, 0.3]

    for multiplier, label in [(1, "d_k=3"), (3, "d_k=27"), (10, "d_k=300")]:
        scaled_vals = [v * multiplier for v in base]
        probs = softmax(scaled_vals)
        max_p = max(probs)
        dist = ", ".join(f"{p:.2f}" for p in probs)
        s.append(title_line(f"  {label:>8}:  scores x{multiplier:<3}  softmax = [{dist}]"))
        s.append(title_line(f"             max = {max_p:.2f}  {'<-- nearly one-hot!' if max_p > 0.7 else ''}"))

    s.append(blank())
    s.append(separator())
    s.append(title_line(f"  Dividing by sqrt(d_k) = sqrt({d_k}) = {scale:.2f}"))
    s.append(title_line("  keeps scores in a healthy range where softmax"))
    s.append(title_line("  can produce a smooth distribution, not a spike."))
    s.append(blank())
    s.append(title_line("  Smooth = blend information from multiple words"))
    s.append(title_line("  Spike  = only copy from one word (too rigid)"))
    s.append(blank())
    slides.append(("Scaling", 4, s))

    # ======== Part 5: Multi-Head ========
    s = []
    s.append(blank())
    s.append(title_line("PART 5: MULTI-HEAD ATTENTION"))
    s.append(title_line("One head asks one type of question. More heads = richer."))
    s.append(separator())
    s.append(blank())
    s.append(title_line('  "sat" asks two DIFFERENT questions simultaneously:'))
    s.append(blank())

    s.append(title_line("  HEAD 1: 'Who is my subject?' (syntactic role)"))
    for j in range(SAT + 1):
        tok = TOKENS[j]
        w = attn_weights_sat[j]
        pct = int(w * 100)
        bar = "#" * int(w * 30)
        s.append(title_line(f'    -> "{tok:9s}"  {pct:2d}%  {bar}'))
    s.append(blank())

    s.append(title_line("  HEAD 2: 'What describes the scene?' (semantic)"))
    for j in range(SAT + 1):
        tok = TOKENS[j]
        w = attn_weights_sat_h2[j]
        pct = int(w * 100)
        bar = "#" * int(w * 30)
        s.append(title_line(f'    -> "{tok:9s}"  {pct:2d}%  {bar}'))
    s.append(blank())

    s.append(separator())
    s.append(title_line("  Each head has its OWN W_Q, W_K matrices."))
    s.append(title_line("  Different weights = different attention patterns."))
    s.append(blank())
    slides.append(("Multi-head", 5, s))

    # ======== Multi-head concatenation ========
    s = []
    s.append(blank())
    s.append(title_line("HOW MULTI-HEAD OUTPUTS COMBINE"))
    s.append(separator())
    s.append(blank())
    s.append(title_line("  1. Each head produces its own output (weighted V sum)"))
    s.append(blank())

    # Compute head 1 output for sat
    h1_out = [0.0, 0.0, 0.0]
    for j in range(SAT + 1):
        for d in range(3):
            h1_out[d] += attn_weights_sat[j] * Vs[j][d]
    h1_out = [round(x, 3) for x in h1_out]
    s.append(title_line(f"  Head 1 output: {fmt_vec(h1_out)}  (3 dims)"))

    # Compute head 2 output (using same V for simplicity)
    h2_out = [0.0, 0.0, 0.0]
    for j in range(SAT + 1):
        for d in range(3):
            h2_out[d] += attn_weights_sat_h2[j] * Vs[j][d]
    h2_out = [round(x, 3) for x in h2_out]
    s.append(title_line(f"  Head 2 output: {fmt_vec(h2_out)}  (3 dims)"))

    s.append(blank())
    s.append(title_line("  2. Concatenate all heads:"))
    concat = h1_out + h2_out
    s.append(title_line(f"     concat = {fmt_vec(concat)}"))
    s.append(title_line(f"     (3 + 3 = 6 dims)"))
    s.append(blank())
    s.append(title_line("  3. Project back to model dimension:"))
    s.append(title_line("     output = concat @ W_O    (6 -> 4 dims)"))
    s.append(blank())
    s.append(separator())
    s.append(title_line("  Real models:"))
    s.append(title_line("    GPT-2:    12 heads, each 64-dim = 768 total"))
    s.append(title_line("    GPT-3:    96 heads, each 128-dim = 12,288 total"))
    s.append(title_line("    GPT-4:    likely 128+ heads"))
    s.append(blank())
    s.append(title_line("  Each head learns a DIFFERENT type of relationship."))
    s.append(blank())
    slides.append(("Concat heads", 5, s))

    # ======== Part 6: Parallelism ========
    s = []
    s.append(blank())
    s.append(title_line("PART 6: PARALLELISM -- WHY GPUs LOVE ATTENTION"))
    s.append(separator())
    s.append(blank())
    s.append(title_line("  RNN: each step DEPENDS on the previous step"))
    s.append(blank())
    s.append(title_line("    h1 = f(x1)"))
    s.append(title_line("    h2 = f(x2, h1)     <-- must wait for h1"))
    s.append(title_line("    h3 = f(x3, h2)     <-- must wait for h2"))
    s.append(title_line("    ...                 <-- sequential, can't parallelize"))
    s.append(blank())
    s.append(separator())
    s.append(blank())
    s.append(title_line("  Attention: ALL pairs computed at once"))
    s.append(blank())
    s.append(title_line("    Q = X @ W_Q        <-- one matrix multiply"))
    s.append(title_line("    K = X @ W_K        <-- one matrix multiply"))
    s.append(title_line("    scores = Q @ K^T   <-- one matrix multiply"))
    s.append(title_line("    ...all pairs in parallel!"))
    s.append(blank())
    s.append(separator())
    s.append(title_line("  Matrix multiplies are what GPUs do best."))
    s.append(title_line("  This is why transformers can train on trillions of tokens"))
    s.append(title_line("  while RNNs top out around billions."))
    s.append(blank())
    slides.append(("Parallelism", 6, s))

    # ======== Parallelism visual ========
    s = []
    s.append(blank())
    s.append(title_line("SEQUENTIAL vs PARALLEL: VISUAL"))
    s.append(separator())
    s.append(blank())
    s.append(title_line("  RNN (8 tokens = 8 sequential steps):"))
    s.append(blank())
    s.append(title_line("  Time ->"))
    s.append(title_line("  GPU:  [The][cat][that][ I ][saw][yes][sat][dwn]"))
    s.append(title_line("         t=1  t=2  t=3  t=4  t=5  t=6  t=7  t=8"))
    s.append(title_line("  Total: 8 time steps (each waits for previous)"))
    s.append(blank())
    s.append(separator())
    s.append(blank())
    s.append(title_line("  Attention (8 tokens = 1 parallel step):"))
    s.append(blank())
    s.append(title_line("  Time ->"))
    s.append(title_line("  GPU:  [The|cat|that| I |saw|yes|sat|dwn]"))
    s.append(title_line("         t=1  ALL AT ONCE (matrix multiply)"))
    s.append(title_line("  Total: 1 time step"))
    s.append(blank())
    s.append(separator())
    s.append(title_line("  8x faster for 8 tokens."))
    s.append(title_line("  1000x faster for 1000 tokens."))
    s.append(title_line("  This is why attention SCALED to GPT-3/4 sizes."))
    s.append(blank())
    slides.append(("Parallel visual", 6, s))

    # ======== Part 7: O(n^2) cost ========
    s = []
    s.append(blank())
    s.append(title_line("PART 7: THE TRADEOFF -- O(n^2) COST"))
    s.append(title_line("Every token attending to every token has a price"))
    s.append(separator())
    s.append(blank())
    s.append(title_line("  Score matrix = n x n (every pair of tokens):"))
    s.append(blank())

    costs = [
        (512, "262K", "1 MB"),
        (2048, "4.2M", "16 MB"),
        (8192, "67M", "256 MB"),
        (32768, "1.07B", "4 GB"),
        (131072, "17.2B", "64 GB"),
    ]

    s.append(title_line(f"  {'Tokens':>8}  {'Operations':>12}  {'Memory':>8}  Relative"))
    s.append(title_line(f"  {'------':>8}  {'----------':>12}  {'------':>8}  --------"))
    for tok, ops, mem in costs:
        rel = (tok / 512) ** 2
        bar = "#" * min(25, int(math.log2(rel + 1) * 4))
        s.append(title_line(f"  {tok:>8,}  {ops:>12}  {mem:>8}  {rel:>6.0f}x  {bar}"))

    s.append(blank())
    s.append(separator())
    s.append(title_line("  Double the context -> 4x the cost (quadratic)."))
    s.append(title_line("  This is why models have context window LIMITS."))
    s.append(blank())
    slides.append(("O(n^2)", 7, s))

    # ======== Solutions to O(n^2) ========
    s = []
    s.append(blank())
    s.append(title_line("HOW REAL MODELS HANDLE O(n^2)"))
    s.append(separator())
    s.append(blank())
    s.append(title_line("  Flash Attention:"))
    s.append(title_line("    Don't store the full n x n matrix in memory."))
    s.append(title_line("    Compute attention in tiles, recompute on backward pass."))
    s.append(title_line("    Same result, 2-4x less memory."))
    s.append(blank())
    s.append(title_line("  KV Cache (for generation):"))
    s.append(title_line("    When generating token by token, store past K and V."))
    s.append(title_line("    New token only computes 1 new Q against cached K/V."))
    s.append(title_line("    Avoids recomputing the entire sequence each step."))
    s.append(blank())
    s.append(title_line("  Sparse Attention:"))
    s.append(title_line("    Only attend to nearby tokens + selected distant ones."))
    s.append(title_line("    Reduces O(n^2) to O(n * sqrt(n)) or O(n * log(n))."))
    s.append(blank())
    s.append(separator())
    s.append(title_line("  These are optimizations of the SAME attention idea."))
    s.append(title_line("  The core mechanism is unchanged since 2017."))
    s.append(blank())
    slides.append(("Solutions", 7, s))

    # ======== Summary ========
    s = []
    s.append(blank())
    s.append(title_line("SUMMARY: WHY ATTENTION IS THE #1 FEATURE"))
    s.append(separator())
    s.append(blank())
    s.append(title_line("  Before attention (RNN):"))
    s.append(title_line("    - Sequential processing (slow, can't parallelize)"))
    s.append(title_line("    - Fixed-size bottleneck (information lost)"))
    s.append(title_line("    - Long-range connections decay"))
    s.append(title_line("    - Max practical size: ~1B parameters"))
    s.append(blank())
    s.append(title_line("  After attention (Transformer):"))
    s.append(title_line("    - Parallel processing (GPU-friendly)"))
    s.append(title_line("    - Every token sees every token (no bottleneck)"))
    s.append(title_line("    - Direct connections at any distance"))
    s.append(title_line("    - Scales to 100B+ parameters"))
    s.append(blank())
    s.append(separator())
    s.append(blank())
    s.append(title_line("  Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k)) V"))
    s.append(blank())
    s.append(title_line("  One equation. Solved the bottleneck."))
    s.append(title_line("  Enabled parallelism. Unlocked scaling."))
    s.append(title_line("  Without it, there are no LLMs."))
    s.append(blank())
    s.append(title_line("  End of animation. Press q to quit, arrows to review."))
    s.append(blank())
    slides.append(("Summary", 8, s))

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
    output.append(f"  WHY ATTENTION MATTERS              Slide {idx + 1}/{total}    Part {step}/8")
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

    speeds = [3.0, 2.5, 2.5, 2.0, 2.0, 2.0, 2.0, 2.0,
              2.0, 2.0, 1.5, 1.5, 1.5, 1.5, 1.5, 1.0]
    default_speed = 1.5

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
        print("Animation ended.")


if __name__ == "__main__":
    main()
