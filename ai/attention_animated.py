"""
Self-Attention — Animated ASCII visualization.

Watch how a Transformer computes attention step by step.
Uses the exact sentence and numbers from TRANSFORMER.md.

Sentence: "The hungry cat sat"

Flow:
  1. Start with embeddings (6-dim vectors)
  2. Multiply by W_Q, W_K, W_V to get Query, Key, Value (3-dim)
  3. Compute attention scores: Q dot K for each pair
  4. Scale by sqrt(d_k) and apply softmax
  5. Weighted sum of Values -> context-aware output

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

# -- Data from TRANSFORMER.md --

TOKENS = ["The", "hungry", "cat", "sat"]

EMBEDDINGS = [
    [0.1, 0.3, 0.0, 0.8, 0.2, 0.1],   # The
    [0.7, 0.2, 0.9, 0.1, 0.5, 0.3],   # hungry
    [0.8, 0.1, 0.6, 0.3, 0.9, 0.2],   # cat
    [0.2, 0.8, 0.3, 0.7, 0.1, 0.6],   # sat
]

W_Q = [
    [0.3, 0.1, 0.5],
    [0.1, 0.6, 0.2],
    [0.5, 0.2, 0.4],
    [0.2, 0.4, 0.1],
    [0.4, 0.3, 0.6],
    [0.1, 0.5, 0.3],
]

W_K = [
    [0.4, 0.2, 0.1],
    [0.1, 0.5, 0.3],
    [0.3, 0.1, 0.6],
    [0.6, 0.3, 0.2],
    [0.2, 0.4, 0.5],
    [0.5, 0.1, 0.4],
]

W_V = [
    [0.2, 0.5, 0.3],
    [0.4, 0.1, 0.6],
    [0.1, 0.3, 0.2],
    [0.3, 0.2, 0.4],
    [0.5, 0.4, 0.1],
    [0.1, 0.6, 0.5],
]


def mat_vec(matrix, vec):
    result = []
    cols = len(matrix[0])
    for c in range(cols):
        s = 0.0
        for r in range(len(vec)):
            s += vec[r] * matrix[r][c]
        result.append(round(s, 2))
    return result


def dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def softmax(vals):
    mx = max(vals)
    exps = [math.exp(v - mx) for v in vals]
    s = sum(exps)
    return [e / s for e in exps]


def fmt_vec(v, width=5):
    return "[" + ", ".join(f"{x:+.2f}" for x in v) + "]"


def fmt_vec_short(v):
    return "[" + ", ".join(f"{x:.2f}" for x in v) + "]"


# -- Pre-compute all the math --

Qs = [mat_vec(W_Q, e) for e in EMBEDDINGS]
Ks = [mat_vec(W_K, e) for e in EMBEDDINGS]
Vs = [mat_vec(W_V, e) for e in EMBEDDINGS]

# Attention scores for "sat" (token 3) — can see tokens 0-3 (causal mask)
SAT = 3
raw_scores = [dot(Qs[SAT], Ks[j]) for j in range(SAT + 1)]
d_k = 3
scale = math.sqrt(d_k)
scaled_scores = [s / scale for s in raw_scores]
attn_weights = softmax(scaled_scores)

# Weighted sum of values
output_sat = [0.0, 0.0, 0.0]
for j in range(SAT + 1):
    for d in range(3):
        output_sat[d] += attn_weights[j] * Vs[j][d]
output_sat = [round(x, 2) for x in output_sat]


# -- Build slides --

WIDTH = 72

def title_line(text):
    pad = WIDTH - 4 - len(text)
    return "  " + text + " " * max(pad, 0)


def separator():
    return "  " + "-" * (WIDTH - 4)


def blank():
    return ""


def build_slides():
    slides = []

    # ======== SLIDE GROUP: Title ========
    s = []
    s.append(blank())
    s.append(title_line("SELF-ATTENTION -- STEP BY STEP"))
    s.append(title_line("How a Transformer understands relationships between words"))
    s.append(separator())
    s.append(blank())
    s.append(title_line('Sentence: "The hungry cat sat"'))
    s.append(blank())
    s.append(title_line("Goal: make each token aware of the other tokens"))
    s.append(title_line("so 'sat' knows WHO sat (the cat) and HOW (hungrily)"))
    s.append(blank())
    s.append(separator())
    s.append(title_line("The pipeline:"))
    s.append(title_line("  Embedding -> Q,K,V -> Scores -> Softmax -> Blend Vs"))
    s.append(blank())
    s.append(title_line("We'll follow token 'sat' through each step."))
    s.append(blank())
    slides.append(("Title", 0, s))

    # ======== SLIDE GROUP: Step 1 — Embeddings ========
    s = []
    s.append(blank())
    s.append(title_line("STEP 1: EMBEDDINGS"))
    s.append(title_line("Each token starts as a 6-dimensional vector"))
    s.append(separator())
    s.append(blank())
    for i, tok in enumerate(TOKENS):
        marker = " <<<" if i == SAT else ""
        s.append(title_line(f'  "{tok:7s}"  {fmt_vec(EMBEDDINGS[i])}{marker}'))
    s.append(blank())
    s.append(separator())
    s.append(title_line("These vectors were learned during training."))
    s.append(title_line("Each dimension encodes some learned feature of the word."))
    s.append(blank())
    s.append(title_line("But right now, each token is ISOLATED."))
    s.append(title_line('"sat" has no idea that "cat" is its subject.'))
    s.append(title_line("Attention will fix that."))
    s.append(blank())
    slides.append(("Embeddings", 1, s))

    # ======== SLIDE GROUP: Step 2 — Weight matrices ========
    s = []
    s.append(blank())
    s.append(title_line("STEP 2a: THE WEIGHT MATRICES"))
    s.append(title_line("Three 6x3 matrices stored in the model"))
    s.append(separator())
    s.append(blank())
    s.append(title_line("  W_Q (Query weights)     W_K (Key weights)      W_V (Value weights)"))
    s.append(title_line("  'What am I looking for?' 'What do I contain?'  'What info do I share?'"))
    s.append(blank())
    for r in range(6):
        q_row = "[" + ", ".join(f"{W_Q[r][c]:.1f}" for c in range(3)) + "]"
        k_row = "[" + ", ".join(f"{W_K[r][c]:.1f}" for c in range(3)) + "]"
        v_row = "[" + ", ".join(f"{W_V[r][c]:.1f}" for c in range(3)) + "]"
        s.append(title_line(f"  {q_row}                 {k_row}                 {v_row}"))
    s.append(blank())
    s.append(separator())
    s.append(title_line("These are the SAME for every token, every sentence."))
    s.append(title_line("Training adjusted these over billions of examples."))
    s.append(blank())
    slides.append(("W matrices", 2, s))

    # ======== SLIDE GROUP: Step 2b — Computing Q, K, V ========
    for i, tok in enumerate(TOKENS):
        s = []
        s.append(blank())
        s.append(title_line(f'STEP 2b: COMPUTE Q, K, V FOR "{tok.upper()}"'))
        s.append(title_line("embedding @ W = projected vector (6-dim -> 3-dim)"))
        s.append(separator())
        s.append(blank())
        s.append(title_line(f"  embedding = {fmt_vec(EMBEDDINGS[i])}"))
        s.append(blank())

        # Show Q computation
        q = Qs[i]
        s.append(title_line(f"  Q = embedding @ W_Q"))
        terms_q = []
        for d in range(3):
            parts = " + ".join(f"{EMBEDDINGS[i][r]:.1f}x{W_Q[r][d]:.1f}" for r in range(6))
            terms_q.append(f"    dim{d}: {parts} = {q[d]:+.2f}")
        for t in terms_q:
            s.append(title_line(t))
        s.append(title_line(f"  Q = {fmt_vec_short(q)}"))
        s.append(blank())

        # Show K computation
        k = Ks[i]
        s.append(title_line(f"  K = embedding @ W_K"))
        s.append(title_line(f"  K = {fmt_vec_short(k)}"))
        s.append(blank())

        # Show V computation
        v = Vs[i]
        s.append(title_line(f"  V = embedding @ W_V"))
        s.append(title_line(f"  V = {fmt_vec_short(v)}"))
        s.append(blank())
        s.append(separator())

        if i == SAT:
            s.append(title_line("Now every token has Q, K, V. Time for attention!"))
        else:
            s.append(title_line(f'  "{tok}" now has 3 vectors: Q (question), K (label), V (content)'))
        s.append(blank())
        slides.append((f"QKV {tok}", 2, s))

    # ======== Summary of all Q, K, V ========
    s = []
    s.append(blank())
    s.append(title_line("STEP 2 SUMMARY: ALL Q, K, V VECTORS"))
    s.append(separator())
    s.append(blank())
    s.append(title_line("           Q (query)         K (key)          V (value)"))
    s.append(title_line("         'looking for?'    'I contain...'   'I share...'"))
    s.append(blank())
    for i, tok in enumerate(TOKENS):
        marker = " <--" if i == SAT else ""
        s.append(title_line(f'  "{tok:7s}" {fmt_vec_short(Qs[i])}    {fmt_vec_short(Ks[i])}    {fmt_vec_short(Vs[i])}{marker}'))
    s.append(blank())
    s.append(separator())
    s.append(title_line("Each token went from 1 vector (embedding) to 4 vectors."))
    s.append(title_line("  embedding = who the token IS"))
    s.append(title_line("  Q = what it's LOOKING FOR"))
    s.append(title_line("  K = what it ADVERTISES"))
    s.append(title_line("  V = what it OFFERS if matched"))
    s.append(blank())
    slides.append(("QKV summary", 2, s))

    # ======== SLIDE GROUP: Step 3 — Attention scores ========
    s = []
    s.append(blank())
    s.append(title_line('STEP 3: ATTENTION SCORES FOR "SAT"'))
    s.append(title_line("sat's Q knocks on each token's K (dot product)"))
    s.append(separator())
    s.append(blank())
    s.append(title_line(f"  Q_sat = {fmt_vec_short(Qs[SAT])}"))
    s.append(blank())

    for j in range(SAT + 1):
        tok = TOKENS[j]
        score = raw_scores[j]
        detail = " + ".join(
            f"{Qs[SAT][d]:.2f}x{Ks[j][d]:.2f}" for d in range(3)
        )
        bar_len = int(score * 15)
        bar = "#" * bar_len
        s.append(title_line(f'  sat -> "{tok:7s}":  {detail} = {score:.2f}  {bar}'))

    s.append(blank())
    s.append(separator())
    winner = TOKENS[raw_scores.index(max(raw_scores))]
    s.append(title_line(f'  "{winner}" scores highest! The Q.K dot product'))
    s.append(title_line("  naturally gives higher scores to related tokens."))
    s.append(blank())
    s.append(title_line("  Causal mask: sat can only see The, hungry, cat, sat"))
    s.append(title_line("  (no future tokens like quietly, on, the, warm, mat)"))
    s.append(blank())
    slides.append(("Attention scores", 3, s))

    # ======== SLIDE GROUP: Step 4a — Scaling ========
    s = []
    s.append(blank())
    s.append(title_line("STEP 4a: SCALE THE SCORES"))
    s.append(title_line(f"Divide by sqrt(d_k) = sqrt({d_k}) = {scale:.2f}"))
    s.append(separator())
    s.append(blank())
    s.append(title_line("  Why scale? Without it, large dot products make softmax"))
    s.append(title_line("  too spiky -- one token gets ~100%, others get ~0%."))
    s.append(title_line("  Scaling keeps gradients healthy during training."))
    s.append(blank())

    for j in range(SAT + 1):
        tok = TOKENS[j]
        s.append(title_line(f'  "{tok:7s}":  {raw_scores[j]:.2f} / {scale:.2f} = {scaled_scores[j]:.2f}'))
    s.append(blank())
    s.append(separator())
    s.append(title_line("  The relative ranking stays the same."))
    s.append(title_line("  Scaling just prevents extreme softmax outputs."))
    s.append(blank())
    slides.append(("Scale", 4, s))

    # ======== SLIDE GROUP: Step 4b — Softmax ========
    s = []
    s.append(blank())
    s.append(title_line("STEP 4b: SOFTMAX -> PROBABILITIES"))
    s.append(title_line("Convert scores to weights that sum to 1.0"))
    s.append(separator())
    s.append(blank())
    s.append(title_line("  Formula: softmax(x_i) = e^x_i / sum(e^x_j)"))
    s.append(blank())

    exps = [math.exp(v) for v in scaled_scores]
    exp_sum = sum(exps)
    s.append(title_line("  e^ each scaled score:"))
    for j in range(SAT + 1):
        tok = TOKENS[j]
        s.append(title_line(f'    "{tok:7s}":  e^{scaled_scores[j]:.2f} = {exps[j]:.3f}'))
    s.append(blank())
    s.append(title_line(f"  Sum of e^ values: {exp_sum:.3f}"))
    s.append(blank())

    s.append(title_line("  Divide each by sum:"))
    for j in range(SAT + 1):
        tok = TOKENS[j]
        w = attn_weights[j]
        pct = int(w * 100)
        bar = "#" * int(w * 40)
        s.append(title_line(f'    "{tok:7s}":  {exps[j]:.3f}/{exp_sum:.3f} = {w:.2f} ({pct:2d}%)  {bar}'))
    s.append(blank())

    total = sum(attn_weights)
    s.append(title_line(f"  Total: {total:.2f} (always sums to 1.0)"))
    s.append(blank())
    s.append(separator())
    best = TOKENS[attn_weights.index(max(attn_weights))]
    best_pct = int(max(attn_weights) * 100)
    s.append(title_line(f'  "sat" pays {best_pct}% attention to "{best}" -- its subject!'))
    s.append(blank())
    slides.append(("Softmax", 4, s))

    # ======== SLIDE GROUP: Step 5 — Weighted sum of Values ========
    s = []
    s.append(blank())
    s.append(title_line('STEP 5: BLEND VALUES -> NEW "SAT" VECTOR'))
    s.append(title_line("Multiply each V by its attention weight, then sum"))
    s.append(separator())
    s.append(blank())
    s.append(title_line("  output = w0*V_The + w1*V_hungry + w2*V_cat + w3*V_sat"))
    s.append(blank())

    contributions = []
    for j in range(SAT + 1):
        tok = TOKENS[j]
        w = attn_weights[j]
        v = Vs[j]
        contrib = [round(w * v[d], 2) for d in range(3)]
        contributions.append(contrib)
        pct = int(w * 100)
        s.append(title_line(f'  {w:.2f} x V_{tok:7s} {fmt_vec_short(v)} = {fmt_vec_short(contrib)}  ({pct}%)'))
    s.append(title_line("  " + "-" * 50))

    s.append(title_line(f"  output_sat = {fmt_vec_short(output_sat)}"))
    s.append(blank())
    s.append(separator())
    s.append(title_line("  Before attention: 'sat' = just the word 'sat'"))
    s.append(title_line("  After attention:  'sat' = 'sat' enriched with 'cat' and 'hungry'"))
    s.append(blank())
    s.append(title_line("  The model now knows this is about a hungry cat sitting,"))
    s.append(title_line("  not just any sitting. Context flows through attention."))
    s.append(blank())
    slides.append(("Blend Values", 5, s))

    # ======== SLIDE GROUP: Visual — attention flow ========
    s = []
    s.append(blank())
    s.append(title_line("ATTENTION FLOW VISUALIZATION"))
    s.append(title_line("Line thickness = attention weight"))
    s.append(separator())
    s.append(blank())
    s.append(title_line('  "The"       "hungry"      "cat"         "sat"'))
    s.append(title_line("    |            |            |              |"))
    s.append(title_line("   emb          emb          emb            emb"))
    s.append(title_line("    |            |            |              |"))
    s.append(title_line("  Q,K,V       Q,K,V        Q,K,V          Q,K,V"))
    s.append(title_line("    |            |            |              |"))
    s.append(title_line("    K            K            K           Q  K"))
    s.append(title_line("    |            |            |           |  |"))

    # Show connections with varying thickness
    w_the = int(attn_weights[0] * 100)
    w_hun = int(attn_weights[1] * 100)
    w_cat = int(attn_weights[2] * 100)
    w_sat = int(attn_weights[3] * 100)

    s.append(title_line(f"    +-----({w_the:2d}%)----+---({w_hun:2d}%)---+--({w_cat:2d}%)-+  +({w_sat:2d}%)"))
    s.append(title_line("    |            |            |           |  |"))
    s.append(title_line("    V            V            V           V  |"))
    s.append(title_line("    |            |            |           |  |"))
    s.append(title_line(f"   x{attn_weights[0]:.2f}        x{attn_weights[1]:.2f}        x{attn_weights[2]:.2f}       x{attn_weights[3]:.2f}"))
    s.append(title_line("    |            |            |           |"))
    s.append(title_line("    +------------+------+-----+-----------+"))
    s.append(title_line("                        |"))
    s.append(title_line("                       SUM"))
    s.append(title_line("                        |"))
    s.append(title_line(f"                  output_sat = {fmt_vec_short(output_sat)}"))
    s.append(blank())
    s.append(separator())
    s.append(title_line("  Q asks 'who is relevant to me?'"))
    s.append(title_line("  K answers 'I might be!'"))
    s.append(title_line("  V provides the actual content to blend"))
    s.append(blank())
    slides.append(("Flow diagram", 5, s))

    # ======== SLIDE GROUP: Full attention matrix ========
    s = []
    s.append(blank())
    s.append(title_line("FULL ATTENTION MATRIX (all tokens)"))
    s.append(title_line("Each row: how one token distributes its attention"))
    s.append(separator())
    s.append(blank())

    # Compute attention for all tokens
    all_weights = []
    for i in range(len(TOKENS)):
        row_scores = [dot(Qs[i], Ks[j]) / scale for j in range(i + 1)]
        row_weights = softmax(row_scores)
        # Pad with zeros for masked positions
        padded = list(row_weights) + [0.0] * (len(TOKENS) - i - 1)
        all_weights.append(padded)

    s.append(title_line("              The    hungry   cat     sat"))
    s.append(title_line("            +" + "-" * 38 + "+"))

    for i, tok in enumerate(TOKENS):
        row = f'  "{tok:7s}" |'
        for j in range(len(TOKENS)):
            w = all_weights[i][j]
            if j > i:
                row += "    .   "
            else:
                pct = int(w * 100)
                if pct >= 30:
                    row += f"  [{pct:2d}%]"
                elif pct >= 15:
                    row += f"   {pct:2d}% "
                else:
                    row += f"   {pct:2d}% "
        row += "  |"
        s.append(title_line(row))

    s.append(title_line("            +" + "-" * 38 + "+"))
    s.append(blank())
    s.append(title_line("  [XX%] = strong attention    XX% = moderate     . = masked"))
    s.append(blank())
    s.append(separator())
    s.append(title_line("  Each row sums to 100%. Each token decides independently"))
    s.append(title_line("  how to distribute its attention across visible tokens."))
    s.append(blank())
    slides.append(("Attention matrix", 5, s))

    # ======== SLIDE GROUP: What happens next ========
    s = []
    s.append(blank())
    s.append(title_line("WHAT HAPPENS NEXT?"))
    s.append(separator())
    s.append(blank())
    s.append(title_line("  This was ONE attention head in ONE layer."))
    s.append(blank())
    s.append(title_line("  A real Transformer has:"))
    s.append(title_line("    - Multiple heads (each asks a different question)"))
    s.append(title_line("        Head 1: 'who is my subject?'     sat -> cat"))
    s.append(title_line("        Head 2: 'what describes me?'     cat -> hungry"))
    s.append(title_line("        Head 3: 'where did this happen?' sat -> on, mat"))
    s.append(blank())
    s.append(title_line("    - Multiple layers (each refines understanding)"))
    s.append(title_line("        Layer 1:  basic syntax (adjective near noun)"))
    s.append(title_line("        Layer 16: semantic roles (agent, action, object)"))
    s.append(title_line("        Layer 32: abstract reasoning"))
    s.append(blank())
    s.append(title_line("  GPT-4 class: 96+ layers, 96+ heads per layer"))
    s.append(title_line("  = thousands of different 'questions' asked simultaneously"))
    s.append(blank())
    s.append(separator())
    s.append(title_line("  embedding -> [attention + FFN] x N layers -> prediction"))
    s.append(title_line("                    \\___________________/"))
    s.append(title_line("                     this animation showed"))
    s.append(title_line("                     just the attention part"))
    s.append(title_line("                     of one head, one layer"))
    s.append(blank())
    slides.append(("Next steps", 6, s))

    # ======== SLIDE GROUP: Summary ========
    s = []
    s.append(blank())
    s.append(title_line("SUMMARY: THE ATTENTION MECHANISM"))
    s.append(separator())
    s.append(blank())
    s.append(title_line("  1. EMBEDDING:   each token gets a vector (who am I?)"))
    s.append(title_line("  2. Q, K, V:     three projections via learned W matrices"))
    s.append(title_line("                  Q = what am I looking for?"))
    s.append(title_line("                  K = what do I advertise?"))
    s.append(title_line("                  V = what content do I carry?"))
    s.append(title_line("  3. SCORES:      Q dot K = how relevant is each token?"))
    s.append(title_line("  4. SOFTMAX:     scores -> probabilities (sum to 1)"))
    s.append(title_line("  5. BLEND:       weighted sum of V vectors"))
    s.append(title_line("                  -> one context-aware output vector"))
    s.append(blank())
    s.append(separator())
    s.append(blank())
    s.append(title_line("  The key insight:"))
    s.append(title_line("    Before attention: 'sat' is just the word 'sat'"))
    s.append(title_line("    After attention:  'sat' carries info from the whole sentence"))
    s.append(title_line("    The W matrices learned WHAT to pay attention to"))
    s.append(title_line("    from billions of training examples."))
    s.append(blank())
    s.append(title_line("  End of animation. Press q to quit, arrows to review."))
    s.append(blank())
    slides.append(("Summary", 7, s))

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
    output.append(f"  SELF-ATTENTION ANIMATION          Slide {idx + 1}/{total}    Step {step}/7")
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

    speeds = [3.0, 3.0, 2.5, 2.5, 2.0, 2.0, 2.0, 2.0, 2.0,
              1.5, 1.5, 1.5, 1.0, 1.0]
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
        print("Attention animation ended.")


if __name__ == "__main__":
    main()
