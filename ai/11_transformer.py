"""
Lesson 11: Build a Transformer from Scratch (nanoGPT-style)
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from lesson_player import LessonPlayer, cyan, green

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
torch.manual_seed(42)

player = LessonPlayer("Lesson 11: Build a Transformer from Scratch")

# ── Corpus ──────────────────────────────────────────────
# A tiny piece of Shakespeare — enough for a toy language model.

TEXT = """\
First Citizen:
Before we proceed any further, hear me speak.

All:
Speak, speak.

First Citizen:
You are all resolved rather to die than to famish?

All:
Resolved. resolved.

First Citizen:
First, you know Caius Marcius is chief enemy to the people.

All:
We know't, we know't.

First Citizen:
Let us kill him, and we'll have corn at our own price.
Is't a verdict?

All:
No more talking on't; let it be done: away, away!

Second Citizen:
One word, good citizens.

First Citizen:
We are accounted poor citizens, the patricians good.
What authority surfeits on would relieve us: if they
would yield us but the superfluity, while it were
wholesome, we might guess they relieved us humanely;
but they think we are too dear.
"""

# ── Step 1: the goal ────────────────────────────────────

player.add_step("Step 1: What We're Building", f"""\
  A {cyan('character-level language model')}. Given a string of text,
  predict the next character. Keep feeding the output back in — you
  get generation.

  {cyan('Same idea as GPT, just smaller:')}
    GPT-3:  50,257 tokens, 96 layers, 175B params, trained on the internet
    Ours:    ~65 chars,    2 layers,  ~40k params, trained on Shakespeare

  {cyan('Why character-level?')}
    No tokenizer to train. The "vocab" is just the unique characters.
    Easy to see what's going on. Production models use BPE subwords
    instead, but the math is identical.

  {green('Corpus size:')} {len(TEXT)} characters
  {green('Pipeline:')} text → IDs → embeddings → transformer → logits → next char""")

# ── Step 2: tokenization ────────────────────────────────

chars = sorted(set(TEXT))
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for ch, i in stoi.items()}


def encode(s):
    return [stoi[ch] for ch in s]


def decode(ids):
    return "".join(itos[i] for i in ids)


data = torch.tensor(encode(TEXT), dtype=torch.long)
sample = TEXT[:20]
sample_ids = encode(sample)

player.add_step("Step 2: Tokenization (char → ID)", f"""\
  Build a vocabulary from every unique character. Each char becomes an
  integer ID. Encoding turns text into a tensor of IDs.

  {cyan('Code:')}
    chars = sorted(set(TEXT))
    stoi = {{ch: i for i, ch in enumerate(chars)}}
    itos = {{i: ch for ch, i in stoi.items()}}

    encode("hi") = [stoi[c] for c in "hi"]
    decode([7, 8]) = "".join(itos[i] for i in [7, 8])

  {green('Vocab size:')} {vocab_size}
  {green('Example:')}
    encode({sample!r})
      → {sample_ids}
    decode(...) → {decode(sample_ids)!r}

  {green('Full corpus as a tensor:')} shape = {list(data.shape)}""")

# ── Step 3: batching ────────────────────────────────────

SEQ_LEN = 32
BATCH_SIZE = 16


def get_batch():
    ix = torch.randint(0, len(data) - SEQ_LEN - 1, (BATCH_SIZE,))
    x = torch.stack([data[i:i + SEQ_LEN] for i in ix])
    y = torch.stack([data[i + 1:i + SEQ_LEN + 1] for i in ix])
    return x, y


xb, yb = get_batch()

player.add_step("Step 3: Training Batches", f"""\
  For each training step we grab random slices of the corpus.
  {cyan('Input x')} = characters at positions [i, i+SEQ_LEN).
  {cyan('Target y')} = characters at positions [i+1, i+SEQ_LEN+1) — shifted by 1.

  At every position t, the model must predict x[t+1] given x[:t+1].
  So one batch gives us SEQ_LEN × BATCH_SIZE prediction examples.

  {cyan('Code:')}
    ix = torch.randint(0, len(data) - SEQ_LEN - 1, (BATCH_SIZE,))
    x = torch.stack([data[i:i + SEQ_LEN] for i in ix])
    y = torch.stack([data[i + 1:i + SEQ_LEN + 1] for i in ix])

  {green('Shapes:')}
    x: {list(xb.shape)}   (batch_size × seq_len)
    y: {list(yb.shape)}   (same shape, shifted by 1)

  {green('First row decoded:')}
    x[0]: {decode(xb[0].tolist())!r}
    y[0]: {decode(yb[0].tolist())!r}""")

# ── Step 4: self-attention ──────────────────────────────

player.add_step("Step 4: Self-Attention — the Core Idea", f"""\
  The transformer's key innovation. Each position looks at {cyan('every')}
  earlier position and decides how much to attend to each one.

  {cyan('Three projections per position:')}
    Q (query) — what am I looking for?
    K (key)   — what do I offer?
    V (value) — what do I share?

  {cyan('Attention formula:')}
    attn(Q, K, V) = softmax(Q @ Kᵀ / √d_k) @ V

  Intuition: dot-product Q·K = similarity. Softmax turns similarities
  into weights. Weighted sum of V = contextual representation.

  {cyan('Causal mask:')} for language modeling, position t may only
  look at positions 0..t. We zero out the upper triangle with -inf
  before softmax, so the future can't leak into the past.

  {cyan('Multi-head:')} run H attentions in parallel with smaller
  dimensions (d_model / H each), concatenate. Different heads can
  learn different relationships (syntax, semantics, distance...).""")

# ── Step 5: building blocks ─────────────────────────────


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads, seq_len):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        mask = torch.tril(torch.ones(seq_len, seq_len))
        self.register_buffer("mask", mask.view(1, 1, seq_len, seq_len))

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = attn.masked_fill(self.mask[:, :, :T, :T] == 0, float('-inf'))
        attn = F.softmax(attn, dim=-1)
        out = (attn @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(out)


class Block(nn.Module):
    def __init__(self, d_model, n_heads, seq_len):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, seq_len)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))   # residual 1
        x = x + self.mlp(self.ln2(x))    # residual 2
        return x


player.add_step("Step 5: The Transformer Block", f"""\
  A block = attention + feed-forward, each with a residual and LayerNorm.

  {cyan('Code (simplified):')}
    class Block(nn.Module):
        def forward(self, x):
            x = x + self.attn(self.ln1(x))    # residual around attention
            x = x + self.mlp(self.ln2(x))     # residual around MLP
            return x

  {cyan('The MLP:')}
    Linear(d_model → 4·d_model) → GELU → Linear(4·d_model → d_model)
    Lets the model "think" about each token independently after
    attention has mixed information between tokens.

  {cyan('Residuals (x + f(x)):')} keep gradients flowing in deep nets.
  {cyan('LayerNorm:')} stabilizes training by normalizing activations.
  {cyan('Modern twist:')} we do LayerNorm BEFORE attention/MLP
  ("pre-norm") — easier to train than the original "post-norm".

  Stacking N blocks = deeper model. GPT-3 has 96. We'll use 2.""")

# ── Step 6: full model ──────────────────────────────────


class MiniGPT(nn.Module):
    def __init__(self, vocab_size, d_model=64, n_heads=4, n_layers=2, seq_len=32):
        super().__init__()
        self.seq_len = seq_len
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(seq_len, d_model)
        self.blocks = nn.Sequential(
            *[Block(d_model, n_heads, seq_len) for _ in range(n_layers)]
        )
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, idx):
        B, T = idx.shape
        tok = self.tok_emb(idx)
        pos = self.pos_emb(torch.arange(T, device=idx.device))
        x = tok + pos
        x = self.blocks(x)
        x = self.ln_f(x)
        return self.head(x)

    def generate(self, idx, max_new_tokens, temperature=1.0):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.seq_len:]
            logits = self.forward(idx_cond)[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            next_idx = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_idx], dim=1)
        return idx


model = MiniGPT(vocab_size, d_model=64, n_heads=4, n_layers=2, seq_len=SEQ_LEN)
n_params = sum(p.numel() for p in model.parameters())

player.add_step("Step 6: Assemble the Full Model", f"""\
  {cyan('Forward pass:')}
    1. Token embedding:    ID → d_model vector
    2. Positional embedding: position → d_model vector (add to tokens)
    3. N transformer blocks (attention + MLP, stacked)
    4. Final LayerNorm
    5. Linear projection → logits over vocabulary

  {cyan('Why positional embeddings?')} Attention is permutation-invariant
  — without position info, "cat sat mat" and "mat sat cat" look the
  same. We add a learned vector per position to break the symmetry.

  {cyan('Code:')}
    def forward(self, idx):
        tok = self.tok_emb(idx)                   # (B, T, d)
        pos = self.pos_emb(torch.arange(T))       # (T, d)
        x = tok + pos                             # broadcast add
        x = self.blocks(x)                        # N transformer blocks
        x = self.ln_f(x)
        return self.head(x)                       # (B, T, vocab)

  {green('Model:')} d_model=64, n_heads=4, n_layers=2
  {green('Total parameters:')} {n_params:,}""")

# ── Step 7: training ────────────────────────────────────

optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)

losses = []
log_lines = ""
N_STEPS = 500

model.train()
for step in range(N_STEPS):
    x, y = get_batch()
    logits = model(x)
    loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    losses.append(loss.item())
    if (step + 1) % 100 == 0:
        log_lines += f"    step {step+1:4d}/{N_STEPS}  loss={loss.item():.4f}\n"

plt.figure(figsize=(8, 4))
plt.plot(losses)
plt.xlabel("Training step")
plt.ylabel("Cross-entropy loss")
plt.title("MiniGPT Training Loss")
plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, "images", "11_transformer_loss.png"))
plt.close()

player.add_step("Step 7: Training", f"""\
  Same 4-step loop as every neural net (lesson 10):
    forward → loss → backward → step.

  {cyan('Code:')}
    for step in range(N_STEPS):
        x, y = get_batch()                      # (B, T) each
        logits = model(x)                       # (B, T, vocab)
        loss = F.cross_entropy(
            logits.view(-1, vocab_size),
            y.view(-1),
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

  {cyan('Why flatten?')} CrossEntropy wants (N, C) predictions and (N,)
  targets. We reshape logits to (B·T, vocab) and y to (B·T,).

  {green('Training:')}
{log_lines}
  Loss at step 1 was around ln({vocab_size}) ≈ {math.log(vocab_size):.2f}
  (uniform guess). Dropping below that means the model learned something.

  Loss curve: images/11_transformer_loss.png""")

# ── Step 8: generation ──────────────────────────────────

model.eval()
with torch.no_grad():
    ctx = torch.tensor([[encode("First ")[0]]], dtype=torch.long)
    out = model.generate(ctx, max_new_tokens=250, temperature=0.8)
    sample_cold = decode(out[0].tolist())

    prompt = torch.tensor([encode("First Citizen:\n")], dtype=torch.long)
    out2 = model.generate(prompt, max_new_tokens=200, temperature=0.8)
    sample_prompted = decode(out2[0].tolist())

player.add_step("Step 8: Generation", f"""\
  {cyan('Autoregressive sampling:')} feed the current tokens in, get
  logits for the next position, sample from the softmax, append,
  repeat.

  {cyan('Code:')}
    def generate(self, idx, max_new_tokens, temperature=1.0):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.seq_len:]        # last seq_len tokens
            logits = self.forward(idx_cond)[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            next_idx = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_idx], dim=1)
        return idx

  {cyan('Temperature:')} divides logits before softmax.
    low  (0.3) → sharper, more deterministic
    high (1.2) → flatter, more random

  {green('Cold start (single char seed):')}
{sample_cold}

  {green('With a prompt:')}
{sample_prompted}

  Not Shakespeare — but the model has clearly learned capital letters,
  colons after names, and word-ish shapes. More data + bigger model +
  more training = GPT.""")

# ── Step 9: summary ─────────────────────────────────────

player.add_step("Step 9: What You Just Built", f"""\
  You implemented the same architecture that powers GPT, Claude, Llama
  — just tiny. Every piece is here:

  {green('✓')} Tokenizer (char-level)           → real LLMs use BPE
  {green('✓')} Token + positional embeddings    → same, maybe RoPE
  {green('✓')} Multi-head causal self-attention → same, maybe grouped-query
  {green('✓')} Feed-forward block               → same, maybe SwiGLU
  {green('✓')} Residual connections + LayerNorm → same, maybe RMSNorm
  {green('✓')} Autoregressive generation        → same, plus top-k / top-p

  {cyan('The gap from here to frontier models is scale, not concept:')}
    MiniGPT (this):   ~40k params,  500 training steps
    GPT-2 small:       124M params
    Llama 3 8B:        8B params,   ~15T tokens of training data
    GPT-4 (rumored):   ~1.7T params across mixture-of-experts

  {cyan('Next up (lesson 12):')} take a pre-trained model and adapt it
  to a new domain — {green('fine-tuning')} and {green('LoRA')}.""")

# ── Play ────────────────────────────────────────────────

player.play()
