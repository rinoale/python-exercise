# How Generative AI Actually Works — The Key Mechanics

Modern generative AI (GPT, Claude, Llama, etc.) is not simulating a brain.
It's a stack of specific mathematical operations. This document explains each
one — what it does, why it's needed, and how it connects to the others.

## The big picture

```
"The cat sat on the" → [Tokenize] → [Embed] → [Transformer × N] → [Predict] → "mat"
```

A generative language model does one thing: **predict the next token**.
Everything below exists to make that prediction as good as possible.

---

## 1. Tokenization

**What:** Split text into subword pieces and map each to an integer ID.

```
"unhappiness" → ["un", "happiness"] → [432, 18903]
```

**Why:** Neural networks operate on numbers, not strings. Subword
tokenization (BPE) handles rare words by composing them from known pieces,
so the model doesn't need a separate entry for every word ever written.

---

## 2. Embedding

**What:** Convert each token ID into a high-dimensional vector (e.g., 4096
floats).

```
token 432 ("un") → [0.12, -0.84, 0.03, ..., 0.41]   (4096 numbers)
```

**Why:** A flat integer (432) carries no meaning — it's just an arbitrary
label. An embedding vector places the token in a continuous space where
**similar meanings are near each other**. "king" and "queen" end up close;
"king" and "toaster" end up far apart. These vectors are learned during
training.

---

## 3. Positional Encoding

**What:** Add position information to each embedding so the model knows
token order.

```
embedding("cat") + position(3) → position-aware vector
```

**Why:** Unlike RNNs that process tokens one by one, a Transformer sees all
tokens at once (in parallel). Without positional encoding, "dog bites man"
and "man bites dog" would look identical — the model wouldn't know which
word came first.

---

## 4. Self-Attention (the core innovation)

**What:** For each token, compute how much every other token in the
sequence matters to it, then mix their information accordingly.

**The Q/K/V analogy:**

Think of a library:
- **Query (Q):** "I'm looking for information about X"
- **Key (K):** "I contain information about Y"
- **Value (V):** "Here's my actual content"

Each token generates all three. Attention scores = how well each Query
matches each Key. Those scores determine how much of each Value to absorb.

**The math:**

```
Attention(Q, K, V) = softmax(Q @ Kᵀ / √d_k) @ V
```

Step by step for the sentence "The cat sat on the mat":

```
1. Q @ Kᵀ         → similarity scores between all token pairs
                     "sat" has high score with "cat" (who sat?)
                     "sat" has high score with "mat" (where?)

2. / √d_k         → scale down so scores don't explode
                     (d_k = dimension of keys, e.g., 128)

3. softmax(...)    → convert raw scores to probabilities (sum to 1)
                     "sat" → cat: 0.35, mat: 0.25, the: 0.10, ...

4. ... @ V         → weighted sum of Value vectors
                     "sat" absorbs 35% of cat's meaning,
                     25% of mat's meaning, etc.
```

**Why:** This is how the model builds **context**. The word "bank" alone is
ambiguous. After self-attention, "bank" in "river bank" absorbs meaning
from "river" and becomes a different vector than "bank" in "bank account".

**Causal mask:** In generative models, each token can only attend to tokens
**before** it (not future tokens). Position 5 sees positions 0–4. This is
enforced by setting future attention scores to −∞ before softmax.

```
         The  cat  sat  on  the  mat
The     [  ✓   ✗    ✗   ✗    ✗    ✗ ]
cat     [  ✓   ✓    ✗   ✗    ✗    ✗ ]
sat     [  ✓   ✓    ✓   ✗    ✗    ✗ ]
on      [  ✓   ✓    ✓   ✓    ✗    ✗ ]
the     [  ✓   ✓    ✓   ✓    ✓    ✗ ]
mat     [  ✓   ✓    ✓   ✓    ✓    ✓ ]
```

---

## 5. Multi-Head Attention

**What:** Run self-attention multiple times in parallel, each with
different learned Q/K/V weights.

```
Head 1: learns syntactic relationships    (subject ↔ verb)
Head 2: learns semantic relationships     (adjective ↔ noun)
Head 3: learns positional patterns        (nearby words)
Head 4: learns long-range dependencies    (pronoun ↔ antecedent)
...
Head H: (64–128 heads in large models)
```

**Why:** A single attention can only capture one type of relationship per
layer. Multiple heads let the model track grammar, meaning, position, and
coreference simultaneously. Their outputs are concatenated and projected
back to the original dimension.

```
MultiHead = Concat(head_1, head_2, ..., head_H) @ W_output
```

---

## 6. Feed-Forward Network (FFN / MLP)

**What:** After attention, each token passes through a two-layer neural
network independently.

```
FFN(x) = GELU(x @ W₁ + b₁) @ W₂ + b₂
```

**Why:** Attention mixes information **between** tokens. The FFN processes
each token **individually** to transform that mixed information. Research
suggests FFN layers act as **knowledge storage** — factual associations
("Paris is the capital of France") are encoded in these weights.

The FFN is usually 4× wider than the model dimension (e.g., 4096 → 16384
→ 4096), creating a bottleneck that forces compression of information.

---

## 7. Residual Connections

**What:** Add the input of each sub-layer back to its output.

```
x = x + Attention(x)    # not just Attention(x)
x = x + FFN(x)          # not just FFN(x)
```

**Why:** Deep networks (96+ layers) suffer from the **vanishing gradient
problem** — gradients shrink to near-zero as they propagate backward
through many layers, making early layers nearly impossible to train. The
skip connection creates a direct path for gradients to flow backward,
keeping training stable. Without residual connections, models deeper than
~10 layers barely learn.

---

## 8. Layer Normalization (LayerNorm)

**What:** Normalize each token's vector to mean=0 and std=1.

```
LayerNorm(x) = (x − mean(x)) / std(x) × γ + β
```

γ and β are learnable parameters that let the model undo the normalization
if needed.

**Why:** Without normalization, values can drift to extreme ranges across
layers (very large or very small), making training unstable. LayerNorm
keeps the numbers in a well-behaved range at every layer.

**Pre-norm vs post-norm:** Modern models (GPT-2+) normalize **before**
attention/FFN (pre-norm), which trains more stably than the original
Transformer's post-norm.

---

## 9. The Transformer Block (putting it together)

One Transformer block:

```
Input x
  │
  ├─────────────────┐
  ↓                 │
  LayerNorm         │
  ↓                 │
  Multi-Head        │
  Attention         │
  ↓                 │
  + ←───────────────┘  (residual connection)
  │
  ├─────────────────┐
  ↓                 │
  LayerNorm         │
  ↓                 │
  Feed-Forward      │
  ↓                 │
  + ←───────────────┘  (residual connection)
  │
Output x
```

A full model stacks N of these blocks:
- GPT-2: 12 blocks
- GPT-3: 96 blocks
- Llama 3 70B: 80 blocks

Each block refines the representation. Early layers capture syntax and
local patterns. Middle layers build semantic understanding. Late layers
specialize for the prediction task.

---

## 10. The Output Layer

**What:** Convert the final hidden vector into a probability distribution
over the entire vocabulary.

```
hidden vector (4096 floats)
  ↓
Linear layer (4096 → 50,000)    → logits (raw scores for each token)
  ↓
Softmax                         → probabilities (sum to 1.0)
  ↓
"mat": 0.23, "rug": 0.11, "floor": 0.08, ...
```

**Why:** The model needs to produce a concrete prediction. The linear layer
projects from the model's internal space to vocabulary space. Softmax
converts raw scores to probabilities.

---

## 11. Sampling (choosing the next token)

**What:** Pick one token from the probability distribution.

| Method | How | Result |
|---|---|---|
| Greedy | Always pick highest probability | Deterministic, repetitive |
| Temperature | Divide logits by T before softmax | T<1: sharper, T>1: more random |
| Top-k | Only consider the k most likely tokens | Cuts off unlikely choices |
| Top-p (nucleus) | Consider tokens until cumulative prob ≥ p | Adaptive cutoff |

```
Temperature effect on "The cat sat on the ___":

T=0.1:  mat(0.95) rug(0.03) floor(0.02)   → almost always "mat"
T=1.0:  mat(0.23) rug(0.11) floor(0.08)   → varied but sensible
T=2.0:  mat(0.08) rug(0.06) floor(0.05)   → nearly random
```

---

## 12. Training — How the model learns

### Loss Function: Cross-Entropy

For each position, the model predicts a probability distribution over the
vocabulary. Cross-entropy loss measures how far off the prediction is from
the actual next token:

```
Loss = −log(probability assigned to the correct token)

If model gives "mat" 80% probability and "mat" is correct:
  Loss = −log(0.80) = 0.22  (low — good prediction)

If model gives "mat" 1% probability:
  Loss = −log(0.01) = 4.6   (high — bad prediction)
```

### Backpropagation

Walk backward through every operation, computing how much each weight
contributed to the error:

```
Loss
  ↓  how much did the output layer contribute?
Output weights
  ↓  how much did block 96 contribute?
Block 96
  ↓  ...
Block 1
  ↓  how much did each embedding contribute?
Embeddings
```

This produces a **gradient** for every single weight in the model (billions
of them).

### Gradient Descent (Adam optimizer)

Update each weight to reduce the loss:

```
w_new = w_old − learning_rate × gradient
```

Adam improves on basic gradient descent by tracking momentum (moving
average of past gradients) and adapting the learning rate per-weight. It's
the standard optimizer for Transformers.

### Scale of training

| Model | Parameters | Training tokens | GPU hours |
|---|---|---|---|
| GPT-2 | 1.5B | 40B | ~thousands |
| GPT-3 | 175B | 300B | ~tens of thousands |
| Llama 3 70B | 70B | 15T | ~millions |

---

## 13. The complete pipeline

```
"The cat sat on the"

  ↓ Tokenize
[The] [cat] [sat] [on] [the]

  ↓ Embed + Positional Encoding
[v₁]  [v₂]  [v₃]  [v₄]  [v₅]     (each is a 4096-dim vector)

  ↓ Transformer Block 1
    Self-Attention: each token gathers context from earlier tokens
    FFN: transforms each token's representation
    Residual + LayerNorm: keeps training stable

  ↓ Transformer Block 2
    ... deeper patterns ...

  ↓ ... × N blocks ...

  ↓ Transformer Block N
    Final refined representations

  ↓ Output Linear Layer
    [v₅'] → logits for 50,000 tokens

  ↓ Softmax
    probability distribution

  ↓ Sample
    "mat"

→ "The cat sat on the mat"
→ append "mat", repeat from the top to generate next token
```

---

## Is this efficient? (Back to your question)

These operations — matrix multiplication, softmax, attention — are
**not** imitating a brain. They're linear algebra optimized for GPUs.

But your instinct about inefficiency has merit:

| Concern | Reality |
|---|---|
| Floating-point waste | Quantization (int4/int8) cuts compute 4-8× with ~1% quality loss |
| Every token attends to all prior tokens | O(n²) cost — active research on sparse/linear attention |
| All parameters active for every token | Mixture of Experts (MoE) activates only ~10% per token |
| Billions of multiplications per token | Hardware (GPUs/TPUs) is specifically designed for this |
| Brain does it in 20 watts | Yes — neuromorphic chips try to close this gap |

The honest answer: we found math that works (gradient descent + attention),
and we're making it less wasteful iteratively. Nobody has found a
fundamentally better paradigm yet — but your instinct that one might exist
is shared by serious researchers.
