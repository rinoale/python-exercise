# How Transformer Attention Actually Computes — Step by Step

A concrete walkthrough of self-attention with real numbers.
No external data, no magic — just matrix multiplication.

Example sentence: **"The hungry cat sat quietly on the warm mat"**

## The core question

How does a Transformer figure out that "sat" is related to "cat" and
not to "The" or "on"? The answer: **learned weight matrices** turn
each token into a Query, Key, and Value, then dot products measure
similarity.

**At inference time** (when you use the model), everything comes from:
1. **Your input tokens** (embeddings)
2. **The weight matrices** (W_Q, W_K, W_V) stored in the model file

No lookup table, no database, no training data accessed at runtime.

But those weight matrices **are** the training data — compressed into
numbers. Trillions of sentences were fed through gradient descent to
shape W_Q, W_K, W_V into values that produce meaningful attention
patterns. The training data isn't "alongside" the model at runtime —
it's **inside** it, baked into billions of numbers permanently.

```
Training (already finished, months ago):
  Trillions of sentences → gradient descent → W_Q, W_K, W_V
  The knowledge is baked into the weights.

Inference (when you use the model):
  Your sentence + frozen weights → attention → output
  No training data is consulted. It's already inside the weights.
```

Think of a chef: culinary school (training) took years of tasting
thousands of dishes. But when cooking your order (inference), the chef
doesn't re-read cookbooks — everything is intuition shaped by that
training.

---

## Step 1: Each token starts as an embedding vector

Each token maps to a vector of numbers. Real models use 4096 dimensions;
we use 6 here so the math is readable.

```
Token 0: "The"      → [ 0.1,  0.3,  0.0,  0.8,  0.2,  0.1]
Token 1: "hungry"   → [ 0.7,  0.2,  0.9,  0.1,  0.5,  0.3]
Token 2: "cat"      → [ 0.8,  0.1,  0.6,  0.3,  0.9,  0.2]
Token 3: "sat"      → [ 0.2,  0.8,  0.3,  0.7,  0.1,  0.6]
Token 4: "quietly"  → [ 0.3,  0.6,  0.5,  0.4,  0.2,  0.7]
Token 5: "on"       → [ 0.1,  0.4,  0.1,  0.9,  0.1,  0.2]
Token 6: "the"      → [ 0.1,  0.3,  0.0,  0.8,  0.2,  0.1]
Token 7: "warm"     → [ 0.6,  0.3,  0.8,  0.2,  0.4,  0.5]
Token 8: "mat"      → [ 0.7,  0.2,  0.5,  0.4,  0.8,  0.3]
```

These embeddings are learned during training. Notice:
- "The" and "the" are identical (same word, different position)
- "cat" and "mat" are similar (both nouns, similar roles)
- "hungry" and "warm" are similar (both adjectives)

Positional encoding (added to each vector) would distinguish "The" from
"the" by position, but we skip that here to focus on attention.

---

## Step 2: Generate Q, K, V with learned weight matrices

The model has three weight matrices — `W_Q`, `W_K`, `W_V` — learned
during training. Each is 6×3 (projecting 6-dim embeddings to 3-dim
Q/K/V space). Every token gets multiplied by all three:

```
Q = embedding @ W_Q    ("what am I looking for?")
K = embedding @ W_K    ("what do I contain?")
V = embedding @ W_V    ("what information do I share?")
```

The W matrices (learned — these specific values encode what the model
learned about language relationships during training):

```
W_Q:                W_K:                W_V:
[ 0.3  0.1  0.5]   [ 0.4  0.2  0.1]   [ 0.2  0.5  0.3]
[ 0.1  0.6  0.2]   [ 0.1  0.5  0.3]   [ 0.4  0.1  0.6]
[ 0.5  0.2  0.4]   [ 0.3  0.1  0.6]   [ 0.1  0.3  0.2]
[ 0.2  0.4  0.1]   [ 0.6  0.3  0.2]   [ 0.3  0.2  0.4]
[ 0.4  0.3  0.6]   [ 0.2  0.4  0.5]   [ 0.5  0.4  0.1]
[ 0.1  0.5  0.3]   [ 0.5  0.1  0.4]   [ 0.1  0.6  0.5]
```

Computing Q, K, V for each token (embedding @ W_matrix):

```
          Q                    K                    V
"The"   [0.24, 0.55, 0.20]   [0.55, 0.41, 0.25]   [0.32, 0.27, 0.44]
"hungry"[0.74, 0.47, 0.82]   [0.66, 0.38, 0.85]   [0.55, 0.72, 0.46]
"cat"   [0.69, 0.49, 0.94]   [0.64, 0.47, 0.76]   [0.67, 0.79, 0.42]
"sat"   [0.35, 0.79, 0.43]   [0.64, 0.55, 0.42]   [0.46, 0.48, 0.70]
"quiet" [0.38, 0.68, 0.59]   [0.54, 0.42, 0.59]   [0.35, 0.59, 0.63]
"on"    [0.23, 0.54, 0.17]   [0.57, 0.39, 0.25]   [0.32, 0.25, 0.42]
"the"   [0.24, 0.55, 0.20]   [0.55, 0.41, 0.25]   [0.32, 0.27, 0.44]
"warm"  [0.63, 0.51, 0.73]   [0.62, 0.35, 0.78]   [0.47, 0.68, 0.44]
"mat"   [0.64, 0.48, 0.83]   [0.60, 0.44, 0.70]   [0.59, 0.73, 0.40]
```

**Key point:** These W matrices are the same for every token in every
sentence. They encode the model's learned understanding of what
constitutes a good query and key. Training adjusts these weights over
billions of examples.

---

## Step 3: Compute attention scores (Q @ K^T)

Each token's Query asks: "how relevant is every other token to me?"
It checks by dot-producting its Q with every token's K.

Let's focus on **"sat"** (token 3). With the causal mask, "sat" can
only see tokens 0-3 (itself and past tokens):

```
Q_sat = [0.35, 0.79, 0.43]

sat→"The":     0.35×0.55 + 0.79×0.41 + 0.43×0.25 = 0.62
sat→"hungry":  0.35×0.66 + 0.79×0.38 + 0.43×0.85 = 0.89
sat→"cat":     0.35×0.64 + 0.79×0.47 + 0.43×0.76 = 0.92
sat→"sat":     0.35×0.64 + 0.79×0.55 + 0.43×0.42 = 0.84
sat→"quietly": masked (future token, −∞)
sat→"on":      masked
sat→"the":     masked
sat→"warm":    masked
sat→"mat":     masked
```

Already visible: **"cat" scores highest (0.92)**, "hungry" is next
(0.89), "The" scores lowest (0.62). The Q/K dot product naturally
produces higher scores between "sat" and content words.

Now let's also compute for **"mat"** (token 8, can see everything):

```
Q_mat = [0.64, 0.48, 0.83]

mat→"The":     0.64×0.55 + 0.48×0.41 + 0.83×0.25 = 0.76
mat→"hungry":  0.64×0.66 + 0.48×0.38 + 0.83×0.85 = 1.31
mat→"cat":     0.64×0.64 + 0.48×0.47 + 0.83×0.76 = 1.27
mat→"sat":     0.64×0.64 + 0.48×0.55 + 0.83×0.42 = 1.02
mat→"quietly": 0.64×0.54 + 0.48×0.42 + 0.83×0.59 = 1.04
mat→"on":      0.64×0.57 + 0.48×0.39 + 0.83×0.25 = 0.76
mat→"the":     0.64×0.55 + 0.48×0.41 + 0.83×0.25 = 0.76
mat→"warm":    0.64×0.62 + 0.48×0.35 + 0.83×0.78 = 1.21
mat→"mat":     0.64×0.60 + 0.48×0.44 + 0.83×0.70 = 1.18
```

"mat" attends most strongly to **"hungry" (1.31)**, **"cat" (1.27)**,
and **"warm" (1.21)** — content words with strong semantic roles. It
attends weakly to **"The"/"on"/"the" (0.76)** — function words.

---

## Step 4: Scale and softmax

### Scale

Divide by √d_k (d_k = 3 in our example, √3 ≈ 1.73):

```
"sat" scores (scaled):
  The: 0.62/1.73 = 0.36     hungry: 0.89/1.73 = 0.51
  cat: 0.92/1.73 = 0.53     sat:    0.84/1.73 = 0.49

"mat" scores (scaled):
  The: 0.44    hungry: 0.76    cat: 0.73    sat: 0.59
  quietly: 0.60  on: 0.44    the: 0.44    warm: 0.70    mat: 0.68
```

### Softmax

Convert to probabilities (sum to 1):

```
"sat" attention weights:
  ┌──────────────────────────────────────────────────┐
  │  The: 0.19   hungry: 0.23   cat: 0.37   sat: 0.21  │
  └──────────────────────────────────────────────────┘
                                  ↑ highest

"mat" attention weights:
  ┌──────────────────────────────────────────────────────────────────────────────┐
  │  The: 0.08  hungry: 0.15  cat: 0.14  sat: 0.10  quietly: 0.10             │
  │  on: 0.08   the: 0.08    warm: 0.14  mat: 0.13                            │
  └──────────────────────────────────────────────────────────────────────────────┘
                  ↑ highest     ↑ high                  ↑ high
```

Now the pattern is clear:

**"sat" pays 37% attention to "cat"** (its subject) — more than any
other token. It learned that verbs should attend to their subjects.

**"mat" pays most attention to "hungry" (15%), "cat" (14%), "warm" (14%)**
— the content words that define the scene. Function words like "The",
"on", "the" get only 8% each.

In a real model with 128 dimensions instead of 3, these differences would
be even sharper — "cat" might get 60%+ of "sat"'s attention.

---

## Step 5: Weighted sum of Values

The attention weights determine how much of each token's Value to absorb.

For "sat":

```
output_sat = 0.19 × V_The  +  0.23 × V_hungry  +  0.37 × V_cat  +  0.21 × V_sat

           = 0.19 × [0.32, 0.27, 0.44]      "The"    — small contribution
           + 0.23 × [0.55, 0.72, 0.46]      "hungry" — moderate
           + 0.37 × [0.67, 0.79, 0.42]      "cat"    — dominant
           + 0.21 × [0.46, 0.48, 0.70]      "sat"    — moderate

           = [0.06, 0.05, 0.08]
           + [0.13, 0.17, 0.11]
           + [0.25, 0.29, 0.16]              ← "cat" contributes most
           + [0.10, 0.10, 0.15]

           = [0.54, 0.61, 0.49]
```

This output is no longer just "sat." It's **"sat" enriched with "cat"
and "hungry"** — the model now understands this is about a hungry cat
sitting, not just any sitting.

For "mat":

```
output_mat = 0.08 × V_The  +  0.15 × V_hungry  +  0.14 × V_cat
           + 0.10 × V_sat  +  0.10 × V_quietly +  0.08 × V_on
           + 0.08 × V_the  +  0.14 × V_warm    +  0.13 × V_mat

           = [0.49, 0.54, 0.46]
```

"mat" now carries information from the whole sentence — it knows it's
a warm mat where a hungry cat sat quietly.

---

## The full picture

```
"The hungry cat sat quietly on the warm mat"
  │     │    │    │     │     │   │    │    │
  ↓     ↓    ↓    ↓     ↓     ↓   ↓    ↓    ↓
[emb] [emb] [emb] [emb] [emb] [emb] [emb] [emb] [emb]
  │     │    │    │     │     │   │    │    │
  ├─────┼────┼────┼─────┼─────┼───┼────┼────┤
  ↓     ↓    ↓    ↓     ↓     ↓   ↓    ↓    ↓
 ×W_Q  ×W_Q ×W_Q ×W_Q  ×W_Q  ×W_Q ×W_Q ×W_Q ×W_Q  → Queries
 ×W_K  ×W_K ×W_K ×W_K  ×W_K  ×W_K ×W_K ×W_K ×W_K  → Keys
 ×W_V  ×W_V ×W_V ×W_V  ×W_V  ×W_V ×W_V ×W_V ×W_V  → Values
  │     │    │    │     │     │   │    │    │
  └─────┴────┴────┴─────┴─────┴───┴────┴────┘
                         ↓
              Q @ Kᵀ / √d_k  (all pairs)
                         ↓
                  causal mask (hide future)
                         ↓
                      softmax (→ probabilities)
                         ↓
                   weights @ V (blend values)
                         ↓
              context-aware output per token
```

---

## The attention matrix (all tokens at once)

The model computes attention for ALL tokens simultaneously as a matrix.
With the causal mask applied, it looks like this:

```
                    The  hungry  cat   sat  quiet   on   the  warm  mat
                   ─────────────────────────────────────────────────────
  "The"       sees [ ██    ·      ·     ·     ·     ·     ·     ·    · ]
  "hungry"    sees [ ▓▓   ██      ·     ·     ·     ·     ·     ·    · ]
  "cat"       sees [ ░░   ▓▓    ██     ·     ·     ·     ·     ·    · ]
  "sat"       sees [ ░░   ▓▓    ██    ▓▓     ·     ·     ·     ·    · ]
  "quietly"   sees [ ░░   ▓▓    ██    ▓▓    ▓▓     ·     ·     ·    · ]
  "on"        sees [ ░░   ░░    ▓▓    ▓▓    ░░    ██     ·     ·    · ]
  "the"       sees [ ░░   ░░    ▓▓    ▓▓    ░░    ▓▓    ██     ·    · ]
  "warm"      sees [ ░░   ▓▓    ▓▓    ░░    ░░    ░░    ░░    ██    · ]
  "mat"       sees [ ░░   ██    ▓▓    ░░    ░░    ░░    ░░    ▓▓   ▓▓ ]

  ██ = strong attention (>0.20)
  ▓▓ = moderate attention (0.12-0.20)
  ░░ = weak attention (<0.12)
  ·  = masked (future token, cannot see)
```

Patterns that emerge:
- **"sat" strongly attends to "cat"** — verb finds its subject
- **"mat" strongly attends to "hungry" and "warm"** — noun gathers its modifiers
- **"The"/"on"/"the" get weak attention** — function words carry less content
- **Each row sums to 1.0** — it's a probability distribution

---

## Why Q, K, V instead of just comparing embeddings directly?

You could compute similarity directly between embeddings:

```
similarity("sat", "cat") = embedding_sat · embedding_cat
```

But this gives each token only ONE fixed meaning. The word "bank" would
have the same vector whether it means "river bank" or "bank account."

Q, K, V add a layer of indirection:
- The **Query** asks a specific question ("I'm a verb — who is my subject?")
- The **Key** advertises a specific role ("I'm a noun — I could be a subject")
- The **Value** carries the actual content to blend

The W matrices learn these roles during training. Different attention heads
learn different types of questions:

```
Head 1 (syntax):    "sat" Q asks "who is my subject?"     → attends to "cat"
Head 2 (modifier):  "cat" Q asks "what describes me?"     → attends to "hungry"
Head 3 (location):  "sat" Q asks "where did this happen?" → attends to "on", "mat"
Head 4 (adjective): "mat" Q asks "what describes me?"     → attends to "warm"
```

Without the Q/K/V transformation, you'd only have one way to compare
tokens. With it, you have as many ways as you have attention heads.

---

## Where the "intelligence" lives

When you download a model file (e.g., Llama 3.1 8B = ~16GB), what's
inside is mostly these matrices:

```
Per Transformer block (×32 blocks in a 7B model):
  W_Q:  4096 × 4096  =  16.7M parameters
  W_K:  4096 × 4096  =  16.7M parameters
  W_V:  4096 × 4096  =  16.7M parameters
  W_O:  4096 × 4096  =  16.7M parameters   (output projection)
  W_1:  4096 × 11008 =  45.1M parameters   (FFN up)
  W_2:  11008 × 4096 =  45.1M parameters   (FFN down)
  ─────────────────────────────────────
  Total per block:      ~190M parameters
  × 32 blocks:          ~6.1B parameters

Plus embeddings, layer norms, output layer → ~7B total
```

Every single one of these numbers was adjusted during training by
gradient descent to minimize prediction error on trillions of tokens.

The input tokens are just the trigger. The intelligence is in the weights.

---

## Causal mask — why "sat" can't see future tokens

In generative models, each token can only attend to tokens before it
(and itself). This is enforced by setting future positions to −∞ before
softmax, which makes them 0 after softmax.

```
"The":     can see [The]
"hungry":  can see [The, hungry]
"cat":     can see [The, hungry, cat]
"sat":     can see [The, hungry, cat, sat]
"quietly": can see [The, hungry, cat, sat, quietly]
"on":      can see [The, hungry, cat, sat, quietly, on]
"the":     can see [The, hungry, cat, sat, quietly, on, the]
"warm":    can see [The, hungry, cat, sat, quietly, on, the, warm]
"mat":     can see [The, hungry, cat, sat, quietly, on, the, warm, mat]
```

This is how the model generates text left-to-right — when predicting the
word after "mat", it can use context from the entire sentence. But when
it was generating "sat", it only knew about "The hungry cat" so far.

---

## How training teaches the W matrices what to attend to

Before training, W_Q, W_K, W_V are random numbers. Attention patterns
are meaningless.

```
Untrained model sees "The hungry cat sat quietly on the warm ___":

  "warm" attends to:
    The(0.11) hungry(0.11) cat(0.11) sat(0.12) quietly(0.11)
    on(0.11) the(0.11) warm(0.11) ← nearly uniform, random

  → predicts "banana" with 0.2% confidence
  → correct answer was "mat"
  → loss = −log(0.002) = 6.2  (very high)

Backpropagation adjusts W_Q, W_K, W_V:
  - "warm"'s Query becomes better at asking "what noun do I modify?"
  - "mat"'s Key would become better at advertising "I'm a noun needing an adjective"
  - The dot product between adjective-queries and noun-keys increases

After billions of such adjustments:

  "warm" attends to:
    The(0.03) hungry(0.10) cat(0.25) sat(0.08) quietly(0.05)
    on(0.04) the(0.03) warm(0.42) ← strong self-attention + noun attention

  → predicts "mat" with 18% confidence (top choice)
  → loss = −log(0.18) = 1.7  (much lower)
```

Nobody programs "adjectives should attend to nouns." The model discovers
this pattern because it helps predict the next token. The W matrices
encode these patterns as numbers — that's what "learning" means.

---

## What dot product actually measures

Dot product multiplies matching pairs and sums them:

```
[1, 2, 3] · [4, 5, 6] = 1×4 + 2×5 + 3×6 = 4 + 10 + 18 = 32
```

It measures **similarity of direction** — do two vectors agree on which
dimensions matter?

```
             Action  Romance  Horror
Alice:       [  5,      1,      4  ]    ← loves action & horror
Bob:         [  4,      2,      5  ]    ← loves action & horror
Carol:       [  1,      5,      1  ]    ← loves romance

Alice · Bob   = 5×4 + 1×2 + 4×5 = 42    ← HIGH (similar taste)
Alice · Carol = 5×1 + 1×5 + 4×1 = 14    ← LOW (different taste)
```

In attention, Q and K vectors are shaped by the W matrices so that
related tokens point in similar directions:

```
Q_sat = [high, low, high]      "I care about verb-ness and subject-ness"
K_cat = [high, low, high]      "I'm strong on verb-ness and subject-ness"
                                → dot product is HIGH (agree on what matters)

K_the = [low, high, low]       "I'm strong on article-ness"
                                → dot product is LOW (disagree on what matters)
```

Because LayerNorm keeps all vectors roughly the same total magnitude,
**similarly distributed values produce higher dot products than spiky
mismatches**:

```
Both care about the same dimensions:
[0.5, 0.5, 0.5, 0.5] · [0.5, 0.5, 0.5, 0.5] = 1.00  ← HIGH

Spiky in opposite places:
[1.7, 0.1, 0.1, 0.1] · [0.1, 0.1, 0.1, 1.7] = 0.36  ← LOW
```

The W matrices are trained so that related tokens (verb ↔ subject,
adjective ↔ noun) end up with similar distributions across dimensions,
and unrelated tokens end up spiky in different places. The dot product
then naturally scores related pairs higher.

---

## What do the dimensions mean? (The black box)

Each dimension in a 4096-dim vector could represent a concept. In
theory, "cat" would score high on animal-related dimensions:

```
Dimension:    code   math   fish   animal  furniture  food
"cat"       [ 0.1,   0.1,   0.8,   1.5,    0.1,      0.3 ]
"dog"       [ 0.1,   0.1,   0.4,   1.4,    0.1,      0.2 ]  ← similar to cat
"mat"       [ 0.1,   0.1,   0.1,   0.1,    1.5,      0.1 ]  ← different from cat
"tuna"      [ 0.1,   0.1,   1.5,   0.3,    0.1,      1.2 ]  ← shares "fish" with cat
```

In reality, it's messier. Researchers have found individual dimensions
that activate for interpretable concepts — "is this a number?", "is
this French?", "is this a proper noun?" But most dimensions encode
**combinations** that humans can't name:

```
Dimension 847:  something like "noun that a verb acts upon" + "indoor object" + ???
Dimension 2391: no human interpretation found
```

The model finds whatever patterns help predict the next token. Some
align with human concepts (animal, color, verb). Many don't — they're
useful abstractions that have no word in any human language.

This is exactly why it's called a **black box**. We built it, trained
it, it works — but we can't fully explain why specific numbers have
specific values. The field trying to decode this is called
**mechanistic interpretability** — one of the hardest open problems
in AI safety.

---

## The `@` operator — matrix multiplication in Python

`@` is the matrix multiplication operator, added in Python 3.5
specifically for math/ML code. It's not regular multiplication.

```python
import numpy as np

A = np.array([[1, 2],
              [3, 4]])
B = np.array([[5, 6],
              [7, 8]])

# These are the same thing:
A @ B            # [[19, 22], [43, 50]]
np.matmul(A, B)  # [[19, 22], [43, 50]]

# NOT the same as element-wise multiply:
A * B            # [[ 5, 12], [21, 32]]
```

`@` exists because the function form is painful in complex expressions:

```python
# Without @
output = np.matmul(softmax(np.matmul(Q, K.T) / np.sqrt(d_k)), V)

# With @
output = softmax(Q @ K.T / np.sqrt(d_k)) @ V
```

### What `@` actually does to a vector

Matrix multiplication **transforms** a vector — it moves it to a
different space where it means something new.

```
embedding = [0.8, 0.1, 0.6]     ← "cat" in embedding space

W_Q = [[0.3, 0.1],              ← learned transform
       [0.1, 0.6],
       [0.5, 0.2]]

embedding @ W_Q = [0.55, 0.26]  ← "cat" in query space

The math:
  0.8×0.3 + 0.1×0.1 + 0.6×0.5 = 0.55
  0.8×0.1 + 0.1×0.6 + 0.6×0.2 = 0.26
```

Each output number is a **weighted combination** of the input. The W
matrix controls the recipe — how much of each input dimension
contributes to each output dimension.

### The real purpose: changing perspective

Think of it as a **lens** that reveals different aspects of the same
thing:

```
"cat" embedding = [0.8, 0.1, 0.6]    ← general identity of "cat"

"cat" @ W_Q = [0.55, 0.26]    ← "what is cat looking for?"
"cat" @ W_K = [0.41, 0.73]    ← "what does cat offer?"
"cat" @ W_V = [0.62, 0.38]    ← "what content does cat carry?"
```

Same input, three different W matrices, three different views. The
embedding holds everything about "cat." The `@` transform **selects
and recombines** specific aspects.

It's like sunlight through a prism — the prism doesn't add anything
new. It reorganizes what's already there to reveal structure.

### Every `@` in a Transformer

| Operation | What it does |
|---|---|
| `embedding @ W_Q` | Transform to query space ("what am I looking for?") |
| `embedding @ W_K` | Transform to key space ("what do I offer?") |
| `embedding @ W_V` | Transform to value space ("what content do I carry?") |
| `Q @ K.T` | Dot product between all Q/K pairs (attention scores) |
| `weights @ V` | Blend values by attention weights |
| `x @ W_1` | FFN layer — transform to wider space (knowledge retrieval) |
| `hidden @ W_out` | Output layer — project to vocabulary size |

Every `@` is either **changing perspective** (embedding → Q/K/V),
**measuring similarity** (Q @ K^T), or **blending information**
(weights @ V). None of them "enhance" the vector — they
**reinterpret** it for a specific purpose.

---

## Softmax — turning raw scores into probabilities

### What it does

Softmax takes a list of arbitrary numbers and converts them into
**probabilities that sum to 1**:

```
softmax([2.0, 1.0, 0.1])

Step 1: e^ each value
  e^2.0 = 7.389
  e^1.0 = 2.718
  e^0.1 = 1.105

Step 2: divide each by the sum (7.389 + 2.718 + 1.105 = 11.212)
  7.389 / 11.212 = 0.66
  2.718 / 11.212 = 0.24
  1.105 / 11.212 = 0.10

Result: [0.66, 0.24, 0.10]    ← sums to 1.0
```

The formula: `softmax(xᵢ) = e^xᵢ / Σ(e^xⱼ)`

### Why not just divide by the sum directly?

You could normalize by dividing each score by the total:

```
Raw scores:    [2.0, 1.0, 0.1]
Sum = 3.1
Simple divide: [0.65, 0.32, 0.03]    ← also sums to 1
```

This breaks with negative numbers:

```
Raw scores:    [2.0, -1.0, 0.5]
Sum = 1.5
Simple divide: [1.33, -0.67, 0.33]   ← negative probability? meaningless
```

Softmax uses `e^x` which is **always positive**, so every output is
between 0 and 1 regardless of input. Negative inputs just become small
positive numbers:

```
Raw scores:    [2.0, -1.0, 0.5]
e^ values:     [7.389, 0.368, 1.649]    ← all positive
Sum = 9.406
Softmax:       [0.79, 0.04, 0.18]       ← valid probabilities
```

### The amplification effect

The key property of softmax: **it amplifies differences**. `e^x` is
exponential — small differences in input become large differences in
output:

```
Close scores:      [3.0, 2.8, 2.9]
  Softmax:         [0.37, 0.30, 0.33]    ← fairly even

Spread scores:     [5.0, 2.0, 1.0]
  Softmax:         [0.88, 0.04, 0.08]    ← winner takes almost all

Extreme scores:    [10.0, 2.0, 1.0]
  Softmax:         [0.9997, 0.0002, 0.0001]  ← practically one-hot
```

This is why softmax is used instead of simple normalization — it lets
the model be **decisive**. A slightly higher attention score becomes a
much larger attention weight.

### Where softmax appears in a Transformer

**1. Attention weights** — the main use:

```
Q_sat @ K.T = [0.62, 0.89, 0.92, 0.84]
               The   hungry  cat   sat

softmax →     [0.19, 0.23,  0.37, 0.21]
                             ↑ "cat" wins decisively
```

Without softmax, these raw scores are just numbers — they could be
any size, they don't sum to 1. Softmax turns them into a probability
distribution: "spend 37% of attention on cat, 23% on hungry, ..."

This matters because the next step is a **weighted sum of Values**.
The weights must sum to 1, or the output vector's magnitude would
grow or shrink unpredictably across layers.

**2. Final output layer** — predicting the next token:

```
logits = hidden @ W_out    ← raw scores for 50,000 tokens

logits:   [..., "mat": 4.2, "rug": 2.1, "cat": 1.8, ...]

softmax → [..., "mat": 0.23, "rug": 0.03, "cat": 0.02, ...]
                  ↑ highest probability = most likely next token
```

**3. Classification** — in any model that picks a category:

```
logits:   [dog: 3.1, cat: 5.2, bird: 1.0]
softmax → [dog: 0.11, cat: 0.85, bird: 0.04]
           → model predicts "cat" with 85% confidence
```

### Temperature — controlling softmax sharpness

Dividing logits by a temperature value before softmax changes how
decisive the output is:

```
logits = [4.0, 2.0, 1.0]

T=0.5 (confident):   softmax([8.0, 4.0, 2.0])   = [0.95, 0.04, 0.01]
T=1.0 (normal):      softmax([4.0, 2.0, 1.0])   = [0.84, 0.11, 0.04]
T=2.0 (creative):    softmax([2.0, 1.0, 0.5])   = [0.49, 0.24, 0.11]
T=10.0 (random):     softmax([0.4, 0.2, 0.1])   = [0.37, 0.34, 0.29]
```

Lower temperature → sharper → more predictable output.
Higher temperature → flatter → more random/creative output.

This is the "Temperature" slider in ChatGPT and other AI tools.
It doesn't change the model — it just scales the numbers before
softmax.

### Summary

| Property | Why it matters |
|---|---|
| Outputs are always positive | No negative probabilities |
| Outputs sum to 1 | Valid probability distribution |
| Amplifies differences | Model can be decisive |
| Differentiable | Gradients can flow through it during training |
| Temperature-controllable | Tune creativity vs consistency at inference |

Softmax is the bridge between "raw math" and "meaningful
probabilities." Without it, attention scores and predictions would
be uninterpretable numbers.
