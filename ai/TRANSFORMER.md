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

---

## The Full Transformer Architecture

Everything above covers **self-attention** — one component of a
Transformer. Here's the complete picture, from input to output.

```
"The hungry cat sat quietly on the warm ___"
                    |
                    v
          +------------------+
          |   Tokenizer      |   split text into token IDs
          +------------------+
                    |
          [464, 18823, 5431, 3800, 22920, 389, 279, 8369, ___]
                    |
                    v
          +------------------+
          |   Embedding      |   token ID -> vector (lookup table)
          |   + Positional   |   add position info to each vector
          +------------------+
                    |
          9 vectors, each 4096-dim
                    |
                    v
     +==============================+
     ‖   Transformer Block 1       ‖
     ‖                              ‖
     ‖  1. Layer Norm               ‖
     ‖  2. Multi-Head Attention     ‖
     ‖  3. Residual Connection      ‖
     ‖  4. Layer Norm               ‖
     ‖  5. Feed-Forward Network     ‖
     ‖  6. Residual Connection      ‖
     +==============================+
                    |
     +==============================+
     ‖   Transformer Block 2       ‖
     ‖        (same structure)      ‖
     +==============================+
                    |
                   ...
                    |
     +==============================+
     ‖   Transformer Block 32      ‖  (or 96 for GPT-4 class)
     +==============================+
                    |
          +------------------+
          |   Final LayerNorm |
          +------------------+
                    |
          +------------------+
          |   Output Layer   |   vector -> vocabulary probabilities
          +------------------+
                    |
                    v
          "mat" (highest probability)
```

Each block takes 9 vectors in, produces 9 vectors out. The vectors
get richer with each layer — early layers capture syntax, later layers
capture meaning and reasoning.

---

## Positional Encoding — how the model knows word order

Attention computes Q·K dot products between all token pairs. Dot
products don't care about order — "cat sat" and "sat cat" would
produce identical attention scores without positional information.

The solution: **add position information to each embedding** before
the first Transformer block.

### The problem

```
"The cat sat"    embeddings: [E_the, E_cat, E_sat]
"sat cat The"    embeddings: [E_sat, E_cat, E_the]

Without position info, attention sees the SAME set of vectors.
It can't tell these sentences apart.
```

### Original Transformer: sinusoidal encoding

The 2017 "Attention Is All You Need" paper used sine/cosine waves
at different frequencies. Each position gets a unique pattern:

```
PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

Position 0: [sin(0/1), cos(0/1), sin(0/100), cos(0/100), ...]
           = [0.000,    1.000,    0.000,      1.000,      ...]

Position 1: [sin(1/1), cos(1/1), sin(1/100), cos(1/100), ...]
           = [0.841,    0.540,    0.010,      0.999,      ...]

Position 2: [sin(2/1), cos(2/1), sin(2/100), cos(2/100), ...]
           = [0.909,   -0.416,    0.020,      0.999,      ...]
```

Why sines and cosines? Two useful properties:
1. Every position gets a unique pattern (no two positions are the same)
2. Relative distances are learnable — the model can figure out
   "position 5 is 3 steps after position 2" from the wave patterns

The positional encoding is **added** to the embedding (element-wise):

```
input_to_transformer = embedding + positional_encoding

"cat" at position 2:
  embedding:  [0.8, 0.1, 0.6, 0.3, ...]   ← what the word IS
+ position:   [0.9, -0.4, 0.02, 0.99, ...] ← WHERE the word is
= input:      [1.7, -0.3, 0.62, 1.29, ...] ← both combined
```

### Modern models: Rotary Position Embedding (RoPE)

Most current models (Llama, GPT-4, Claude) use RoPE instead. Rather
than adding a vector, RoPE **rotates** the Q and K vectors based on
position. This encodes position directly into the attention computation.

```
Sinusoidal (original):   input = embedding + PE(position)
RoPE (modern):           Q_rotated = rotate(Q, position)
                         K_rotated = rotate(K, position)
                         score = Q_rotated · K_rotated
```

The key insight: when you dot-product two rotated vectors, the result
depends on their **relative** position (how far apart they are), not
their absolute positions. This means the model naturally generalizes
to longer sequences than it was trained on.

```
Q at position 5 · K at position 3 → depends on distance 2
Q at position 100 · K at position 98 → same distance 2, same effect
```

The details of the rotation involve splitting the vector into pairs
of dimensions and applying 2D rotation matrices at different
frequencies — similar in spirit to sinusoidal encoding, but applied
multiplicatively to Q/K rather than additively to embeddings.

---

## Layer Normalization — keeping numbers stable

After each sub-layer (attention and FFN), the numbers in the vectors
can drift — some dimensions might grow to 1000, others shrink to
0.001. This makes the next layer's computations unstable.

Layer Norm fixes this by normalizing each vector to have mean 0 and
variance 1, then applying learned scale and shift parameters.

### The math

For a single vector x of dimension d:

```
Input:  x = [2.0, -1.0, 3.0, 0.5]

Step 1: Compute mean
  mean = (2.0 + (-1.0) + 3.0 + 0.5) / 4 = 1.125

Step 2: Compute variance
  variance = ((2.0-1.125)^2 + (-1.0-1.125)^2 + (3.0-1.125)^2 + (0.5-1.125)^2) / 4
           = (0.766 + 4.516 + 3.516 + 0.391) / 4
           = 2.297

Step 3: Normalize (subtract mean, divide by std dev)
  x_norm = (x - mean) / sqrt(variance + epsilon)
         = [(2.0-1.125), (-1.0-1.125), (3.0-1.125), (0.5-1.125)] / sqrt(2.297)
         = [0.875, -2.125, 1.875, -0.625] / 1.516
         = [0.577, -1.402, 1.237, -0.412]

Step 4: Scale and shift (gamma and beta are learned parameters)
  output = gamma * x_norm + beta
```

Gamma and beta are learned during training. They let the model decide
"I want this layer's output to have this specific scale and offset"
while keeping the distribution well-behaved.

### Why it matters

Without Layer Norm, deep networks suffer from:

```
Layer 1 output:   [-0.3,  0.5,  0.1, -0.2]     reasonable values
Layer 10 output:  [-12.0, 85.0, -43.0, 67.0]    growing out of control
Layer 30 output:  [NaN,   NaN,  NaN,  NaN]       exploded → training fails
```

With Layer Norm after every sub-layer:

```
Layer 1 output:   [-0.8,  1.2,  0.1, -0.5]      normalized
Layer 10 output:  [-0.6,  0.9,  1.1, -1.4]      still normalized
Layer 30 output:  [-1.1,  0.7,  0.3,  0.1]      still fine
Layer 96 output:  [-0.4,  1.3, -0.8, -0.1]      still fine after 96 layers
```

### Pre-Norm vs Post-Norm

The original Transformer applied Layer Norm **after** each sub-layer
(post-norm). Most modern models apply it **before** (pre-norm),
because it's more stable during training:

```
Post-Norm (original 2017):
  x → Attention → Add residual → LayerNorm → FFN → Add residual → LayerNorm

Pre-Norm (modern, GPT/Llama/Claude):
  x → LayerNorm → Attention → Add residual → LayerNorm → FFN → Add residual
```

The difference is subtle but pre-norm makes training much more stable
for very deep models (96+ layers).

### RMSNorm — a simpler variant

Many recent models (Llama, Mistral) use RMSNorm instead of full
Layer Norm. It skips the mean-subtraction step and just divides by
the root mean square:

```
LayerNorm: subtract mean, divide by std, apply gamma and beta
RMSNorm:   divide by RMS(x), apply gamma only

RMS(x) = sqrt(mean(x^2))

Input:  [2.0, -1.0, 3.0, 0.5]
RMS   = sqrt((4.0 + 1.0 + 9.0 + 0.25) / 4) = sqrt(3.5625) = 1.888
Output = [2.0/1.888, -1.0/1.888, 3.0/1.888, 0.5/1.888] * gamma
       = [1.059, -0.530, 1.589, 0.265] * gamma
```

Simpler, faster, and works just as well in practice. One fewer
subtraction and one fewer learned parameter (no beta).

---

## Residual Connections — the skip highway

A residual connection (also called "skip connection") adds the
input of a sub-layer directly to its output:

```
output = sub_layer(x) + x
```

This simple addition is critical. Without it, deep Transformers
can't be trained at all.

### The problem it solves

In a 96-layer network, the gradient signal must travel all the way
from the output back to layer 1 during training. Each layer
transforms the signal, and if those transformations are imperfect
(especially early in training when weights are random), the gradient
shrinks exponentially:

```
Without residual connections:
  Layer 96 gradient: 1.0
  Layer 80 gradient: 0.1       (shrinking through each layer)
  Layer 50 gradient: 0.0001
  Layer 20 gradient: 0.0000001
  Layer 1 gradient:  ~0        ← vanishing gradient, can't learn

With residual connections:
  Layer 96 gradient: 1.0
  Layer 80 gradient: 0.95      (gradient has a direct highway)
  Layer 50 gradient: 0.85
  Layer 20 gradient: 0.72
  Layer 1 gradient:  0.60      ← still strong, keeps learning
```

### How it works visually

```
         x (input)
         |  \
         |   \  (skip connection — just copy the input)
         |    \
         v     |
  +------------+
  | Attention  |
  +------------+
         |     |
         v     |
       output  |
         |     |
         v     v
        [  +  ]     ← element-wise addition
           |
           v
     x + Attention(x)     ← the residual output
```

### Why addition?

The key insight: the sub-layer only needs to learn the **difference**
(residual) between what it received and what the next layer needs.

```
Without skip: layer must learn the entire transformation
  f(x) = completely new representation

With skip: layer only learns what to ADD
  f(x) + x = original + small adjustment

It's much easier to learn "add a small correction" than
"rebuild everything from scratch."
```

This is like editing a document vs writing from scratch — it's
easier to make corrections to an existing draft than to rewrite
the entire thing.

### Where they appear in a Transformer block

Each block has **two** residual connections:

```
x ──────────────────────┐
|                       |
LayerNorm               |
|                       |
Multi-Head Attention    |
|                       |
+ ←─────────────────────┘  residual #1: x + Attention(LayerNorm(x))
|
y ──────────────────────┐
|                       |
LayerNorm               |
|                       |
Feed-Forward Network    |
|                       |
+ ←─────────────────────┘  residual #2: y + FFN(LayerNorm(y))
|
output
```

Without these skip connections, information from early layers
would be completely overwritten by later processing. With them,
every layer can access the original signal plus all accumulated
modifications.

---

## Multi-Head Attention — asking multiple questions at once

The attention we computed earlier was **single-head** — one Q, one K,
one V per token. But a single head can only ask one type of question.

"sat" might need to find:
- Its subject ("cat")
- Its modifier ("quietly")
- Its location ("on the mat")

One attention head can't do all three simultaneously. Multi-head
attention runs **multiple attention heads in parallel**, each with
its own W_Q, W_K, W_V matrices.

### How it works

Instead of one large Q/K/V projection, split into h smaller ones:

```
Single-head (what we computed):
  W_Q: 4096 × 4096    one big transformation
  Q = embedding @ W_Q  → one 4096-dim query

Multi-head with h=32 heads:
  W_Q_1: 4096 × 128    head 1's query weights
  W_Q_2: 4096 × 128    head 2's query weights
  ...
  W_Q_32: 4096 × 128   head 32's query weights

  Each head produces a 128-dim Q, K, V (4096 / 32 = 128)
```

Each head runs the full attention mechanism independently:

```
Head 1 (syntax):
  Q_sat asks "who is my subject?"
  → attends to "cat" (37%), "sat" (28%), "hungry" (20%), "The" (15%)

Head 2 (modifier):
  Q_sat asks "what describes how I happened?"
  → attends to "quietly" (45%), "sat" (20%), "hungry" (18%), ...

Head 3 (location):
  Q_sat asks "where did this happen?"
  → attends to "on" (30%), "mat" (25%), "the" (20%), ...

Head 4 (adjective relay):
  Q_sat asks "what adjectives are nearby?"
  → attends to "hungry" (35%), "warm" (30%), "quietly" (20%), ...

...32 heads total, each asking a different learned question
```

### Concatenate and project

After all heads compute their outputs, concatenate them and project
back to the original dimension:

```
head_1 output: 128-dim vector (sat enriched with subject info)
head_2 output: 128-dim vector (sat enriched with modifier info)
head_3 output: 128-dim vector (sat enriched with location info)
...
head_32 output: 128-dim vector

Concatenate: [head_1 | head_2 | ... | head_32] = 4096-dim vector

Project: concatenated @ W_O = 4096-dim final output
```

W_O (output projection) is another learned matrix (4096 × 4096).
It combines the information from all heads into a single
representation.

### The full multi-head attention equation

```
MultiHead(Q, K, V) = Concat(head_1, ..., head_h) @ W_O

where head_i = Attention(Q @ W_Q_i, K @ W_K_i, V @ W_V_i)

and Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k)) @ V
```

### What different heads learn

Researchers have studied what individual heads learn. Some
consistent patterns found across models:

```
"Previous token" heads:   always attend to the token right before
"Induction" heads:        copy patterns (if "AB...A" seen, attend to B)
"Syntax" heads:           verb→subject, noun→adjective, etc.
"Positional" heads:       attend to specific relative positions
"Rare token" heads:       attend to unusual/important words
"Duplicate" heads:        find repeated tokens in the sequence
```

No one programs these behaviors. They emerge from training because
they help predict the next token. Each head discovers a useful
"question" to ask.

### Computation cost

Multi-head attention doesn't cost more than single-head — it's the
same total amount of computation, just divided differently:

```
Single-head:   W_Q is 4096×4096, one attention on 4096-dim space
Multi-head:    32 W_Q_i each 4096×128, 32 attentions on 128-dim space

Total parameters: 4096 × 4096 = 16.7M either way
Total computation: roughly the same
But multi-head captures 32 different relationship types
```

The free lunch: same cost, much richer representation.

---

## Feed-Forward Network — where knowledge lives

After attention blends information between tokens, each token
passes through a **feed-forward network (FFN)** independently.
This is where the model applies its learned knowledge.

### Structure

The FFN is a simple 2-layer neural network applied to each token
separately:

```
FFN(x) = activation(x @ W_1 + b_1) @ W_2 + b_2

W_1: 4096 × 11008    (expand to wider space)
W_2: 11008 × 4096    (compress back)
```

The expansion factor is typically ~2.7x (4096 → 11008 for Llama).
This wider intermediate space is where knowledge is stored.

### Step by step

```
Input:  x = [0.5, -0.3, 0.8, ...]    (4096-dim, output from attention)

Step 1: Expand to wider space
  hidden = x @ W_1 + b_1              (4096 → 11008)
  hidden = [2.1, -0.5, 0.0, 3.4, -1.2, ...]   (11008-dim)

Step 2: Activation function (non-linearity)
  hidden = activation(hidden)
  hidden = [2.1, 0.0, 0.0, 3.4, 0.0, ...]     (negative values → 0)

Step 3: Compress back
  output = hidden @ W_2 + b_2         (11008 → 4096)
  output = [0.7, -0.1, 1.2, ...]      (4096-dim)
```

### Why expand and compress?

The expansion creates a high-dimensional space where the model can
represent complex patterns. Think of it as "unpacking" the
compressed representation into a space where simple operations
(multiply, add, threshold) can implement complex logic.

```
4096-dim:   too compressed to represent "Paris is capital of France"
            as a simple pattern

11008-dim:  enough room for individual neurons to represent
            specific facts:
              neuron 4821: activates for "European capital" context
              neuron 7293: activates for "French language" context
              neuron 1054: activates for "geographical entity" context
```

When the right combination of neurons fires, the output encodes
the answer.

### Activation functions

The activation function introduces **non-linearity** — without it,
stacking layers would just be multiplying matrices, which collapses
to a single matrix (no depth benefit).

**ReLU** (original, simple):
```
ReLU(x) = max(0, x)

Input:  [ 2.1, -0.5,  0.0,  3.4, -1.2]
Output: [ 2.1,  0.0,  0.0,  3.4,  0.0]

Negative → 0, positive → unchanged
Simple but "dead neurons" problem: once a neuron outputs 0,
it may never recover during training
```

**GELU** (used in GPT, BERT):
```
GELU(x) = x * P(X <= x)   where X ~ Normal(0,1)

Approximation: GELU(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))

Input:  [ 2.1, -0.5,  0.0,  3.4, -1.2]
Output: [ 2.0, -0.15, 0.0,  3.4, -0.17]

Like ReLU but smoother — small negative values get small negative
outputs instead of hard zero. "Softer" gating.
```

**SwiGLU** (used in Llama, modern models):
```
SwiGLU(x) = (x @ W_gate * sigmoid(x @ W_gate)) * (x @ W_up)

This uses a gating mechanism — one path decides "how much to let
through" (the gate) while another path provides the content.
Two matrices instead of one, but better performance in practice.

Most modern LLMs use SwiGLU because it consistently outperforms
GELU and ReLU at the same parameter count.
```

### Attention vs FFN — who does what?

This is a fundamental division of labor:

```
Attention:  "Who should I listen to?"
            Moves information BETWEEN tokens.
            Captures relationships and context.
            "sat" learns it relates to "cat"

FFN:        "What do I know about this?"
            Processes each token INDEPENDENTLY.
            Applies stored knowledge.
            "cat sat on" → probably "the" or "a" next (grammar)
            "capital of France" → "Paris" (factual knowledge)
```

Research has shown:

```
Attention layers:   encode syntax, coreference, relationships
                    "which words relate to which"

FFN layers:         encode facts, patterns, rules
                    "what typically follows this context"
                    ~2/3 of total parameters live in FFN
```

When you fine-tune a model on new facts, it's primarily the FFN
weights that change. When you fine-tune for a new language's
grammar, it's primarily the attention weights.

### Key-Value memory interpretation

There's an elegant way to think about FFN layers. Each row of W_1
is a "key" (pattern to match) and the corresponding column of W_2
is a "value" (information to add):

```
W_1 row 4821:  pattern that matches "European capital city" contexts
W_2 col 4821:  vector that encodes "Paris, London, Berlin" type info

If input matches pattern 4821 → activation is high →
that row's knowledge gets added to the output
```

This makes FFN layers function like a **learned lookup table** with
11,008 entries per layer, 32+ layers deep. The total "memory
capacity" of a 7B model is roughly:

```
32 layers × 11,008 neurons × 2 (key+value) ≈ 700K "memory slots"
Each slot stores a fuzzy pattern→response mapping
```

This is why larger models know more facts — they have more FFN
neurons to store knowledge in.

---

## The Output Layer — predicting the next token

After all Transformer blocks have processed the sequence, the final
step converts the last token's vector into a probability distribution
over the entire vocabulary.

### Step by step

```
Input: final hidden state for last token
       h = [0.7, -0.1, 1.2, ...]     (4096-dim)

Step 1: Final Layer Norm
       h_norm = LayerNorm(h)           (4096-dim, normalized)

Step 2: Project to vocabulary size
       logits = h_norm @ W_out         (4096 → 50,257 for GPT-2)
                                       (4096 → 128,256 for Llama 3)

       W_out is 4096 × vocab_size — the largest single matrix
       in the model

       logits = [2.1, -0.5, 0.3, ..., 4.8, ..., 1.2]
                 "a"   "b"   "c"       "mat"      "zoo"
                 (50,257 raw scores)

Step 3: Softmax (with temperature)
       probs = softmax(logits / temperature)

       probs = [0.001, 0.000, 0.000, ..., 0.23, ..., 0.000]
                "a"                        "mat"
                (50,257 probabilities summing to 1.0)

Step 4: Sample or pick the highest
       Greedy (temperature → 0): always pick highest → "mat"
       Sampling (temperature > 0): random draw weighted by probs
```

### Weight tying

Many models **share** the embedding matrix and the output matrix.
The embedding table maps token ID → vector. The output layer maps
vector → token scores. These are inverse operations, so using the
same matrix (transposed) for both saves parameters and often works
better:

```
Embedding:    token_id → vector      (vocab_size × 4096)
Output:       vector → token_scores  (4096 × vocab_size)

With weight tying:
  W_out = W_embedding^T

This halves the embedding-related parameters.
A 128K vocabulary at 4096 dims = 524M parameters saved.
```

### Only the last token matters (for generation)

During text generation, only the **last token's** output is used
to predict the next word. But all tokens must be processed because
attention allows information to flow from early tokens to later ones:

```
"The hungry cat sat quietly on the warm ___"
  |     |     |    |     |     |   |    |    |
  v     v     v    v     v     v   v    v    v
[All 9 tokens processed through all layers]
  |     |     |    |     |     |   |    |    |
  x     x     x    x     x     x   x    x    v
                                          take this
                                          last vector
                                               |
                                               v
                                         logits → "mat"
```

The other tokens' final vectors are discarded during generation
(but they were essential during attention — they provided the
context that shaped the last token's representation).

### KV Cache — why you don't recompute everything

When generating token by token, the model would normally need to
reprocess the entire sequence for each new token. The KV cache
stores the K and V vectors from previous tokens so they don't
need to be recomputed:

```
Generating "The hungry cat sat"

Token 1: "The"
  Compute Q, K, V for "The"
  Store K_The, V_The in cache
  Predict → "hungry"

Token 2: "The hungry"
  Load K_The, V_The from cache (no recomputation)
  Compute Q, K, V for "hungry"
  Store K_hungry, V_hungry in cache
  Attention uses: Q_hungry with [K_The, K_hungry]
  Predict → "cat"

Token 3: "The hungry cat"
  Load K_The, K_hungry, V_The, V_hungry from cache
  Compute Q, K, V for "cat" only
  Store K_cat, V_cat in cache
  Attention uses: Q_cat with [K_The, K_hungry, K_cat]
  Predict → "sat"
```

Without KV cache: generating N tokens requires N × N/2 computations
With KV cache: generating N tokens requires N computations

This is why the KV cache eats so much GPU memory for long contexts —
it stores K and V for every token, every layer, every head:

```
Llama 3 8B, 8192 token context:
  32 layers × 32 heads × 2 (K+V) × 8192 tokens × 128 dims × 2 bytes
  = ~4 GB just for the KV cache
```

---

## Putting it all together — one complete forward pass

Let's trace "The hungry cat sat" through the entire model (simplified
to 2 layers, 2 heads, dim=4 for readability):

```
INPUT: "The hungry cat sat ___"

=== TOKENIZER ===
["The", "hungry", "cat", "sat"] → [464, 18823, 5431, 3800]

=== EMBEDDING + POSITION ===
Token 464   → [0.1, 0.3, 0.8, 0.2] + PE(0) → [0.1, 1.3, 0.8, 0.2]
Token 18823 → [0.7, 0.2, 0.1, 0.5] + PE(1) → [1.5, 0.2, 0.1, 0.5]
Token 5431  → [0.8, 0.1, 0.3, 0.9] + PE(2) → [0.8, -0.3, 0.3, 0.9]
Token 3800  → [0.2, 0.8, 0.7, 0.1] + PE(3) → [0.2, 0.8, -0.2, 0.1]

=== TRANSFORMER BLOCK 1 ===

--- LayerNorm ---
Normalize each vector to mean=0, var=1

--- Multi-Head Attention (2 heads) ---
Head 1 (learns syntax):    "sat" attends to "cat" (38%)
Head 2 (learns modifiers): "sat" attends to "hungry" (33%)
Concatenate heads → project with W_O
Each token now carries info from related tokens

--- Residual Connection ---
output = attention_output + input   (preserve original signal)

--- LayerNorm ---
Normalize again

--- FFN ---
Expand 4 → 11 dims, apply SwiGLU, compress 11 → 4 dims
Each token independently retrieves relevant knowledge

--- Residual Connection ---
output = ffn_output + input

=== TRANSFORMER BLOCK 2 ===
(Same structure, different learned weights)
Deeper patterns: "sat" now knows it's about a hungry cat,
Block 2's attention can use this richer representation to find
even more abstract relationships.

=== OUTPUT ===
Take last token "sat"'s final vector
LayerNorm → multiply by W_out (4 → 50257) → logits
Softmax → probabilities
Top prediction: "quietly" or "on" or "down"
```

### Every named operation and its purpose

```
Operation            Parameters              Purpose
---------            ----------              -------
Embedding table      vocab_size × d_model    token ID → vector
Positional encoding  (varies by method)      encode word order
LayerNorm            2 × d_model per norm    stabilize values
W_Q, W_K, W_V        3 × d_model × d_model   create Q, K, V
W_O                  d_model × d_model       combine attention heads
W_1 (FFN up)         d_model × d_ff         expand to knowledge space
W_2 (FFN down)       d_ff × d_model         compress back
W_out (output)       d_model × vocab_size    vector → token probs
```

### Parameter count breakdown (Llama 3 8B as example)

```
Component                     Parameters      % of total
---------                     ----------      ----------
Embedding table               128K × 4096     = 524M       6.5%
Per block (×32):
  Attention (Q,K,V,O)         4 × 4096²       = 67M/block
  FFN (W_1, W_2, W_gate)      3 × 4096×11008  = 135M/block
  LayerNorms                  4 × 4096        = 16K/block
  Block total                                 = 202M/block
All blocks                    32 × 202M       = 6,464M     80.8%
Output projection             4096 × 128K     = 524M       6.5%
Other (RoPE, norms)                           ≈ 500M       6.2%
                                              ----------
Total                                         ≈ 8,000M     100%
```

The FFN layers dominate — they're the "memory" of the model.
Attention is the "thinking" — cheaper but essential for connecting
information across the sequence.

---

## The complete Transformer equation

Everything in this document, compressed into math:

```
Given input tokens x_1, x_2, ..., x_n:

1. h_0 = Embedding(x) + PositionalEncoding(x)

2. For each layer l = 1, ..., L:

   a. Attention sub-layer:
      h_l' = h_{l-1} + MultiHeadAttention(LayerNorm(h_{l-1}))

      where MultiHeadAttention = Concat(head_1, ..., head_H) @ W_O
      and   head_i = softmax(Q_i @ K_i^T / sqrt(d_k)) @ V_i
      and   Q_i = LayerNorm(h) @ W_Q_i
            K_i = LayerNorm(h) @ W_K_i
            V_i = LayerNorm(h) @ W_V_i

   b. FFN sub-layer:
      h_l = h_l' + FFN(LayerNorm(h_l'))

      where FFN(x) = (SwiGLU(x @ W_gate) * (x @ W_up)) @ W_down

3. output = softmax(LayerNorm(h_L) @ W_out / temperature)
```

That's it. Every Transformer — GPT-4, Claude, Llama, Gemini — is
a variation of this formula. The differences are in the sizes
(how many layers, how many heads, how wide), the specific activation
functions, the positional encoding method, and the training data.
The architecture is the same.

The remarkable thing: this relatively simple structure, trained on
enough text with enough parameters, produces what looks like
understanding. Nobody designed "reasoning" or "creativity" into it.
Those capabilities emerged from billions of gradient descent steps
optimizing a single objective: predict the next token.
