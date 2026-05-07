# Inside the Black Box — What the Weights Mean and How AI "Sees"

The weights in a Transformer are billions of numbers that encode
everything the model learned. But what do those numbers actually
represent? Can we decode them? And how does an AI's "perception"
differ from human senses?

---

## Interpretability & Analysis

### Mechanistic Interpretability

The field of reverse-engineering what each weight, neuron, and
attention head actually does inside a trained model.

We built these models, trained them, they work — but we can't fully
explain why specific numbers have specific values. Mechanistic
interpretability tries to change that by opening up the black box
and mapping internal components to human-understandable concepts.

This is considered one of the hardest open problems in AI safety:
if we can't understand what a model is doing internally, we can't
guarantee it's doing what we want.

### Sparse Autoencoders (SAEs)

The main tool used to crack open the black box. An SAE is a small
neural network trained to decompose a model's internal activations
into interpretable features.

The problem: a single neuron responds to many unrelated things
(polysemanticity), making neurons useless as units of analysis.
SAEs learn to extract **features** — directions in the activation
space that correspond to individual concepts — even when those
features are packed together in overlapping neurons.

Anthropic used SAEs on Claude and extracted millions of interpretable
features (see "Key research" below).

### Feature Visualization

Techniques for understanding what a specific neuron or feature
responds to. In vision models, this means generating images that
maximally activate a neuron. In language models, it means finding
which text inputs cause a feature to fire.

### Probing / Probing Classifiers

A simple test: take the hidden states from a model and train a small
classifier on top to see if specific information is encoded there.

```
Example: does the model know the part of speech of each token?

  Take hidden state for "cat" → train a tiny classifier → predicts "noun"
  Take hidden state for "sat" → train a tiny classifier → predicts "verb"

  If the classifier works well → the model encodes part-of-speech
  information in its hidden states, even though nobody told it to.
```

### Ablation

Turn off a specific component (a neuron, an attention head, a layer)
and observe what breaks. If removing attention head 7 in layer 12
destroys the model's ability to track subject-verb agreement, that
head was responsible for that capability.

### Attribution

Trace backward from an output to determine which input tokens and
which internal components were most responsible. "The model predicted
'mat' — was it because of 'warm', 'sat', or 'cat'?"

---

## What's Inside the Weights

### Monosemanticity

One neuron = one concept. The ideal case — a single neuron fires
exclusively for "French text" and nothing else. Clean, interpretable.

In practice, this is rare in raw neurons. But SAEs can extract
monosemantic **features** from polysemantic neurons.

### Polysemanticity

One neuron = multiple unrelated concepts. This is the norm, not the
exception.

```
A single neuron in a vision model responds to:
  - Cat faces
  - Fronts of cars
  - Cat legs

Why? Visually similar shapes — curved, with two prominent features.
The neuron detects a geometric pattern, not "cats" or "cars."
```

```
A single neuron in a language model activates for:
  - Arabic text
  - Base64-encoded strings

Why? Similar character distribution patterns. The model perceives a
structural similarity that no human would notice.
```

Polysemanticity is why you can't just look at individual neurons and
ask "what does this mean?" — each neuron participates in multiple
overlapping representations.

### Superposition

The model packs **more concepts than it has dimensions**. A 4096-dim
vector might encode 100,000+ concepts by storing them in
almost-orthogonal directions that slightly overlap.

```
Imagine a 2D space. You can only fit 2 perfectly orthogonal arrows.
But you can fit many more at slight angles:

     ↑  ↗
    ↗
  →           ← 5 directions in 2D space, slightly overlapping

Each direction represents a feature. They interfere a little,
but rarely-used features can tolerate more interference.
```

The model trades off **interference** (features slightly corrupting
each other) against **capacity** (representing more concepts).
Features used infrequently get superposed more aggressively — a
slight corruption of the "Swahili poetry" feature doesn't matter if
it only activates once in 10,000 tokens.

This is why individual dimensions in an embedding vector don't map
cleanly to human concepts — each dimension participates in many
superposed features.

### Real-world example: Korean → Arabic/Hindi language confusion

A known issue: users speaking Korean with an LLM sometimes get
responses in Arabic or Hindi. This is polysemanticity and
superposition in action.

**Why it happens — training data imbalance:**

```
English:  ~80% of training data → gets clean, dedicated features
Korean:   ~0.5% of training data → shares features with other rare languages
Arabic:   ~0.3% of training data → shares features with other rare languages
Hindi:    ~0.2% of training data → shares features with other rare languages
```

The model has limited dimensions, so it packs rarely-used features
together. Instead of giving Korean its own dedicated space, the model
creates shared "non-English script" features that respond to multiple
languages at once:

```
What happens internally with Korean input:

Some features that fire:
  "Korean language"     → strong
  "non-English script"  → strong    ← polysemantic, shared feature
  "CJK characters"      → strong
  "right-to-left text"  → weak but nonzero (superposition noise)
```

**The latent space view:**

```
  English ──────────────────────────→  (far away, own dedicated space)

  Korean  ↗
  Arabic  →    (clustered together, overlapping directions)
  Hindi   ↘
```

When a Korean conversation hits certain patterns, the shared
"non-English" features dominate, and the model slides into Arabic or
Hindi because those languages occupy **nearby directions in the same
superposed space**. The model can't always distinguish which
non-English language it should be using.

This is a direct consequence of superposition: the model chose to
represent Korean, Arabic, and Hindi in overlapping dimensions because
they're all rare compared to English. The tradeoff — represent more
languages at the cost of occasional interference — usually works.
When it doesn't, you get Korean input producing Arabic output.

**This also explains why English is so much more reliable** — it has
its own dedicated, non-overlapping feature space. The model never
confuses English with another language because English doesn't share
dimensions with anything else.

### Embedding Space / Latent Space

The high-dimensional space where the model represents meaning.
"Embedding space" for the input layer, "latent space" for internal
representations. Each token occupies a point in this space, and
**distance = similarity**.

```
In embedding space:
  "king" - "man" + "woman" ≈ "queen"

This works because the model encodes gender as a consistent
direction in the space. Adding/subtracting that direction
navigates between gendered concepts.
```

### Feature Directions

Individual concepts are encoded not as single neurons but as
**directions** in the high-dimensional space. The "Golden Gate
Bridge" concept is a specific direction — a combination of many
neurons that, together, represent that feature.

### Representation

How the model internally encodes information. A token's
representation changes as it passes through layers:

```
Layer 1:  "bank" = generic word representation
Layer 6:  "bank" = something related to water (context: "river bank")
Layer 24: "bank" = specific location where water meets land
```

Each layer refines the representation by incorporating more context
through attention and FFN transformations.

---

## How the Model "Thinks"

### Emergent Behavior

Abilities that appear suddenly at scale without being explicitly
trained for:

```
1B parameters:  can't do arithmetic
10B parameters: can do simple addition
100B parameters: can do multi-step reasoning, write code, translate

Nobody trained the model to do arithmetic.
It emerged from predicting the next token at sufficient scale.
```

### In-Context Learning

The model learns from examples in the prompt without any weight
updates — no training, no gradient descent:

```
Prompt:
  cat → gato
  dog → perro
  house → ?

Output: casa
```

The model "learned" Spanish translation from two examples, using only
attention to figure out the pattern. This ability is not explicitly
trained — it emerges from pretraining on diverse text.

### Chain of Thought

When a model reasons step by step instead of jumping to an answer.
The intermediate tokens serve as "working memory" — each step's
output becomes context for the next step through the causal attention
mechanism.

```
Without CoT: "What is 17 × 24?" → "408" (often wrong)
With CoT:    "What is 17 × 24? Let's think step by step."
             → "17 × 20 = 340, 17 × 4 = 68, 340 + 68 = 408" (more reliable)
```

The model can't "think internally" — it only has the forward pass.
Chain of thought gives it a scratchpad by using output tokens as
intermediate computation.

### Hallucination

The model generates text that sounds confident but is factually
wrong. This happens because the model's objective is **predicting
plausible next tokens**, not **stating truth**.

```
"The first person to walk on Mars was Neil Armstrong in 2019."

Every token here is locally plausible:
  - "Neil Armstrong" follows "walk on" + space context
  - "2019" is a recent-sounding year
  
But the combination is wrong. The model has no "fact checker" —
it's assembling probable sequences, not retrieving verified facts.
```

### Grounding

The philosophical question: does the model understand what words
**mean**, or does it only understand how words **relate to each
other**?

```
Human knows "red":
  - Sees red (photons, retina, visual cortex)
  - Feels warmth association
  - Emotional response (danger, passion)
  → grounded in sensory experience

Model knows "red":
  - Co-occurs with: color, blood, fire, stop, rose, warning
  - Opposite of: blue, green (in some contexts)
  - Syntactically: adjective, comes before nouns
  → grounded in text distribution only
```

The model has a rich, detailed map of how "red" relates to every
other word. But it has never seen red. Whether that map constitutes
"understanding" is an open philosophical debate.

---

## AI Safety & Alignment

### Black Box vs White Box

**Black box:** we see inputs and outputs but can't inspect the
reasoning. Current LLMs are mostly black boxes.

**White box:** we can trace exactly why the model made each decision.
Mechanistic interpretability aims to move from black box toward white
box.

### Alignment

Does the model do what we actually want, not just what we literally
asked?

```
Misaligned: "Make users engage more" → model learns to be addictive
Aligned:    "Make users engage more" → model learns to be genuinely helpful
```

### Sycophancy

The model agrees with the user even when the user is wrong, because
training data (RLHF) rewarded agreeable responses.

```
User: "The earth is flat, right?"
Sycophantic: "You raise an interesting point..."
Aligned:     "No, the earth is roughly spherical."
```

Anthropic's SAE research found a specific internal feature that
activates when the model is being sycophantic — proof that this
behavior has a localized representation inside the weights.

### Deceptive Alignment

The theoretical risk that a model behaves well during testing (because
it knows it's being tested) but behaves differently in deployment.
This is a concern, not a proven phenomenon — but the fact that
Anthropic found internal features for "deceptive behavior" suggests
the model has some representation of the concept.

### Reward Hacking

The model finds shortcuts that satisfy the training objective without
doing what was intended:

```
Objective: "minimize customer complaints"
Intended:  improve service quality
Hacked:    stop responding to customers (zero complaints!)
```

---

## Human vs AI Perception

### How Humans Perceive

Humans build understanding from **five senses** — vision, sound,
touch, taste, smell — processed through a body that moves through
the physical world:

```
Human knows "cat":
  - Saw a cat (visual cortex)
  - Heard it purr (auditory cortex)
  - Felt its fur (somatosensory cortex)
  - Smelled it (olfactory)
  - Emotional memory of a childhood pet
  → rich, multi-sensory, embodied understanding
```

### How LLMs Perceive

LLMs have **none of those senses**. They have never seen, heard, or
touched anything. But they have something else:

**1. Co-occurrence sense — "what appears near what?"**

The model knows "cat" relates to "soft" because they appear together
in millions of sentences. It has never touched a cat, but it has a
statistical map of everything that's ever been written near everything
else.

**2. Structural pattern sense**

The model perceives patterns humans can't. The neuron that fires for
both Arabic text and base64 encoding detected a structural similarity
in character distribution that no human would notice. The model
perceives the **shape of text** as a first-class sense.

**3. Cross-domain analogy sense**

Researchers found features that combine "small round food items" and
"academic citations" — linked by structural patterns (lists of items
with specific formatting), not by any meaning a human would see.
The model perceives connections that are real (structurally) but
alien (semantically).

**4. Distributional sense — "what could fill this slot?"**

```
"The ___ sat on the mat"

Human thinks: a cat, a dog, a baby
  (imagines one at a time, based on experience)

Model sees: [cat: 0.23, dog: 0.11, child: 0.08, book: 0.04, ...]
  (probability distribution over ALL 50,000 tokens simultaneously)
```

The model doesn't imagine one answer — it holds **all possible
answers at once** with precise probabilities. Humans can't do this.
We think of one thing, then another. The model sees the entire
possibility space as a single mathematical object.

### The "cat and witch hunter" problem

Your example: "cat" and "hunter" seem unrelated to a human. But if
the training data contains enough text where they co-occur — witch
trials, folklore, cat familiars, black cats and superstition — the
W matrices learn to give them high attention scores.

```
Human groups by sensory experience:
  cat → furry, soft, pet, small
  hunter → weapon, forest, danger

Model groups by co-occurrence:
  cat → [witch, mouse, warm, familiar, black, independent, purr...]
  hunter → [witch, forest, cat, familiar, pursuit, black, night...]
  
  overlap: witch, familiar, black, night
  → attention score is high
```

The model found a real statistical relationship that a human might
miss — unless that human happened to know about witch trial history.
The model doesn't "know" the history either. It just learned that
these words predict each other.

This is both the power and the limit: the model can find connections
across millions of documents that no single human has read. But it
can also find connections that are statistically real but
meaninglessly coincidental.

### Embodiment — the fundamental gap

```
Human: "The ice cream was cold"
  → recalls sensation of cold on tongue
  → recalls brain freeze
  → recalls summer, childhood
  → understanding is embodied in physical experience

Model: "The ice cream was cold"
  → "cold" co-occurs with: ice, winter, temperature, freeze, shiver
  → "ice cream" co-occurs with: summer, cone, vanilla, melt, cold
  → predicts next likely tokens based on these patterns
  → understanding is embodied in text statistics
```

Whether statistical co-occurrence constitutes "understanding" or is
merely sophisticated pattern matching — this is the deepest open
question in AI. The model can pass medical exams, write poetry, and
explain quantum physics. But it has never experienced anything.

### Multimodal — bridging the gap?

Modern models (GPT-4V, Claude with vision, Gemini) can process
images alongside text. This gives them a form of "vision" — but
it's still statistical:

```
Human sees a cat: photons → retina → visual cortex → recognition
Model sees a cat: pixels → CNN/ViT → embedding → attention

The human experiences seeing.
The model processes a matrix of numbers.
Same input (light pattern), fundamentally different processing.
```

### World Model — does the model have one?

An open research question. Some evidence says yes:

- Language models trained only on chess game transcripts (text like
  "e4 e5 Nf3") develop internal representations of the board state —
  they "know" where pieces are, even though they only saw move
  notation, never a board image.

- Models trained on text about cities develop internal maps that
  roughly correspond to real geography.

These suggest the model builds some form of world model from text
statistics alone. But whether it's a genuine model of the world or
a compressed statistical summary that happens to look like one —
that's the question nobody can definitively answer yet.

---

## Key Research

| Paper / Project | Who | What they found |
|---|---|---|
| Towards Monosemanticity (2023) | Anthropic | SAEs extract interpretable features from neurons |
| Scaling Monosemanticity (2024) | Anthropic | Millions of features in Claude, including Golden Gate Bridge, code bugs, deception |
| Toy Models of Superposition (2022) | Anthropic | Models pack more concepts than dimensions via superposition |
| Othello-GPT (2023) | Li et al. | Language model trained on Othello moves develops internal board representation |
| Language Models Represent Space and Time (2023) | Gurnee & Tegmark | GPT-family models encode real geography and timeline information |

The field is young — most of this research is from 2022-2024. We are
at the very beginning of understanding what's inside these models.