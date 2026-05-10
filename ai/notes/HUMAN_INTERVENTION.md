# Human Intervention Points in Building an LLM

Every stage of building an LLM involves human decisions.
The model doesn't build itself — humans decide what it learns, how it sees, and what it becomes.

## The Full Map

```
STAGE                          HUMAN DECISIONS                              IMPACT
─────                          ───────────────                              ──────

1. Data Collection             What sources? What languages?                ██████████
                               What domains? How much?
                               Web crawl? Books? Code? Academic papers?

2. Data Curation               What to filter out? (spam, misinfo, toxic)   ██████████
                               What to weight up? (textbooks, code)
                               How to deduplicate?
                               How to balance topics?
                               → This is the #1 factor in model quality.

3. Tokenizer Design            BPE vs SentencePiece vs others?              ████████
                               Vocabulary size? (32k? 100k?)
                               Domain-specific tokens? (code, medical)
                               Multi-language support?
                               → Determines what units the model sees.

4. Embedding Attributes        Dimension? (768? 1536? 3072?)                ███████
                               Number of layers? (12? 48? 96?)
                               Number of attention heads? (12? 96?)
                               Context window size? (2k? 128k?)
                               → Determines model capacity.

5. Training Objective          Next-token prediction? (GPT)                 ███████
                               Masked token prediction? (BERT)
                               Contrastive learning? (CLIP)
                               Combination of objectives?
                               → Determines what the model learns to do.

6. Training Hyperparameters    Learning rate and schedule?                   █████
                               Batch size?
                               Warmup steps?
                               Weight decay?
                               When to stop training?
                               → Fine-tuning the learning process itself.

7. Transformer Architecture    Currently: no changes since 2017.             ██
                               softmax(QK^T / sqrt(d_k)) V
                               Same in GPT, BERT, ViT, Whisper, etc.
                               → Settled for now. But see "Future" below.

8. Fine-Tuning                 What instruction-response pairs?              ████████
                               What behaviors to teach?
                               What format? (chat, QA, code, etc.)
                               How much fine-tuning data?
                               → Turns a text predictor into an assistant.

9. RLHF / Alignment           What does "helpful" mean?                     ████████
                               What's off-limits?
                               How to rank response quality?
                               Who are the human evaluators?
                               → Turns an assistant into a SAFE assistant.

10. RAG / Deployment           What knowledge base to retrieve from?         ██████
                               What retrieval strategy?
                               What system prompt?
                               What guardrails?
                               → The last mile of human control.
```

## What Most People See vs What Actually Matters

```
What users see:           Fine-tuning (8) → RLHF (9) → RAG (10)
                          "I can customize the model here"

What actually determines
model quality:            Data (1-2) → Tokenizer (3) → Architecture (4-5)
                          "This was decided months ago by a team of 50"

What's been unchanged
for 8 years:              Transformer (7)
                          "Same math since 2017"
```

## The Invisible Decisions

Steps 1-6 are invisible to end users but represent billions of dollars of decisions:

- **Anthropic/OpenAI/Google** each have dedicated teams for data curation alone
- The choice to include or exclude a data source changes model behavior for everyone
- A tokenizer decision made once affects every prompt forever after
- These early decisions are the hardest to change later (would require full retraining)

## Future: What Might Change?

The transformer mechanism (step 7) has been frozen since 2017.
But "settled" doesn't mean "final." Potential areas of change:

### O(n^2) Attention Problem
The score matrix is n x n (every token vs every token).
Double the context = 4x the compute and memory.

```
  Tokens     Score matrix     Memory (fp32)
  ──────     ────────────     ─────────────
  512        262K             1 MB
  2,048      4.2M             16 MB
  8,192      67M              256 MB
  32,768     1.07B            4 GB
  131,072    17.2B            64 GB
  1,000,000  1T               4 TB          ← impossible with current attention
```

Current workarounds (Flash Attention, KV Cache, Sparse Attention)
optimize the constant factor but don't change the O(n^2) fundamental.
A true breakthrough would change the mechanism itself.

### Why O(n^2) Exists: The Price of Being Blind and Deaf

Humans don't process all words equally. In "the cat sat down":
- "the" is nearly invisible — a filler word, skipped in speech ("th")
- "cat" and "sat" are emphasized with intonation
- A human attends to maybe 2 out of 4 words

This is because humans have THREE channels working simultaneously:

```
  Audio:   "th ca--t SAT down"
           Intonation signals what's important.
           "the" is barely pronounced. "SAT" is stressed.
           You know what matters BEFORE you process meaning.

  Visual:  You see the cat. You see it sitting.
           Half the meaning comes from eyes, not words.
           "it" refers to "the cat"? Obvious — you're looking at it.

  Text:    "the cat sat down"
           Your brain already knows from audio + visual
           which words to focus on. Most words are confirmation
           of what you already understood from sound and sight.
```

An LLM has ONLY text. No intonation. No visual context. No pointing.
So it must compensate by exhaustively checking every relationship:

```
  LLM:    "the cat sat down"
           Is "the" important here? Check against all 4 words.
           Is "cat" important here? Check against all 4 words.
           Is "sat" important here? Check against all 4 words.
           Is "down" important here? Check against all 4 words.

           4 × 4 = 16 pairs checked.
           A human checked maybe 4.
```

The O(n^2) cost is the price of having a single modality.
The model brute-forces every pair because it has no other signal
for what matters. No tone of voice. No gesture. No shared visual scene.

This also explains why:

- **Sparse attention works** — it mimics human selective attention
  (skip "the", focus on "cat"). But it must guess which tokens to skip
  because it can't hear intonation telling it.

- **Multi-modal models are the real path forward** — GPT-4V, Gemini
  add vision and audio. With these extra channels, the model gets the
  same shortcuts humans have. If it can see the cat, it doesn't need
  12 attention heads to figure out what "it" refers to.

- **Context windows keep growing** — without audio/visual shortcuts,
  the model needs more surrounding text to disambiguate.
  Humans resolve "apple" (fruit vs company) from a glance at the speaker.
  The model needs 50 tokens of context.

The Cambridge reading phenomenon proves this further:
"Tihs snetncee is srcmabled" — humans read it instantly because we
process word SHAPES visually, not character-by-character.
An LLM has no visual pattern matching. It must tokenize precisely
and attend to every token because text is all it has.

Future prediction: once models truly integrate audio and visual input
as first-class modalities (not bolted-on), they may become naturally
sparse — like humans are — and the O(n^2) problem may dissolve
not through a math trick, but through richer input.

### Other Potential Changes
- **Static weights at inference** — current models use the same weights for every input.
  Maybe weights should adapt per input (dynamic inference).
- **Fixed depth** — every input goes through all 96 layers.
  Maybe simple inputs should skip layers (early exit).
- **Discrete tokens** — everything is forced through a tokenizer.
  Maybe continuous representations work better for some domains.
- **Single modality** — most models handle one type of input.
  True unified multimodal architecture (not bolted-on) may need new mechanisms.

## Key Insight

Building an LLM is not "pick an architecture and train."
It's a chain of 10+ human decisions, each compounding on the previous.
The architecture (attention) is the most understood and least changed part.
The data and tokenizer — the parts furthest from the user — matter the most.
