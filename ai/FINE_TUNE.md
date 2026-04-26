# Fine-Tuning — The Practical Guide

Fine-tuning takes a pretrained model and adjusts its weights on your
specific data so it behaves differently. This document covers everything
you need to know to actually do it.

## What fine-tuning is (and isn't)

```
Pretraining:    Internet-scale data  →  general knowledge    (months, millions $)
Fine-tuning:    Your specific data   →  specialized behavior (hours, hundreds $)
```

Pretraining builds the brain. Fine-tuning teaches it a specific job.

You are NOT training from scratch. You're taking a model that already
understands language and nudging its weights toward your use case.

---

## When to fine-tune vs alternatives

Before fine-tuning, ask if a cheaper approach works:

| Approach | Cost | When to use |
|---|---|---|
| **Prompt engineering** | Free | Change behavior with instructions alone |
| **Few-shot examples** | Free | Show 3-5 examples in the prompt |
| **RAG** (retrieval) | Low | Model needs access to your documents |
| **Fine-tuning** | Medium | Change the model's style, format, or domain knowledge |
| **Full pretraining** | Extreme | You need a fundamentally different model |

Fine-tune when:
- You want a consistent output format the model can't achieve via prompting
- You need domain-specific language (medical, legal, code)
- You want to distill a large model's behavior into a smaller, cheaper one
- Latency matters and you can't afford long prompts with examples

Don't fine-tune when:
- You just need the model to access your data (use RAG)
- A good system prompt solves it (test this first)
- You have fewer than ~100 high-quality examples

---

## Types of fine-tuning

### Full Fine-Tuning

**What:** Update every weight in the model.

```
All 7 billion parameters are unfrozen and adjusted.
```

**Pros:** Maximum flexibility, best possible quality.
**Cons:** Needs enormous GPU memory. A 7B model needs ~56GB just for
weights in float32, plus optimizer states (2-3× more). Practically
requires multiple A100/H100 GPUs.

**Risk:** Most prone to catastrophic forgetting — the model can lose
its general abilities while specializing.

### LoRA (Low-Rank Adaptation)

**What:** Freeze all original weights. Add tiny trainable matrices
alongside specific layers.

```
Original:     y = W @ x           (W is frozen, not updated)
With LoRA:    y = W @ x + (B @ A) @ x    (only A and B are trained)

W: 4096 × 4096 = 16.7M parameters  (frozen)
A: 4096 × 16   = 65K parameters    (trained)
B: 16 × 4096   = 65K parameters    (trained)

Total trainable: 130K vs 16.7M = 0.78% of original
```

**Key terms:**

| Term | What it means |
|---|---|
| **Rank (r)** | Size of the low-rank matrices. r=8 or r=16 is common. Higher = more capacity but more memory |
| **Alpha (α)** | Scaling factor. The LoRA output is multiplied by α/r. Higher α = stronger adaptation |
| **Target modules** | Which layers get LoRA adapters. Usually attention layers (q_proj, k_proj, v_proj, o_proj) |
| **Adapter** | The pair of small matrices (A, B). Can be saved separately (~10-50MB vs multi-GB model) |
| **Merging** | Bake the adapter back into the base model: W_new = W + α/r × (B @ A) |

**Why it works:** Weight changes during fine-tuning tend to be low-rank —
they don't need the full dimensionality. LoRA exploits this by only
learning the "difference" in a compressed form.

**Pros:** Trains on a single GPU (even 24GB). Adapter files are tiny.
Can swap different adapters on the same base model.
**Cons:** Slightly less expressive than full fine-tuning.

### QLoRA (Quantized LoRA)

**What:** Same as LoRA, but the frozen base model is quantized to 4-bit.

```
Base model:  7B params × 4-bit = ~3.5GB  (instead of ~14GB in float16)
LoRA adapters: trained in float16/bfloat16 (full precision)
```

**Why:** Cuts memory by ~4× more. A 70B model fits on a single 48GB GPU.
This is what most individual fine-tuners actually use.

**Is the result the same as LoRA?** Almost. The frozen weights are only
*read* during training (forward pass), never updated. Reading them at
4-bit introduces tiny rounding errors, but the LoRA adapters are still
trained in full precision (bf16). Research shows ~0.1-0.3% quality
difference on benchmarks — practically negligible for most use cases.

**Key term — NF4 (NormalFloat4):** The specific 4-bit format used. It's
designed for normally distributed weights, which neural networks tend to
have. Better quality than naive 4-bit truncation.

### Other methods (less common)

| Method | What |
|---|---|
| **Prefix Tuning** | Add trainable "virtual tokens" to the beginning of each layer |
| **Adapters (Houlsby)** | Insert small trainable modules between existing layers |
| **IA³** | Learn to rescale activations instead of adding matrices |
| **Full + FSDP** | Full fine-tuning distributed across many GPUs with sharding |

---

## The fine-tuning vocabulary

### Data terms

| Term | Meaning |
|---|---|
| **Training set** | Data the model learns from |
| **Validation set** | Held-out data to monitor overfitting during training |
| **Test set** | Final evaluation data, never seen during training |
| **Instruction dataset** | Pairs of (instruction, response) |
| **Chat dataset** | Multi-turn conversations with roles (system, user, assistant) |
| **Completion dataset** | Raw text the model learns to continue |
| **Data quality** | The single most important factor. 1,000 perfect examples beat 100,000 noisy ones |

### Training hyperparameters

| Term | What it controls | Typical value |
|---|---|---|
| **Learning rate** | How big each weight update is | 1e-5 to 5e-5 (full), 1e-4 to 3e-4 (LoRA) |
| **Epochs** | How many times to loop through the entire dataset | 1-5 (more = risk of overfitting) |
| **Batch size** | How many examples to process before updating weights | 4-32 (limited by GPU memory) |
| **Gradient accumulation** | Simulate larger batch size by accumulating gradients over multiple mini-batches | 4-8 steps |
| **Warmup steps** | Gradually increase learning rate from 0 to target | 5-10% of total steps |
| **Weight decay** | Penalize large weights to prevent overfitting | 0.01-0.1 |
| **Max sequence length** | Truncate or pad sequences to this length | 512-4096 tokens |
| **Scheduler** | How learning rate changes over time | cosine (decrease smoothly) or linear |

**Why learning rate matters most:** Too high → model forgets everything
(catastrophic forgetting) or training diverges. Too low → barely learns
anything from your data. This is the first thing to tune.

```
Learning rate too high:
  Step 0: "What is Python?" → "A programming language"     ← correct
  Step 100: "What is Python?" → "skdjf the output format"  ← destroyed

Learning rate just right:
  Step 0: "What is Python?" → "A programming language"     ← still correct
  Step 100: "What is Python?" → "A programming language used for..."  ← improved
```

### Weight and gradient terms

| Term | Meaning |
|---|---|
| **Frozen weights** | Parameters that don't update (requires_grad=False) |
| **Trainable parameters** | Parameters that get updated via gradients |
| **Gradient** | Direction and magnitude to adjust each weight |
| **Gradient clipping** | Cap gradient magnitude to prevent exploding updates (max_grad_norm=1.0) |
| **Mixed precision (fp16/bf16)** | Use half-precision floats for speed. bf16 is preferred (larger range, less overflow) |
| **Loss curve** | Plot of loss over training steps. Should go down, then flatten |

### Problems and solutions

| Problem | What happens | How to detect | Fix |
|---|---|---|---|
| **Catastrophic forgetting** | Model loses general knowledge | Ask general questions → nonsense | Lower learning rate, fewer epochs, use LoRA |
| **Overfitting** | Memorizes training data, fails on new inputs | Training loss ↓ but validation loss ↑ | More data, fewer epochs, add dropout, weight decay |
| **Underfitting** | Doesn't learn your task | Both training and validation loss stay high | More epochs, higher learning rate, more data |
| **Mode collapse** | Always generates the same output | All outputs look identical | Lower learning rate, more diverse data |
| **Data leakage** | Test data accidentally in training set | Suspiciously good evaluation scores | Properly split data before training |

---

## Data preparation (the hardest part)

Data quality determines 80% of fine-tuning success. The model, method,
and hyperparameters are secondary.

### Dataset formats

**Instruction format** (most common for fine-tuning):

```json
[
  {
    "instruction": "Summarize this medical report in plain English",
    "input": "Patient presents with acute myocardial infarction...",
    "output": "The patient had a heart attack..."
  },
  {
    "instruction": "Translate to formal legal language",
    "input": "The tenant didn't pay rent for 3 months",
    "output": "The lessee has been in default of rental payments..."
  }
]
```

**Chat format** (for conversational fine-tuning):

```json
{
  "conversations": [
    {"role": "system", "content": "You are a medical assistant."},
    {"role": "user", "content": "What does hypertension mean?"},
    {"role": "assistant", "content": "Hypertension means high blood pressure..."},
    {"role": "user", "content": "Is it dangerous?"},
    {"role": "assistant", "content": "It can be if untreated..."}
  ]
}
```

**Completion format** (for continued pretraining):

```text
Raw text that the model learns to predict token by token.
Used when you want the model to absorb domain knowledge
(e.g., feed it all your company's documentation).
```

### Chat templates

Each model family has its own special token format. You must match it
exactly, or the model learns garbage boundaries.

```
# Llama 3 format
<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a helpful assistant.<|eot_id|>
<|start_header_id|>user<|end_header_id|>
Hello<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
Hi there!<|eot_id|>

# ChatML format (Mistral, others)
<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
Hello<|im_end|>
<|im_start|>assistant
Hi there!<|im_end|>
```

The tokenizer handles this — you don't write these tags manually. But you
need to know they exist because a template mismatch silently ruins training.

### Data quality checklist

- [ ] Consistent format across all examples
- [ ] No contradictory examples (same input, different correct output)
- [ ] Responses match the style you actually want
- [ ] No sensitive/private data unless intentional
- [ ] Balanced — not 90% one category and 10% everything else
- [ ] Minimum ~100 examples for LoRA, ~1000 for full fine-tuning
- [ ] Reviewed by a human (at least a random sample)

---

## The practical workflow

### Step 1: Choose base model

| Size | VRAM needed (QLoRA) | Good for |
|---|---|---|
| 1-3B (Llama 3.2, Phi-3) | 6-8GB | Edge, mobile, fast experiments |
| 7-8B (Llama 3.1, Mistral) | 10-16GB | Best balance of quality and cost |
| 13-14B | 16-24GB | Noticeably smarter than 7B |
| 70B | 40-48GB | Near-frontier quality |

### Step 2: Prepare data

```python
# Typical data prep script
import json

raw_data = load_your_data()

formatted = []
for item in raw_data:
    formatted.append({
        "instruction": item["question"],
        "output": item["ideal_answer"]
    })

# Split: 90% train, 10% validation
train = formatted[:int(len(formatted) * 0.9)]
val = formatted[int(len(formatted) * 0.9):]

with open("train.json", "w") as f:
    json.dump(train, f)
with open("val.json", "w") as f:
    json.dump(val, f)
```

### Step 3: Load model + configure LoRA

Both LoRA and QLoRA use the same LoRA config. The only difference is
**how you load the base model**.

#### Option A: LoRA (base model in float16)

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer

model_name = "meta-llama/Llama-3.1-8B-Instruct"

# --- Load model in float16 (no quantization) ---
# Needs ~16GB VRAM for 7B model
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,    # half precision — saves memory vs float32
    device_map="auto",             # automatically place layers on available GPUs
)
tokenizer = AutoTokenizer.from_pretrained(model_name)
```

#### Option B: QLoRA (base model in 4-bit)

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, BitsAndBytesConfig
from peft import LoraConfig, prepare_model_for_kbit_training
from trl import SFTTrainer

model_name = "meta-llama/Llama-3.1-8B-Instruct"

# --- This is what makes it QLoRA instead of LoRA ---
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,              # quantize frozen weights to 4-bit
    bnb_4bit_quant_type="nf4",      # NormalFloat4 — best quality 4-bit format
    bnb_4bit_compute_dtype=torch.bfloat16,  # compute in bf16 during forward pass
    bnb_4bit_use_double_quant=True, # quantize the quantization constants too (saves ~0.4GB)
)

# Load model in 4-bit — only ~3.5GB VRAM for 7B model
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,  # ← this one line is the difference from LoRA
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Prepare model for training after quantization
# (freezes base, enables gradient checkpointing for memory savings)
model = prepare_model_for_kbit_training(model)
```

#### Side-by-side comparison

```
                        LoRA                    QLoRA
─────────────────────────────────────────────────────────────
Base model precision    float16 / bfloat16      4-bit (NF4)
Base model VRAM (7B)    ~14 GB                  ~3.5 GB
Adapter precision       float16 / bfloat16      float16 / bfloat16  (same)
Adapter VRAM            ~50-200 MB              ~50-200 MB          (same)
Optimizer states        ~2-4 GB                 ~2-4 GB             (same)
Total VRAM (7B)         ~20 GB                  ~8 GB
Training quality        Full precision          ~99.7% of LoRA
Training speed          Faster                  ~10-20% slower (dequantize on each forward pass)
Extra dependency        —                       bitsandbytes
```

QLoRA trades ~0.3% quality and ~15% speed for 60% less memory.
For most people on consumer GPUs, that tradeoff is obvious.

#### LoRA config (same for both)

```python
lora_config = LoraConfig(
    r=16,                          # rank — how much capacity the adapter has
                                   #   r=8:  minimal, fast, less expressive
                                   #   r=16: good default
                                   #   r=64: maximum, slower, more expressive
    lora_alpha=32,                 # scaling factor — usually set to 2×r
                                   #   effective weight = alpha/r = 32/16 = 2.0
                                   #   higher = adapter has stronger influence
    target_modules=[               # which layers get LoRA adapters
        "q_proj", "k_proj",        #   attention: query, key
        "v_proj", "o_proj",        #   attention: value, output
        "gate_proj", "up_proj",    #   FFN layers (optional but recommended)
        "down_proj"                #   more modules = more capacity + more VRAM
    ],
    lora_dropout=0.05,             # dropout on adapter outputs (regularization)
    task_type="CAUSAL_LM",         # tells PEFT this is a language model
    bias="none",                   # don't train bias terms (standard for LLMs)
)
```

**How to choose target_modules:**

```
Minimal (least VRAM, fastest):
  target_modules=["q_proj", "v_proj"]
  ~0.5% of parameters trained

Standard (good balance):
  target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
  ~1% of parameters trained

Full (best quality, most VRAM):
  target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                   "gate_proj", "up_proj", "down_proj"]
  ~2-3% of parameters trained
```

#### Training config (same for both)

```python
training_args = TrainingArguments(
    output_dir="./output",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,  # effective batch = 4 × 4 = 16
    learning_rate=2e-4,             # LoRA uses higher LR than full fine-tuning
    warmup_steps=100,               # ramp up LR gradually at start
    weight_decay=0.01,
    bf16=True,                      # use bfloat16 for training
    logging_steps=10,               # print loss every 10 steps
    eval_strategy="steps",
    eval_steps=100,                 # run validation every 100 steps
    save_steps=200,                 # save checkpoint every 200 steps
    lr_scheduler_type="cosine",     # smoothly decrease LR over training
    gradient_checkpointing=True,    # trade compute for memory (slower but less VRAM)
)
```

### Step 4: Train

```python
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    peft_config=lora_config,
    max_seq_length=2048,
)

trainer.train()

# Save the adapter (small file, ~50MB)
trainer.model.save_pretrained("./my-adapter")
tokenizer.save_pretrained("./my-adapter")
```

**What gets saved:** Only the LoRA adapter weights (~10-50MB), not
the entire base model. To use it later, you load the base model +
adapter separately.

### Step 5: Evaluate

```python
# Load base model + adapter
from peft import PeftModel

base = AutoModelForCausalLM.from_pretrained(model_name)
model = PeftModel.from_pretrained(base, "./my-adapter")

# Test on held-out examples
for example in test_set:
    output = generate(model, example["instruction"])
    print(f"Expected: {example['output']}")
    print(f"Got:      {output}")
    print()
```

### Step 6: Merge and deploy (optional)

```python
# Merge adapter into base model for faster inference
merged = model.merge_and_unload()
merged.save_pretrained("./my-model-merged")

# Now it's a standalone model — no adapter loading needed
```

---

## Tools and frameworks

| Tool | What it does | When to use |
|---|---|---|
| **Hugging Face Transformers + PEFT** | The standard library. Load models, apply LoRA, train | Default choice |
| **TRL (Transformer Reinforcement Learning)** | SFTTrainer, DPO, RLHF on top of Transformers | When you need alignment training |
| **Unsloth** | 2× faster QLoRA training with optimized kernels | When speed matters on single GPU |
| **Axolotl** | YAML-config-driven training. Wraps everything above | When you want config files, not code |
| **LLaMA-Factory** | GUI + CLI for fine-tuning many model families | When you want a web UI |
| **OpenAI / Anthropic API** | Upload data, they fine-tune for you | When you don't want to manage GPUs |

### Hardware requirements

| Method | 7B model | 13B model | 70B model |
|---|---|---|---|
| Full fine-tune (fp16) | 2× A100 80GB | 4× A100 80GB | 16× A100 80GB |
| LoRA (fp16) | 1× A100 40GB | 1× A100 80GB | 4× A100 80GB |
| QLoRA (4-bit) | 1× RTX 4090 24GB | 1× A100 40GB | 1× A100 80GB |

Cloud GPU costs (approximate, 2025):
- A100 80GB: ~$2/hr (Lambda, RunPod, Vast.ai)
- H100 80GB: ~$3-4/hr
- A 7B QLoRA fine-tune on 10K examples ≈ 2-4 hours ≈ $4-16

---

## Alignment fine-tuning (RLHF / DPO)

Standard fine-tuning (SFT) teaches the model WHAT to say.
Alignment teaches it HOW to say it — helpfully, safely, honestly.

### SFT (Supervised Fine-Tuning)

```
Input:  "Write a poem about cats"
Output: "Soft paws upon the windowsill..."

The model learns: given this prompt, produce this response.
```

This is the fine-tuning described above. It's step 1.

### RLHF (Reinforcement Learning from Human Feedback)

```
Step 1: Generate multiple responses to the same prompt
Step 2: Humans rank them (response A > response B)
Step 3: Train a "reward model" to predict human preferences
Step 4: Use RL (PPO) to optimize the language model against the reward model
```

**Why it's hard:** PPO is unstable. You need a separate reward model.
Training involves four models in memory at once (policy, reference,
reward, value). Expensive and finicky.

### DPO (Direct Preference Optimization)

```
Input: prompt + chosen_response + rejected_response
The model learns to increase probability of chosen, decrease rejected.
No reward model needed — the preference is baked into the loss function.
```

**Why DPO is winning:** Same quality as RLHF, much simpler. One training
loop, no reward model, no RL instability. This is what most people use
now for alignment.

**Data format for DPO:**

```json
{
  "prompt": "Explain quantum computing",
  "chosen": "Quantum computing uses qubits that can be 0 and 1 simultaneously...",
  "rejected": "Quantum computing is when computers use quantum physics to be faster..."
}
```

---

## Continued pretraining vs instruction fine-tuning

Two different goals, often confused:

### Continued pretraining (domain adaptation)

**Goal:** Teach the model new knowledge.

```
Feed it 10GB of medical textbooks, legal documents, or your codebase.
The model learns to predict the next token in this domain.
No instruction/response pairs — just raw text.
```

**Result:** The model "knows" your domain but doesn't know how to
follow instructions about it. You typically do instruction fine-tuning
after this.

### Instruction fine-tuning (SFT)

**Goal:** Teach the model to follow instructions in a specific way.

```
(instruction, response) pairs.
The model learns your desired behavior, format, and style.
```

### The full pipeline (what companies actually do)

```
Base model (Llama 3.1 8B)
  ↓  continued pretraining on domain text
Domain model (knows medical/legal/code knowledge)
  ↓  instruction fine-tuning on (instruction, response) pairs
Instruction model (follows instructions in your domain)
  ↓  DPO/RLHF on preference data
Aligned model (helpful, safe, your style)
  ↓  quantize (4-bit/8-bit)
Production model (fast, cheap to serve)
```

Most individual fine-tuners skip continued pretraining and start from
an already-instruct-tuned model (like Llama-3.1-8B-Instruct).

---

## Evaluation — how to know if it worked

### Quantitative metrics

| Metric | What it measures |
|---|---|
| **Training loss** | How well the model fits training data (should decrease) |
| **Validation loss** | How well it generalizes (should decrease, then plateau) |
| **Perplexity** | e^loss — lower = model is more confident on correct tokens |
| **BLEU / ROUGE** | N-gram overlap with reference text (crude but standard) |
| **Pass@k** | For code: does generated code pass test cases? |

### The loss curve — what to look for

```
Good:
  Training loss:    ████▓▓▒▒░░░░     (decreases, plateaus)
  Validation loss:  ████▓▓▒▒░░░░     (follows training, plateaus)

Overfitting:
  Training loss:    ████▓▓▒▒░░░░     (keeps decreasing)
  Validation loss:  ████▓▓▒▒▓▓██     (starts increasing ← stop here)

Underfitting:
  Training loss:    ████████████      (barely moves)
  Validation loss:  ████████████      (barely moves)
```

### Qualitative evaluation (more important)

Metrics don't capture what you actually care about. Always:

1. Test with 20-50 prompts that represent real use cases
2. Compare base model output vs fine-tuned output side by side
3. Check for catastrophic forgetting — ask general knowledge questions
4. Check edge cases — what happens with unusual inputs?
5. Have someone who didn't make the dataset evaluate blindly

---

## Common mistakes

| Mistake | Why it fails | Fix |
|---|---|---|
| Too few examples | Model doesn't learn the pattern | Minimum 100 for LoRA, 1000 for full |
| Too many epochs | Memorizes training data | Use 1-3 epochs, watch validation loss |
| Learning rate too high | Destroys pretrained knowledge | Start at 2e-4 for LoRA, 2e-5 for full |
| Bad data quality | Garbage in, garbage out | Review examples manually |
| Wrong chat template | Model learns wrong token boundaries | Use tokenizer.apply_chat_template() |
| No validation set | Can't detect overfitting | Always hold out 10% |
| Fine-tuning when prompting works | Wasted time and money | Test prompt engineering first |
| Training on only positive examples | Model doesn't learn what NOT to do | Add DPO with rejected examples |

---

## Knowledge map — what a fine-tuner needs to know

```
Must know:
├── Python basics (you're here)
├── PyTorch fundamentals (tensors, autograd, training loops)
├── Hugging Face ecosystem (transformers, datasets, tokenizers)
├── LoRA / QLoRA (how and why)
├── Data preparation (formats, cleaning, quality)
├── Hyperparameter tuning (learning rate, epochs, batch size)
├── Evaluation (loss curves, qualitative testing)
└── When NOT to fine-tune (prompting, RAG as alternatives)

Should know:
├── Linux + CLI (SSH, tmux, GPU monitoring with nvidia-smi)
├── Distributed training (multi-GPU, DeepSpeed, FSDP)
├── Quantization (GPTQ, AWQ, bitsandbytes)
├── DPO / alignment training
├── Weights & Biases or MLflow (experiment tracking)
└── Model serving (vLLM, TGI, Ollama)

Nice to know:
├── Flash Attention (memory-efficient attention)
├── Mixture of Experts (MoE) fine-tuning
├── Continued pretraining strategies
├── Synthetic data generation (using a strong model to generate training data)
├── Merging models (TIES, DARE, model soups)
└── The Transformer architecture in depth (covered in GEN_AI.md)
```
