"""
Lesson 12: Fine-tuning — Adapting a Pre-trained Model

Reuses the MiniGPT architecture from lesson 11. We:
  1. Pretrain it on Shakespeare (domain A).
  2. Fine-tune the full model on Python code (domain B).
  3. Fine-tune with LoRA (tiny adapters) instead, and compare.
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import math
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from lesson_player import LessonPlayer, cyan, green

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
torch.manual_seed(42)

player = LessonPlayer("Lesson 12: Fine-tuning a Pre-trained Model")

# ── Corpora ─────────────────────────────────────────────

SHAKESPEARE = """\
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
"""

PYTHON_CODE = """\
def hello(name):
    return f"Hello, {name}!"

def add(a, b):
    return a + b

def square(x):
    return x * x

def is_even(n):
    return n % 2 == 0

class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y
    def distance(self, other):
        dx = self.x - other.x
        dy = self.y - other.y
        return (dx * dx + dy * dy) ** 0.5

def factorial(n):
    if n <= 1:
        return 1
    return n * factorial(n - 1)
"""

# Unified vocab: characters from both corpora.
ALL_CHARS = sorted(set(SHAKESPEARE + PYTHON_CODE))
vocab_size = len(ALL_CHARS)
stoi = {ch: i for i, ch in enumerate(ALL_CHARS)}
itos = {i: ch for ch, i in stoi.items()}


def encode(s):
    return [stoi[ch] for ch in s if ch in stoi]


def decode(ids):
    return "".join(itos[i] for i in ids)

# ── Model (copied from lesson 11) ───────────────────────

SEQ_LEN = 32
D_MODEL = 64
N_HEADS = 4
N_LAYERS = 2


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads, seq_len):
        super().__init__()
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
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class MiniGPT(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, n_layers, seq_len):
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

    def generate(self, idx, max_new_tokens, temperature=0.8):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.seq_len:]
            logits = self.forward(idx_cond)[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            next_idx = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_idx], dim=1)
        return idx


def make_batcher(text, batch_size=16, seq_len=SEQ_LEN):
    data = torch.tensor(encode(text), dtype=torch.long)

    def get_batch():
        ix = torch.randint(0, len(data) - seq_len - 1, (batch_size,))
        x = torch.stack([data[i:i + seq_len] for i in ix])
        y = torch.stack([data[i + 1:i + seq_len + 1] for i in ix])
        return x, y
    return get_batch


def train(model, get_batch, steps, lr=3e-3):
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr)
    losses = []
    model.train()
    for _ in range(steps):
        x, y = get_batch()
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    return losses


def sample(model, prompt, n=200):
    model.eval()
    with torch.no_grad():
        ctx = torch.tensor([encode(prompt)], dtype=torch.long)
        out = model.generate(ctx, max_new_tokens=n)
        return decode(out[0].tolist())

# ── Step 1: the case for fine-tuning ────────────────────

player.add_step("Step 1: Why Fine-tune?", f"""\
  Pretraining a large language model costs {cyan('millions of dollars')}
  in GPU time. You will never do that. But you don't have to.

  {cyan('The deal:')} someone else pretrains a model on the whole
  internet. That model learned language, grammar, reasoning, world
  facts. You take it and adapt it to {green('your')} domain with a tiny
  fraction of the data and compute.

  {cyan('What fine-tuning is good for:')}
    - Style / tone / format (customer-service voice, your docs)
    - Following instructions in your domain's language
    - Sticking to a schema (JSON, SQL, a specific output shape)

  {cyan('What fine-tuning is NOT great at:')}
    - Teaching the model NEW facts. For that, use RAG (retrieval).
    - Fixing reasoning bugs in a small base model. Use a bigger one.

  {green('Cost order of magnitude:')}
    Pretrain GPT-3 style:   $5-10M, months, thousands of GPUs
    Full fine-tune a 7B:    $100s, hours, 8 GPUs
    LoRA fine-tune a 7B:    $10s, hour, 1 GPU""")

# ── Step 2: HF ecosystem ────────────────────────────────

player.add_step("Step 2: The Hugging Face Ecosystem", f"""\
  In practice, fine-tuning in Python looks like this:

  {cyan('Install:')}
    pip install transformers peft datasets accelerate

  {cyan('Load a pretrained model:')}
    from transformers import AutoModelForCausalLM, AutoTokenizer
    model = AutoModelForCausalLM.from_pretrained("gpt2")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

  {cyan('Apply LoRA:')}
    from peft import LoraConfig, get_peft_model
    cfg = LoraConfig(r=8, target_modules=["c_attn"])
    model = get_peft_model(model, cfg)

  {cyan('Train:')}
    from transformers import Trainer, TrainingArguments
    trainer = Trainer(model=model, args=..., train_dataset=...)
    trainer.train()

  {cyan('Why Hugging Face?')} huggingface.co hosts 500k+ open models.
  Same API for all of them. `transformers` handles architecture, `peft`
  handles LoRA/adapters, `datasets` handles data loading.

  This lesson implements the ideas by hand — so you understand what
  those lines are actually doing. In real work you'd use the library.""")

# ── Step 3: our setup ───────────────────────────────────

player.add_step("Step 3: Our Setup — MiniGPT as the Base Model", f"""\
  We reuse the character-level transformer from lesson 11, but build
  vocab from the union of {cyan('two')} corpora so the model can emit
  chars from either domain.

  {cyan('Domain A (pretrain):')} Shakespeare dialogue
  {cyan('Domain B (fine-tune):')} Python code snippets

  {cyan('Shared vocab:')} we take {green(f'{vocab_size} unique chars')} from
  the union of both texts. The model won't use Python-specific chars
  like {{ }} ( ) during pretraining — but they're in the vocab, ready
  to be activated during fine-tuning.

  {cyan('Model config:')}
    d_model  = {D_MODEL}
    n_heads  = {N_HEADS}
    n_layers = {N_LAYERS}
    seq_len  = {SEQ_LEN}

  {cyan('Plan:')}
    1. Pretrain on Shakespeare
    2. Save a snapshot of the pretrained weights
    3. Fine-tune ALL weights on Python → compare
    4. Restart from the snapshot, fine-tune ONLY LoRA adapters → compare""")

# ── Step 4: pretrain ────────────────────────────────────

model = MiniGPT(vocab_size, D_MODEL, N_HEADS, N_LAYERS, SEQ_LEN)
shake_batch = make_batcher(SHAKESPEARE)
pretrain_losses = train(model, shake_batch, steps=500, lr=3e-3)

pretrained_state = copy.deepcopy(model.state_dict())
total_params = sum(p.numel() for p in model.parameters())

shake_out_after_pretrain = sample(model, "First Citizen:\n", n=180)
code_out_after_pretrain = sample(model, "def ", n=180)

player.add_step("Step 4: Pretrain on Shakespeare", f"""\
  Same training loop as lesson 11: 500 steps on Shakespeare.

  {green('Final pretrain loss:')} {pretrain_losses[-1]:.3f}
  {green('Total parameters:')} {total_params:,}

  {cyan("Generate from 'First Citizen:\\n' — in domain:")}
{shake_out_after_pretrain}

  {cyan("Generate from 'def ' — out of domain:")}
{code_out_after_pretrain}

  The model can handle Shakespeare. Prompted with `def ` it tries to
  continue in Shakespeare style — it has no idea what Python looks
  like yet. This is our {green('pretrained')} starting point.""")

# ── Step 5: full fine-tune ──────────────────────────────

code_batch = make_batcher(PYTHON_CODE)
ft_losses = train(model, code_batch, steps=300, lr=1e-3)
trainable_full = sum(p.numel() for p in model.parameters() if p.requires_grad)

shake_out_after_full = sample(model, "First Citizen:\n", n=180)
code_out_after_full = sample(model, "def ", n=180)

player.add_step("Step 5: Full Fine-tune on Python", f"""\
  Keep training the SAME model, now on Python code. All weights update.

  {cyan('Code:')}
    code_batch = make_batcher(PYTHON_CODE)
    train(model, code_batch, steps=300, lr=1e-3)

  {cyan('Lower learning rate:')} 1e-3 vs 3e-3 for pretraining. Standard
  practice — you don't want to blow away the pretrained knowledge.

  {green('Trainable parameters:')} {trainable_full:,}  (the whole model)
  {green('Final fine-tune loss:')} {ft_losses[-1]:.3f}

  {cyan("Generate from 'def ' — now in the new domain:")}
{code_out_after_full}

  {cyan("Generate from 'First Citizen:\\n' — old domain:")}
{shake_out_after_full}

  The model now produces code-ish output. It also partially forgot
  Shakespeare — that's {green('catastrophic forgetting')}, one reason
  full fine-tuning can be dangerous.""")

# ── Step 6: LoRA concept ────────────────────────────────

player.add_step("Step 6: LoRA — Low-Rank Adaptation", f"""\
  Full fine-tuning updates every weight. For a 7B-param model, that
  means 7B gradients, 7B optimizer states, ~30GB of VRAM. Heavy.

  {cyan("LoRA's insight:")} the {green('change')} needed to adapt a
  pretrained weight matrix is usually low-rank. So instead of updating
  W directly, learn a tiny correction:

      W_final = W_pretrained + α · (B @ A)

  where W is (d × d), A is (r × d), B is (d × r), and {cyan('r << d')}.

  {cyan('For d=64, r=8:')}
    Full fine-tune:  64 × 64 = 4,096 params per weight
    LoRA adapter:    64 × 8 + 8 × 64 = 1,024 params per weight
    Savings:         4x smaller here — for d=4096, r=8 it's 256x

  {cyan('Key tricks:')}
    - Base weights are FROZEN (requires_grad=False)
    - Only A and B are trained
    - B is initialized to 0, so the adapter starts as a no-op
    - Merge back at inference: W + BA, no latency cost

  {cyan('Code sketch:')}
    class LoRALinear(nn.Module):
        def forward(self, x):
            return self.base(x) + (x @ self.A.T @ self.B.T) * self.scale""")

# ── Step 7: LoRA fine-tune ──────────────────────────────


class LoRALinear(nn.Module):
    """Wraps an nn.Linear; freezes it and adds a low-rank adapter."""
    def __init__(self, base: nn.Linear, rank=8, alpha=16):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False
        in_f = base.in_features
        out_f = base.out_features
        self.lora_A = nn.Parameter(torch.zeros(rank, in_f))
        self.lora_B = nn.Parameter(torch.zeros(out_f, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.scaling = alpha / rank

    def forward(self, x):
        return self.base(x) + (x @ self.lora_A.T @ self.lora_B.T) * self.scaling


def inject_lora(model, rank=8, alpha=16):
    """Wrap every Linear inside each block with a LoRALinear."""
    for p in model.parameters():
        p.requires_grad = False
    for block in model.blocks:
        block.attn.qkv = LoRALinear(block.attn.qkv, rank, alpha)
        block.attn.proj = LoRALinear(block.attn.proj, rank, alpha)
        block.mlp[0] = LoRALinear(block.mlp[0], rank, alpha)
        block.mlp[2] = LoRALinear(block.mlp[2], rank, alpha)
    for name, p in model.named_parameters():
        if "lora_" in name:
            p.requires_grad = True
    return model


# Restart from the pretrained snapshot, then add LoRA.
model_lora = MiniGPT(vocab_size, D_MODEL, N_HEADS, N_LAYERS, SEQ_LEN)
model_lora.load_state_dict(pretrained_state)
model_lora = inject_lora(model_lora, rank=8, alpha=16)

trainable_lora = sum(p.numel() for p in model_lora.parameters() if p.requires_grad)
frozen_lora = sum(p.numel() for p in model_lora.parameters() if not p.requires_grad)
lora_losses = train(model_lora, code_batch, steps=500, lr=5e-3)

shake_out_after_lora = sample(model_lora, "First Citizen:\n", n=180)
code_out_after_lora = sample(model_lora, "def ", n=180)

# Loss comparison plot
plt.figure(figsize=(9, 4))
plt.plot(ft_losses, label=f"Full fine-tune ({trainable_full:,} params)")
plt.plot(lora_losses, label=f"LoRA fine-tune ({trainable_lora:,} params)")
plt.xlabel("Fine-tuning step")
plt.ylabel("Cross-entropy loss")
plt.title("Full fine-tune vs LoRA fine-tune")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, "images", "12_finetune_loss.png"))
plt.close()

pct = 100 * trainable_lora / (trainable_lora + frozen_lora)

player.add_step("Step 7: Fine-tune with LoRA", f"""\
  Restart from the pretrained snapshot. Wrap every attention Linear
  with a LoRA adapter. Freeze everything except the adapters. Train.

  {cyan('Code:')}
    model = MiniGPT(...)
    model.load_state_dict(pretrained_state)    # restore pretrained
    model = inject_lora(model, rank=8)         # freeze base, add adapters
    train(model, code_batch, steps=300, lr=5e-3)

  {green('Parameter accounting:')}
    Frozen (base):  {frozen_lora:,}
    Trainable:      {trainable_lora:,}  ({pct:.1f}% of total)
    Final loss:     {lora_losses[-1]:.3f}

  {cyan("Generate from 'def ' after LoRA fine-tune:")}
{code_out_after_lora}

  {cyan("Generate from 'First Citizen:\\n' after LoRA fine-tune:")}
{shake_out_after_lora}

  Notice: the LoRA model shifts toward Python but often keeps more of
  its Shakespeare knowledge — fewer weights moved, less forgetting.

  Loss comparison: images/12_finetune_loss.png""")

# ── Step 8: summary ─────────────────────────────────────

player.add_step("Step 8: Summary", f"""\
  {green('What you built:')}
    ✓ Pretrained a base transformer on domain A
    ✓ Full fine-tuned it on domain B (all weights)
    ✓ Implemented LoRA by hand — low-rank adapter wrapping nn.Linear
    ✓ Fine-tuned with LoRA on a fraction of the parameters

  {cyan('Fine-tuning comparison:')}
    Approach      Trainable params    Forgetting    Storage
    Full          {trainable_full:>5,}              high          full checkpoint
    LoRA (r=8)    {trainable_lora:>5,} ({pct:.1f}%)       lower         adapter only

  {cyan('Production path (the actual libraries):')}
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model
    model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3-8B")
    model = get_peft_model(model, LoraConfig(r=8, alpha=16, ...))
    # train; save only adapter weights (~few MB instead of 16GB)

  {cyan('Beyond this course:')}
    - RAG  → give the model access to your docs at inference time
    - RLHF / DPO → align model behavior with human preferences
    - Distillation → compress a big model into a small one
    - Quantization → run big models on tiny hardware (int4, int8)

  You now understand the building blocks of modern LLM engineering.""")

# ── Play ────────────────────────────────────────────────

player.play()
