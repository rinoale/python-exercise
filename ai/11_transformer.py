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
  {green('Pipeline:')} text → IDs → embeddings → transformer → logits → next char""",

korean=f"""\
  {cyan('문자 단위 언어 모델')}을 만듭니다. 텍스트를 받아서
  다음 문자를 예측합니다. 출력을 다시 입력으로 넣으면 — 문장이 생성됩니다.

  {cyan('GPT와 같은 원리, 규모만 작음:')}
    GPT-3:  50,257 토큰, 96층, 1750억 파라미터, 인터넷 전체로 학습
    우리:    ~65 문자,   2층,  ~4만 파라미터, 셰익스피어로 학습

  {cyan('왜 문자 단위인가?')}
    토크나이저를 따로 학습할 필요 없음. "어휘"는 고유 문자 목록 그 자체.
    내부 동작을 쉽게 볼 수 있음. 실제 모델은 BPE 서브워드를 사용하지만
    수학적 원리는 동일.

  {green('코퍼스 크기:')} {len(TEXT)}자
  {green('처리 흐름:')} 텍스트 → ID → 임베딩 → 트랜스포머 → 로짓 → 다음 문자""")

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

  {cyan('Type:')}
    TEXT = "First Citizen:\\nBefore we proceed any further, hear me speak."
    chars = sorted(set(TEXT))
    len(chars)
    stoi = {{ch: i for i, ch in enumerate(chars)}}
    itos = {{i: ch for ch, i in stoi.items()}}
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda ids: "".join(itos[i] for i in ids)
    encode("hi")
    decode(encode("hi"))
    encode("First")
    decode(encode("First"))

  {green('Try also:')}
    stoi
    itos
    [ch for ch in sorted(set("hello world"))]

  {green('In our full model:')} vocab_size={vocab_size}, corpus={len(TEXT)} chars""",

detail=f"""\
  {green('Term: Tokenization')}
      Converting text into numbers a model can process.
      Three levels of tokenization exist:

      {cyan('Character-level')} (what we use):
          "hello" → [7, 4, 11, 11, 14]
          Vocab is tiny (~65 chars). Simple but inefficient —
          each character is one token, so sequences are long.

      {cyan('Word-level:')}
          "hello world" → [4231, 891]
          Fast, but can't handle misspellings or new words.

      {cyan('Subword (BPE)')} (what GPT/Claude use):
          "unhappiness" → ["un", "happiness"]  → [342, 8901]
          Best of both worlds. Common words stay whole,
          rare words get split into known pieces.

  {cyan('Our stoi / itos dictionaries are the vocabulary.')}
      stoi = string-to-integer (encoder)
      itos = integer-to-string (decoder)""",

korean=f"""\
  고유한 문자마다 하나의 정수 ID를 부여합니다.
  인코딩은 텍스트를 숫자 텐서로 변환합니다.

  {cyan('Type:')}
    TEXT = "First Citizen:\\nBefore we proceed any further, hear me speak."
    chars = sorted(set(TEXT))
    len(chars)
    stoi = {{ch: i for i, ch in enumerate(chars)}}
    itos = {{i: ch for ch, i in stoi.items()}}
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda ids: "".join(itos[i] for i in ids)
    encode("hi")
    decode(encode("hi"))
    encode("First")
    decode(encode("First"))

  {green('추가 실습:')}
    stoi
    itos
    [ch for ch in sorted(set("hello world"))]

  {green('전체 모델에서:')} vocab_size={vocab_size}, 코퍼스={len(TEXT)}자""",

detail_korean=f"""\
  {green('용어: 토큰화')}
      텍스트를 모델이 처리할 수 있는 숫자로 변환하는 것.
      세 가지 수준이 있음:

      {cyan('문자 단위')} (우리가 사용):
          "hello" → [7, 4, 11, 11, 14]
          어휘가 작음 (~65자). 단순하지만 비효율적 —
          문자 하나가 토큰 하나라 시퀀스가 길어짐.

      {cyan('단어 단위:')}
          "hello world" → [4231, 891]
          빠르지만 오타나 신조어 처리 불가.

      {cyan('서브워드 (BPE)')} (GPT/Claude가 사용):
          "unhappiness" → ["un", "happiness"]  → [342, 8901]
          두 장점 모두. 흔한 단어는 그대로, 드문 단어는 알려진 조각으로 분리.

  {cyan('stoi / itos 딕셔너리가 어휘.')}
      stoi = 문자열→정수 (인코더)
      itos = 정수→문자열 (디코더)""")

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

  {cyan('Type:')}
    import torch
    data = torch.tensor([0,1,2,3,4,5,6,7,8,9])
    SEQ_LEN, BATCH_SIZE = 4, 2
    ix = torch.randint(0, len(data) - SEQ_LEN - 1, (BATCH_SIZE,))
    ix
    x = torch.stack([data[i:i + SEQ_LEN] for i in ix])
    y = torch.stack([data[i + 1:i + SEQ_LEN + 1] for i in ix])
    x
    y
    x.shape

  {green('Notice:')} y is x shifted by 1. If x=[2,3,4,5], y=[3,4,5,6].
  The model learns to predict the next token at each position.

  {green('In our full model:')} BATCH_SIZE=16, SEQ_LEN=32
    x shape: {list(xb.shape)}, y shape: {list(yb.shape)}""",

korean=f"""\
  학습 스텝마다 코퍼스에서 무작위 조각을 가져옵니다.
  {cyan('입력 x')} = 위치 [i, i+SEQ_LEN)의 문자들.
  {cyan('정답 y')} = 위치 [i+1, i+SEQ_LEN+1)의 문자들 — 1칸 밀린 것.

  위치 t에서 모델은 x[:t+1]을 보고 x[t+1]을 예측해야 합니다.
  배치 하나에서 SEQ_LEN × BATCH_SIZE개의 예측 연습이 생깁니다.

  {cyan('Type:')}
    import torch
    data = torch.tensor([0,1,2,3,4,5,6,7,8,9])
    SEQ_LEN, BATCH_SIZE = 4, 2
    ix = torch.randint(0, len(data) - SEQ_LEN - 1, (BATCH_SIZE,))
    ix
    x = torch.stack([data[i:i + SEQ_LEN] for i in ix])
    y = torch.stack([data[i + 1:i + SEQ_LEN + 1] for i in ix])
    x
    y
    x.shape

  {green('주목:')} y는 x를 1칸 밀린 것. x=[2,3,4,5]이면 y=[3,4,5,6].
  모델은 각 위치에서 다음 토큰을 예측하는 법을 배움.

  {green('전체 모델:')} BATCH_SIZE=16, SEQ_LEN=32
    x 형태: {list(xb.shape)}, y 형태: {list(yb.shape)}""")

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
  learn different relationships (syntax, semantics, distance...).

  {cyan('Type: (3 tokens, d_k=2: walk through the formula step by step)')}
    import torch, torch.nn.functional as F, math
    Q = torch.tensor([[0.3, 0.8], [0.5, 0.1], [0.9, 0.4]])
    K = torch.tensor([[0.1, 0.2], [0.4, 0.7], [0.2, 0.5]])
    V = torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]])
    attn = Q @ K.T / math.sqrt(2)
    attn                                # similarity scores: who attends to whom?
    mask = torch.tril(torch.ones(3, 3))
    mask                                # lower triangle = 1 (past), upper = 0 (future)
    attn = attn.masked_fill(mask == 0, float('-inf'))
    attn                                # future positions are now -inf
    weights = F.softmax(attn, dim=-1)
    weights                             # -inf → 0 after softmax. rows sum to 1
    weights[0]                          # token 0: can only see itself → [1, 0, 0]
    weights[2]                          # token 2: attends to all three tokens
    output = weights @ V
    output                              # weighted mix of V — this is the attention output""",

detail=f"""\
  {green('Step-by-step attention example:')}

      Input: "The cat sat"  (3 tokens, processing "sat")

      Q for "sat":  what am I looking for?  →  [0.3, 0.8, ...]
      K for "The":  what do I offer?        →  [0.1, 0.2, ...]
      K for "cat":  what do I offer?        →  [0.4, 0.7, ...]

      Similarity:
          Q("sat") · K("The") = 0.3×0.1 + 0.8×0.2 = 0.19  (low)
          Q("sat") · K("cat") = 0.3×0.4 + 0.8×0.7 = 0.68  (high!)

      After softmax:  weights = [0.24, 0.76]
          "sat" attends 76% to "cat", 24% to "The"

      Output = 0.24 × V("The") + 0.76 × V("cat")
          "sat" now carries information about "cat."

  {green('Why √d_k scaling?')}
      Without scaling, large d_k makes dot products huge,
      which pushes softmax into near-one-hot.
      Dividing by √d_k keeps gradients healthy.

  {green('Term: Causal Mask')}
      During training, all positions are processed in parallel.
      But position 3 shouldn't see positions 4, 5, ... (the future).
      The mask sets those weights to -inf → softmax makes them 0.
      This simulates left-to-right processing.""",

korean=f"""\
  트랜스포머의 핵심 혁신. 각 위치가 {cyan('모든')} 이전 위치를 보고
  각각에 얼마나 주의를 기울일지 결정합니다.

  {cyan('각 위치마다 3개의 투영:')}
    Q (query, 쿼리) — 내가 찾고 있는 것은?
    K (key, 키)     — 내가 제공하는 것은?
    V (value, 값)   — 내가 공유하는 것은?

  {cyan('어텐션 공식:')}
    attn(Q, K, V) = softmax(Q @ Kᵀ / √d_k) @ V

  직관: Q·K 내적 = 유사도. softmax가 유사도를 가중치로 변환.
  V의 가중합 = 문맥을 반영한 표현.

  {cyan('인과 마스크:')} 언어 모델에서 위치 t는
  0..t만 볼 수 있음. softmax 전에 상삼각을 -inf로 채워서
  미래 정보가 과거로 누출되지 않게 합니다.

  {cyan('멀티헤드:')} H개의 어텐션을 병렬로 실행, 각각 더 작은
  차원(d_model / H)으로. 서로 다른 관계를 학습할 수 있음
  (구문, 의미, 거리...).

  {cyan('Type: (3개 토큰, d_k=2: 공식을 단계별로 실행)')}
    import torch, torch.nn.functional as F, math
    Q = torch.tensor([[0.3, 0.8], [0.5, 0.1], [0.9, 0.4]])
    K = torch.tensor([[0.1, 0.2], [0.4, 0.7], [0.2, 0.5]])
    V = torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]])
    attn = Q @ K.T / math.sqrt(2)
    attn                                # 유사도 점수: 누가 누구에게 주의?
    mask = torch.tril(torch.ones(3, 3))
    mask                                # 하삼각 = 1 (과거), 상삼각 = 0 (미래)
    attn = attn.masked_fill(mask == 0, float('-inf'))
    attn                                # 미래 위치는 이제 -inf
    weights = F.softmax(attn, dim=-1)
    weights                             # -inf → softmax 후 0. 행의 합은 1
    weights[0]                          # 토큰 0: 자기 자신만 봄 → [1, 0, 0]
    weights[2]                          # 토큰 2: 세 토큰 모두에 주의
    output = weights @ V
    output                              # V의 가중합 — 어텐션 출력""",

detail_korean=f"""\
  {green('어텐션 단계별 예시:')}

      입력: "The cat sat"  (3개 토큰, "sat" 처리 중)

      "sat"의 Q:  내가 찾는 것은?     →  [0.3, 0.8, ...]
      "The"의 K:  내가 제공하는 것은?  →  [0.1, 0.2, ...]
      "cat"의 K:  내가 제공하는 것은?  →  [0.4, 0.7, ...]

      유사도:
          Q("sat") · K("The") = 0.3×0.1 + 0.8×0.2 = 0.19  (낮음)
          Q("sat") · K("cat") = 0.3×0.4 + 0.8×0.7 = 0.68  (높음!)

      softmax 후:  가중치 = [0.24, 0.76]
          "sat"는 "cat"에 76%, "The"에 24% 주의

      출력 = 0.24 × V("The") + 0.76 × V("cat")
          "sat"이 이제 "cat"의 정보를 담고 있음.

  {green('왜 √d_k로 나누는가?')}
      스케일링 없이 d_k가 크면 내적이 커져서
      softmax가 거의 원핫에 가까워짐.
      √d_k로 나누면 그래디언트가 안정적.

  {green('인과 마스크')}
      학습 중에는 모든 위치가 병렬로 처리됨.
      하지만 위치 3은 위치 4, 5, ... (미래)를 보면 안 됨.
      마스크가 해당 가중치를 -inf로 설정 → softmax 후 0이 됨.
      왼쪽에서 오른쪽으로의 처리를 시뮬레이션.""")

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

  {cyan('Type: (see residuals and LayerNorm in action)')}
    import torch, torch.nn as nn
    d_model = 8
    x = torch.randn(1, 4, d_model)
    ln = nn.LayerNorm(d_model)
    normed = ln(x)
    normed.mean(dim=-1)                 # ≈ 0 per token (LayerNorm normalizes)
    normed.std(dim=-1)                  # ≈ 1 per token
    mlp = nn.Sequential(nn.Linear(d_model, 4 * d_model), nn.GELU(), nn.Linear(4 * d_model, d_model))
    mlp_out = mlp(ln(x))
    mlp_out.shape                       # same shape as x: (1, 4, 8)
    residual = x + mlp_out              # x + f(x): original info preserved
    (residual - x)[0, 0]               # difference = what the MLP added

  {cyan('The MLP:')}
    Linear(d_model → 4·d_model) → GELU → Linear(4·d_model → d_model)
    Lets the model "think" about each token independently after
    attention has mixed information between tokens.

  {cyan('Residuals (x + f(x)):')} keep gradients flowing in deep nets.
  {cyan('LayerNorm:')} stabilizes training by normalizing activations.
  {cyan('Modern twist:')} we do LayerNorm BEFORE attention/MLP
  ("pre-norm") — easier to train than the original "post-norm".

  Stacking N blocks = deeper model. GPT-3 has 96. We'll use 2.""",

detail=f"""\
  {green('Term: Residual Connection (skip connection)')}
      x = x + f(x)    instead of    x = f(x)

      Why? In deep networks, gradients vanish as they backpropagate
      through many layers. The residual shortcut lets gradients
      flow directly through the + operation, skipping layers.
      Without residuals, deep transformers can't train.

  {green('Term: LayerNorm')}
      Normalizes each sample to mean=0, std=1 across features.
      Stabilizes training — without it, activations can explode
      or vanish across layers.

      Pre-norm (what we use): normalize BEFORE attention/MLP.
      Post-norm (original paper): normalize AFTER.
      Pre-norm is easier to train and is now the standard.

  {green('Term: GELU activation')}
      Like ReLU but smooth — doesn't have a hard kink at 0.
      GELU(x) ≈ x × sigmoid(1.702x)
      Used in GPT-2, BERT, and most modern transformers.
      ReLU works too, but GELU gives slightly better results.""",

korean=f"""\
  블록 = 어텐션 + 피드포워드, 각각 잔차 연결(residual)과 LayerNorm 포함.

  {cyan('Type: (잔차 연결과 LayerNorm을 직접 확인)')}
    import torch, torch.nn as nn
    d_model = 8
    x = torch.randn(1, 4, d_model)
    ln = nn.LayerNorm(d_model)
    normed = ln(x)
    normed.mean(dim=-1)                 # ≈ 0 (LayerNorm이 정규화)
    normed.std(dim=-1)                  # ≈ 1
    mlp = nn.Sequential(nn.Linear(d_model, 4 * d_model), nn.GELU(), nn.Linear(4 * d_model, d_model))
    mlp_out = mlp(ln(x))
    mlp_out.shape                       # x와 같은 형태: (1, 4, 8)
    residual = x + mlp_out              # x + f(x): 원래 정보 보존
    (residual - x)[0, 0]               # 차이 = MLP가 추가한 것

  {cyan('MLP:')}
    Linear(d_model → 4·d_model) → GELU → Linear(4·d_model → d_model)
    어텐션이 토큰 간 정보를 섞은 후, 각 토큰이 독립적으로 "생각"하는 단계.

  {cyan('잔차 연결 (x + f(x)):')} 깊은 네트워크에서 그래디언트 흐름 유지.
  {cyan('LayerNorm:')} 활성화 값을 정규화하여 학습 안정화.
  {cyan('현대적 방식:')} 어텐션/MLP 전에 LayerNorm 적용
  ("pre-norm") — 원래 논문의 "post-norm"보다 학습이 쉬움.

  N개 블록을 쌓으면 더 깊은 모델. GPT-3은 96개. 우리는 2개.""",

detail_korean=f"""\
  {green('용어: 잔차 연결 (skip connection)')}
      x = x + f(x)    대신    x = f(x)

      왜? 깊은 네트워크에서 그래디언트가 여러 레이어를 거치며
      역전파될 때 소실됨. 잔차 지름길은 + 연산을 통해
      그래디언트가 레이어를 건너뛰며 직접 흐르게 함.
      잔차 연결 없이는 깊은 트랜스포머를 학습할 수 없음.

  {green('용어: LayerNorm')}
      각 샘플의 feature를 mean=0, std=1로 정규화.
      학습 안정화 — 이것 없이는 활성화 값이 레이어마다
      폭발하거나 소실될 수 있음.

      Pre-norm (우리가 사용): 어텐션/MLP 전에 정규화.
      Post-norm (원래 논문): 어텐션/MLP 후에 정규화.
      Pre-norm이 학습이 더 쉽고 현재 표준.

  {green('용어: GELU 활성화')}
      ReLU와 비슷하지만 부드러움 — 0에서 꺾이지 않음.
      GELU(x) ≈ x × sigmoid(1.702x)
      GPT-2, BERT 등 대부분의 현대 트랜스포머에서 사용.
      ReLU도 작동하지만 GELU가 약간 더 나은 결과.""")

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
  Steps 2-5 gave us the pieces. Now we wire them into one class:

  {cyan('MiniGPT architecture (read-only diagram, not for typing):')}
    class MiniGPT(nn.Module):
        __init__:
            tok_emb  = Embedding(vocab_size, d_model)    # step 2
            pos_emb  = Embedding(seq_len, d_model)       # position info
            blocks   = N × Block(attention + FFN)        # step 5
            ln_f     = LayerNorm(d_model)                # final norm
            head     = Linear(d_model, vocab_size)       # → logits

        forward(token_ids):
            tok = tok_emb(ids)          # (B, T, d_model)
            pos = pos_emb(0..T-1)       # (T, d_model)
            x = tok + pos               # combine token + position
            x = blocks(x)              # N transformer blocks
            x = ln_f(x)                # final LayerNorm
            return head(x)             # (B, T, vocab_size) logits

  {green('This lesson already built one for you:')}
    model = MiniGPT(vocab_size=45, d_model=64, n_heads=4, n_layers=2, seq_len=32)
    Total parameters: {n_params:,}
    (defined at line ~551 of this file — open it to see the real code)

  {cyan('Try it in REPL:')}
    __file__ = "ai/11_transformer.py"
    exec(open(__file__).read())
    model                               # trained MiniGPT, ready to use
    model(torch.tensor([[0,1,2,3]]))    # feed token IDs, get logits

  {cyan('Why positional embeddings?')} Attention is permutation-invariant
  — without position info, "cat sat mat" and "mat sat cat" look the
  same. We add a learned vector per position to break the symmetry.

  {cyan('Type: (prove that position embedding breaks permutation invariance)')}
    import torch, torch.nn as nn
    vocab_size, d_model, T = 10, 8, 4
    tok_emb = nn.Embedding(vocab_size, d_model)
    pos_emb = nn.Embedding(T, d_model)
    a = torch.tensor([[2, 5, 1, 7]])
    b = torch.tensor([[7, 1, 5, 2]])    # same tokens, reversed order
    tok_emb(a)[0, 0]                    # token 2 at position 0
    tok_emb(b)[0, 3]                    # token 2 at position 3 — same vector!
    without_pos_a = tok_emb(a).sum()
    without_pos_b = tok_emb(b).sum()
    without_pos_a == without_pos_b      # True — without position, order is invisible
    pos = pos_emb(torch.arange(T))
    with_pos_a = (tok_emb(a) + pos).sum()
    with_pos_b = (tok_emb(b) + pos).sum()
    with_pos_a == with_pos_b            # False — position makes order matter""",

korean=f"""\
  2~5단계에서 만든 부품을 하나의 클래스로 조립합니다:

  {cyan('MiniGPT 구조 (읽기 전용 다이어그램, 직접 입력용 아님):')}
    class MiniGPT(nn.Module):
        __init__:
            tok_emb  = Embedding(vocab_size, d_model)    # 2단계
            pos_emb  = Embedding(seq_len, d_model)       # 위치 정보
            blocks   = N × Block(어텐션 + FFN)           # 5단계
            ln_f     = LayerNorm(d_model)                # 최종 정규화
            head     = Linear(d_model, vocab_size)       # → 로짓

        forward(token_ids):
            tok = tok_emb(ids)          # (B, T, d_model)
            pos = pos_emb(0..T-1)       # (T, d_model)
            x = tok + pos               # 토큰 + 위치 결합
            x = blocks(x)              # N개 트랜스포머 블록
            x = ln_f(x)                # 최종 LayerNorm
            return head(x)             # (B, T, vocab_size) 로짓

  {green('이 레슨이 이미 하나 만들어 놓음:')}
    model = MiniGPT(vocab_size=45, d_model=64, n_heads=4, n_layers=2, seq_len=32)
    총 파라미터: {n_params:,}
    (이 파일의 ~551번째 줄에 정의됨 — 파일을 열어 실제 코드 확인)

  {cyan('REPL에서 사용해보기:')}
    __file__ = "ai/11_transformer.py"
    exec(open(__file__).read())
    model                               # 학습된 MiniGPT, 바로 사용 가능
    model(torch.tensor([[0,1,2,3]]))    # 토큰 ID 입력 → 로짓 출력

  {cyan('왜 위치 임베딩?')} 어텐션은 순서에 무관.
  위치 정보가 없으면 "cat sat mat"과 "mat sat cat"이 동일하게 보임.
  위치별 학습 벡터를 더해서 대칭을 깨뜨립니다.

  {cyan('Type: (위치 임베딩이 순서 불변성을 깨는 것을 증명)')}
    import torch, torch.nn as nn
    vocab_size, d_model, T = 10, 8, 4
    tok_emb = nn.Embedding(vocab_size, d_model)
    pos_emb = nn.Embedding(T, d_model)
    a = torch.tensor([[2, 5, 1, 7]])
    b = torch.tensor([[7, 1, 5, 2]])    # 같은 토큰, 역순
    tok_emb(a)[0, 0]                    # 위치 0의 토큰 2
    tok_emb(b)[0, 3]                    # 위치 3의 토큰 2 — 같은 벡터!
    without_pos_a = tok_emb(a).sum()
    without_pos_b = tok_emb(b).sum()
    without_pos_a == without_pos_b      # True — 위치 없이는 순서가 보이지 않음
    pos = pos_emb(torch.arange(T))
    with_pos_a = (tok_emb(a) + pos).sum()
    with_pos_b = (tok_emb(b) + pos).sum()
    with_pos_a == with_pos_b            # False — 위치가 순서를 구분하게 만듦""")

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

  {cyan('Type: (see why we flatten, and watch loss drop)')}
    import torch, torch.nn as nn, torch.nn.functional as F
    B, T, vocab_size, d_model = 2, 4, 10, 8
    tok_emb = nn.Embedding(vocab_size, d_model)
    head = nn.Linear(d_model, vocab_size)
    optimizer = torch.optim.AdamW(list(tok_emb.parameters()) + list(head.parameters()), lr=1e-2)
    x = torch.tensor([[2, 5, 1, 7], [0, 3, 6, 9]])
    y = torch.tensor([[5, 1, 7, 4], [3, 6, 9, 2]])
    logits = head(tok_emb(x))
    logits.shape                        # (2, 4, 10) = B, T, vocab
    logits.view(-1, vocab_size).shape    # (8, 10) = B*T predictions, each over vocab
    y.view(-1)                          # [5,1,7,4,3,6,9,2] = 8 target IDs
    loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))
    loss                                # before training: ~ln(10) ≈ 2.3 (random guess)
    for i in range(50):
        logits = head(tok_emb(x))
        loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    loss                                # after 50 steps: much lower — model learned x→y

  {green('Training:')}
{log_lines}
  Loss at step 1 was around ln({vocab_size}) ≈ {math.log(vocab_size):.2f}
  (uniform guess). Dropping below that means the model learned something.

  Loss curve: images/11_transformer_loss.png""",

korean=f"""\
  레슨 10과 동일한 4단계 루프:
    순전파 → 손실 → 역전파 → 가중치 갱신.

  {cyan('Type: (왜 flatten하는지 보고, 손실이 줄어드는 것을 확인)')}
    import torch, torch.nn as nn, torch.nn.functional as F
    B, T, vocab_size, d_model = 2, 4, 10, 8
    tok_emb = nn.Embedding(vocab_size, d_model)
    head = nn.Linear(d_model, vocab_size)
    optimizer = torch.optim.AdamW(list(tok_emb.parameters()) + list(head.parameters()), lr=1e-2)
    x = torch.tensor([[2, 5, 1, 7], [0, 3, 6, 9]])
    y = torch.tensor([[5, 1, 7, 4], [3, 6, 9, 2]])
    logits = head(tok_emb(x))
    logits.shape                        # (2, 4, 10) = B, T, vocab
    logits.view(-1, vocab_size).shape    # (8, 10) = B*T개 예측, 각각 vocab에 대해
    y.view(-1)                          # [5,1,7,4,3,6,9,2] = 8개 정답 ID
    loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))
    loss                                # 학습 전: ~ln(10) ≈ 2.3 (무작위 추측)
    for i in range(50):
        logits = head(tok_emb(x))
        loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    loss                                # 50스텝 후: 훨씬 낮음 — 모델이 x→y를 학습

  {green('학습 결과:')}
{log_lines}
  1스텝의 손실은 약 ln({vocab_size}) ≈ {math.log(vocab_size):.2f}
  (균등 추측). 이 값 아래로 떨어지면 모델이 무언가를 배운 것.

  손실 곡선: images/11_transformer_loss.png""")

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

  {cyan('Type: (generate one token at a time, autoregressive loop)')}
    import torch, torch.nn as nn, torch.nn.functional as F
    vocab_size, d_model = 10, 8
    tok_emb = nn.Embedding(vocab_size, d_model)
    head = nn.Linear(d_model, vocab_size)
    idx = torch.tensor([[2, 5, 1]])     # start with 3 tokens
    logits = head(tok_emb(idx))
    logits.shape                        # (1, 3, 10) — prediction at every position
    last_logits = logits[:, -1, :]      # only need the LAST position to generate
    last_logits.shape                   # (1, 10) — scores for 10 possible next tokens
    probs = F.softmax(last_logits / 0.8, dim=-1)
    probs                               # probability distribution over next token
    next_id = torch.multinomial(probs, num_samples=1)
    next_id                             # sampled next token
    idx = torch.cat([idx, next_id], dim=1)
    idx                                 # [2, 5, 1, ?] — appended! repeat to keep generating
    F.softmax(last_logits / 0.3, dim=-1)  # low temp: one token dominates
    F.softmax(last_logits / 2.0, dim=-1)  # high temp: more spread out

  {green('Cold start (single char seed):')}
{sample_cold}

  {green('With a prompt:')}
{sample_prompted}

  {cyan('Try with the real model (after exec):')}
    prompt = torch.tensor([encode("First Citizen:\\n")], dtype=torch.long)
    out = model.generate(prompt, max_new_tokens=100, temperature=0.8)
    decode(out[0].tolist())             # readable text!

  {cyan('The flow:')}
    "First Citizen:\\n"                  # your prompt (input)
      ↓ encode()                        # text → token IDs
    [18, 27, 36, 37, 38, ...]          # numbers the model understands
      ↓ model.generate()               # predict next, append, repeat
    [18, 27, ..., 10, 25, 33, ...]     # original + generated IDs
      ↓ decode()                        # token IDs → text
    "First Citizen:\\nWhat is the..."    # readable output

  Not Shakespeare — but the model has clearly learned capital letters,
  colons after names, and word-ish shapes. More data + bigger model +
  more training = GPT.""",

korean=f"""\
  {cyan('자기회귀 샘플링:')} 현재 토큰을 입력하고, 마지막 위치의
  로짓을 받고, softmax로 확률 계산, 샘플링, 이어붙이기를 반복.

  {cyan('Type: (한 번에 하나씩 토큰 생성, 자기회귀 루프)')}
    import torch, torch.nn as nn, torch.nn.functional as F
    vocab_size, d_model = 10, 8
    tok_emb = nn.Embedding(vocab_size, d_model)
    head = nn.Linear(d_model, vocab_size)
    idx = torch.tensor([[2, 5, 1]])     # 3개 토큰으로 시작
    logits = head(tok_emb(idx))
    logits.shape                        # (1, 3, 10) — 모든 위치에서 예측
    last_logits = logits[:, -1, :]      # 생성에는 마지막 위치만 필요
    last_logits.shape                   # (1, 10) — 10개 가능한 다음 토큰의 점수
    probs = F.softmax(last_logits / 0.8, dim=-1)
    probs                               # 다음 토큰의 확률 분포
    next_id = torch.multinomial(probs, num_samples=1)
    next_id                             # 샘플링된 다음 토큰
    idx = torch.cat([idx, next_id], dim=1)
    idx                                 # [2, 5, 1, ?] — 추가됨! 반복하면 계속 생성
    F.softmax(last_logits / 0.3, dim=-1)  # 낮은 온도: 하나가 지배적
    F.softmax(last_logits / 2.0, dim=-1)  # 높은 온도: 더 고르게 분포

  {green('콜드 스타트 (단일 문자 시드):')}
{sample_cold}

  {green('프롬프트 사용:')}
{sample_prompted}

  {cyan('실제 모델로 시도 (exec 후):')}
    prompt = torch.tensor([encode("First Citizen:\\n")], dtype=torch.long)
    out = model.generate(prompt, max_new_tokens=100, temperature=0.8)
    decode(out[0].tolist())             # 읽을 수 있는 텍스트!

  {cyan('흐름:')}
    "First Citizen:\\n"                  # 프롬프트 (입력)
      ↓ encode()                        # 텍스트 → 토큰 ID
    [18, 27, 36, 37, 38, ...]          # 모델이 이해하는 숫자
      ↓ model.generate()               # 다음 예측, 이어붙이기, 반복
    [18, 27, ..., 10, 25, 33, ...]     # 원본 + 생성된 ID
      ↓ decode()                        # 토큰 ID → 텍스트
    "First Citizen:\\nWhat is the..."    # 읽을 수 있는 출력

  셰익스피어는 아니지만 — 대문자, 이름 뒤 콜론, 단어 비슷한 형태를
  분명히 학습. 더 많은 데이터 + 큰 모델 + 더 긴 학습 = GPT.""")

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

  {cyan('Each piece has a human parallel — press d to see them.')}

  {cyan('Next up (lesson 12):')} take a pre-trained model and adapt it
  to a new domain — {green('fine-tuning')} and {green('LoRA')}.""",

detail=f"""\
  {green('Each architecture maps to how you process language:')}

  {cyan('Tokenizer')} — chunking, not spelling
      You don't read l-e-t-t-e-r-s. You see "un|believe|able" as
      meaningful chunks. BPE does the same — finds reusable pieces
      so the model reads efficiently, not character by character.

  {cyan('Token + positional embeddings')} — what + where
      "Dog bites man" vs "Man bites dog" — same words, opposite
      meaning. You always track WHAT was said and WHERE in the
      sentence. Token embedding = what. Position embedding = where.

  {cyan('Self-attention')} — context-dependent understanding
      "The cat sat on the mat." You know "sat" = past tense of sit,
      not the SAT exam, because your brain scores "sat" against
      "cat" and "mat" (high relevance) vs "exam" (not present).
      Multi-head = doing this for grammar, meaning, and reference
      simultaneously, like parallel tracks in your mind.

  {cyan('Feed-forward block')} — private thinking after discussion
      Attention mixes information between words (group discussion).
      Then the FFN lets each word process what it learned alone
      — like sitting quietly after a meeting to form your own
      conclusions before the next round of conversation.

  {cyan('Residual connections + LayerNorm')} — notes + composure
      Residual = you keep your original understanding and add new
      insights on top. You never throw away what you already knew.
      LayerNorm = staying level-headed. Prevents you from getting
      too fixated on one thing or losing track of another.

  {cyan('Autoregressive generation')} — speaking one word at a time
      You say a word, then pick the next based on everything so
      far. You can't unsay words. Top-k/top-p = you don't always
      pick the most obvious word — sometimes you choose a slightly
      surprising one to keep things interesting.""",

korean=f"""\
  GPT, Claude, Llama를 구동하는 것과 동일한 아키텍처를 구현했습니다
  — 단지 작을 뿐. 모든 부품이 여기 있음:

  {green('✓')} 토크나이저 (문자 단위)          → 실제 LLM은 BPE 사용
  {green('✓')} 토큰 + 위치 임베딩             → 동일, RoPE일 수도
  {green('✓')} 멀티헤드 인과 셀프어텐션        → 동일, grouped-query일 수도
  {green('✓')} 피드포워드 블록                 → 동일, SwiGLU일 수도
  {green('✓')} 잔차 연결 + LayerNorm           → 동일, RMSNorm일 수도
  {green('✓')} 자기회귀 생성                   → 동일, top-k / top-p 추가

  {cyan('여기서 최첨단 모델까지의 차이는 규모일 뿐, 개념이 아님:')}
    MiniGPT (이것):   ~4만 파라미터,  500 학습 스텝
    GPT-2 small:       1.24억 파라미터
    Llama 3 8B:        80억 파라미터,  ~15T 토큰의 학습 데이터
    GPT-4 (추정):      ~1.7T 파라미터 (mixture-of-experts)

  {cyan('각 부품은 사람의 행동과 대응됨 — d를 눌러 확인.')}

  {cyan('다음 (레슨 12):')} 사전학습된 모델을 새로운 도메인에 적응시키기
  — {green('파인튜닝')}과 {green('LoRA')}.""",

detail_korean=f"""\
  {green('각 아키텍처는 사람이 언어를 처리하는 방식과 대응됨:')}

  {cyan('토크나이저')} — 철자가 아니라 덩어리로 읽기
      글자를 하나씩 읽지 않음. "믿|을|수|없|는"처럼 의미 있는
      덩어리로 봄. BPE도 마찬가지 — 재사용 가능한 조각을 찾아서
      모델이 효율적으로 읽게 함.

  {cyan('토큰 + 위치 임베딩')} — 무엇 + 어디
      "개가 사람을 물었다" vs "사람이 개를 물었다" — 같은 단어,
      반대 의미. 무엇이 말해졌는지와 문장 어디에 있는지를
      항상 추적함. 토큰 임베딩 = 무엇. 위치 임베딩 = 어디.

  {cyan('셀프어텐션')} — 문맥에 따른 이해
      "고양이가 매트 위에 앉았다." "앉았다"가 시험이 아니라
      "앉다"의 과거형인 걸 아는 이유는, 뇌가 "앉았다"를
      "고양이", "매트"와 비교해서 높은 관련성을 매기기 때문.
      멀티헤드 = 문법, 의미, 참조를 동시에 처리하는 것.
      마음속 병렬 트랙과 같음.

  {cyan('피드포워드 블록')} — 토론 후 혼자 생각하기
      어텐션은 단어 간 정보를 섞음 (그룹 토론).
      그 다음 FFN은 각 단어가 배운 것을 혼자 처리하게 함
      — 회의 후 조용히 앉아 다음 대화 전에
      자기만의 결론을 내리는 것과 같음.

  {cyan('잔차 연결 + LayerNorm')} — 메모 + 평정심
      잔차 = 원래 이해를 유지하고 새 통찰을 위에 추가.
      이미 알던 것을 절대 버리지 않음.
      LayerNorm = 평정심 유지. 한 가지에 너무 집착하거나
      다른 것을 놓치는 것을 방지.

  {cyan('자기회귀 생성')} — 한 단어씩 말하기
      한 단어를 말하고, 지금까지의 모든 것을 바탕으로
      다음 단어를 고름. 이미 한 말은 되돌릴 수 없음.
      Top-k/top-p = 항상 가장 뻔한 단어만 고르지 않음
      — 때로는 약간 의외의 단어를 골라 흥미롭게 만듦.""")

# ── Play ────────────────────────────────────────────────

player.play()
