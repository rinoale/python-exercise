"""
Lesson 13: Domain Data Pipeline — From Elasticsearch to Fine-tuning

Goal: learn the end-to-end workflow for building a domain-specialized
      generative AI model from your own data.

      1. Pull data from Elasticsearch
      2. Clean and prepare training text
      3. Fine-tune a model on your domain

This lesson uses the Mabinogi game data pipeline as a real example.
The code patterns apply to any domain: game logs, customer tickets,
internal docs, chat history, etc.

Type along:
    python3
    >>> from elasticsearch import Elasticsearch
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import math
import copy
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from lesson_player import LessonPlayer, cyan, green

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
torch.manual_seed(42)

player = LessonPlayer("Lesson 13: Domain Data Pipeline — From Elasticsearch to Fine-tuning")

# ── Step 1 ──────────────────────────────────────────────

player.add_step("Step 1: The Real-World Pipeline", f"""\
  Lessons 11-12 used tiny hardcoded strings as training data.
  Real projects start with {cyan('raw data in external systems')}.

  {green('Your pipeline:')}
      Elasticsearch (raw data)
          ↓
      Export & clean (Python scripts)
          ↓
      Training text file (one domain corpus)
          ↓
      Tokenize (chars, BPE, or existing tokenizer)
          ↓
      Fine-tune a model
          ↓
      Generate domain-specific output

  {cyan('What kind of data works for fine-tuning?')}
      - Chat logs       → model learns the community's language/slang
      - Item listings   → model learns item names, price patterns
      - Wiki/guide text → model learns game mechanics explanations
      - Q&A pairs       → model learns to answer domain questions

  {green('Rule of thumb:')}
      The model will generate text that LOOKS LIKE your training data.
      Chat logs in  → chat-style output out.
      Wiki text in  → wiki-style output out.
      Garbage in    → garbage out.""",

korean=f"""\
  레슨 11-12에서는 하드코딩된 짧은 문자열을 학습 데이터로 사용.
  실제 프로젝트는 {cyan('외부 시스템의 원시 데이터')}에서 시작합니다.

  {green('파이프라인:')}
      Elasticsearch (원시 데이터)
          ↓
      추출 & 정제 (파이썬 스크립트)
          ↓
      학습용 텍스트 파일 (도메인 코퍼스)
          ↓
      토큰화 (문자, BPE, 또는 기존 토크나이저)
          ↓
      모델 파인튜닝
          ↓
      도메인 특화 출력 생성

  {cyan('어떤 데이터가 파인튜닝에 적합한가?')}
      - 채팅 로그       → 커뮤니티 언어/슬랭 학습
      - 아이템 목록     → 아이템 이름, 가격 패턴 학습
      - 위키/가이드     → 게임 메커니즘 설명 학습
      - Q&A 쌍          → 도메인 질문에 답하기 학습

  {green('경험 법칙:')}
      모델은 학습 데이터처럼 생긴 텍스트를 생성합니다.
      채팅 로그 입력 → 채팅 스타일 출력.
      위키 텍스트 입력 → 위키 스타일 출력.
      쓰레기 입력    → 쓰레기 출력.""")

# ── Step 2 ──────────────────────────────────────────────

player.add_step("Step 2: Pulling Data from Elasticsearch", f"""\
  {cyan('Install:')}
      pip install elasticsearch

  {cyan('Connect:')}
      from elasticsearch import Elasticsearch
      es = Elasticsearch("http://localhost:9200")
      es.info()    # check connection

  {cyan('Search — pull chat messages:')}
      resp = es.search(
          index="mabinogi-chat",
          body={{
              "query": {{"match_all": {{}}}},
              "size": 1000,
              "sort": [{{"timestamp": "asc"}}]
          }}
      )
      hits = resp["hits"]["hits"]
      messages = [h["_source"]["message"] for h in hits]

  {cyan('Scroll — pull ALL data (beyond 10,000 limit):')}
      resp = es.search(
          index="mabinogi-chat",
          body={{"query": {{"match_all": {{}}}}, "size": 5000}},
          scroll="2m"
      )
      scroll_id = resp["_scroll_id"]
      all_hits = resp["hits"]["hits"]

      while True:
          resp = es.scroll(scroll_id=scroll_id, scroll="2m")
          if not resp["hits"]["hits"]:
              break
          all_hits.extend(resp["hits"]["hits"])

  {green('This gives you raw documents. Next: clean them.')}""",

detail=f"""\
  {green('Pulling price history:')}
      resp = es.search(
          index="mabinogi-prices",
          body={{
              "query": {{
                  "bool": {{
                      "must": [
                          {{"term": {{"item_name": "Spirit Weapon"}}}},
                          {{"range": {{"timestamp": {{"gte": "2024-01-01"}}}}}}
                      ]
                  }}
              }},
              "size": 10000,
              "sort": [{{"timestamp": "asc"}}]
          }}
      )

  {green('Aggregation — average price per item per month:')}
      resp = es.search(
          index="mabinogi-prices",
          body={{
              "size": 0,
              "aggs": {{
                  "by_item": {{
                      "terms": {{"field": "item_name", "size": 100}},
                      "aggs": {{
                          "monthly": {{
                              "date_histogram": {{
                                  "field": "timestamp",
                                  "calendar_interval": "month"
                              }},
                              "aggs": {{
                                  "avg_price": {{"avg": {{"field": "price"}}}}
                              }}
                          }}
                      }}
                  }}
              }}
          }}
      )

  {cyan('Tip:')} use aggregations for analytics, scroll for full export.""",

korean=f"""\
  {cyan('설치:')}
      pip install elasticsearch

  {cyan('연결:')}
      from elasticsearch import Elasticsearch
      es = Elasticsearch("http://localhost:9200")
      es.info()    # 연결 확인

  {cyan('검색 — 채팅 메시지 가져오기:')}
      resp = es.search(
          index="mabinogi-chat",
          body={{
              "query": {{"match_all": {{}}}},
              "size": 1000,
              "sort": [{{"timestamp": "asc"}}]
          }}
      )
      hits = resp["hits"]["hits"]
      messages = [h["_source"]["message"] for h in hits]

  {cyan('스크롤 — 전체 데이터 가져오기 (10,000개 제한 초과):')}
      resp = es.search(
          index="mabinogi-chat",
          body={{"query": {{"match_all": {{}}}}, "size": 5000}},
          scroll="2m"
      )
      scroll_id = resp["_scroll_id"]
      all_hits = resp["hits"]["hits"]

      while True:
          resp = es.scroll(scroll_id=scroll_id, scroll="2m")
          if not resp["hits"]["hits"]:
              break
          all_hits.extend(resp["hits"]["hits"])

  {green('이렇게 하면 원시 문서를 얻음. 다음: 정제.')}'""",

detail_korean=f"""\
  {green('가격 이력 가져오기:')}
      resp = es.search(
          index="mabinogi-prices",
          body={{
              "query": {{
                  "bool": {{
                      "must": [
                          {{"term": {{"item_name": "Spirit Weapon"}}}},
                          {{"range": {{"timestamp": {{"gte": "2024-01-01"}}}}}}
                      ]
                  }}
              }},
              "size": 10000,
              "sort": [{{"timestamp": "asc"}}]
          }}
      )

  {green('집계 — 아이템별 월평균 가격:')}
      resp = es.search(
          index="mabinogi-prices",
          body={{
              "size": 0,
              "aggs": {{
                  "by_item": {{
                      "terms": {{"field": "item_name", "size": 100}},
                      "aggs": {{
                          "monthly": {{
                              "date_histogram": {{
                                  "field": "timestamp",
                                  "calendar_interval": "month"
                              }},
                              "aggs": {{
                                  "avg_price": {{"avg": {{"field": "price"}}}}
                              }}
                          }}
                      }}
                  }}
              }}
          }}
      )

  {cyan('팁:')} 분석에는 집계, 전체 내보내기에는 스크롤 사용.""")

# ── Step 3 ──────────────────────────────────────────────

player.add_step("Step 3: Cleaning Data for Training", f"""\
  Raw data is never ready for training. You need to clean it.

  {cyan('Common cleaning steps:')}

      1. {green('Remove noise')}
         - Bot messages, system announcements, empty messages
         - Repeated spam ("WTS WTS WTS WTS WTS...")
         - Messages that are just URLs or numbers

      2. {green('Normalize text')}
         - Consistent encoding (UTF-8)
         - Strip excessive whitespace
         - Decide: keep or remove emojis/special chars

      3. {green('Format for training')}
         - One document per line, or structured format
         - Add separators between conversations

  {cyan('Example cleaning script:')}
      import re

      def clean_message(msg):
          if not msg or len(msg) < 3:
              return None
          if msg.startswith("[System]"):
              return None
          msg = re.sub(r'http\\S+', '', msg)       # remove URLs
          msg = re.sub(r'(.+?)\\1{{3,}}', r'\\1', msg)  # remove spam repeats
          msg = msg.strip()
          if len(msg) < 3:
              return None
          return msg

      cleaned = [clean_message(m) for m in messages]
      cleaned = [m for m in cleaned if m is not None]

  {cyan('Save to file:')}
      with open("mabinogi_chat.txt", "w") as f:
          f.write("\\n".join(cleaned))""",

detail=f"""\
  {green('Formatting strategies:')}

  {cyan('Strategy 1: Plain text')} (for pretraining / language modeling)
      Just concatenate all text. The model learns the overall style.

      player1: anyone selling black dragon knight?
      player2: got one, 50m
      player1: deal, ch7 dunbarton
      player2: omw

  {cyan('Strategy 2: Structured pairs')} (for instruction fine-tuning)
      Format as question-answer pairs.

      Q: What is the best enchant for a knuckle?
      A: Smash enchant for damage, or Piercing for PvP.

      Q: How much is a Celtic Royal Knight Armor?
      A: Usually 200-300M depending on stats and enchants.

  {cyan('Strategy 3: Price narration')} (for price-aware generation)
      Convert price data into natural language.

      Item: Spirit Weapon
      Jan 2024: 10M avg, Feb: 12M (+20%), Mar: 15M (+25%)
      Trend: rising. Likely cause: new content update.

  {green('Which strategy to pick:')}
      Plain text → good for generating chat-style responses
      Structured pairs → good for Q&A / assistant behavior
      Mix both → the model learns both styles""",

korean=f"""\
  원시 데이터는 학습에 바로 쓸 수 없음. 정제가 필요합니다.

  {cyan('일반적인 정제 단계:')}

      1. {green('노이즈 제거')}
         - 봇 메시지, 시스템 공지, 빈 메시지
         - 반복 스팸 ("WTS WTS WTS WTS WTS...")
         - URL이나 숫자만 있는 메시지

      2. {green('텍스트 정규화')}
         - 일관된 인코딩 (UTF-8)
         - 과도한 공백 제거
         - 이모지/특수문자 유지 또는 제거 결정

      3. {green('학습용 형식')}
         - 한 줄에 하나의 문서, 또는 구조화된 형식
         - 대화 사이에 구분자 추가

  {cyan('정제 스크립트 예시:')}
      import re

      def clean_message(msg):
          if not msg or len(msg) < 3:
              return None
          if msg.startswith("[System]"):
              return None
          msg = re.sub(r'http\\S+', '', msg)       # URL 제거
          msg = re.sub(r'(.+?)\\1{{3,}}', r'\\1', msg)  # 스팸 반복 제거
          msg = msg.strip()
          if len(msg) < 3:
              return None
          return msg

  {cyan('파일로 저장:')}
      with open("mabinogi_chat.txt", "w") as f:
          f.write("\\n".join(cleaned))""",

detail_korean=f"""\
  {green('포맷팅 전략:')}

  {cyan('전략 1: 일반 텍스트')} (사전학습 / 언어 모델링용)
      모든 텍스트를 이어붙임. 모델이 전체적인 스타일을 학습.

      player1: 흑룡기사 파는 사람?
      player2: 있어, 5000만
      player1: 거래, 7채 던바튼
      player2: 감

  {cyan('전략 2: 구조화된 쌍')} (지시 파인튜닝용)
      질문-답변 쌍으로 구성.

      Q: 너클 최고 인챈트가 뭐야?
      A: 데미지는 스매시 인챈트, PvP는 피어싱.

      Q: 켈틱 로얄 나이트 아머 얼마야?
      A: 보통 2~3억, 스탯과 인챈트에 따라 다름.

  {cyan('전략 3: 가격 설명')} (가격 인식 생성용)
      가격 데이터를 자연어로 변환.

      아이템: 스피릿 웨폰
      2024년 1월: 평균 1000만, 2월: 1200만 (+20%), 3월: 1500만 (+25%)
      추세: 상승. 원인: 새 콘텐츠 업데이트.

  {green('어떤 전략을 선택할지:')}
      일반 텍스트 → 채팅 스타일 응답 생성에 적합
      구조화된 쌍 → Q&A / 어시스턴트 동작에 적합
      둘 다 혼합 → 모델이 두 스타일 모두 학습""")

# ── Step 4 ──────────────────────────────────────────────

player.add_step("Step 4: Data Quality > Data Quantity", f"""\
  {cyan('Common mistake:')} throw ALL your data at the model.
  More data isn't always better — cleaner data is.

  {green('The quality pyramid:')}

      ┌───────────────┐
      │  Curated Q&A  │  ← best: hand-picked, high quality
      │   (hundreds)  │     teaches specific behavior
      ├───────────────┤
      │  Clean chat   │  ← good: filtered, deduplicated
      │  (thousands)  │     teaches language/style
      ├───────────────┤
      │   Raw dump    │  ← risky: spam, noise, toxic content
      │  (millions)   │     can teach bad patterns
      └───────────────┘

  {cyan('Practical approach — use layers:')}
      1. Pretrain (or use existing model) on general text
      2. Fine-tune on your clean chat corpus (style/vocabulary)
      3. Fine-tune again on curated Q&A (behavior/accuracy)

  {green('How much data do you need?')}
      Character-level model (lesson 11 style):  10K-100K characters
      LoRA fine-tune on small model:             1K-10K examples
      Full fine-tune on 7B model:                10K-100K examples

  {cyan('For your Mabinogi data:')}
      Chat history → filter to high-quality conversations
      Price data   → convert to natural language descriptions
      Combine both → the model learns game vocabulary + price awareness""",

korean=f"""\
  {cyan('흔한 실수:')} 모든 데이터를 모델에 다 넣기.
  더 많은 데이터가 항상 좋은 것은 아님 — 더 깨끗한 데이터가 좋음.

  {green('품질 피라미드:')}

      ┌───────────────┐
      │  엄선된 Q&A   │  ← 최고: 직접 선별, 고품질
      │   (수백 개)   │     특정 행동을 가르침
      ├───────────────┤
      │  정제된 채팅  │  ← 양호: 필터링, 중복 제거
      │  (수천 개)    │     언어/스타일 학습
      ├───────────────┤
      │   원시 덤프   │  ← 위험: 스팸, 노이즈, 유해 컨텐츠
      │  (수백만 개)  │     나쁜 패턴을 학습할 수 있음
      └───────────────┘

  {cyan('실용적 접근 — 단계별:')}
      1. 사전학습 (또는 기존 모델 사용) — 일반 텍스트
      2. 정제된 채팅 코퍼스로 파인튜닝 (스타일/어휘)
      3. 엄선된 Q&A로 다시 파인튜닝 (행동/정확도)

  {green('얼마나 많은 데이터가 필요한가?')}
      문자 단위 모델 (레슨 11 스타일):  1만-10만 문자
      소형 모델 LoRA 파인튜닝:          1천-1만 예시
      7B 모델 풀 파인튜닝:              1만-10만 예시""")

# ── Step 5: live demo with simulated data ───────────────

MABINOGI_CHAT = """\
anyone selling a celtic royal knight armor?
got one, +20 max damage roll, 250m
that's fair, what channel?
ch7 dunbarton, ign Elsinore
omw, thanks
np, enjoy the armor
does anyone know when the next update drops?
probably next week, they usually patch on thursdays
the spirit weapon market is crazy right now
yeah prices doubled since the new dungeon came out
I bought mine for 10m last month, now they're 25m
should have stocked up lol
WTB> enchant scroll rank 1 smash, paying 15m
I have one, whisper me
looking for a party for peaca abyss
what's your total level?
cumulative 28k, got full celtic set
that should be fine, we need one more DPS
I can join, got a chain blade build
perfect, meet at peaca entrance in 5
the new raid boss is insane, anyone cleared it yet?
my guild got it down to 30% but we wiped
you need at least 8 people with divine weapons
the enrage timer is the real problem, need more DPS
"""

MABINOGI_PRICES = """\
Celtic Royal Knight Armor: Jan 200M, Feb 220M, Mar 250M. Rising trend due to limited supply from raids.
Spirit Weapon: Jan 10M, Feb 15M, Mar 25M. Sharp increase after new dungeon content added spirit drops.
Rank 1 Smash Enchant: Jan 12M, Feb 14M, Mar 15M. Stable with slight upward trend.
Black Dragon Knight Armor: Jan 80M, Feb 75M, Mar 70M. Declining as newer gear outclasses it.
Divine Weapon Fragment: Jan 5M, Feb 8M, Mar 12M. New raid boss drops these, demand exceeds supply.
Chain Blade Reforge: Jan 30M, Feb 28M, Mar 32M. Fluctuating based on PvP season activity.
"""

MABINOGI_QA = """\
Q: What is the best armor for end-game content?
A: Celtic Royal Knight Armor is currently the top choice. It offers the highest defense and magic defense stats, plus special set bonuses when paired with Celtic accessories.

Q: How much is a Spirit Weapon worth?
A: Prices have been rising sharply. Currently around 25M, up from 10M two months ago. The new dungeon update increased demand significantly.

Q: What level should I be for Peaca Abyss?
A: Recommended cumulative level is 25k+. You'll want a full party of 4-8 players with end-game gear. Divine weapons help a lot with the boss mechanics.

Q: Is chain blade good for PvP?
A: Chain blade is one of the top PvP weapons right now. The dorcha snatch combo is very strong. Pair it with Rank 1 Smash enchant for maximum burst damage.

Q: When do patches usually happen?
A: Mabinogi typically patches on Thursdays. Major content updates happen every 1-2 months, with maintenance usually scheduled for early morning hours.
"""

ALL_TEXT = MABINOGI_CHAT + "\n" + MABINOGI_PRICES + "\n" + MABINOGI_QA

ALL_CHARS = sorted(set(ALL_TEXT))
vocab_size = len(ALL_CHARS)
stoi = {ch: i for i, ch in enumerate(ALL_CHARS)}
itos = {i: ch for ch, i in stoi.items()}


def encode(s):
    return [stoi[ch] for ch in s if ch in stoi]


def decode(ids):
    return "".join(itos[i] for i in ids)


data_stats = f"""\
      Chat messages:  {len(MABINOGI_CHAT)} chars
      Price data:     {len(MABINOGI_PRICES)} chars
      Q&A pairs:      {len(MABINOGI_QA)} chars
      Total:          {len(ALL_TEXT)} chars
      Vocabulary:     {vocab_size} unique characters"""

player.add_step("Step 5: Prepare the Training Corpus", f"""\
  In a real project, you'd export from Elasticsearch (step 2-3).
  Here we simulate with sample Mabinogi data to show the full flow.

  {cyan('Three data sources combined:')}
{data_stats}

  {cyan('The training text includes:')}
      - Player chat (buying/selling, party finding, game discussion)
      - Price history narrated as text
      - Q&A pairs about game mechanics

  {cyan('Building the vocabulary:')}
      chars = sorted(set(all_text))
      stoi = {{ch: i for i, ch in enumerate(chars)}}
      itos = {{i: ch for ch, i in stoi.items()}}

  {green('This is the same tokenization from lesson 11,')}
  just applied to your domain data instead of Shakespeare.

  {cyan('In production you would:')}
      - Use the pretrained model's existing tokenizer (BPE)
      - NOT build a new vocabulary from scratch
      - Character-level is fine for learning, not for real quality""",

korean=f"""\
  실제 프로젝트에서는 Elasticsearch에서 내보내기 (2-3단계).
  여기서는 전체 흐름을 보여주기 위해 마비노기 샘플 데이터를 시뮬레이션.

  {cyan('세 가지 데이터 소스 결합:')}
{data_stats}

  {cyan('학습 텍스트 포함 내용:')}
      - 플레이어 채팅 (거래, 파티 모집, 게임 토론)
      - 텍스트로 서술된 가격 히스토리
      - 게임 메커니즘 Q&A 쌍

  {cyan('어휘 구축:')}
      chars = sorted(set(all_text))
      stoi = {{ch: i for i, ch in enumerate(chars)}}
      itos = {{i: ch for ch, i in stoi.items()}}

  {green('레슨 11과 같은 토큰화,')}
  셰익스피어 대신 여러분의 도메인 데이터에 적용한 것.

  {cyan('실무에서는:')}
      - 사전학습 모델의 기존 토크나이저(BPE) 사용
      - 처음부터 새 어휘를 만들지 않음
      - 문자 단위는 학습용으로 괜찮지만, 실제 품질에는 부적합""")

# ── Step 6: model + training ────────────────────────────

SEQ_LEN = 64
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


data_tensor = torch.tensor(encode(ALL_TEXT), dtype=torch.long)


def get_batch(batch_size=32):
    ix = torch.randint(0, len(data_tensor) - SEQ_LEN - 1, (batch_size,))
    x = torch.stack([data_tensor[i:i + SEQ_LEN] for i in ix])
    y = torch.stack([data_tensor[i + 1:i + SEQ_LEN + 1] for i in ix])
    return x, y


model = MiniGPT(vocab_size, D_MODEL, N_HEADS, N_LAYERS, SEQ_LEN)
total_params = sum(p.numel() for p in model.parameters())
opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
losses = []

model.train()
for step in range(800):
    x, y = get_batch()
    logits = model(x)
    loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))
    opt.zero_grad()
    loss.backward()
    opt.step()
    losses.append(loss.item())

plt.figure(figsize=(10, 4))
plt.plot(losses, alpha=0.3, color='blue')
window = 50
if len(losses) >= window:
    smoothed = [sum(losses[i:i+window])/window for i in range(len(losses)-window+1)]
    plt.plot(range(window-1, len(losses)), smoothed, color='blue', linewidth=2, label='smoothed')
plt.xlabel("Training step")
plt.ylabel("Loss")
plt.title("Training on Mabinogi Data")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, "images", "13_mabinogi_training.png"))
plt.close()


def sample(prompt, n=200):
    model.eval()
    with torch.no_grad():
        ctx = torch.tensor([encode(prompt)], dtype=torch.long)
        out = model.generate(ctx, max_new_tokens=n)
        return decode(out[0].tolist())


gen_chat = sample("anyone selling ", n=150)
gen_price = sample("Spirit Weapon:", n=150)
gen_qa = sample("Q: What is the best", n=200)

player.add_step("Step 6: Train on Domain Data", f"""\
  Same MiniGPT architecture from lesson 11, now trained on
  Mabinogi data.  800 steps, takes ~30 seconds on CPU.

  {green('Model:')} {total_params:,} parameters
  {green('Training data:')} {len(ALL_TEXT):,} characters
  {green('Final loss:')} {losses[-1]:.3f}
  {green('Loss plot:')} images/13_mabinogi_training.png

  {cyan("Generate from 'anyone selling ':")}
{gen_chat}

  {cyan("Generate from 'Spirit Weapon:':")}
{gen_price}

  {cyan("Generate from 'Q: What is the best':")}
{gen_qa}

  {green('The output is rough')} — this is a tiny model on tiny data.
  But notice it picks up game vocabulary: item names, prices,
  player interaction patterns, Q&A format.""",

korean=f"""\
  레슨 11과 같은 MiniGPT 아키텍처, 이번엔 마비노기 데이터로 학습.
  800스텝, CPU에서 ~30초.

  {green('모델:')} {total_params:,} 파라미터
  {green('학습 데이터:')} {len(ALL_TEXT):,}자
  {green('최종 손실:')} {losses[-1]:.3f}
  {green('손실 그래프:')} images/13_mabinogi_training.png

  {cyan("'anyone selling '로 생성:")}
{gen_chat}

  {cyan("'Spirit Weapon:'로 생성:")}
{gen_price}

  {cyan("'Q: What is the best'로 생성:")}
{gen_qa}

  {green('출력이 거칠지만')} — 작은 모델에 적은 데이터.
  하지만 게임 어휘를 습득: 아이템 이름, 가격,
  플레이어 상호작용 패턴, Q&A 형식.""")

# ── Step 7 ──────────────────────────────────────────────

player.add_step("Step 7: From Toy to Production", f"""\
  This lesson's model is a proof of concept.
  Here's how to make it actually useful.

  {cyan('Level 1: Better toy model')} (what you can do now)
      - Export MORE data from Elasticsearch (100K+ characters)
      - Train longer (2000+ steps)
      - Increase model size (d_model=128, n_layers=4)
      - Use longer sequences (seq_len=128)

  {cyan('Level 2: Fine-tune a real model')} (recommended path)
      pip install transformers peft datasets

      from transformers import AutoModelForCausalLM, AutoTokenizer
      from peft import LoraConfig, get_peft_model

      model = AutoModelForCausalLM.from_pretrained("gpt2")
      tokenizer = AutoTokenizer.from_pretrained("gpt2")

      lora_config = LoraConfig(
          r=16,
          lora_alpha=32,
          target_modules=["c_attn", "c_proj"],
          lora_dropout=0.05,
      )
      model = get_peft_model(model, lora_config)

      # prepare your Mabinogi text as a HuggingFace Dataset
      # train with Trainer or a manual loop
      # save only the adapter (~few MB)

  {cyan('Level 3: Fine-tune a larger model')} (best quality)
      Use a 1-3B model (Llama 3.2 1B, Phi-3-mini, Qwen2.5)
      with LoRA + quantization (QLoRA) to fit on a single GPU.
      Format your data as instruction pairs for chat behavior.""",

detail=f"""\
  {green('Choosing a base model:')}

      Model               Size    Good for
      gpt2                124M    Learning, quick experiments
      gpt2-medium         355M    Better quality, still fast
      Llama-3.2-1B        1B      Good quality, fits on 1 GPU
      Phi-3-mini          3.8B    Strong reasoning, needs QLoRA
      Llama-3.1-8B        8B      Best quality, needs good GPU

  {green('QLoRA — fine-tune big models on small GPUs:')}
      from transformers import BitsAndBytesConfig
      bnb_config = BitsAndBytesConfig(
          load_in_4bit=True,
          bnb_4bit_compute_dtype=torch.float16,
      )
      model = AutoModelForCausalLM.from_pretrained(
          "meta-llama/Llama-3.2-1B",
          quantization_config=bnb_config,
      )
      # 1B model: ~2GB VRAM instead of ~4GB
      # 8B model: ~6GB VRAM instead of ~16GB

  {green('Data format for instruction tuning:')}
      [
        {{
          "instruction": "What is a Spirit Weapon worth?",
          "output": "Currently around 25M, up from 10M..."
        }},
        ...
      ]""",

korean=f"""\
  이 레슨의 모델은 개념 증명.
  실제로 유용하게 만드는 방법:

  {cyan('레벨 1: 더 나은 토이 모델')} (지금 할 수 있는 것)
      - ES에서 더 많은 데이터 내보내기 (10만+ 문자)
      - 더 오래 학습 (2000+ 스텝)
      - 모델 크기 키우기 (d_model=128, n_layers=4)
      - 더 긴 시퀀스 (seq_len=128)

  {cyan('레벨 2: 실제 모델 파인튜닝')} (권장 경로)
      pip install transformers peft datasets

      from transformers import AutoModelForCausalLM, AutoTokenizer
      from peft import LoraConfig, get_peft_model

      model = AutoModelForCausalLM.from_pretrained("gpt2")
      tokenizer = AutoTokenizer.from_pretrained("gpt2")

      lora_config = LoraConfig(
          r=16,
          lora_alpha=32,
          target_modules=["c_attn", "c_proj"],
          lora_dropout=0.05,
      )
      model = get_peft_model(model, lora_config)

  {cyan('레벨 3: 더 큰 모델 파인튜닝')} (최고 품질)
      1-3B 모델 (Llama 3.2 1B, Phi-3-mini, Qwen2.5)에
      LoRA + 양자화 (QLoRA)로 단일 GPU에서 학습.
      채팅 행동을 위해 instruction pair 형식으로 데이터 준비.""",

detail_korean=f"""\
  {green('베이스 모델 선택:')}

      모델               크기    용도
      gpt2                124M    학습용, 빠른 실험
      gpt2-medium         355M    더 나은 품질, 여전히 빠름
      Llama-3.2-1B        1B      좋은 품질, GPU 1개로 가능
      Phi-3-mini          3.8B    강한 추론, QLoRA 필요
      Llama-3.1-8B        8B      최고 품질, 좋은 GPU 필요

  {green('QLoRA — 작은 GPU로 큰 모델 파인튜닝:')}
      from transformers import BitsAndBytesConfig
      bnb_config = BitsAndBytesConfig(
          load_in_4bit=True,
          bnb_4bit_compute_dtype=torch.float16,
      )
      model = AutoModelForCausalLM.from_pretrained(
          "meta-llama/Llama-3.2-1B",
          quantization_config=bnb_config,
      )
      # 1B 모델: ~2GB VRAM (원래 ~4GB)
      # 8B 모델: ~6GB VRAM (원래 ~16GB)

  {green('지시 튜닝용 데이터 형식:')}
      [
        {{
          "instruction": "스피릿 웨폰 얼마야?",
          "output": "현재 약 2500만, 1000만에서 올랐어..."
        }},
        ...
      ]""")

# ── Step 8 ──────────────────────────────────────────────

player.add_step("Step 8: The Full Elasticsearch-to-Model Pipeline", f"""\
  {green('Complete workflow for your Mabinogi project:')}

  {cyan('1. Export data')}
      python export_chat.py      # scroll through ES → chat.txt
      python export_prices.py    # aggregate prices → prices.txt

  {cyan('2. Clean data')}
      python clean_data.py       # remove noise, normalize, deduplicate
      # output: mabinogi_corpus.txt

  {cyan('3. Create training pairs')} (for instruction tuning)
      python make_pairs.py       # convert to Q&A format
      # output: mabinogi_qa.jsonl

  {cyan('4. Fine-tune')}
      python finetune.py         # LoRA fine-tune on your data
      # output: adapter_weights/ (~10MB)

  {cyan('5. Inference')}
      python generate.py "What's the price trend for Spirit Weapons?"
      # loads base model + adapter → generates answer

  {green('What you end up with:')}
      A model that speaks Mabinogi — knows item names, price trends,
      community slang, and game mechanics. Small adapter file that
      sits on top of a public base model.

  {green('Limitations to be honest about:')}
      - Small models hallucinate facts. Don't trust prices as truth.
      - For factual accuracy, combine with RAG (retrieve real data
        from ES at query time, feed it as context to the model).
      - The model reflects its training data, including any bias.""",

korean=f"""\
  {green('마비노기 프로젝트의 전체 워크플로우:')}

  {cyan('1. 데이터 내보내기')}
      python export_chat.py      # ES 스크롤 → chat.txt
      python export_prices.py    # 가격 집계 → prices.txt

  {cyan('2. 데이터 정제')}
      python clean_data.py       # 노이즈 제거, 정규화, 중복 제거
      # 출력: mabinogi_corpus.txt

  {cyan('3. 학습 쌍 생성')} (instruction 튜닝용)
      python make_pairs.py       # Q&A 형식으로 변환
      # 출력: mabinogi_qa.jsonl

  {cyan('4. 파인튜닝')}
      python finetune.py         # 데이터로 LoRA 파인튜닝
      # 출력: adapter_weights/ (~10MB)

  {cyan('5. 추론')}
      python generate.py "스피릿 웨폰 가격 추세는?"
      # 기본 모델 + 어댑터 로드 → 답변 생성

  {green('결과물:')}
      마비노기를 아는 모델 — 아이템 이름, 가격 추세, 커뮤니티
      슬랭, 게임 메커니즘을 알고 있음. 공개 기본 모델 위에
      올라가는 작은 어댑터 파일.

  {green('솔직히 말하면 한계:')}
      - 작은 모델은 사실을 꾸며냄. 가격을 진실로 믿지 마세요.
      - 사실 정확성을 위해 RAG와 결합 (쿼리 시 ES에서 실제 데이터를
        검색하여 모델에 컨텍스트로 제공).
      - 모델은 학습 데이터를 반영, 편향 포함.""")

# ── Step 9 ──────────────────────────────────────────────

player.add_step("Step 9: Summary — The Complete Course", f"""\
  {green('You\'ve completed the full journey:')}

      Lessons 01-03:  Linear regression — y = ax + b
      Lessons 04-08:  Classical ML — trees, clustering, preprocessing
      Lessons 09-10:  Neural networks — from sklearn to PyTorch
      Lesson 11:      Transformer — attention, the architecture behind LLMs
      Lesson 12:      Fine-tuning — LoRA, adapting pretrained models
      Lesson 13:      Pipeline — from raw data to domain-specialized AI

  {green('What you can now do:')}
      ✓ Understand how ML models learn from data
      ✓ Build and train neural networks with PyTorch
      ✓ Understand transformer architecture (what GPT/Claude are)
      ✓ Fine-tune models on your own domain data
      ✓ Design a data pipeline from Elasticsearch to generation

  {cyan('Your next steps for the Mabinogi project:')}
      1. Export your real data from Elasticsearch
      2. Clean it using the patterns from step 3
      3. Start with gpt2 + LoRA (quick experiment)
      4. Graduate to a larger model when you want better quality
      5. Add RAG for factual accuracy on prices

  {green('The foundation you built here applies to everything.')}
  Whether you're building a game assistant, a customer service bot,
  or a code generator — the pipeline is the same:
  data → clean → tokenize → train → generate.""",

korean=f"""\
  {green('전체 여정을 완료했습니다:')}

      레슨 01-03:  선형 회귀 — y = ax + b
      레슨 04-08:  고전 ML — 트리, 클러스터링, 전처리
      레슨 09-10:  신경망 — sklearn에서 PyTorch까지
      레슨 11:     트랜스포머 — 어텐션, LLM의 핵심 아키텍처
      레슨 12:     파인튜닝 — LoRA, 사전학습 모델 적응
      레슨 13:     파이프라인 — 원시 데이터에서 도메인 특화 AI까지

  {green('이제 할 수 있는 것:')}
      ✓ ML 모델이 데이터에서 배우는 방법 이해
      ✓ PyTorch로 신경망 구축 및 학습
      ✓ 트랜스포머 아키텍처 이해 (GPT/Claude의 원리)
      ✓ 자체 도메인 데이터로 모델 파인튜닝
      ✓ Elasticsearch에서 생성까지 데이터 파이프라인 설계

  {cyan('마비노기 프로젝트 다음 단계:')}
      1. Elasticsearch에서 실제 데이터 내보내기
      2. 3단계 패턴으로 정제
      3. gpt2 + LoRA로 시작 (빠른 실험)
      4. 더 높은 품질이 필요하면 더 큰 모델로 전환
      5. 가격 정확성을 위해 RAG 추가

  {green('여기서 쌓은 기초는 모든 곳에 적용됩니다.')}
  게임 어시스턴트, 고객 서비스 봇, 코드 생성기 — 무엇을
  만들든 파이프라인은 동일:
  데이터 → 정제 → 토큰화 → 학습 → 생성.""")

# ── Play ────────────────────────────────────────────────

player.play()
