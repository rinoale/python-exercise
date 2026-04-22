# AI Course — Supplementary Notes

Companion notes to `COURSE.md`. Topics that came up while working through
the lessons but don't belong inside any single lesson file.

---

## Why is Python the language of AI?

Short version: **Python is a driver's seat over a C++ engine.**

### Python isn't doing the math

When you write `x @ y` in PyTorch, Python spends microseconds dispatching
the call. The actual matrix multiply happens in **cuBLAS / CUDA kernels
written in C++ and CUDA C**.

PyTorch's own source is roughly 60% C++ and 35% Python. Same story for
TensorFlow and JAX. Python is the API layer — not the compute layer.

### Why Python won anyway

- **Iteration speed.** Research cycles: try idea → see result → try next
  idea. Python gives you seconds per cycle; compiled languages give you
  minutes. For rapid experimentation, this compounds massively.
- **Ecosystem inertia.** NumPy (2005) → SciPy → pandas → scikit-learn →
  Theano → TensorFlow → PyTorch. Twenty years of layered tooling.
  Hugging Face alone hosts 500k+ models with a Python-first API.
- **Glue code doesn't need speed.** The 1% of your program written in
  Python (load data, configure model, call `.backward()`) is not the
  bottleneck. The 99% inside PyTorch already *is* C++.

### But the trend is moving toward C++ / Rust — for inference

- `llama.cpp` — C++ inference, runs Llama on a MacBook
- `mistral.rs`, `candle` — Rust inference frameworks
- `vLLM`, `TensorRT-LLM` — C++ core under a Python API
- NVIDIA `TensorRT` — compiled C++ inference graphs
- Mojo (new language) — Python syntax with C++ performance

### The split that's emerging

| Stage | Language | Why |
|---|---|---|
| Training | Python + PyTorch | Research iteration wins |
| Serving / inference at scale | C++ / Rust | Latency and cost win |
| Edge / local | C++ | No Python runtime on your laptop |

### Would rewriting in C++ be faster?

- Faster to **develop**? No — slower, because you'd re-implement the
  ecosystem from scratch.
- Faster at **runtime**? No — PyTorch already *is* C++ underneath. The
  ceiling is the same.
- Porting a **trained** model to a C++ inference engine for production
  serving? That's real and increasingly common.

So Python's dominance is not about math libraries being better in
Python — it's about everything *around* the math: data loading, model
definition, experiment tracking, distributed training, sharing.
