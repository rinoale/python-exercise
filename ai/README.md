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

---

## Terms — Mathematical & ML Vocabulary

All terms from lessons 01–12.

### Statistics & Probability

| Math Term | ML Term / Definition |
|---|---|
| Mean | Average of all values |
| Median | Middle value; robust to outliers |
| Mode | Most frequent value |
| Variance | How spread out values are from the mean |
| Standard Deviation | √variance; in original units |
| 68-95-99.7 Rule | ~68% within ±1σ, ~95% within ±2σ, ~99.7% within ±3σ |
| Covariance | How x and y move together |
| Correlation | Covariance / (σx × σy); always between −1 and +1 |

### Linear Algebra & Calculus

| Math Term | ML Term / Definition |
|---|---|
| Dot Product | Q·K = similarity measure |
| Gradient | How much the loss changes when a weight changes |
| Chain Rule | dLoss/dW₁ = dLoss/dOut × dOut/dH₂ × … |
| Low-Rank Decomposition | W = W_pretrained + α·(B @ A) |
| Euclidean Distance | d = √((x₁−x₂)² + (y₁−y₂)²) |
| Vectorization | Math on entire arrays at once instead of looping |

### Regression

| Math Term | ML Term / Definition |
|---|---|
| Slope | Weight; rise / run = Δy / Δx |
| Intercept | Bias; where the line crosses the y-axis |
| Residual | actual_y − predicted_y |
| Ordinary Least Squares (OLS) | Minimize Σ(residual²) |
| Sum of Squared Residuals (SSR) | Model's total error |
| Total Sum of Squares (TSS) | Baseline (mean-only) model's error |
| R² Score | 1 − SSR/TSS; how much variation the model explains |
| Polynomial | Model with higher-degree terms: x², x³, … |

### Error Metrics

| Math Term | ML Term / Definition |
|---|---|
| Loss | How wrong the predictions are |
| Mean Absolute Error (MAE) | Average of \|error\| |
| Mean Squared Error (MSE) | Average of error² |
| Root Mean Squared Error (RMSE) | √MSE |
| Cross-Entropy Loss | High probability on correct answer → low loss |

### Activation Functions

| Math Term | ML Term / Definition |
|---|---|
| Sigmoid | σ(x) = 1/(1+e⁻ˣ); squashes to (0, 1) |
| Softmax | Generalizes sigmoid to K classes; outputs sum to 1 |
| ReLU | max(0, x) |
| Tanh | Squashes to (−1, 1) |
| GELU | GELU(x) ≈ x × sigmoid(1.702x) |

### Classification

| Math Term | ML Term / Definition |
|---|---|
| Classification | Predict a category |
| Binary Classification | 2 categories |
| Multi-class Classification | 3+ categories |
| Logistic Regression | Classifier: score → probability → class |
| Accuracy | Correct / total |
| Confusion Matrix | Shows which classes get mixed up |
| Precision | Of predicted X, how many were actually X? |
| Recall | Of all actual X, how many did I find? |
| F1-score | Harmonic mean of precision and recall |
| Decision Boundary | Lines/curves that separate classes |

### Trees & Ensembles

| Math Term | ML Term / Definition |
|---|---|
| Decision Tree | Flowchart of yes/no questions |
| Split | Divide data by feature/threshold |
| Leaf Node | Terminal node; makes a prediction |
| Gini Impurity | 1 − Σ(pᵢ²); how mixed a group is |
| Information Gain | Parent Gini − weighted children Gini |
| Entropy | −Σ(pᵢ log₂ pᵢ); another impurity measure |
| Ensemble Learning | Combining multiple weak models into one strong model |
| Bagging | Random subsets → average/vote; reduces variance |
| Boosting | Sequential models fixing previous errors; reduces bias |
| Stacking | Different model types + meta-model on outputs |
| Random Forest | Many trees voting together |
| Feature Importance | How much each feature reduces impurity |

### Clustering

| Math Term | ML Term / Definition |
|---|---|
| Supervised Learning | Has labeled data: (input, correct_answer) |
| Unsupervised Learning | Unlabeled data: just inputs, no answers |
| Clustering | Grouping similar data points together |
| K-Means | Place K centroids → assign nearest → move to mean → repeat |
| Centroid | Center point (mean) of a cluster |
| Convergence | When the algorithm stops changing |
| Inertia | Σ(distance from each point to its centroid)² |
| Elbow Method | Plot inertia vs K, find the bend |
| Silhouette Score | s = (b−a)/max(a,b); cluster quality |
| DBSCAN | Density-based; finds clusters of any shape |
| GMM | Like K-Means but with soft (probabilistic) assignments |

### Neural Networks

| Math Term | ML Term / Definition |
|---|---|
| Neuron | output = weight × input + bias |
| Neural Network | Layers of simple math operations |
| Activation Function | Adds non-linearity between layers |
| Hidden Layer | Intermediate layer between input and output |
| Forward Pass | Push data through the network → prediction |
| Backpropagation | Compute gradients to adjust weights |
| Gradient Descent | w_new = w_old − learning_rate × gradient |
| Epoch | One pass through all training data |
| Learning Rate | Controls how big each weight update is |
| MLP (Multi-Layer Perceptron) | Fully-connected neural network |
| Capacity | Ability to learn complex patterns |
| Hyperparameter | Setting you choose before training |
| Parameter | Learned from data (weights and biases) |

### Transformers & Language Models

| Math Term | ML Term / Definition |
|---|---|
| Tokenization | Converting text into numbers |
| Vocabulary | The set of unique tokens |
| Encoding / Decoding | Text → IDs / IDs → text |
| BPE (Byte Pair Encoding) | Subword tokenization |
| Embedding | Discrete token → continuous vector |
| Positional Embedding | Position → vector |
| Self-Attention | Each position looks at every earlier position |
| Query / Key / Value | Q = looking for, K = offering, V = sharing |
| Attention Formula | softmax(Q @ Kᵀ / √d_k) @ V |
| Causal Mask | Position t only sees positions 0..t |
| Multi-head Attention | H parallel attentions with smaller dimensions |
| Transformer Block | Attention + feed-forward |
| Residual Connection | x = x + f(x); skip connection |
| LayerNorm | Normalizes each sample to mean=0, std=1 |
| Pre-norm / Post-norm | Normalize before or after attention/MLP |
| Logits | Raw scores before softmax |
| Autoregressive Sampling | Predict next token, feed back, repeat |
| Temperature | Controls randomness in sampling |
| Batch | Group of examples processed together |
| Sequence Length | Number of tokens in a sequence |

### Data Preprocessing

| Math Term | ML Term / Definition |
|---|---|
| NaN | Missing data indicator |
| Imputation | Replacing missing values with estimates |
| StandardScaler | Transforms to mean=0, std=1 |
| MinMaxScaler | Transforms to range [0, 1] |
| LabelEncoder | Text → integer |
| OneHotEncoder | Text → binary columns |
| Pipeline | Chain preprocessing + model into one object |
| Feature Scaling | Normalize features to the same scale |
| Data Leakage | When test data influences training |

### Core ML Concepts

| Math Term | ML Term / Definition |
|---|---|
| Feature | Input variable (x) — what you know |
| Label / Target | Output variable (y) — what you predict |
| Model | The learned formula |
| Weight | Learned coefficient |
| Bias | Baseline value when input = 0 |
| Training | Finding the best weights from data |
| Inference | Using the trained model to predict |
| Overfitting | Memorizes noise in training data |
| Underfitting | Model too simple to capture the pattern |
| Generalization | Ability to predict on unseen data |
| Bias-Variance Tradeoff | Total error = bias² + variance + noise |
| Feature Engineering | Creating new features from existing ones |
| Cross-Validation | Split data K ways, average the scores |

### Fine-tuning & Advanced

| Math Term | ML Term / Definition |
|---|---|
| Pretraining | Train on large-scale data |
| Fine-tuning | Adapt a pretrained model to a new domain |
| Transfer Learning | Reuse weights from pretrained model |
| Catastrophic Forgetting | Model forgets old domain during fine-tuning |
| LoRA (Low-Rank Adaptation) | Learn tiny correction matrices instead of updating W |
| LoRA Rank / Alpha | r = adapter size, α = scaling factor |
| Frozen Weights | requires_grad=False; weights don't update |
| RAG (Retrieval-Augmented Generation) | Give model access to docs at inference time |
| RLHF | Align model with human preferences via reinforcement learning |
| DPO (Direct Preference Optimization) | Alternative to RLHF |
| Distillation | Compress big model into small one |
| Quantization | Run models with reduced precision (int4, int8) |

### PyTorch Essentials

| Math Term | ML Term / Definition |
|---|---|
| Tensor | NumPy array + GPU + autograd support |
| Autograd | Automatic differentiation |
| Computational Graph | Graph of operations; .backward() walks in reverse |
| nn.Module | Base class for all PyTorch networks |
| nn.Linear | Fully connected layer (W @ x + b) |
| zero_grad() | Reset gradients before backward pass |
| backward() | Backpropagation; compute all gradients |
| step() | Gradient descent; update all weights |
| eval() | Switch to evaluation mode |
| torch.no_grad() | Disable gradient tracking for inference |
| .to(device) | Move tensor/model to CPU or GPU |
| .pth | Standard file extension for saved PyTorch model weights |

---

## simple_ocr.py — Train an OCR Model on Real Data

A standalone script that trains a simple CNN+CTC OCR model using the
Mabinogi attribute training data (3,826 images). Not a lesson — a working
training pipeline you can modify.

### Usage

```bash
# Train the model (uses GPU if available)
python simple_ocr.py

# Predict text from a single image
python simple_ocr.py predict path/to/image.png
```

### How it works

```
Image (공격, 20×13px)
  ↓  resize to 32×128, normalize to 0-1
Conv2d layers (3×3 sliding window detects edges, strokes, shapes)
  ↓  produces 32 horizontal "columns" of visual features
Linear layer (predicts character probabilities at each position)
  ↓  32 positions × 71 classes (70 chars + CTC blank)
CTC decode: [-, -, 공, 공, -, 격, 격, -, -] → "공격"
```

### Comparison with the real model

| | simple_ocr.py | deep-text-recognition-benchmark |
|---|---|---|
| Architecture | CNN(3 layers) → Linear | TPS → ResNet → BiLSTM → CTC |
| Parameters | ~129K | ~7M |
| Training | 50 epochs, ~30s | 5000 iterations |
| Accuracy | ~95-99% | ~99.5%+ |
