# AI / ML / LLM Cheatsheet

Quick-reference for every major term covered in the `ai/` lessons.
Each entry: **what** it is, **why** we use it, **how** it works, and **where** to find it in this repo.

---

## 1. Mathematics & Statistics

| Term | What / Why / How | Reference |
|------|-----------------|-----------|
| **Mean** | Average of values. Baseline prediction when you know nothing else. `sum(x) / len(x)` | `01_numpy_basics.py`, `jupyter/ch01_what_is_ml.ipynb` |
| **Variance** | How spread out values are from the mean. Tells you if data is tight or noisy. `mean((x - mean)²)` | `01_numpy_basics.py` |
| **Standard Deviation** | √variance — spread in original units. Easier to interpret than variance. | `02_pandas_usage.py` |
| **Covariance** | Do two variables move together? Positive = both rise, negative = one rises other falls. `mean((x-μx)(y-μy))` | `01_numpy_basics.py`, `02_pandas_usage.py` |
| **Correlation** | Normalized covariance, always between -1 and +1. Unit-free comparison of relationships. `cov(x,y) / (std_x × std_y)` | `02_pandas_usage.py` |
| **Dot Product** | Multiply matching pairs and sum. Measures how similarly two vectors point. Core of attention: Q·K tells how much two tokens should attend. | `jupyter/cores/dot_product.ipynb`, `TRANSFORMER.md`, `BLACK_BOX.md` |
| **Matrix Multiplication** | `@` operator — transforms vectors through layers. Every neural network layer is `W @ x + b`. | `10_pytorch_intro.py`, `GEN_AI.md` |
| **Gradient** | Direction + magnitude to adjust weights. "Which way and how far do I move each knob to reduce error?" `∂Loss/∂weight` | `09_neural_network.py`, `10_pytorch_intro.py` |
| **Chain Rule** | Multiply derivatives along the path. Backpropagation is just the chain rule applied repeatedly. `dL/dW₁ = dL/dOut × dOut/dH × dH/dW₁` | `BLACK_BOX.md`, `GEN_AI.md`, `jupyter/ch07_backpropagation.ipynb` |
| **Euclidean Distance** | Straight-line distance between two points. K-Means uses it to assign points to nearest centroid. `√(Σ(a-b)²)` | `08_clustering.py`, `jupyter/ch05_clustering.ipynb` |
| **Cosine Similarity** | Dot product of unit vectors — measures angle, ignores magnitude. Used in embedding search: "how similar are these meanings?" | `embedding_training_animated.py`, `jupyter/RAG/embedding_training.ipynb`, `jupyter/cores/dot_product.ipynb` |
| **Low-Rank Decomposition** | Approximate a big matrix with two small ones: `W_new = W + α(B @ A)`. LoRA uses this to fine-tune cheaply. | `FINE_TUNE.md`, `12_fine_tuning.py` |
| **Vectorization** | Math on entire arrays at once instead of looping. NumPy does this 10–100× faster than Python loops. | `01_numpy_basics.py` |

---

## 2. Loss Functions & Error Metrics

| Term | What / Why / How | Reference |
|------|-----------------|-----------|
| **Loss Function** | Single number measuring "how wrong." Training = minimize this number. | `09_neural_network.py`, `GEN_AI.md` |
| **Residual** | `actual - predicted`. Vertical distance from point to line. | `03_linear_regression.py`, `jupyter/ch02_regression.ipynb` |
| **MSE (Mean Squared Error)** | Average of error². Penalizes large errors heavily. Has clean derivative for gradient descent. | `01_numpy_basics.py`, `jupyter/ch02_regression.ipynb` |
| **MAE (Mean Absolute Error)** | Average of |error|. Easier to interpret, more robust to outliers. | `01_numpy_basics.py` |
| **RMSE** | √MSE — back in original units. Combines MSE's gradient friendliness with interpretability. | `01_numpy_basics.py` |
| **R² Score** | `1 - SSR/TSS`. Fraction of variation explained by your model (0 = useless, 1 = perfect). | `03_linear_regression.py`, `jupyter/ch02_regression.ipynb` |
| **Cross-Entropy Loss** | `-log(P(correct))`. High confidence on correct answer = low loss. Standard for classification and language models. | `09_neural_network.py`, `10_pytorch_intro.py`, `GEN_AI.md` |
| **Perplexity** | `e^loss`. "How many tokens is the model choosing between?" Lower = more confident on correct tokens. | `jupyter/ch13_large_language_models.ipynb` |
| **Contrastive Loss** | Pull similar pairs closer, push dissimilar pairs apart in embedding space. Trains sentence embeddings. | `embedding_training_animated.py`, `jupyter/RAG/embedding_training.ipynb` |

---

## 3. Activation Functions

| Term | What / Why / How | Reference |
|------|-----------------|-----------|
| **Sigmoid** | `1/(1+e⁻ˣ)` — squashes any number to (0, 1). Used for binary probability: "is it a cat or not?" | `jupyter/cores/sigmoid.ipynb`, `09_neural_network.py`, `xor_from_scratch.py` |
| **ReLU** | `max(0, x)` — dead simple, kills negatives. Most popular hidden-layer activation because it trains fast and avoids vanishing gradients. | `jupyter/cores/relu.ipynb`, `09_neural_network.py`, `10_pytorch_intro.py` |
| **Softmax** | Converts raw scores to probabilities that sum to 1. Amplifies the biggest score. "Pick the most likely class/token." | `jupyter/cores/softmax.ipynb`, `05_classification.py`, `TRANSFORMER.md` |
| **Tanh** | Squashes to (-1, 1). Centered version of sigmoid. Used in older RNNs and some gates. | `09_neural_network.py` |
| **GELU** | Smooth ReLU ≈ `x × sigmoid(1.702x)`. Default in GPT/BERT. Small negative values get a tiny gradient instead of zero. | `11_transformer.py`, `GEN_AI.md` |
| **Softmax Temperature** | Divide logits by T before softmax. T < 1 = sharp/confident, T > 1 = random/creative. Controls generation diversity. | `11_transformer.py`, `GEN_AI.md` |

---

## 4. Regression

| Term | What / Why / How | Reference |
|------|-----------------|-----------|
| **Linear Regression** | `y = wx + b`. Find the best straight line through data. Simplest predictive model. | `03_linear_regression.py`, `jupyter/ch02_regression.ipynb` |
| **Slope (Weight)** | How much y changes per unit of x. The "multiplier" the model learns. | `01_numpy_basics.py`, `03_linear_regression.py` |
| **Intercept (Bias)** | y-value when x = 0. Shifts the line up or down. | `03_linear_regression.py` |
| **OLS (Ordinary Least Squares)** | Minimize Σ(residual²). Closed-form solution for linear regression. | `01_numpy_basics.py` |
| **Polynomial Regression** | Add x², x³, ... features so a linear model can fit curves. More degrees = more flexible = more overfitting risk. | `04_polynomial_regression.py`, `jupyter/ch02_regression.ipynb` |

---

## 5. Classification

| Term | What / Why / How | Reference |
|------|-----------------|-----------|
| **Classification** | Predict a category, not a number. "spam or not?" "cat, dog, or bird?" | `05_classification.py`, `jupyter/ch03_classification.ipynb` |
| **Logistic Regression** | Linear score → sigmoid → probability → class. Despite the name, it's a classifier. | `05_classification.py`, `jupyter/ch03_classification.ipynb` |
| **Confusion Matrix** | Table showing which classes get confused with which. Diagonal = correct predictions. | `05_classification.py`, `jupyter/ch03_classification.ipynb` |
| **Accuracy** | Correct / total. Simple but misleading with imbalanced classes (99% "not fraud" gets 99% accuracy). | `05_classification.py`, `09_neural_network.py` |
| **Precision** | Of predicted positives, how many were actually positive? "When I say yes, am I right?" | `05_classification.py`, `jupyter/ch03_classification.ipynb` |
| **Recall** | Of all actual positives, how many did I find? "Did I miss any?" | `05_classification.py`, `jupyter/ch03_classification.ipynb` |
| **F1 Score** | Harmonic mean of precision and recall. Balances both when you can't afford to ignore either. | `05_classification.py`, `jupyter/ch03_classification.ipynb` |
| **Decision Boundary** | The line/curve separating classes in feature space. | `05_classification.py` |

---

## 6. Trees & Ensembles

| Term | What / Why / How | Reference |
|------|-----------------|-----------|
| **Decision Tree** | Flowchart of yes/no questions. Splits data by feature + threshold at each node. Interpretable but overfits easily. | `07_trees_and_forests.py`, `jupyter/ch04_trees_and_forests.ipynb` |
| **Gini Impurity** | `1 - Σ(pᵢ²)`. Measures how mixed a group is. Lower = purer. Used to pick the best split. | `07_trees_and_forests.py`, `jupyter/ch04_trees_and_forests.ipynb` |
| **Information Gain** | Parent impurity - weighted children impurity. Pick splits that maximize this. | `07_trees_and_forests.py` |
| **Random Forest** | Many decision trees voting. Each tree trained on random data subset + random feature subset. Reduces overfitting. | `07_trees_and_forests.py`, `jupyter/ch04_trees_and_forests.ipynb` |
| **Bagging** | Bootstrap aggregating. Train on random subsets, average results. Reduces variance. | `07_trees_and_forests.py` |
| **Boosting** | Train models sequentially, each fixing previous errors. Reduces bias. (XGBoost, LightGBM) | `07_trees_and_forests.py` |
| **Feature Importance** | How much each feature reduces impurity across all trees. "Which inputs matter most?" | `07_trees_and_forests.py`, `jupyter/ch04_trees_and_forests.ipynb` |

---

## 7. Clustering (Unsupervised)

| Term | What / Why / How | Reference |
|------|-----------------|-----------|
| **K-Means** | Place K centroids → assign nearest points → move centroids to mean → repeat until stable. | `08_clustering.py`, `jupyter/ch05_clustering.ipynb` |
| **Centroid** | Center point (mean) of a cluster. | `08_clustering.py` |
| **Elbow Method** | Plot inertia vs K. The "bend" = best K. Too few clusters = high inertia, too many = diminishing returns. | `08_clustering.py`, `jupyter/ch05_clustering.ipynb` |
| **Silhouette Score** | `(b-a)/max(a,b)` where a = within-cluster distance, b = nearest-other-cluster distance. -1 to +1. | `08_clustering.py` |
| **DBSCAN** | Density-based clustering. Finds clusters of any shape, auto-detects K, labels outliers as noise. | `08_clustering.py` |
| **Inertia** | Sum of squared distances to centroids. Lower = tighter clusters. | `08_clustering.py` |

---

## 8. Preprocessing & Validation

| Term | What / Why / How | Reference |
|------|-----------------|-----------|
| **Train/Test Split** | Hold out ~20% of data for honest evaluation. Never train on test data. | `03_linear_regression.py`, `jupyter/ch02_regression.ipynb` |
| **Cross-Validation** | Split K ways, train K times, average scores. More reliable than a single split. | `04_polynomial_regression.py`, `jupyter/ch02_regression.ipynb` |
| **Overfitting** | Model memorizes training noise. Great train score, bad test score. Cure: more data, simpler model, regularization. | `04_polynomial_regression.py`, `jupyter/ch02_regression.ipynb` |
| **Underfitting** | Model too simple to capture patterns. Bad everywhere. Cure: more features, more capacity. | `04_polynomial_regression.py` |
| **Bias-Variance Tradeoff** | Total error = bias² + variance + noise. Simple models have high bias; complex ones have high variance. | `04_polynomial_regression.py` |
| **StandardScaler** | Transform to mean=0, std=1. Makes features comparable regardless of original units. | `06_preprocessing.py`, `jupyter/ch03_classification.ipynb` |
| **MinMaxScaler** | Transform to [0, 1] range. Good when you need bounded values. | `06_preprocessing.py` |
| **OneHotEncoding** | Category → binary columns. "red" → [1,0,0], "blue" → [0,1,0]. For nominal (no-order) data. | `06_preprocessing.py` |
| **LabelEncoder** | Category → integer. "small"=0, "medium"=1, "large"=2. For ordinal (ordered) data. | `06_preprocessing.py` |
| **Data Leakage** | Test data influences training. Must split BEFORE preprocessing. Silent killer of real-world performance. | `06_preprocessing.py` |
| **Pipeline** | Chain preprocessing + model into one object. Prevents leakage and simplifies code. | `06_preprocessing.py` |

---

## 9. Neural Networks

| Term | What / Why / How | Reference |
|------|-----------------|-----------|
| **Neuron** | `output = activation(weight × input + bias)`. Atomic unit of neural networks. | `09_neural_network.py`, `jupyter/ch06_neural_networks.ipynb` |
| **Hidden Layer** | Layer between input and output. More layers = deeper = can learn more complex patterns. | `09_neural_network.py`, `jupyter/ch06_neural_networks.ipynb` |
| **Forward Pass** | Push data through network layer by layer → get prediction. | `09_neural_network.py`, `10_pytorch_intro.py` |
| **Backpropagation** | Compute gradients by walking the chain rule backwards through the network. "How much did each weight contribute to the error?" | `09_neural_network.py`, `xor_from_scratch.py`, `jupyter/ch07_backpropagation.ipynb`, `jupyter/xor_backpropagation.ipynb` |
| **Gradient Descent** | `w_new = w_old - lr × gradient`. Move weights in the direction that reduces loss. Repeat until converged. | `09_neural_network.py`, `GEN_AI.md`, `jupyter/ch07_backpropagation.ipynb` |
| **Learning Rate** | Step size for weight updates. Too big = overshoot, too small = slow. Typical: 1e-3 to 1e-5. | `09_neural_network.py`, `FINE_TUNE.md` |
| **Epoch** | One complete pass through all training data. Usually train for many epochs until loss plateaus. | `09_neural_network.py`, `10_pytorch_intro.py` |
| **Batch Size** | Number of examples per gradient update. Larger = more stable but more memory. | `FINE_TUNE.md`, `11_transformer.py` |
| **MLP (Multi-Layer Perceptron)** | Fully-connected neural network. Every neuron connects to every neuron in next layer. | `09_neural_network.py`, `jupyter/ch06_neural_networks.ipynb` |
| **XOR Problem** | Classic proof that single-layer networks can't learn non-linear patterns. Need hidden layers. | `xor_from_scratch.py`, `xor_animated.py`, `jupyter/xor_backpropagation.ipynb` |
| **Adam Optimizer** | Adaptive learning rate + momentum. Standard for transformers. Adjusts each weight's step size individually. | `10_pytorch_intro.py`, `FINE_TUNE.md` |
| **Dropout** | Randomly disable neurons during training. Forces network to not rely on any single neuron. Prevents overfitting. | `FINE_TUNE.md` |
| **Gradient Clipping** | Cap gradient magnitude. Prevents exploding gradients that destabilize training. | `FINE_TUNE.md` |

---

## 10. PyTorch Essentials

| Term | What / Why / How | Reference |
|------|-----------------|-----------|
| **Tensor** | NumPy array + GPU support + automatic differentiation. The data structure of deep learning. | `10_pytorch_intro.py`, `jupyter/ch08_deep_learning.ipynb` |
| **Autograd** | Automatically compute gradients. Call `.backward()` and PyTorch traces the computational graph in reverse. | `10_pytorch_intro.py`, `jupyter/ch08_deep_learning.ipynb` |
| **nn.Module** | Base class for all PyTorch models. Define layers in `__init__`, wire them in `forward()`. | `10_pytorch_intro.py`, `11_transformer.py` |
| **nn.Linear** | Fully connected layer: `W @ x + b`. The dense building block. | `10_pytorch_intro.py`, `11_transformer.py` |
| **nn.Embedding** | Lookup table: token ID → vector. `nn.Embedding(vocab_size, dim)` | `11_transformer.py` |
| **zero_grad() → backward() → step()** | The training loop: clear old gradients → compute new ones → update weights. | `10_pytorch_intro.py` |
| **torch.no_grad()** | Disable gradient tracking for inference. Saves memory and speed. | `10_pytorch_intro.py` |

---

## 11. Tokenization & Embeddings

| Term | What / Why / How | Reference |
|------|-----------------|-----------|
| **Tokenization** | Text → token IDs. "Hello world" → [15496, 995]. The first step in any NLP pipeline. | `11_transformer.py`, `jupyter/ch10_tokenization.ipynb` |
| **Token** | Subword piece. "unhappiness" → ["un", "happiness"]. Not always whole words. | `11_transformer.py`, `jupyter/ch10_tokenization.ipynb` |
| **BPE (Byte Pair Encoding)** | Start with characters, merge most frequent pairs. Balances vocabulary size vs. sequence length. Used by GPT. | `11_transformer.py`, `jupyter/ch10_tokenization.ipynb` |
| **Vocabulary** | Set of all known tokens. GPT-2: ~50k tokens. Unknown words get split into known subwords. | `11_transformer.py`, `jupyter/ch10_tokenization.ipynb` |
| **Embedding** | Token ID → dense vector (e.g., 768 dimensions). Similar meanings → nearby vectors. The model's "understanding" of a token. | `11_transformer.py`, `jupyter/ch11_embeddings.ipynb`, `embedding_training_animated.py` |
| **Token Embedding** | "What is this token?" Identity information. | `11_transformer.py`, `jupyter/ch11_embeddings.ipynb` |
| **Positional Embedding** | "Where is this token?" Position information. Without it, "dog bites man" = "man bites dog". | `11_transformer.py`, `jupyter/ch11_embeddings.ipynb`, `TRANSFORMER.md` |
| **Embedding Space** | High-dimensional space where meaning lives. Distance = similarity. Directions encode relationships (king - man + woman ≈ queen). | `BLACK_BOX.md`, `jupyter/ch11_embeddings.ipynb` |
| **Sentence Embedding** | Compress entire sentence into one vector. Used for search, similarity, RAG retrieval. | `embedding_training_animated.py`, `jupyter/RAG/embedding_training.ipynb` |

---

## 12. Attention & Transformers

| Term | What / Why / How | Reference |
|------|-----------------|-----------|
| **Self-Attention** | Each token looks at every other token and decides how much to "pay attention" to it. Core innovation of Transformers. | `11_transformer.py`, `TRANSFORMER.md`, `jupyter/ch12_attention_transformers.ipynb` |
| **Query (Q)** | "What am I looking for?" Each token generates a query vector. | `jupyter/cores/attention_qkv.ipynb`, `TRANSFORMER.md`, `11_transformer.py` |
| **Key (K)** | "What do I contain?" Each token generates a key vector. Q·K = relevance score. | `jupyter/cores/attention_qkv.ipynb`, `TRANSFORMER.md`, `11_transformer.py` |
| **Value (V)** | "What information do I share?" Weighted by attention scores. The actual content passed forward. | `jupyter/cores/attention_qkv.ipynb`, `TRANSFORMER.md`, `11_transformer.py` |
| **Attention Formula** | `softmax(Q @ Kᵀ / √d_k) @ V`. Dot product for relevance → normalize → weighted sum of values. | `11_transformer.py`, `TRANSFORMER.md`, `GEN_AI.md` |
| **Causal Mask** | Set future positions to -∞ before softmax. Token at position t can only see 0..t. Essential for text generation. | `11_transformer.py`, `TRANSFORMER.md` |
| **Multi-Head Attention** | H parallel attention heads, each with different Q/K/V weights. Each head can learn different patterns (syntax, semantics, position). | `11_transformer.py`, `jupyter/ch12_attention_transformers.ipynb`, `GEN_AI.md` |
| **Transformer Block** | Self-Attention → Add & LayerNorm → Feed-Forward → Add & LayerNorm. Stack N of these. GPT-2 = 12 blocks. | `11_transformer.py`, `jupyter/ch12_attention_transformers.ipynb` |
| **Feed-Forward Network (FFN)** | Two dense layers per token inside each block. Acts as "knowledge storage" — where facts are encoded. | `11_transformer.py`, `GEN_AI.md` |
| **Residual Connection** | `x + f(x)` instead of just `f(x)`. Keeps gradients flowing through deep networks. Skips around each sub-layer. | `11_transformer.py`, `GEN_AI.md` |
| **LayerNorm** | Normalize each token's activations to mean=0, std=1. Stabilizes training of deep networks. | `11_transformer.py`, `GEN_AI.md` |
| **Logits** | Raw scores before softmax, one per vocabulary token (~50k numbers). Highest logit → most likely next token. | `11_transformer.py`, `GEN_AI.md` |

---

## 13. Text Generation

| Term | What / Why / How | Reference |
|------|-----------------|-----------|
| **Autoregressive Sampling** | Predict next token → append → repeat. The model never sees the future, only the past. | `11_transformer.py`, `GEN_AI.md` |
| **Temperature** | Divide logits by T before softmax. T=0.1 = almost deterministic, T=1.5 = creative/random. | `11_transformer.py`, `GEN_AI.md` |
| **Top-k Sampling** | Only consider the k most likely tokens. Filters out low-probability garbage. | `GEN_AI.md` |
| **Top-p (Nucleus) Sampling** | Keep tokens until cumulative probability ≥ p. Adaptive: sometimes 5 tokens, sometimes 50. | `GEN_AI.md` |
| **Language Model** | Predicts next token given previous tokens. Trained on massive text data. Foundation of GPT, LLaMA, etc. | `11_transformer.py`, `jupyter/ch13_large_language_models.ipynb` |

---

## 14. Sequence Models (Pre-Transformer)

| Term | What / Why / How | Reference |
|------|-----------------|-----------|
| **RNN (Recurrent Neural Network)** | Process tokens one by one, passing hidden state forward. Struggles with long sequences (vanishing gradients). | `jupyter/ch09_sequence_models.ipynb` |
| **LSTM (Long Short-Term Memory)** | RNN with gates (forget, input, output) that control information flow. Can remember longer sequences. | `jupyter/ch09_sequence_models.ipynb` |
| **BiLSTM** | LSTM reading both left-to-right and right-to-left. Captures full context. Used in OCR. | `simple_ocr.py`, `jupyter/ch09_sequence_models.ipynb` |
| **Vanishing Gradient** | Gradients shrink to ~0 in deep/long networks. RNN's fatal flaw. Transformers solve this with residual connections + attention. | `jupyter/ch09_sequence_models.ipynb` |

---

## 15. LLM Training & Fine-Tuning

| Term | What / Why / How | Reference |
|------|-----------------|-----------|
| **Pretraining** | Train on massive text (internet-scale). Learns language, facts, reasoning. Costs millions of $. | `FINE_TUNE.md`, `jupyter/ch13_large_language_models.ipynb` |
| **Fine-Tuning** | Adapt pretrained model to your specific domain/task. Much cheaper than pretraining. Hours, not months. | `12_fine_tuning.py`, `FINE_TUNE.md`, `jupyter/ch15_fine_tuning.ipynb` |
| **Transfer Learning** | Reuse knowledge from pretraining. "You already know English — now learn medical English." | `FINE_TUNE.md` |
| **LoRA (Low-Rank Adaptation)** | Freeze base weights, add tiny trainable matrices alongside. ~0.5–1% of parameters. Fits on consumer GPUs. | `12_fine_tuning.py`, `FINE_TUNE.md`, `jupyter/ch15_fine_tuning.ipynb` |
| **QLoRA** | LoRA but base model quantized to 4-bit. Fits 70B parameter model on 48GB GPU. | `FINE_TUNE.md` |
| **LoRA Rank (r)** | Size of low-rank matrices. r=8–16 typical. Higher = more capacity but more memory. | `FINE_TUNE.md` |
| **LoRA Alpha (α)** | Scaling factor. Effective weight contribution = α/r. Usually α = 2×r. | `FINE_TUNE.md` |
| **Frozen Weights** | Parameters not updated (`requires_grad=False`). Base model stays intact during LoRA fine-tuning. | `FINE_TUNE.md`, `12_fine_tuning.py` |
| **Full Fine-Tuning** | Update every weight. Maximum flexibility. Needs huge GPU memory (>80GB for 7B model). | `FINE_TUNE.md` |
| **Continued Pretraining** | Feed domain-specific text to teach new knowledge before fine-tuning on tasks. | `FINE_TUNE.md`, `13_domain_data_pipeline.py` |
| **Catastrophic Forgetting** | Model forgets general knowledge while specializing. Mitigation: LoRA, low learning rate, diverse data. | `12_fine_tuning.py`, `FINE_TUNE.md` |
| **SFT (Supervised Fine-Tuning)** | Train on (instruction, response) pairs. Teaches the model WHAT to say and in what format. | `FINE_TUNE.md` |
| **RLHF** | Humans rank outputs → train reward model → use RL to make model prefer higher-ranked responses. Teaches HOW to say it. | `FINE_TUNE.md`, `jupyter/ch14_rlhf.ipynb` |
| **DPO (Direct Preference Optimization)** | Simpler alternative to RLHF. Train directly on (prompt, chosen, rejected) triples. No reward model needed. | `FINE_TUNE.md` |
| **Quantization** | Reduce precision (float32 → int4/int8). Makes models smaller and faster at small accuracy cost. | `FINE_TUNE.md` |
| **Mixed Precision (fp16/bf16)** | Use half-precision floats during training. 2× speed, half memory, negligible accuracy loss. | `FINE_TUNE.md` |

---

## 16. RAG (Retrieval-Augmented Generation)

| Term | What / Why / How | Reference |
|------|-----------------|-----------|
| **RAG** | Retrieve relevant documents → inject into prompt → generate grounded answer. No retraining needed. | `jupyter/RAG/rag_example.ipynb`, `jupyter/ch18_rag.ipynb`, `FINE_TUNE.md` |
| **Vector Store** | Database of embeddings. Search by similarity: "find documents closest to this question." | `jupyter/RAG/rag_example.ipynb`, `jupyter/RAG/rag_advanced.ipynb` |
| **Chunking** | Split documents into smaller pieces for embedding. Too big = loses relevance, too small = loses context. | `jupyter/RAG/rag_advanced.ipynb`, `jupyter/RAG/rag_tuning_guide.ipynb` |
| **Retrieval** | Given a query embedding, find the top-k most similar document chunks. | `jupyter/RAG/rag_example.ipynb` |
| **Reranking** | After retrieval, use a more expensive model to re-score and reorder results for higher precision. | `jupyter/RAG/rag_advanced.ipynb`, `jupyter/RAG/rag_tuning_guide.ipynb` |
| **Hybrid Search** | Combine dense (embedding) and sparse (keyword/BM25) search. Covers both semantic and exact matches. Tune `hybrid_weight`: keyword-heavy (0.7) for medical/code, vector-heavy (0.7) for customer support. | `jupyter/RAG/rag_advanced.ipynb`, `jupyter/RAG/rag_tuning_guide.ipynb` |
| **Keyword Search (BM25)** | Count and match exact words. Good at exact terms (drug names, function names). Bad at synonyms ("car" won't find "automobile"). | `jupyter/RAG/rag_advanced.ipynb` |
| **Vector Search (Dense)** | Compare embedded vectors by cosine similarity. Good at meaning ("fix a crash" finds "debugging segfaults"). Bad at distinguishing exact values (eGFR 30 vs 45). | `jupyter/RAG/rag_advanced.ipynb`, `jupyter/RAG/rag_example.ipynb` |
| **Metadata Filtering** | Filter by structured fields (drug, category, date) BEFORE vector search. Don't search all 10k docs when you already know the drug name. | `jupyter/RAG/rag_advanced.ipynb` |
| **Query Decomposition** | LLM splits complex question into sub-queries, searches each separately, combines results. "Patient on metformin with eGFR 35 needs CT contrast" → 2 searches. | `jupyter/RAG/rag_advanced.ipynb` |
| **Cross-Encoder Reranking** | Stage 1: fast rough search (top 20). Stage 2: cross-encoder reads query+doc TOGETHER word-by-word, rescores precisely (top 3). Much more accurate than embedding alone. | `jupyter/RAG/rag_advanced.ipynb` |
| **Forced Citation** | System prompt forces LLM to cite source [D001] for every claim, or say "NOT FOUND." Prevents hallucination, enables audit. | `jupyter/RAG/rag_advanced.ipynb` |
| **Confidence Scoring** | Check if best match score > threshold. High → answer. Low → say "I don't know" and escalate to human. Knowing when NOT to answer. | `jupyter/RAG/rag_advanced.ipynb` |
| **Temporal Awareness** | Boost recent documents, warn if source is old. Medical guidelines change — using outdated docs can be dangerous. | `jupyter/RAG/rag_advanced.ipynb` |

---

## 17. Interpretability & Safety

| Term | What / Why / How | Reference |
|------|-----------------|-----------|
| **Black Box** | Can't inspect internal reasoning. We see input → output but not why. Most modern neural networks. | `BLACK_BOX.md` |
| **Mechanistic Interpretability** | Reverse-engineer what each weight/neuron/layer actually does. Like opening the black box. | `BLACK_BOX.md` |
| **Probing** | Train a tiny classifier on hidden states to test what information is encoded. "Does layer 5 know part-of-speech?" | `BLACK_BOX.md` |
| **Ablation** | Turn off a component, observe what breaks. Shows what each part is responsible for. | `BLACK_BOX.md` |
| **Sparse Autoencoder (SAE)** | Decompose neuron activations into interpretable features. Tool for mechanistic interpretability. | `BLACK_BOX.md` |
| **Superposition** | Model packs more concepts than it has dimensions by using overlapping directions. Why single neurons aren't interpretable. | `BLACK_BOX.md` |
| **Polysemanticity** | One neuron = multiple unrelated concepts. The norm in neural networks. Opposite of monosemanticity. | `BLACK_BOX.md` |
| **Feature Direction** | Concepts encoded as directions in activation space, not single neurons. "Truth" might be a specific direction. | `BLACK_BOX.md` |
| **Hallucination** | Model generates confident-sounding but false information. Major LLM limitation. RAG helps mitigate. | `BLACK_BOX.md` |
| **Alignment** | Does the model do what we actually want, not just what we literally asked? Central safety concern. | `BLACK_BOX.md` |
| **Sycophancy** | Model agrees with user even when user is wrong. A form of misalignment. | `BLACK_BOX.md` |
| **Reward Hacking** | Model finds shortcuts satisfying the training objective but missing the intent. Gaming the metric. | `BLACK_BOX.md` |
| **Emergent Behavior** | Abilities appearing suddenly at scale without explicit training. Chain-of-thought, in-context learning. | `BLACK_BOX.md` |
| **In-Context Learning** | Learn from examples in the prompt without updating weights. Emergent in large models. | `BLACK_BOX.md` |
| **Chain of Thought** | Prompting model to reason step-by-step. Dramatically improves complex reasoning. | `BLACK_BOX.md` |

---

## 18. Computer Vision & OCR

| Term | What / Why / How | Reference |
|------|-----------------|-----------|
| **CNN (Convolutional Neural Network)** | Small weight matrix slides across image detecting local patterns (edges, shapes, textures). | `simple_ocr.py`, `jupyter/ch08_deep_learning.ipynb` |
| **Conv2d** | 2D convolution layer. 3×3 kernel slides over image producing feature maps. | `simple_ocr.py` |
| **OCR** | Optical Character Recognition. Extract text from images. Pipeline: detect → recognize → decode. | `simple_ocr.py` |
| **CTC (Connectionist Temporal Classification)** | Decode variable-length output from fixed-length input. Handles alignment automatically. | `simple_ocr.py` |
| **ResNet** | Deep CNN with skip connections. Enables training very deep networks (50+ layers). | `simple_ocr.py` |

---

## 19. Tools & Libraries

| Term | What / Why / How | Reference |
|------|-----------------|-----------|
| **NumPy** | Array math library. Foundation of all scientific Python. Vectorized operations. | `01_numpy_basics.py` |
| **Pandas** | DataFrames for tabular data. Load, clean, explore, transform datasets. | `02_pandas_usage.py` |
| **scikit-learn** | Classical ML library. Regression, classification, clustering, preprocessing, evaluation. | `03_linear_regression.py` – `08_clustering.py` |
| **PyTorch** | Deep learning framework. Tensors + autograd + GPU. Build and train neural networks. | `10_pytorch_intro.py`, `11_transformer.py` |
| **Hugging Face Transformers** | Load pretrained models with 3 lines. Fine-tune, inference, share. Industry standard. | `FINE_TUNE.md`, `12_fine_tuning.py` |
| **PEFT** | Parameter-efficient fine-tuning library. Implements LoRA, QLoRA, adapters. | `FINE_TUNE.md` |
| **Elasticsearch** | Search engine for storing and retrieving domain documents. Used in data pipelines. | `13_domain_data_pipeline.py` |

---

## Choosing the Right Algorithm: "How do I find similar things?"

Many algorithms measure "similarity" but they're different tools for different jobs.
The **dot product** is the common thread — most similarity methods are built on it.

### Decision Flowchart

```
What's your task?
  |
  |-- "Group similar things together" (no labels)
  |     -> Clustering: K-Means, DBSCAN
  |     -> Similarity: Euclidean distance between data points
  |
  |-- "Which category does this belong to?" (has labels)
  |     -> KNN: find K nearest neighbors, majority vote
  |     -> Decision Tree / Random Forest: learn yes/no rules
  |     -> Logistic Regression: learn a dividing line
  |
  |-- "Find similar documents/sentences" (search)
  |     -> Embedding + Cosine similarity / Dot product
  |     -> Used in: RAG retrieval, recommendation, search
  |
  |-- "Which words in a sentence matter for THIS word?" (context)
  |     -> Attention (Q, K, V): dot product between token vectors
  |     -> Used in: Transformers, LLMs
```

### How each algorithm measures similarity

| Algorithm | Similarity method | Input type | What for | Reference |
|-----------|------------------|------------|----------|-----------|
| **K-Means** | Euclidean distance (straight line between points) | Structured data (tables) | Group unlabeled data into clusters | `08_clustering.py` |
| **DBSCAN** | Euclidean distance (density of nearby points) | Structured data | Find clusters of any shape, detect outliers | `08_clustering.py` |
| **KNN** | Euclidean distance to K nearest labeled points | Structured data | Classify: "3 of 5 nearest neighbors are cat -> cat" | `05_classification.py` |
| **Decision Tree** | Gini impurity (how mixed is each group after split?) | Structured data | Classify by learning if/else rules | `07_trees_and_forests.py` |
| **Random Forest** | Same as tree, but many trees vote | Structured data | More accurate classification, less overfitting | `07_trees_and_forests.py` |
| **Cosine similarity** | Angle between two vectors (ignores magnitude) | Embedding vectors | "Is this sentence similar to that one?" | `jupyter/cores/dot_product.ipynb`, `jupyter/RAG/embedding_training.ipynb` |
| **Dot product** | Multiply matching pairs and sum | Vectors | Raw similarity score (considers magnitude) | `jupyter/cores/dot_product.ipynb` |
| **Attention (QKV)** | Dot product of Q and K, softmax, weighted sum of V | Token embeddings | "Which other tokens should I attend to?" | `jupyter/cores/attention_qkv.ipynb`, `TRANSFORMER.md` |

### Three worlds of similarity

**Structured data** (tables with columns like age, height, income):
- Algorithms: K-Means, KNN, Decision Tree, Random Forest
- Distance: Euclidean (straight line between points)
- Example: find 5 nearest patients in database, check their diagnosis

**Embedding vectors** (learned representations of text/images):
- Algorithms: Cosine similarity, dot product
- Distance: angle between vectors in high-dimensional space
- Example: "king" and "queen" have cosine similarity 0.95

**Inside a neural network** (attention mechanism):
- Algorithm: Scaled dot product attention
- Distance: Q @ K^T / sqrt(d_k)
- Example: in "The cat sat because it was tired", "it" attends to "cat"

### They're not interchangeable

You don't choose between KNN and Attention for the same problem. The task determines the category:

| If you need to... | Use | NOT |
|---|---|---|
| Group customer segments | K-Means | Attention |
| Classify spam vs. not-spam | Tree / Forest / KNN | Cosine similarity |
| Search similar documents | Embedding + Cosine | K-Means |
| Build a language model | Attention (QKV) | Random Forest |
| Recommend similar products | Embedding + Cosine OR KNN | Decision Tree |

The last row shows where they CAN overlap: recommendation can use either embedding similarity (learned vectors) or KNN (structured features). The choice depends on your data format.

---

## Quick Lookup: "I want to understand X"

| If you want to understand... | Start here |
|------------------------------|-----------|
| How a neural network learns | `xor_from_scratch.py` → `jupyter/xor_backpropagation.ipynb` → `jupyter/ch07_backpropagation.ipynb` |
| What attention actually does | `jupyter/cores/dot_product.ipynb` → `jupyter/cores/attention_qkv.ipynb` → `TRANSFORMER.md` |
| How sigmoid/ReLU/softmax work | `jupyter/cores/sigmoid.ipynb` → `jupyter/cores/relu.ipynb` → `jupyter/cores/softmax.ipynb` |
| The full Transformer architecture | `11_transformer.py` → `jupyter/ch12_attention_transformers.ipynb` → `GEN_AI.md` |
| How to fine-tune an LLM | `FINE_TUNE.md` → `12_fine_tuning.py` → `jupyter/ch15_fine_tuning.ipynb` |
| How RAG works | `jupyter/RAG/rag_example.ipynb` → `jupyter/RAG/rag_advanced.ipynb` → `jupyter/RAG/rag_tuning_guide.ipynb` |
| What's inside the black box | `BLACK_BOX.md` |
| How embeddings are trained | `embedding_training_animated.py` → `jupyter/RAG/embedding_training.ipynb` |
| Classical ML fundamentals | `03_linear_regression.py` → `05_classification.py` → `07_trees_and_forests.py` → `08_clustering.py` |
