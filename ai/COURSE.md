# Python AI/ML Course

Learn AI and machine learning step by step, from raw data to neural networks.

## Prerequisites

- Python basics (see `../python-exercise.py` and other files in parent directory)
- `pyenv local 3.14.2`
- `pip install numpy pandas matplotlib scikit-learn torch`

## How to run

```bash
python ai/01_numpy_basics.py
python ai/02_pandas_usage.py
# ... etc
```

---

## The Big Picture: How AI Got Here

Everything in AI is the same idea at different scales:
**find the best weights to predict output from input.**

```
1 weight       →  y = ax           (lesson 01)
2 weights      →  y = ax + b       (lesson 03)
64 weights     →  pixels → digit   (lesson 09)
175 billion    →  text → text      (GPT-3, 2020)
undisclosed    →  text → text      (GPT-4 / Claude, 2023~)
```

### Timeline

```
1950s   Linear Regression
        y = ax + b. Statistics, not yet called "AI".
        → Lessons 01-03

1980s   Neural Networks & Backpropagation
        Multiple layers of weights. Backprop: algorithm to adjust
        all weights at once. Limited by hardware.
        → Lessons 09-10

1990s   Classical ML
        Decision trees, SVMs, random forests, clustering.
        Good enough for tabular data. Still widely used today.
        → Lessons 04-08

2012    Deep Learning Breakthrough
        CNNs (Convolutional Neural Networks) win ImageNet.
        Input: millions of pixels → Output: "cat"
        GPU training makes deep networks practical.

2017    Transformers ("Attention Is All You Need")
        New architecture. No more processing words one by one.
        Looks at all words at once, learns which words relate.
        Foundation of everything that followed.

2018~   Pre-trained Language Models
        BERT, GPT-2. Train on massive text, fine-tune for tasks.
        "Transfer learning" — one model, many uses.

2020    GPT-3 (175 billion parameters)
        Showed that scaling up parameters + data = new abilities.
        Few-shot learning: give examples in the prompt, no fine-tuning.

2022~   Generative AI Era
        ChatGPT, Claude, Gemini, etc.
        Input: human language → Output: human language
        Generated token by token ("what's the most likely next word?").
        Trained on trillions of words, months on thousands of GPUs.
```

### Where this course fits

```
[Lessons 01-03]  Linear Regression         ←  the foundation
       ↓
[Lessons 04-08]  Classical ML              ←  the practical toolkit
       ↓
[Lessons 09-10]  Neural Networks / PyTorch ←  the bridge to deep learning
       ↓
   (beyond this course)
       ↓
       CNNs         →  image recognition
       RNNs / LSTM  →  sequences, time series
       Transformers →  language models (GPT, Claude)
       RLHF         →  aligning AI with human preferences
```

The core principle never changes — just the scale.

### References

Papers that mark each milestone. All freely available online.

| Year | Milestone | Paper | Link |
|------|-----------|-------|------|
| 1986 | Backpropagation | Rumelhart, Hinton, Williams. "Learning representations by back-propagating errors." *Nature* 323 | [nature.com](https://www.nature.com/articles/323533a0) |
| 2001 | Random Forests | Breiman. "Random Forests." *Machine Learning* 45(1) | [berkeley.edu (PDF)](https://www.stat.berkeley.edu/~breiman/randomforest2001.pdf) |
| 2012 | Deep Learning / CNNs | Krizhevsky, Sutskever, Hinton. "ImageNet Classification with Deep Convolutional Neural Networks." *NeurIPS 2012* | [papers.nips.cc](https://papers.nips.cc/paper/2012/hash/c399862d3b9d6b76c8436e924a68c45b-Abstract.html) |
| 2017 | Transformers | Vaswani et al. "Attention Is All You Need." *NeurIPS 2017* | [arxiv.org/abs/1706.03762](https://arxiv.org/abs/1706.03762) |
| 2018 | BERT | Devlin et al. "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding." *NAACL 2019* | [arxiv.org/abs/1810.04805](https://arxiv.org/abs/1810.04805) |
| 2019 | GPT-2 (1.5B params) | Radford et al. "Language Models are Unsupervised Multitask Learners." *OpenAI Technical Report* | [openai.com (PDF)](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) |
| 2020 | GPT-3 (175B params) | Brown et al. "Language Models are Few-Shot Learners." *NeurIPS 2020* | [arxiv.org/abs/2005.14165](https://arxiv.org/abs/2005.14165) |

**Note on GPT-4 / Claude parameter counts:** OpenAI and Anthropic have not
disclosed their model sizes. Leaked rumors suggest GPT-4 may be a
mixture-of-experts model, but nothing is confirmed. The "trillions" figure
often cited online is unverified.

---

## Lessons

### 01 - NumPy Basics (`01_numpy_basics.py`)
- Creating arrays with `np.array()`
- Mean, centering data (subtract mean)
- Element-wise operations (multiply, sum)
- Computing slope manually: `a = sum(xy) / sum(xx)`

### 02 - Pandas Usage (`02_pandas_usage.py`)
- Loading CSV with `pd.read_csv()`
- Exploring data: `head()`, `describe()`, column access
- Centering a DataFrame: `df - df.mean()`
- Scatter plot and line plot with matplotlib
- Computing slope from DataFrame columns

### 03 - Linear Regression (`03_linear_regression.py`)
- Full formula: `y = ax + b` (slope + intercept)
- Making predictions on new data
- R² score: measuring model accuracy
- scikit-learn `LinearRegression` — same result in 3 lines
- Train/test split: why we need separate test data
- Visualization: regression line + residual plot

### 04 - Polynomial Regression & Overfitting (`04_polynomial_regression.py`)
- When a straight line isn't enough
- `PolynomialFeatures` to create x², x³, ... columns
- Fitting curves of degree 2, 5, 15
- Overfitting: model memorizes noise instead of learning patterns
- Underfitting vs overfitting visualized
- Cross-validation: a smarter way to evaluate

### 05 - Logistic Regression & Classification (`05_classification.py`)
- Regression vs classification: predicting numbers vs categories
- The iris dataset: 3 flower species, 4 measurements
- Logistic regression: draws decision boundaries
- Accuracy, confusion matrix, classification report
- Visualizing decision boundaries in 2D

### 06 - Data Preprocessing (`06_preprocessing.py`)
- Why preprocessing matters: garbage in, garbage out
- Handling missing values: detect and fill
- Feature scaling: StandardScaler, MinMaxScaler
- Encoding categories: LabelEncoder, OneHotEncoder
- Building a pipeline: chain preprocessing + model together

### 07 - Decision Trees & Random Forests (`07_trees_and_forests.py`)
- Decision tree: a flowchart that makes predictions
- Visualizing the tree structure
- Problem: single trees overfit easily
- Random forest: many trees vote together (ensemble)
- Feature importance: which inputs matter most
- Comparing tree vs forest accuracy

### 08 - K-Means Clustering (`08_clustering.py`)
- Supervised vs unsupervised learning
- K-Means: grouping data without labels
- Choosing K with the elbow method
- Visualizing clusters and centroids
- Silhouette score: measuring cluster quality
- Real-world use: customer segmentation

### 09 - Neural Network Basics (`09_neural_network.py`)
- What is a neural network? Layers of connected nodes
- sklearn `MLPClassifier`: neural network in scikit-learn
- Handwritten digit recognition (digits dataset)
- Hidden layers and activation functions
- Comparing architectures: shallow vs deep
- Visualizing what the network learned

### 10 - Intro to Deep Learning with PyTorch (`10_pytorch_intro.py`)
- Why PyTorch? Flexibility beyond scikit-learn
- Tensors: PyTorch's version of numpy arrays
- Building a model with `nn.Module`
- Training loop: forward pass, loss, backward pass, update
- Handwritten digit classification (same task, new tool)
- GPU vs CPU: how deep learning scales

### 11 - Build a Transformer from Scratch (`11_transformer.py`)
- Character-level tokenization: chars → IDs → tensors
- Token and positional embeddings
- Causal self-attention (Q/K/V, scaled dot product, causal mask)
- Multi-head attention + feed-forward + residuals + LayerNorm
- The full MiniGPT: stack N blocks, project to vocab
- Training loop and autoregressive text generation
- Train on a tiny Shakespeare corpus — runs on CPU in under a minute

### 12 - Fine-tuning a Pre-trained Model (`12_fine_tuning.py`)
- Why fine-tune? Cost comparison: pretrain vs full fine-tune vs LoRA
- Hugging Face ecosystem (`transformers`, `peft`, `datasets`)
- Reuse MiniGPT from lesson 11 as the base model
- Pretrain on domain A (Shakespeare), full fine-tune on domain B (Python)
- Catastrophic forgetting: what full fine-tuning costs you
- LoRA explained and implemented by hand (low-rank adapter over `nn.Linear`)
- Compare: trainable parameter counts, loss curves, generation quality
