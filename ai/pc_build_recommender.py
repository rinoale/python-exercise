"""
PC Build Recommender — Embedding model trained from scratch on PC component data.

Trains a simple embedding model where components and use-case keywords
that appear in similar builds get similar vectors. Then recommends
full builds based on natural language queries.

This is a toy model to demonstrate how embedding-based recommendation works.
A real system (like OpenAI embeddings + Pinecone) differs in every stage:

  Toy model                          Real system
  --------------------------------   ----------------------------------------
  118 hand-picked tags               30K+ token vocabulary (learned from text)
  13 curated builds                  millions of documents
  String matching tokenizer          Neural tokenizer (understands "highend" = "high end")
  32-dim vectors, average pooling    768-1536 dim, transformer with attention
  Brute-force cosine loop            Vector DB with ANN index (O(log n) search)
  Retrain every run                  Train once, save model file (.safetensors)

Usage:
    python pc_build_recommender.py

Try queries like:
    "3070 ti for gamer"
    "budget developer setup"
    "4090 machine learning"
    "cheap gaming pc"
    "video editing workstation"
"""
import numpy as np

np.random.seed(42)


# ---------------------------------------------------------------------------
# Knowledge base: PC builds for different use cases
# Each build is a list of tags — components + descriptive keywords.
# The embedding model learns from CO-OCCURRENCE: tags in the same build
# get pulled together in vector space.
#
# TOY: 13 hand-curated builds with manually chosen tags.
# REAL: millions of documents, no manual tagging — the neural tokenizer
#       and transformer handle raw text directly.
# ---------------------------------------------------------------------------

BUILDS = [
    # --- Gaming builds ---
    {
        "name": "Budget Gaming",
        "tags": ["budget", "gaming", "gamer", "casual", "esports", "fps",
                 "rtx_3060", "ryzen_5600", "16gb_ram", "b650_mobo",
                 "ssd_500gb", "650w_psu", "air_cooler",
                 "fhd_monitor", "single_monitor"],
        "parts": {
            "GPU":     "RTX 3060 12GB           $270",
            "CPU":     "Ryzen 5 5600            $130",
            "RAM":     "16GB DDR5 5600MHz       $50",
            "Board":   "B650 Micro-ATX          $120",
            "Storage": "SSD 500GB NVMe          $45",
            "PSU":     "650W 80+ Bronze         $60",
            "Cooler":  "Tower Air Cooler        $30",
            "Monitor": "24\" FHD 165Hz           $170",
            "Total":   "                        ~$875",
        },
    },
    {
        "name": "Mid-Range Gaming",
        "tags": ["mid", "gaming", "gamer", "1440p", "qhd", "fps", "aaa",
                 "rtx_3070ti", "ryzen_7800x3d", "16gb_ram", "b650_mobo",
                 "ssd_1tb", "750w_psu", "aio_240_cooler",
                 "qhd_monitor", "single_monitor"],
        "parts": {
            "GPU":     "RTX 3070 Ti 8GB         $400",
            "CPU":     "Ryzen 7 7800X3D         $340",
            "RAM":     "16GB DDR5 6000MHz       $55",
            "Board":   "B650 ATX                $150",
            "Storage": "SSD 1TB NVMe            $75",
            "PSU":     "750W 80+ Gold           $90",
            "Cooler":  "AIO 240mm Liquid        $80",
            "Monitor": "27\" QHD 165Hz           $280",
            "Total":   "                        ~$1,470",
        },
    },
    {
        "name": "High-End Gaming",
        "tags": ["high", "gaming", "gamer", "4k", "ultra", "aaa", "vr",
                 "rtx_4080", "ryzen_7800x3d", "32gb_ram", "x670_mobo",
                 "ssd_2tb", "850w_psu", "aio_360_cooler",
                 "4k_monitor", "single_monitor"],
        "parts": {
            "GPU":     "RTX 4080 16GB           $950",
            "CPU":     "Ryzen 7 7800X3D         $340",
            "RAM":     "32GB DDR5 6000MHz       $95",
            "Board":   "X670E ATX               $270",
            "Storage": "SSD 2TB NVMe            $130",
            "PSU":     "850W 80+ Gold           $120",
            "Cooler":  "AIO 360mm Liquid        $120",
            "Monitor": "32\" 4K 144Hz            $500",
            "Total":   "                        ~$2,525",
        },
    },
    {
        "name": "Ultimate Gaming",
        "tags": ["ultimate", "extreme", "gaming", "gamer", "4k", "ultra",
                 "rtx_4090", "i9_14900k", "32gb_ram", "z790_mobo",
                 "ssd_2tb", "1000w_psu", "aio_360_cooler",
                 "4k_monitor", "single_monitor"],
        "parts": {
            "GPU":     "RTX 4090 24GB           $1,600",
            "CPU":     "Intel i9-14900K         $530",
            "RAM":     "32GB DDR5 7200MHz       $130",
            "Board":   "Z790 ATX                $350",
            "Storage": "SSD 2TB NVMe Gen5       $200",
            "PSU":     "1000W 80+ Platinum      $180",
            "Cooler":  "AIO 360mm Liquid        $150",
            "Monitor": "32\" 4K 240Hz OLED       $900",
            "Total":   "                        ~$4,040",
        },
    },

    # --- Developer builds ---
    {
        "name": "Budget Developer",
        "tags": ["budget", "developer", "dev", "coding", "programming",
                 "web", "backend", "frontend",
                 "rtx_3060", "i5_13600k", "intel", "32gb_ram", "b760_mobo",
                 "ssd_1tb", "650w_psu", "air_cooler",
                 "fhd_monitor", "dual_monitor"],
        "parts": {
            "GPU":     "RTX 3060 12GB           $270",
            "CPU":     "Intel i5-13600K         $240",
            "RAM":     "32GB DDR5 5600MHz       $85",
            "Board":   "B760 ATX                $130",
            "Storage": "SSD 1TB NVMe            $75",
            "PSU":     "650W 80+ Bronze         $60",
            "Cooler":  "Tower Air Cooler        $35",
            "Monitor": "2x 24\" FHD IPS          $300",
            "Total":   "                        ~$1,195",
        },
    },
    {
        "name": "Mid-Range Developer",
        "tags": ["mid", "developer", "dev", "coding", "programming",
                 "web", "backend", "docker", "containers", "vm",
                 "rtx_3070ti", "i7_13700k", "intel", "32gb_ram", "b760_mobo",
                 "ssd_1tb", "hdd_2tb", "750w_psu", "aio_240_cooler",
                 "fhd_monitor", "dual_monitor"],
        "parts": {
            "GPU":     "RTX 3070 Ti 8GB         $400",
            "CPU":     "Intel i7-13700K         $350",
            "RAM":     "32GB DDR5 6000MHz       $95",
            "Board":   "B760 ATX                $140",
            "Storage": "SSD 1TB NVMe + HDD 2TB  $120",
            "PSU":     "750W 80+ Gold           $90",
            "Cooler":  "AIO 240mm Liquid        $80",
            "Monitor": "2x 27\" FHD IPS          $380",
            "Total":   "                        ~$1,655",
        },
    },
    {
        "name": "Senior Developer / DevOps",
        "tags": ["senior", "developer", "dev", "devops", "coding",
                 "programming", "docker", "containers", "vm", "kubernetes",
                 "rtx_4070", "i9_14900k", "intel", "64gb_ram", "z790_mobo",
                 "ssd_2tb", "hdd_5tb", "850w_psu", "aio_360_cooler",
                 "fhd_monitor", "triple_monitor"],
        "parts": {
            "GPU":     "RTX 4070 12GB           $550",
            "CPU":     "Intel i9-14900K         $530",
            "RAM":     "64GB DDR5 5600MHz       $170",
            "Board":   "Z790 ATX                $280",
            "Storage": "SSD 2TB NVMe + HDD 5TB  $210",
            "PSU":     "850W 80+ Gold           $120",
            "Cooler":  "AIO 360mm Liquid        $120",
            "Monitor": "3x 24\" FHD IPS          $450",
            "Total":   "                        ~$2,430",
        },
    },

    # --- Content creation ---
    {
        "name": "Video Editor / Content Creator",
        "tags": ["content", "creator", "video", "editing", "premiere",
                 "davinci", "youtube", "streaming", "render",
                 "rtx_4080", "i9_14900k", "intel", "64gb_ram", "z790_mobo",
                 "ssd_2tb", "hdd_5tb", "850w_psu", "aio_360_cooler",
                 "4k_monitor", "single_monitor"],
        "parts": {
            "GPU":     "RTX 4080 16GB           $950",
            "CPU":     "Intel i9-14900K         $530",
            "RAM":     "64GB DDR5 5600MHz       $170",
            "Board":   "Z790 ATX                $280",
            "Storage": "SSD 2TB NVMe + HDD 5TB  $210",
            "PSU":     "850W 80+ Gold           $120",
            "Cooler":  "AIO 360mm Liquid        $120",
            "Monitor": "32\" 4K IPS Color-Acc    $600",
            "Total":   "                        ~$2,980",
        },
    },
    {
        "name": "3D Artist / Blender",
        "tags": ["3d", "artist", "blender", "modeling", "render",
                 "animation", "creative", "design",
                 "rtx_4090", "i9_14900k", "intel", "64gb_ram", "z790_mobo",
                 "ssd_2tb", "hdd_5tb", "1000w_psu", "aio_360_cooler",
                 "4k_monitor", "dual_monitor"],
        "parts": {
            "GPU":     "RTX 4090 24GB           $1,600",
            "CPU":     "Intel i9-14900K         $530",
            "RAM":     "64GB DDR5 5600MHz       $170",
            "Board":   "Z790 ATX                $350",
            "Storage": "SSD 2TB NVMe + HDD 5TB  $210",
            "PSU":     "1000W 80+ Platinum      $180",
            "Cooler":  "AIO 360mm Liquid        $150",
            "Monitor": "32\" 4K IPS + 27\" QHD    $900",
            "Total":   "                        ~$4,090",
        },
    },

    # --- ML / Data Science ---
    {
        "name": "Data Scientist",
        "tags": ["data", "science", "scientist", "ml", "machine_learning",
                 "python", "jupyter", "pandas", "analysis",
                 "rtx_4070", "i7_13700k", "intel", "64gb_ram", "z790_mobo",
                 "ssd_2tb", "hdd_2tb", "750w_psu", "aio_240_cooler",
                 "qhd_monitor", "dual_monitor"],
        "parts": {
            "GPU":     "RTX 4070 12GB           $550",
            "CPU":     "Intel i7-13700K         $350",
            "RAM":     "64GB DDR5 5600MHz       $170",
            "Board":   "Z790 ATX                $280",
            "Storage": "SSD 2TB NVMe + HDD 2TB  $170",
            "PSU":     "750W 80+ Gold           $90",
            "Cooler":  "AIO 240mm Liquid        $80",
            "Monitor": "2x 27\" QHD IPS          $500",
            "Total":   "                        ~$2,190",
        },
    },
    {
        "name": "ML Engineer / Deep Learning",
        "tags": ["ml", "machine_learning", "deep_learning", "ai",
                 "training", "model", "pytorch", "tensorflow", "gpu_compute",
                 "rtx_4090", "i9_14900k", "intel", "128gb_ram", "z790_mobo",
                 "ssd_2tb", "hdd_5tb", "1000w_psu", "aio_360_cooler",
                 "4k_monitor", "dual_monitor"],
        "parts": {
            "GPU":     "RTX 4090 24GB           $1,600",
            "CPU":     "Intel i9-14900K         $530",
            "RAM":     "128GB DDR5 4800MHz      $320",
            "Board":   "Z790 ATX                $350",
            "Storage": "SSD 2TB NVMe + HDD 5TB  $210",
            "PSU":     "1000W 80+ Platinum      $180",
            "Cooler":  "AIO 360mm Liquid        $150",
            "Monitor": "32\" 4K + 27\" QHD        $780",
            "Total":   "                        ~$4,120",
        },
    },

    # --- Budget / Office ---
    {
        "name": "Office / Productivity",
        "tags": ["office", "budget", "cheap", "basic", "productivity",
                 "word", "excel", "email", "browsing",
                 "integrated_gpu", "i5_13600k", "intel", "16gb_ram",
                 "b760_mobo", "ssd_500gb", "450w_psu", "stock_cooler",
                 "fhd_monitor", "single_monitor"],
        "parts": {
            "GPU":     "Intel UHD (integrated)  $0",
            "CPU":     "Intel i5-13400          $200",
            "RAM":     "16GB DDR5 4800MHz       $45",
            "Board":   "B760 Micro-ATX          $100",
            "Storage": "SSD 500GB NVMe          $45",
            "PSU":     "450W 80+ Bronze         $40",
            "Cooler":  "Stock Cooler            $0",
            "Monitor": "24\" FHD IPS             $150",
            "Total":   "                        ~$580",
        },
    },
    {
        "name": "Home Server / NAS",
        "tags": ["server", "nas", "storage", "backup", "home_lab",
                 "plex", "media", "raid",
                 "integrated_gpu", "i5_13600k", "intel", "32gb_ram",
                 "b760_mobo", "ssd_500gb", "hdd_5tb", "hdd_5tb_2",
                 "650w_psu", "air_cooler",
                 "fhd_monitor", "single_monitor"],
        "parts": {
            "GPU":     "Intel UHD (integrated)  $0",
            "CPU":     "Intel i5-13600          $200",
            "RAM":     "32GB DDR5 4800MHz       $85",
            "Board":   "B760 ATX                $130",
            "Storage": "SSD 500GB + 2x HDD 5TB  $255",
            "PSU":     "650W 80+ Bronze         $60",
            "Cooler":  "Tower Air Cooler        $30",
            "Monitor": "24\" FHD (for setup)     $150",
            "Total":   "                        ~$910",
        },
    },
]

# Synonyms: map common words to the tags used in builds.
# TOY: brute-force string matching dict. Only recognizes words we manually added.
#      "highend" matches "high" only because of substring search in parse_query().
# REAL: a neural tokenizer (BPE/WordPiece) that breaks ANY text into subwords.
#       It understands "highend" = "high" + "end" without a synonym dict,
#       because it learned from billions of text examples during pre-training.
SYNONYMS = {
    "3060":      "rtx_3060",
    "3070":      "rtx_3070ti",
    "3070ti":    "rtx_3070ti",
    "4070":      "rtx_4070",
    "4080":      "rtx_4080",
    "4090":      "rtx_4090",
    "5600":      "ryzen_5600",
    "5600x":     "ryzen_5600",
    "7800x3d":   "ryzen_7800x3d",
    "13600k":    "i5_13600k",
    "13700k":    "i7_13700k",
    "14900k":    "i9_14900k",
    "i5":        "i5_13600k",
    "i7":        "i7_13700k",
    "i9":        "i9_14900k",
    "ryzen":     "ryzen_7800x3d",
    "amd":       "ryzen_7800x3d",
    "16gb":      "16gb_ram",
    "32gb":      "32gb_ram",
    "64gb":      "64gb_ram",
    "128gb":     "128gb_ram",
    "game":      "gaming",
    "gamer":     "gamer",
    "games":     "gaming",
    "play":      "gaming",
    "code":      "coding",
    "coder":     "coding",
    "program":   "programming",
    "programmer":"programming",
    "develop":   "developer",
    "edit":      "editing",
    "editor":    "editing",
    "stream":    "streaming",
    "streamer":  "streaming",
    "render":    "render",
    "rendering": "render",
    "cheap":     "budget",
    "affordable":"budget",
    "low":       "budget",
    "expensive": "high",
    "top":       "ultimate",
    "best":      "ultimate",
    "pro":       "senior",
    "professional": "senior",
    "deep":      "deep_learning",
    "learning":  "machine_learning",
    "workstation": "developer",
    "youtube":   "youtube",
    "video":     "video",
    "blender":   "blender",
    "modeling":  "modeling",
    "server":    "server",
    "nas":       "nas",
    "plex":      "plex",
    "office":    "office",
    "work":      "office",
    "basic":     "basic",
    "1080p":     "fhd_monitor",
    "1440p":     "qhd_monitor",
    "fhd":       "fhd_monitor",
    "qhd":       "qhd_monitor",
    "4k":        "4k_monitor",
}


# ---------------------------------------------------------------------------
# Embedding model: learns vectors for each tag from build co-occurrence
#
# TOY: just an embedding table (118 tags x 32 dims = 3,776 parameters).
#      Each tag is a single row in the table. Sentence = average of tag vectors.
# REAL: a transformer model (e.g., BERT) with 100M+ parameters.
#      Uses 12+ attention layers to create context-aware vectors where
#      the same word gets DIFFERENT vectors depending on surrounding words.
# ---------------------------------------------------------------------------

class PCEmbeddingModel:
    def __init__(self, dim=32):
        self.dim = dim
        self.vocab = {}
        self.embeddings = None

    def build_vocab(self, builds):
        # TOY: vocabulary is just the set of all tags from our 13 builds.
        # REAL: 30K+ subword tokens learned from billions of text samples.
        all_tags = set()
        for build in builds:
            all_tags.update(build["tags"])
        self.vocab = {tag: i for i, tag in enumerate(sorted(all_tags))}
        self.embeddings = np.random.randn(len(self.vocab), self.dim) * 0.1

    def train(self, builds, epochs=500, lr=0.05):
        losses = []
        for epoch in range(epochs):
            epoch_loss = 0
            np.random.shuffle(builds)

            for build in builds:
                tag_ids = [self.vocab[t] for t in build["tags"] if t in self.vocab]
                if len(tag_ids) < 2:
                    continue

                vecs = self.embeddings[tag_ids]
                center = vecs.mean(axis=0)

                # Pull tags in same build toward their center
                for idx in tag_ids:
                    diff = center - self.embeddings[idx]
                    self.embeddings[idx] += lr * diff * 0.1
                    epoch_loss += np.sum(diff ** 2)

                # Push away tags NOT in this build
                other_ids = [i for i in range(len(self.vocab)) if i not in tag_ids]
                if other_ids:
                    n_neg = min(len(other_ids), len(tag_ids) * 2)
                    neg_ids = np.random.choice(other_ids, n_neg, replace=False)
                    for neg_id in neg_ids:
                        diff = self.embeddings[neg_id] - center
                        dist = np.sum(diff ** 2)
                        if dist < 2.0:
                            self.embeddings[neg_id] += lr * diff * 0.05

            # Normalize all vectors to unit length
            norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-8)
            self.embeddings = self.embeddings / norms

            losses.append(epoch_loss / len(builds))
        return losses

    def embed_tags(self, tags):
        # TOY: average pooling — just average all tag vectors into one.
        #      Loses word order: "budget gaming" = "gaming budget".
        # REAL: transformer attention layers mix token vectors context-aware,
        #       so word order and relationships between words are preserved.
        ids = [self.vocab[t] for t in tags if t in self.vocab]
        if not ids:
            return np.zeros(self.dim)
        vec = self.embeddings[ids].mean(axis=0)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec

    def embed_build(self, build):
        return self.embed_tags(build["tags"])


def parse_query(text):
    # TOY: string matching tokenizer — splits on spaces, looks up each word in
    #      SYNONYMS dict, falls back to substring matching. Cannot understand
    #      words it hasn't seen ("highend" only works because "high" is a substring).
    # REAL: neural tokenizer (BPE/WordPiece) splits ANY text into known subwords.
    #       Then the transformer model understands meaning from context, not from
    #       a hand-written synonym dictionary.
    words = text.lower().replace(",", " ").replace("-", " ").split()
    tags = []
    for word in words:
        word = word.strip()
        if not word or word in ("for", "a", "an", "the", "with", "and", "pc", "build", "setup", "computer"):
            continue
        if word in SYNONYMS:
            tags.append(SYNONYMS[word])
        elif word in model.vocab:
            tags.append(word)
        else:
            # Fallback: substring match (fragile — "highend" matches "high")
            for vocab_word in model.vocab:
                if word in vocab_word or vocab_word in word:
                    tags.append(vocab_word)
                    break
            else:
                for syn_key, syn_val in SYNONYMS.items():
                    if word in syn_key or syn_key in word:
                        tags.append(syn_val)
                        break
    return list(set(tags))


def cosine_sim(a, b):
    dot = np.dot(a, b)
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def recommend(query_text, top_k=3):
    tags = parse_query(query_text)
    if not tags:
        print(f"\n  Could not understand query. Try keywords like:")
        print(f"    'gaming', '3070 ti', 'developer', 'budget', '4k', 'ml', etc.")
        return

    query_vec = model.embed_tags(tags)

    # Brute-force search: loop through all builds and compute cosine similarity.
    # With 13 builds this is instant. At scale (millions of docs), you'd use a
    # vector DB (Pinecone, pgvector, ChromaDB) which builds an index for O(log n) search.
    scores = []
    for build in BUILDS:
        build_vec = model.embed_build(build)
        sim = cosine_sim(query_vec, build_vec)
        scores.append((sim, build))

    scores.sort(key=lambda x: -x[0])

    print(f"\n  Parsed tags: {tags}")
    print(f"\n  {'='*65}")
    print(f"  Top {top_k} Recommended Builds:")
    print(f"  {'='*65}")

    for rank, (sim, build) in enumerate(scores[:top_k]):
        bar_len = int(sim * 30)
        bar = "#" * bar_len + "." * (30 - bar_len)
        print(f"\n  #{rank+1}  {build['name']}")
        print(f"      Match: [{bar}] {sim:.1%}")
        print(f"      {'─'*50}")
        for part, desc in build["parts"].items():
            print(f"      {part:<8} {desc}")
        print()


# ---------------------------------------------------------------------------
# Train and run
# ---------------------------------------------------------------------------

print("PC Build Recommender — Training embedding model...\n")

model = PCEmbeddingModel(dim=32)
model.build_vocab(BUILDS)
losses = model.train(BUILDS, epochs=500, lr=0.05)

# In production, save the trained model to a file so you don't retrain every time:
#   np.savez("pc_model.npz", embeddings=model.embeddings, vocab=list(model.vocab.keys()))
# Then load with:
#   data = np.load("pc_model.npz", allow_pickle=True)
# Real models use formats like .safetensors (HuggingFace), .bin (PyTorch), or .onnx (portable).

print(f"  Vocabulary: {len(model.vocab)} tags")
print(f"  Builds:     {len(BUILDS)} profiles")
print(f"  Training:   500 epochs")
print(f"  Loss:       {losses[0]:.2f} -> {losses[-1]:.2f}\n")

# Show what the model learned
print("  What the model learned (tag similarities):\n")
test_pairs = [
    ("gaming",     "gamer"),
    ("gaming",     "developer"),
    ("rtx_4090",   "ultimate"),
    ("rtx_4090",   "budget"),
    ("rtx_3060",   "budget"),
    ("coding",     "dual_monitor"),
    ("coding",     "single_monitor"),
    ("64gb_ram",   "deep_learning"),
    ("16gb_ram",   "deep_learning"),
    ("intel",      "developer"),
]

for t1, t2 in test_pairs:
    if t1 in model.vocab and t2 in model.vocab:
        v1 = model.embeddings[model.vocab[t1]]
        v2 = model.embeddings[model.vocab[t2]]
        sim = cosine_sim(v1, v2)
        bar = "#" * int(max(0, sim) * 20)
        print(f"    {t1:<18} <-> {t2:<18} {sim:>6.3f}  {bar}")

print(f"\n{'='*65}")
print("  Try queries like:")
print('    "3070 ti for gamer"')
print('    "3070 ti for developer"')
print('    "budget gaming pc"')
print('    "4090 machine learning"')
print('    "cheap office computer"')
print('    "video editing workstation"')
print('    "blender 3d artist"')
print(f"{'='*65}\n")

while True:
    try:
        query = input("  Query (or 'quit'): ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\n  Bye!")
        break

    if not query or query.lower() in ("quit", "exit", "q"):
        print("  Bye!")
        break

    recommend(query)
