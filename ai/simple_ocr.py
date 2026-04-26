"""
Simple OCR Training — How OCR Models Actually Work

Uses your real Mabinogi training data (3,826 images) to build an OCR model
from scratch in PyTorch. This is a simplified version of what your
deep-text-recognition-benchmark model does.

    Your real model:   TPS → ResNet(45 layers) → BiLSTM → CTC
    This script:       Simple CNN(3 layers) → Linear → CTC

Different complexity, same principle.

Usage:
    python simple_ocr.py                       # train and show results
    python simple_ocr.py predict <image_path>  # predict on a single image
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.expanduser(
    "~/workspace/ocr_training_data/mabinogi/ocr/attr_mc_model/v5/train_data"
)
CHARS_FILE = os.path.expanduser(
    "~/workspace/ocr_training_data/mabinogi/ocr/attr_mc_model/v5/unique_chars.txt"
)
MODEL_PATH = os.path.join(BASE_DIR, "simple_ocr_model.pth")

# ── Character Set ──────────────────────────────────────────
#
# Read the same character file your real model uses.
# These are all characters the model can output:
#   %()+./0123456789:~가격공구남내능대런레력련률리마미방밸법벨보부상성수숙술스싱아어용은이재전점제증지컬크킬탄템투티피해호환횟

with open(CHARS_FILE) as f:
    CHARS = f.read().strip()

# Index 0 = CTC blank (a special "no character" token — explained below)
BLANK = 0
char_to_idx = {ch: i + 1 for i, ch in enumerate(CHARS)}
idx_to_char = {i + 1: ch for i, ch in enumerate(CHARS)}
idx_to_char[BLANK] = ""
NUM_CLASSES = len(CHARS) + 1  # characters + 1 CTC blank

IMG_HEIGHT = 32
IMG_WIDTH = 128

# Use GPU if available, otherwise CPU.
# .to(DEVICE) moves tensors/models to the chosen hardware.
# Training on GPU is faster because it runs thousands of multiplications
# in parallel — a 3070 Ti has 6,144 cores vs CPU's ~16 cores.
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ══════════════════════════════════════════════════════════
# Step 1: Dataset — Load Images and Labels
# ══════════════════════════════════════════════════════════
#
# Your images are tiny grayscale crops: "공격" (20×13), "크리티컬" (39×13).
# We resize them all to the same height (32px) and pad to a fixed width
# (128px) so they can go into the network as a batch.
#
# This is exactly what your real pipeline does too — the config says
# imgH=32, imgW=200.

class OCRDataset(Dataset):
    def __init__(self, data_dir):
        self.img_dir = os.path.join(data_dir, "images")
        self.lbl_dir = os.path.join(data_dir, "labels")
        self.samples = []

        for f in sorted(os.listdir(self.img_dir)):
            if not f.endswith(".png"):
                continue
            lbl_file = os.path.join(self.lbl_dir, f.replace(".png", ".txt"))
            if not os.path.exists(lbl_file):
                continue
            with open(lbl_file) as fh:
                label = fh.read().strip()
            if all(ch in char_to_idx for ch in label) and len(label) > 0:
                self.samples.append((f, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        fname, label = self.samples[idx]

        # Load as grayscale
        img = Image.open(os.path.join(self.img_dir, fname)).convert("L")

        # Resize: keep aspect ratio, fit to IMG_HEIGHT
        w, h = img.size
        new_w = min(int(w * IMG_HEIGHT / h), IMG_WIDTH)
        new_w = max(new_w, 1)
        img = img.resize((new_w, IMG_HEIGHT), Image.BILINEAR)

        # To tensor, normalize to [0, 1]
        img_array = np.array(img, dtype=np.float32) / 255.0

        # Pad to fixed width with zeros (black) on the right
        padded = np.zeros((IMG_HEIGHT, IMG_WIDTH), dtype=np.float32)
        padded[:, :new_w] = img_array

        # Shape: (1, 32, 128) — 1 channel (grayscale), 32 high, 128 wide
        tensor = torch.FloatTensor(padded).unsqueeze(0)

        # Encode label: "공격" → [1, 2] (indices into character set)
        label_encoded = torch.IntTensor([char_to_idx[ch] for ch in label])

        return tensor, label_encoded, len(label)


def collate_fn(batch):
    images, labels, label_lengths = zip(*batch)
    images = torch.stack(images)
    labels = torch.cat(labels)
    label_lengths = torch.IntTensor(label_lengths)
    return images, labels, label_lengths


# ══════════════════════════════════════════════════════════
# Step 2: The Model — CNN + Linear
# ══════════════════════════════════════════════════════════
#
# Lesson 10 used nn.Linear — it looks at ALL 64 pixels at once.
# For images, that's wasteful. A "3" in the top-left should be
# recognized the same as a "3" in the center.
#
# CNN (Convolutional Neural Network) fixes this:
#   nn.Linear:  one big weight matrix for ALL pixels
#   nn.Conv2d:  a SMALL weight matrix (3×3) that SLIDES across the image
#
# Imagine a 3×3 magnifying glass scanning left to right, top to bottom.
# It detects local patterns (edges, curves, strokes) regardless of position.
#
#   Layer 1: detects simple things — horizontal edge, vertical edge, corner
#   Layer 2: combines edges — "L shape", "curve", "loop"
#   Layer 3: combines shapes — "ㄱ stroke", "circle", "공 radical"
#
# After the CNN, we read features left-to-right and predict one character
# per horizontal position. This is how OCR reads text spatially.

class SimpleOCR(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.cnn = nn.Sequential(
            # Input: (1, 32, 128) — 1 channel, 32 height, 128 width
            nn.Conv2d(1, 32, kernel_size=3, padding=1),   # → (32, 32, 128)
            nn.ReLU(),
            nn.MaxPool2d(2, 2),                           # → (32, 16, 64)

            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # → (64, 16, 64)
            nn.ReLU(),
            nn.MaxPool2d(2, 2),                           # → (64, 8, 32)

            nn.Conv2d(64, 128, kernel_size=3, padding=1), # → (128, 8, 32)
            nn.ReLU(),
            nn.MaxPool2d((2, 1)),                         # → (128, 4, 32)
            # Height shrinks to 4 but width stays at 32.
            # We keep the width because that's our "time axis" —
            # each of the 32 horizontal positions will predict a character.
        )

        # Each horizontal position has 128 channels × 4 height = 512 features.
        # Project to num_classes to predict a character at each position.
        self.fc = nn.Linear(128 * 4, num_classes)

    def forward(self, x):
        # x: (batch, 1, 32, 128)
        features = self.cnn(x)
        # features: (batch, 128, 4, 32)

        batch, channels, height, width = features.shape

        # Collapse channels and height: (batch, 128, 4, 32) → (batch, 512, 32)
        features = features.view(batch, channels * height, width)

        # Rearrange to: (batch, 32, 512) — 32 positions, each with 512 features
        features = features.permute(0, 2, 1)

        # Predict character at each position: (batch, 32, 512) → (batch, 32, 69)
        output = self.fc(features)

        # CTC expects time-first: (32, batch, 69)
        output = output.permute(1, 0, 2)

        return F.log_softmax(output, dim=2)


# ══════════════════════════════════════════════════════════
# Step 3: CTC Loss — The Key to OCR
# ══════════════════════════════════════════════════════════
#
# The HARD PROBLEM of OCR:
#   Image width = 128 pixels → after CNN, 32 positions
#   Label "공격" = 2 characters
#   How do you align 32 predictions to 2 characters?
#
# You could try to figure out "공 is at pixel 5-15, 격 is at pixel 16-25"
# but that's extremely difficult and fragile.
#
# CTC (Connectionist Temporal Classification) solves this elegantly:
#   1. The model predicts a character at EACH of the 32 positions
#   2. Most positions output "blank" (meaning "nothing here")
#   3. To decode, merge consecutive duplicates and remove blanks
#
# Example — model output for "공격":
#   Position: 1  2  3  4  5  6  7  8  9 10 11 ... 32
#   Output:   -  -  -  공 공 공  -  -  격 격  -  ...  -
#   (where - is the "blank" token)
#
#   Step 1: Remove consecutive duplicates: - 공 - 격 -
#   Step 2: Remove blanks:                 공 격
#   Result: "공격" ✓
#
# The beauty: the model doesn't need to know WHERE each character is.
# CTC considers ALL possible alignments and finds the best one.
# This is exactly what your real model uses too (Prediction: CTC in config).


# ══════════════════════════════════════════════════════════
# Step 4: Training Loop
# ══════════════════════════════════════════════════════════

def train_model():
    print("=" * 60)
    print("Simple OCR Training")
    print("=" * 60)
    print(f"Data:       {DATA_DIR}")
    print(f"Characters: {len(CHARS)} ({CHARS[:20]}...)")
    print(f"Classes:    {NUM_CLASSES} (68 chars + 1 CTC blank)")
    print()

    dataset = OCRDataset(DATA_DIR)
    print(f"Total samples: {len(dataset)}")

    # Split into train (90%) and test (10%)
    n = len(dataset)
    n_train = int(n * 0.9)
    n_test = n - n_train
    train_set, test_set = torch.utils.data.random_split(
        dataset, [n_train, n_test],
        generator=torch.Generator().manual_seed(42)
    )
    print(f"Train: {n_train}, Test: {n_test}")

    train_loader = DataLoader(
        train_set, batch_size=64, shuffle=True, collate_fn=collate_fn
    )
    test_loader = DataLoader(
        test_set, batch_size=64, shuffle=False, collate_fn=collate_fn
    )

    model = SimpleOCR(NUM_CLASSES).to(DEVICE)  # move all weights to GPU
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel: {total_params:,} parameters")
    print(f"Device: {DEVICE}")
    print("Architecture: Conv(1→32) → Conv(32→64) → Conv(64→128) → Linear(512→69)")
    print()

    # CTC Loss — blank=0 means index 0 is the "blank" token
    criterion = nn.CTCLoss(blank=BLANK, zero_infinity=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Same 5-step loop as lesson 10, just with CTC loss instead of CrossEntropy
    print("Training...")
    print("-" * 60)

    for epoch in range(50):
        model.train()
        total_loss = 0
        n_batches = 0

        for images, labels, label_lengths in train_loader:
            # Move batch to GPU — data is loaded on CPU, model lives on GPU,
            # they must be on the same device for matrix multiplication.
            images = images.to(DEVICE)

            # 1. FORWARD: push images through the network
            outputs = model(images)
            # outputs shape: (32, batch_size, 69)
            #   32 = time steps (horizontal positions after CNN)
            #   69 = probability of each character at each position

            # CTC needs to know the output sequence length for each sample
            output_lengths = torch.full(
                (images.size(0),), outputs.size(0), dtype=torch.long
            )

            # 2. LOSS: CTC computes error across all possible alignments
            loss = criterion(outputs, labels, output_lengths, label_lengths)

            # 3. CLEAR old gradients
            optimizer.zero_grad()

            # 4. BACKWARD: compute gradients
            loss.backward()

            # Clip gradients — prevents exploding gradients in CTC training
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)

            # 5. UPDATE weights
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / n_batches

        # Evaluate every 5 epochs
        if (epoch + 1) % 5 == 0:
            correct, total = evaluate(model, test_loader)
            print(f"  Epoch {epoch+1:3d}/50"
                  f"  loss={avg_loss:.4f}"
                  f"  test={correct}/{total} ({correct/total:.1%})")

    print("-" * 60)
    print()

    # Save the trained model
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")

    return model, test_loader, dataset


# ══════════════════════════════════════════════════════════
# Step 5: CTC Greedy Decoding
# ══════════════════════════════════════════════════════════
#
# After training, the model outputs 32 predictions (one per position).
# We need to turn those into a text string.
#
# Greedy decoding:
#   1. At each position, pick the character with highest probability
#   2. Remove consecutive duplicates: 공공공 → 공
#   3. Remove blanks
#
# Example:
#   Raw:    [-, -, 공, 공, 공, -, -, 격, 격, -, -, -, ...]
#   Dedup:  [-, 공, -, 격, -, ...]
#   Clean:  [공, 격]
#   Result: "공격"

def decode_output(output):
    _, indices = output.max(dim=1)

    chars = []
    prev = BLANK
    for idx in indices:
        idx = idx.item()
        if idx != prev and idx != BLANK:
            chars.append(idx_to_char[idx])
        prev = idx

    return "".join(chars)


def evaluate(model, loader):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels, label_lengths in loader:
            images = images.to(DEVICE)
            outputs = model(images).cpu()  # back to CPU for decoding

            offset = 0
            for i in range(images.size(0)):
                predicted = decode_output(outputs[:, i, :])

                length = label_lengths[i].item()
                actual_indices = labels[offset:offset + length]
                actual = "".join(idx_to_char[j.item()] for j in actual_indices)
                offset += length

                if predicted == actual:
                    correct += 1
                total += 1

    model.train()
    return correct, total


# ══════════════════════════════════════════════════════════
# Step 6: Predict on a Single Image
# ══════════════════════════════════════════════════════════

def predict_image(model, image_path):
    model.eval()

    img = Image.open(image_path).convert("L")
    w, h = img.size
    new_w = min(int(w * IMG_HEIGHT / h), IMG_WIDTH)
    new_w = max(new_w, 1)
    img = img.resize((new_w, IMG_HEIGHT), Image.BILINEAR)

    img_array = np.array(img, dtype=np.float32) / 255.0
    padded = np.zeros((IMG_HEIGHT, IMG_WIDTH), dtype=np.float32)
    padded[:, :new_w] = img_array
    tensor = torch.FloatTensor(padded).unsqueeze(0).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        output = model(tensor).cpu()
        return decode_output(output[:, 0, :])


# ══════════════════════════════════════════════════════════
# Step 7: Show Results
# ══════════════════════════════════════════════════════════

def show_predictions(model):
    print("── Sample Predictions ──")
    print()

    img_dir = os.path.join(DATA_DIR, "images")
    lbl_dir = os.path.join(DATA_DIR, "labels")

    results = []
    for fname in sorted(os.listdir(img_dir)):
        if not fname.endswith(".png"):
            continue
        lbl_file = os.path.join(lbl_dir, fname.replace(".png", ".txt"))
        if not os.path.exists(lbl_file):
            continue
        with open(lbl_file) as f:
            actual = f.read().strip()
        if not all(ch in char_to_idx for ch in actual):
            continue
        predicted = predict_image(model, os.path.join(img_dir, fname))
        results.append((fname, actual, predicted, predicted == actual))

    # Show some correct and some wrong
    correct = [r for r in results if r[3]]
    wrong = [r for r in results if not r[3]]

    print(f"  Overall: {len(correct)}/{len(results)} correct ({len(correct)/len(results):.1%})")
    print()

    print("  Correct predictions:")
    for fname, actual, predicted, _ in correct[:15]:
        print(f"    ✓ {fname}:  '{actual}' → '{predicted}'")

    if wrong:
        print()
        print("  Wrong predictions:")
        for fname, actual, predicted, _ in wrong[:15]:
            print(f"    ✗ {fname}:  '{actual}' → '{predicted}'")

    print()
    print("=" * 60)
    print("To predict on a new image:")
    print(f"  python {os.path.basename(__file__)} predict <image_path>")
    print()
    print("To load the trained model in your own code:")
    print(f"  model = SimpleOCR({NUM_CLASSES})")
    print(f"  model.load_state_dict(torch.load('{MODEL_PATH}'))")
    print(f"  result = predict_image(model, 'path/to/image.png')")


# ══════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════

if __name__ == "__main__":

    if len(sys.argv) >= 3 and sys.argv[1] == "predict":
        # Predict mode: load saved model and predict on an image
        if not os.path.exists(MODEL_PATH):
            print(f"No trained model found at {MODEL_PATH}")
            print("Run without arguments first to train.")
            sys.exit(1)
        model = SimpleOCR(NUM_CLASSES).to(DEVICE)
        model.load_state_dict(torch.load(MODEL_PATH, weights_only=True, map_location=DEVICE))
        result = predict_image(model, sys.argv[2])
        print(result)
    else:
        # Train mode
        model, test_loader, dataset = train_model()
        print()
        show_predictions(model)
