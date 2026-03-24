# End-to-End Flow: Image Classifier (Cats vs Dogs)

## Overview

This document explains the complete pipeline from raw images to a deployed deep learning model, comparing training from scratch vs transfer learning.

---

## Flow Diagram

```
Raw Images (cats/ and dogs/ folders)
          │
          ▼
┌──────────────────────┐
│   Data Loading       │  ← CatsDogsDataset (custom PyTorch Dataset)
│   (PIL Images)       │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│  Data Augmentation   │  ← Random crop, horizontal flip,
│  (training only)     │    rotation, color jitter
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│  Normalize           │  ← ImageNet mean/std
│  (mean, std)         │    [0.485, 0.456, 0.406]
└──────────┬───────────┘
           │
     ┌─────┴──────┐
     ▼            ▼
  DataLoader   DataLoader
  (train)       (val)
     │
     ▼
 ┌───────────────────────────────────────────┐
 │          MODEL COMPARISON                  │
 │                                           │
 │  ┌─────────────┐    ┌──────────────────┐  │
 │  │ Simple CNN  │    │ ResNet18         │  │
 │  │ (scratch)   │    │ (Transfer Learn) │  │
 │  └──────┬──────┘    └────────┬─────────┘  │
 └─────────┼────────────────────┼────────────┘
           │                    │
           ▼                    ▼
   Training Loop          Training Loop
   (from scratch)         (frozen backbone)
           │                    │
           ▼                    ▼
       ~70-80%              ~95%+ accuracy
           │                    │
           └──────────┬─────────┘
                      │
                      ▼
           ┌──────────────────┐
           │  Best Model      │  ← best_model.pth checkpoint
           │  Export to ONNX  │  ← model.onnx (cross-platform)
           └──────────────────┘
```

---

## Step-by-Step Breakdown

### Step 1: Dataset & Data Loading
**File**: `src/dataset.py` → `CatsDogsDataset`, `get_data_loaders()`

The dataset structure expected:
```
data/
├── cats/
│   ├── cats_0001.jpg
│   └── ...
└── dogs/
    ├── dogs_0001.jpg
    └── ...
```

If the real Kaggle dataset isn't available, synthetic images are auto-generated for demonstration. In production, use the [Kaggle Cats vs Dogs dataset](https://www.kaggle.com/c/dogs-vs-cats).

```python
# 80% train, 20% validation split
train_loader, val_loader = get_data_loaders(
    data_dir='data/',
    batch_size=32,
    image_size=224,
    val_split=0.2
)
```

---

### Step 2: Data Augmentation
**File**: `src/dataset.py` → `get_transforms(train=True)`

Augmentation artificially increases the effective dataset size:

```
Original image (224×224)
        │
        ├── Random crop from 256×256 → 224×224  (random positioning)
        ├── Random horizontal flip (50% chance)
        ├── Random rotation ±15°
        ├── Color jitter (brightness, contrast, saturation ±20%)
        └── Normalize: (pixel - mean) / std
```

Why augment?
- Prevents overfitting with small datasets
- Makes model invariant to orientation, lighting
- Training gets: every image seen differently each epoch

**Validation transforms** skip augmentation (only resize + normalize) to get consistent metrics.

---

### Step 3: Model A — Simple CNN (From Scratch)
**File**: `src/model.py` → `SimpleCNN`

```
Input: [batch, 3, 224, 224]
   │
   ▼  Conv2d(3→32, 3×3) + BatchNorm + ReLU + MaxPool(2×2)
   ▼  Output: [batch, 32, 112, 112]
   │
   ▼  Conv2d(32→64, 3×3) + BatchNorm + ReLU + MaxPool(2×2)
   ▼  Output: [batch, 64, 56, 56]
   │
   ▼  Conv2d(64→128, 3×3) + BatchNorm + ReLU + MaxPool(2×2)
   ▼  Output: [batch, 128, 28, 28]
   │
   ▼  Conv2d(128→256, 3×3) + BatchNorm + ReLU + MaxPool(2×2)
   ▼  Output: [batch, 256, 14, 14]
   │
   ▼  Flatten → [batch, 50176]
   │
   ▼  Dropout(0.5) → FC(50176→512) → ReLU → Dropout(0.5) → FC(512→2)
   │
   ▼  Output: [batch, 2]  (cat logits, dog logits)
```

**Problem**: Starting from random weights requires a lot of data to learn meaningful features. Achieves ~70-80% accuracy.

---

### Step 4: Model B — Transfer Learning (ResNet18)
**File**: `src/model.py` → `ResNetTransfer`

Instead of learning from scratch, we **steal** learned features from ImageNet:

```
Pretrained ResNet18 (trained on 1.2M ImageNet images)
   │
   ├── Frozen backbone (all layers locked)
   │   └── Already knows: edges, textures, shapes, object parts
   │
   └── Custom classifier head (only these layers train):
       Dropout(0.5) → FC(512→256) → ReLU → Dropout(0.3) → FC(256→2)
```

**Why it works so well**:
- Low-level features (edges, corners) are universal across vision tasks
- Only the final classification layers need to adapt to cats/dogs
- Achieves ~95%+ with far fewer training examples

**Fine-tuning strategy**:
```
Phase 1: Freeze all backbone → train classifier head only (fast, stable)
Phase 2: Unfreeze last few layers → fine-tune with very low LR (optional)
```

---

### Step 5: Training Loop
**File**: `src/train.py` → `Trainer.train()`

```
For each epoch:
  ┌─── Training Phase ────────────────────────────────────────┐
  │  For each batch:                                          │
  │    1. Forward pass:  output = model(images)               │
  │    2. Compute loss:  loss = CrossEntropy(output, labels)  │
  │    3. Backward pass: loss.backward()  (compute gradients) │
  │    4. Update weights: optimizer.step()                    │
  └───────────────────────────────────────────────────────────┘
         │
         ▼
  ┌─── Validation Phase ───────────────────────────────────────┐
  │  For each batch (no gradient computation):                │
  │    output = model(images)                                  │
  │    Compute loss and accuracy                               │
  └───────────────────────────────────────────────────────────┘
         │
         ▼
  If val_acc improved → save best_model.pth
  If no improvement for 5 epochs → Early Stop
```

**Optimizers**:
- Classifier head: `lr = 0.001`
- Backbone (if unfrozen): `lr = 0.0001` (10× smaller)

---

### Step 6: Evaluation

At the end of training:

| Model | Val Accuracy | Notes |
|-------|-------------|-------|
| Simple CNN | 70-80% | Limited by small dataset |
| ResNet18 Transfer | 95%+ | Pre-learned features |

---

### Step 7: ONNX Export
**File**: `src/model.py` → `export_to_onnx()`

Converting to ONNX makes the model runnable on any platform:

```
PyTorch Model (.pt)
      │
      ▼  torch.onnx.export()
      │  - Traces the computation graph
      │  - Serializes ops in ONNX format
      ▼
ONNX Model (.onnx)
      │
      ├── Run with: onnxruntime (Python, C++, Java, Android...)
      └── Convert to TFLite for mobile
```

---

## Data Flow Through Code

```
data/cats/*.jpg + data/dogs/*.jpg
  └── CatsDogsDataset.__getitem__()
        └── PIL.Image.open() → apply transforms → Tensor [3, 224, 224]
              └── DataLoader batches → [32, 3, 224, 224]
                    └── Trainer.train_epoch()
                          ├── model(inputs) → [32, 2] logits
                          ├── CrossEntropyLoss(logits, labels)
                          ├── loss.backward() → gradients
                          └── optimizer.step() → updated weights
```

---

## Running the Full Flow

```bash
# Step 1: Install dependencies
pip install -r requirements.txt

# Step 2: Train (auto-generates synthetic data if needed)
python src/main.py

# Expected output:
# Device: mps / cuda / cpu
# 1. Preparing dataset...  → 320 images per class
# 3. Training Simple CNN   → Epoch 1-5, val_acc improving
# 5. Training ResNet18     → Epoch 1-10, val_acc ~95%
# 6. ONNX exported         → models/resnet/model.onnx
```

---

## Key Learnings

| Concept | Where Used |
|---------|-----------|
| CNN architecture | Conv + BN + ReLU + MaxPool layers |
| Transfer learning | ResNet18 pretrained on ImageNet |
| Data augmentation | Random crops, flips, color jitter |
| Batch Normalization | Faster training, more stable |
| Early stopping | Prevents overfitting |
| ONNX export | Cross-platform deployment |
| Learning rate differential | Backbone vs classifier head LR |
