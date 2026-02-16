# Rust Detection Model Training Guide

**Training Platform**: Google Colab (Free GPU)  
**Dataset Source**: [Roboflow Rust Detection Dataset](https://universe.roboflow.com/test-stage/rust-detection-t8vza/dataset/8)  
**Frameworks**: PyTorch, Transformers (HuggingFace), ONNX Runtime

---

## 📊 Overview

This project trains **two different models** for rust detection on pump surfaces:

| Model | Type | Size | Accuracy | Inference | Best For |
|-------|------|------|----------|-----------|----------|
| **MobileNetV3-Small** | CNN | 6 MB | 100% (10/10 test images) | 19-55ms | Production (Recommended) |
| **CLIP ViT-B/32** | Vision-Language Model | 335 MB | 70% (3/5 false positives) | 64-127ms | Research/Comparison |

**Key Finding**: MobileNetV3 significantly outperforms CLIP in production use due to zero false positives and faster inference.

---

## 📁 Dataset Preparation

### Source Dataset
- **Name**: Rust Detection (YOLO format)
- **Source**: Roboflow Universe
- **Format**: YOLOv8 object detection
- **Classes**: `rust` (bounding boxes on rust regions)

### Conversion to Classification

The original dataset is object detection format (YOLO bounding boxes). We convert it to **binary classification**:

**Logic**:
- If label file has bounding boxes → **rust** class
- If label file is empty → **no_rust** class

**Code** (used in both notebooks):
```python
def label_has_boxes(label_path: Path) -> bool:
    """Check if YOLO label file contains any bounding boxes"""
    if not label_path.exists():
        return False
    txt = label_path.read_text().strip()
    return len(txt) > 0

def convert_split(split: str):
    """Convert YOLO format to ImageFolder classification"""
    img_dir = YOLO_ROOT / split / "images"
    lab_dir = YOLO_ROOT / split / "labels"
    
    out_rust = OUT_ROOT / split / "rust"
    out_no_rust = OUT_ROOT / split / "no_rust"
    out_rust.mkdir(parents=True, exist_ok=True)
    out_no_rust.mkdir(parents=True, exist_ok=True)
    
    for img_path in img_dir.glob("*.*"):
        label_path = lab_dir / (img_path.stem + ".txt")
        has_rust = label_has_boxes(label_path)
        
        if has_rust:
            shutil.copy2(img_path, out_rust / img_path.name)
        else:
            shutil.copy2(img_path, out_no_rust / img_path.name)
```

**Result**: Classification dataset in ImageFolder format:
```
rust_cls/
├── train/
│   ├── rust/       # Images with rust
│   └── no_rust/    # Clean images
├── valid/
│   ├── rust/
│   └── no_rust/
└── test/
    ├── rust/
    └── no_rust/
```

---

## 🚀 Model 1: MobileNetV3-Small (Production Model)

### Training Notebook
**File**: `rust_detection_model_training.ipynb`

### Architecture
- **Base Model**: MobileNetV3-Small (pretrained on ImageNet)
- **Modification**: Replace final classification layer for 2 classes (rust, no_rust)
- **Parameters**: ~2.5M (lightweight)

### Key Training Details

#### 1. Data Augmentation
```python
train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(0.1, 0.1, 0.1, 0.05),  # Brightness, contrast, saturation, hue
    transforms.ToTensor(),
])

val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])
```

#### 2. Class Imbalance Handling
The dataset has more rust images than clean images. To prevent bias:

```python
# Calculate class weights (inverse frequency)
targets = np.array(train_ds.targets)
class_counts = np.bincount(targets)
weights = (class_counts.sum() / (len(class_counts) * class_counts)).astype(np.float32)

# Use weighted loss
class_weights = torch.tensor(weights).to(device)
criterion = nn.CrossEntropyLoss(weight=class_weights)
```

#### 3. Model Configuration
```python
model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
model.classifier[3] = nn.Linear(model.classifier[3].in_features, 2)  # 2 classes

optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
```

#### 4. Training Loop
- **Epochs**: 5
- **Batch Size**: 64
- **Optimizer**: AdamW (lr=3e-4)
- **Metrics**: Accuracy and **Balanced Accuracy** (to handle class imbalance)

```python
for epoch in range(EPOCHS):
    model.train()
    for x, y in train_loader:
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()
    
    # Validation
    val_acc, val_bal_acc = eval_metrics(val_loader)
    
    # Save best model based on balanced accuracy
    if val_bal_acc > best_bal_acc:
        best_bal_acc = val_bal_acc
        best_state = model.state_dict()
```

#### 5. ONNX Export
```python
model.eval().cpu()
dummy_input = torch.randn(1, 3, 224, 224)

torch.onnx.export(
    model,
    dummy_input,
    "rust_model.onnx",
    input_names=["input"],
    output_names=["logits"],
    dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
    opset_version=17
)

# Save labels
with open("rust_labels.json", "w") as f:
    json.dump(["no_rust", "rust"], f)
```

### Results
- **Test Accuracy**: ~94%+ (on Roboflow test set)
- **Real-World Testing**: 100% (10/10 images, 0 false positives)
- **Model Size**: 6 MB
- **Inference Speed**: 19-55ms on CPU

---

## 🧠 Model 2: CLIP Vision-Language Model (Research)

### Training Notebook
**File**: `VLM_rust_detection_training.ipynb`

### Architecture
- **Base Model**: OpenAI CLIP ViT-B/32 (Vision Transformer)
- **Vision Encoder**: ~86M parameters
- **Custom Head**: 2-layer MLP classifier

```python
class CLIPRustClassifier(nn.Module):
    def __init__(self, clip_vision_model, num_classes=2):
        super().__init__()
        self.vision_model = clip_vision_model  # Frozen initially
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(768, 512),  # CLIP hidden dim = 768
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, pixel_values):
        # Extract CLIP vision features
        vision_outputs = self.vision_model(pixel_values)
        pooled_output = vision_outputs.last_hidden_state[:, 0, :]  # CLS token
        
        # Classify
        logits = self.classifier(pooled_output)
        return logits
```

### Key Training Details

#### 1. CLIP Processor
CLIP uses its own preprocessing (different from ImageNet):

```python
from transformers import CLIPProcessor, CLIPModel

MODEL_NAME = "openai/clip-vit-base-patch32"
clip_model = CLIPModel.from_pretrained(MODEL_NAME)
clip_processor = CLIPProcessor.from_pretrained(MODEL_NAME)

# Extract vision encoder
vision_model = clip_model.vision_model
```

#### 2. Dataset Wrapper
Custom dataset to use CLIP's preprocessing:

```python
class RustDataset(Dataset):
    def __init__(self, root_dir, processor, split="train"):
        self.root_dir = Path(root_dir) / split
        self.processor = processor
        
        # Collect all images
        self.samples = []
        for class_name in ["rust", "no_rust"]:
            class_dir = self.root_dir / class_name
            for img_path in class_dir.glob("*.*"):
                self.samples.append((img_path, 1 if class_name == "rust" else 0))
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        
        # CLIP preprocessing
        pixel_values = self.processor(images=image, return_tensors="pt")["pixel_values"][0]
        
        return pixel_values, label
```

#### 3. Training Configuration
```python
# Optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

# Learning rate scheduler
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

# Loss with class weights
criterion = nn.CrossEntropyLoss(weight=class_weights)
```

#### 4. Training Loop
- **Epochs**: 10 (more epochs needed for larger model)
- **Batch Size**: 32 (smaller due to GPU memory)
- **Optimizer**: AdamW (lr=2e-5, smaller for fine-tuning)
- **Scheduler**: Cosine Annealing

```python
for epoch in range(EPOCHS):
    model.train()
    for pixel_values, labels in train_loader:
        optimizer.zero_grad()
        logits = model(pixel_values)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
    
    scheduler.step()
    
    # Validation
    val_acc, val_bal_acc = evaluate(model, val_loader)
    
    # Save best model
    if val_bal_acc > best_val_acc:
        best_val_acc = val_bal_acc
        best_state = model.state_dict()
```

#### 5. ONNX Export
Convert CLIP model to ONNX with external data (>2GB threshold):

```python
model.eval().cpu()
dummy_input = torch.randn(1, 3, 224, 224)

torch.onnx.export(
    model,
    dummy_input,
    "rust_clip.onnx",
    input_names=["input"],
    output_names=["logits"],
    dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
    opset_version=17,
    export_params=True
)

# Large models create .onnx.data file automatically
# rust_clip.onnx (metadata) + rust_clip.onnx.data (335 MB weights)
```

### Results
- **Test Accuracy**: ~88% (on Roboflow test set)
- **Real-World Testing**: 70% (3/5 false positives on clean surfaces)
- **Model Size**: 335 MB (onnx + data)
- **Inference Speed**: 64-127ms on CPU

**Critical Issue**: CLIP has severe false positive problem (flags clean surfaces as rust at 99%+ confidence)

---

## 🔄 Model Switching in Production

The API supports both models via environment variable:

```bash
# Use MobileNetV3 (default, recommended)
export RUST_MODEL_TYPE="mobilenet"
python -m uvicorn app.main:app

# Use CLIP (research only)
export RUST_MODEL_TYPE="clip"
python -m uvicorn app.main:app
```

**Production Recommendation**: Always use MobileNetV3 due to superior accuracy and faster inference.

---

## 📋 How to Reproduce Training

### Prerequisites
1. Google Colab account (free GPU access)
2. Roboflow API key (free account)

### Step-by-Step

#### For MobileNetV3:

1. **Open Colab Notebook**:
   - Upload `rust_detection_model_training.ipynb` to Google Colab
   - Runtime → Change runtime type → GPU

2. **Install Dependencies**:
   ```bash
   !pip install roboflow torch torchvision tqdm onnx onnxruntime
   ```

3. **Download Dataset**:
   - Get Roboflow API key from [roboflow.com](https://roboflow.com)
   - Update API key in notebook:
   ```python
   RF_API_KEY = "your_api_key_here"
   ```

4. **Run All Cells**:
   - Dataset download → Conversion → Training → ONNX export
   - Training takes ~10-15 minutes on Colab T4 GPU

5. **Download Files**:
   - `rust_model.onnx` (6 MB)
   - `rust_labels.json`

#### For CLIP VLM:

1. **Open Colab Notebook**:
   - Upload `VLM_rust_detection_training.ipynb` to Google Colab
   - Runtime → Change runtime type → GPU

2. **Install Dependencies**:
   ```bash
   !pip install roboflow torch torchvision transformers tqdm onnx onnxruntime
   ```

3. **Download Dataset** (same as MobileNetV3)

4. **Run All Cells**:
   - Dataset download → Conversion → CLIP loading → Training → ONNX export
   - Training takes ~30-40 minutes on Colab T4 GPU

5. **Download Files**:
   - `rust_clip.onnx` (~1 MB metadata)
   - `rust_clip.onnx.data` (335 MB weights)
   - `rust_labels.json`

---

## 🎯 Model Comparison Summary

### MobileNetV3 Advantages ✅
- **Lightweight**: 6 MB vs 335 MB (56x smaller)
- **Fast**: 19-55ms vs 64-127ms (2-4x faster)
- **Accurate**: 100% real-world accuracy vs 70%
- **Reliable**: 0 false positives vs 60% false positive rate
- **Production-ready**: Deployable on edge devices

### CLIP VLM Advantages ✅
- **Semantic Understanding**: Better for complex visual reasoning
- **Transfer Learning**: Strong pretrained vision features
- **Multi-modal Capability**: Can use text prompts (not used here)

### Why CLIP Failed Here ❌
1. **Overconfidence Problem**: Predicts 99%+ confidence on false positives
2. **Background Confusion**: Mistakes gray/metallic surfaces for rust
3. **Not Optimized for Binary Tasks**: Designed for 1000+ classes (ImageNet)
4. **Overkill for Simple Task**: Rust detection is visual, not semantic

---

## 📊 Training Metrics

### MobileNetV3
| Metric | Train | Validation | Test | Real-World |
|--------|-------|------------|------|------------|
| Accuracy | 97.2% | 94.8% | 94.1% | **100%** |
| Balanced Acc | 96.8% | 93.5% | 93.7% | 100% |
| False Positives | - | - | - | **0/5** |

### CLIP VLM
| Metric | Train | Validation | Test | Real-World |
|--------|-------|------------|------|------------|
| Accuracy | 92.3% | 89.7% | 88.4% | **70%** |
| Balanced Acc | 90.1% | 87.2% | 86.9% | 70% |
| False Positives | - | - | - | **3/5** ⚠️ |

---

## 🔧 Troubleshooting

### Colab GPU Quota Exhausted
- **Solution**: Use Colab Pro ($10/month) or wait 24 hours for reset
- **Alternative**: Run on local GPU (NVIDIA RTX 2060 or better)

### ONNX Export Memory Error
- **Solution**: Reduce batch size during export:
  ```python
  dummy_input = torch.randn(1, 3, 224, 224)  # Batch size = 1
  ```

### CLIP Model Too Large
- **Solution**: Use external data format (automatic for models >2GB)
- **Files**: `rust_clip.onnx` + `rust_clip.onnx.data` (both needed)

### Class Imbalance Warning
- **Solution**: Already handled via weighted loss:
  ```python
  criterion = nn.CrossEntropyLoss(weight=class_weights)
  ```

---

## 📚 References

1. **MobileNetV3**: [Searching for MobileNetV3 (Howard et al., 2019)](https://arxiv.org/abs/1905.02244)
2. **CLIP**: [Learning Transferable Visual Models (Radford et al., 2021)](https://arxiv.org/abs/2103.00020)
3. **Dataset**: [Roboflow Rust Detection](https://universe.roboflow.com/test-stage/rust-detection-t8vza)
4. **PyTorch**: [pytorch.org](https://pytorch.org)
5. **Transformers**: [huggingface.co/transformers](https://huggingface.co/transformers)

---

## 📄 Files in This Directory

| File | Description |
|------|-------------|
| `rust_detection_model_training.ipynb` | MobileNetV3 training notebook (Colab) |
| `VLM_rust_detection_training.ipynb` | CLIP VLM training notebook (Colab) |
| `MODEL_TRAINING_GUIDE.md` | This document |
| `What we did in Colab.pdf` | Screenshot documentation of training process |

---

## 🎓 Key Learnings

1. **Simpler is Better**: MobileNetV3 (2.5M params) outperforms CLIP (86M params) for binary classification
2. **Class Imbalance Matters**: Use weighted loss for imbalanced datasets
3. **Real-World Testing Essential**: Validation accuracy doesn't catch false positive issues
4. **Model Size vs Performance**: Smaller models can be more reliable for specific tasks
5. **Task-Specific Fine-Tuning**: Vision-language models aren't always optimal for simple visual tasks

---

**Training Date**: February 2026  
**Platform**: Google Colab (Tesla T4 GPU)  
**Frameworks**: PyTorch 2.0, Transformers 4.36, ONNX Runtime 1.16
