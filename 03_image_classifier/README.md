# Image Classifier - Cats vs Dogs

A deep learning image classifier comparing training from scratch vs transfer learning with PyTorch.

## Features

- **Custom CNN**: Train a CNN from scratch to understand fundamentals
- **Transfer Learning**: Use pretrained ResNet for better accuracy
- **Data Augmentation**: Random crops, flips, rotations, color jitter
- **ONNX Export**: Export model for deployment on any platform
- **Early Stopping**: Prevent overfitting with validation monitoring

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Train the models
python src/main.py
```

## Model Comparison

| Model | Val Accuracy | Training Time | Parameters |
|-------|--------------|---------------|------------|
| Simple CNN | ~70-80% | Longer | ~5M |
| ResNet18 (transfer) | ~95%+ | Shorter | 11M (frozen) |

## Project Structure

```
03_image_classifier/
├── data/                   # Dataset (auto-generated)
│   ├── cats/
│   └── dogs/
├── models/                 # Saved models
│   ├── simple_cnn/
│   │   ├── best_model.pth
│   │   └── final_model.pth
│   └── resnet/
│       ├── best_model.pth
│       ├── final_model.pth
│       └── model.onnx
├── notebooks/              # Jupyter notebooks
├── src/
│   ├── __init__.py
│   ├── dataset.py          # Data loading and transforms
│   ├── model.py            # CNN and ResNet models
│   ├── train.py            # Training loop
│   └── main.py             # Main pipeline
├── requirements.txt
└── README.md
```

## Architecture

### Simple CNN (from scratch)
```
Input (3, 224, 224)
    ↓
Conv2d(3→32) + BN + ReLU + MaxPool
    ↓
Conv2d(32→64) + BN + ReLU + MaxPool
    ↓
Conv2d(64→128) + BN + ReLU + MaxPool
    ↓
Conv2d(128→256) + BN + ReLU + MaxPool
    ↓
Flatten + Dropout + FC(512) + ReLU + Dropout + FC(2)
    ↓
Output (2 classes)
```

### ResNet18 Transfer Learning
```
Pretrained ResNet18 (frozen)
    ↓
Custom Classifier:
  Dropout(0.5) → FC(512→256) → ReLU → Dropout(0.3) → FC(256→2)
    ↓
Output (2 classes)
```

## Data Augmentation

Training transforms:
- Resize to 256×256, random crop to 224×224
- Random horizontal flip (50%)
- Random rotation (±15°)
- Color jitter (brightness, contrast, saturation)
- Normalize with ImageNet statistics

## Key Concepts Learned

1. **CNN Architecture**: Convolution, pooling, batch normalization
2. **Transfer Learning**: Using pretrained models as feature extractors
3. **Data Augmentation**: Increasing effective dataset size
4. **PyTorch DataLoader**: Efficient batched data loading
5. **Learning Rate Scheduling**: StepLR for gradual LR decay
6. **Early Stopping**: Prevent overfitting
7. **ONNX Export**: Cross-platform model deployment

## Usage

### Training
```python
from src.main import main
model, history = main()
```

### Inference
```python
from src.main import predict_image

label, confidence = predict_image('path/to/image.jpg')
print(f"Predicted: {label} ({confidence:.2%})")
```

### ONNX Inference
```python
import onnxruntime as ort
import numpy as np

session = ort.InferenceSession('models/resnet/model.onnx')
input_name = session.get_inputs()[0].name
output = session.run(None, {input_name: image_array})
```

## Extending the Project

- Use the full Kaggle Cats vs Dogs dataset
- Try different architectures (VGG, EfficientNet)
- Implement Grad-CAM for visualization
- Add multi-class classification
- Deploy to mobile with TFLite/CoreML

## License

MIT License
