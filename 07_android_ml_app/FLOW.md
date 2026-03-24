# End-to-End Flow: Android ML App (On-Device Inference)

## Overview

This document explains the complete pipeline from a trained Python model to running real-time inference on an Android device — entirely offline.

---

## Flow Diagram

```
Trained PyTorch / TensorFlow Model
          │
          ▼
┌──────────────────────┐
│  Model Export        │
│                      │
│  PyTorch → ONNX      │  ← torch.onnx.export()
│  TF/Keras → TFLite   │  ← TFLiteConverter
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│  Post-Training       │  ← Float32 → Int8
│  Quantization        │    4× smaller, 2-4× faster
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│  Verification        │  ← Run test inference
│  & Benchmarking      │    Measure latency on host
└──────────┬───────────┘
           │
           ▼
  Copy .tflite / .onnx to Android app assets/
           │
           ▼
┌──────────────────────┐
│  Android App         │
│  (Kotlin)            │
│                      │
│  CameraX → Frame     │
│  Bitmap → Tensor     │
│  Interpreter.run()   │
│  Display result      │
└──────────────────────┘
```

---

## Step-by-Step Breakdown

### Step 1: Start with a Trained Model

Use the model from Project 03 (Image Classifier) or any trained model.

Requirements for mobile:
- Input size must be fixed (224×224×3 for our classifier)
- No Python-specific ops (custom layers may not export)
- Model should be in evaluation mode (`model.eval()`)

---

### Step 2A: Export to TFLite
**File**: `model_export/export_to_tflite.py`

```python
# Create or load a Keras/TF model
model = create_simple_classifier()  # or load your trained model

# Convert to TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save
with open("classifier.tflite", "wb") as f:
    f.write(tflite_model)
```

**TFLite format**:
- Flat binary format optimized for mobile
- Supports CPU, GPU, NPU delegates on Android
- Much smaller than full TensorFlow

---

### Step 2B: Export to ONNX (from PyTorch)
**File**: `model_export/export_to_tflite.py` → `convert_pytorch_to_onnx()`

```python
model.eval()
dummy_input = torch.randn(1, 3, 224, 224)   # Example input

torch.onnx.export(
    model,
    dummy_input,
    "model.onnx",
    opset_version=11,
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input": {0: "batch_size"}}  # Variable batch size
)
```

**How ONNX export works**:
```
PyTorch traces execution with the dummy input
→ Records every op and its connections
→ Serializes as an ONNX computation graph
→ Platform-independent format
```

---

### Step 3: Quantization
**File**: `model_export/export_to_tflite.py` → `convert_to_tflite(quantize=True)`

```python
converter.optimizations = [tf.lite.Optimize.DEFAULT]

def representative_dataset():
    for _ in range(100):
        yield [np.random.rand(1, 224, 224, 3).astype(np.float32)]

converter.representative_dataset = representative_dataset
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.uint8   # Input as uint8
converter.inference_output_type = tf.uint8  # Output as uint8
```

**What quantization does**:
```
Float32 weights:  1.23456789  →  stored as float (32 bits)
Int8 weights:     1.23456789  →  mapped to integer 0-255 (8 bits)

Weight lookup:    float_value = scale × (int_value - zero_point)
```

**Results**:
| Metric | Float32 | Int8 |
|--------|---------|------|
| Model size | 4 MB | 1 MB |
| Inference latency | 50ms | 15ms |
| Accuracy drop | baseline | ~1% |

---

### Step 4: Verification & Benchmarking
**File**: `model_export/benchmark.py`

```
Verify the model works:
  interpreter.allocate_tensors()
  interpreter.set_tensor(input_idx, test_input)
  interpreter.invoke()
  output = interpreter.get_tensor(output_idx)

Benchmark 100 inference runs:
  latencies = [time each run]
  Report: mean, P50, P95, P99, min, max
```

Typical results on modern CPU:
```
Float32 model:  mean=45ms, P95=55ms, FPS=22
Int8 model:     mean=12ms, P95=15ms, FPS=83
```

---

### Step 5: Android Integration
**File**: `android_app/ImageClassifier.kt`

#### Setup in build.gradle.kts

```kotlin
implementation("org.tensorflow:tensorflow-lite:2.14.0")
implementation("org.tensorflow:tensorflow-lite-support:0.4.4")
implementation("org.tensorflow:tensorflow-lite-gpu:2.14.0") // GPU delegate
```

#### Model Loading

```kotlin
val modelBuffer = FileUtil.loadMappedFile(context, "classifier.tflite")
val options = Interpreter.Options().apply {
    setNumThreads(4)                    // Use 4 CPU threads
    // addDelegate(GpuDelegate())        // Optional: GPU
    // addDelegate(NnApiDelegate())      // Optional: NPU/DSP
}
interpreter = Interpreter(modelBuffer, options)
```

**Memory-mapped loading**: The model file is directly memory-mapped from the app's assets. No copying, faster initialization.

---

### Step 6: Preprocessing in Android

```kotlin
fun preprocessImage(bitmap: Bitmap): TensorImage {
    val imageProcessor = ImageProcessor.Builder()
        .add(ResizeOp(224, 224, ResizeOp.ResizeMethod.BILINEAR)) // Resize
        .build()
    
    var tensorImage = TensorImage(DataType.FLOAT32)
    tensorImage.load(bitmap)                           // Load bitmap
    tensorImage = imageProcessor.process(tensorImage) // Apply ops
    
    // Normalization is baked into the model (or apply manually)
    return tensorImage
}
```

**Critical**: The preprocessing **must match** what was used during training. Wrong normalization = wrong predictions.

---

### Step 7: Inference

```kotlin
fun classify(bitmap: Bitmap): List<Pair<String, Float>> {
    val input = preprocessImage(bitmap)
    val output = TensorBuffer.createFixedSize(intArrayOf(1, 2), DataType.FLOAT32)
    
    interpreter.run(input.buffer, output.buffer.rewind())
    
    val probabilities = output.floatArray  // [0.95, 0.05]
    return labels.zip(probabilities.toList())
        .sortedByDescending { it.second }
}
// Returns: [("cat", 0.95f), ("dog", 0.05f)]
```

---

### Step 8: Real-Time CameraX Integration

```
CameraX captures frame
      │
      ▼  ImageAnalysis callback (every frame)
      │
      ▼  imageProxy.toBitmap()
      │
      ▼  ImageClassifier.classify(bitmap)
      │  └── preprocess → interpreter.run() → postprocess
      │
      ▼  Update UI on main thread
      │  "Cat: 95%"
      │
      ▼  imageProxy.close()  ← Must close or camera stops!
```

---

## Full Pipeline Summary

```
Python (Training)                    Android (Inference)
─────────────────────────────────    ────────────────────────────
train model (03_image_classifier)
      │
export_to_tflite.py
      │
classifier.tflite ──────────────────► app/src/main/assets/
                                              │
                                      ImageClassifier.kt
                                              │
                                      CameraX frame
                                              │
                                      preprocess bitmap
                                              │
                                      interpreter.run()
                                              │
                                      display: "Dog: 92%"
```

---

## Running the Full Flow

```bash
# Step 1: Install Python dependencies
pip install -r requirements.txt

# Step 2: Export model
python model_export/export_to_tflite.py

# Step 3: Benchmark
python model_export/benchmark.py

# Step 4: Copy to Android project
cp model_export/classifier_float32.tflite path/to/android/app/src/main/assets/

# Step 5: Build Android app in Android Studio
# Add ImageClassifier.kt to your project
# Update build.gradle.kts with TFLite dependencies
```

---

## Key Learnings

| Concept | Where Used |
|---------|-----------|
| ONNX export | PyTorch model → cross-platform format |
| TFLite conversion | Keras/TF → mobile format |
| Int8 quantization | 4× smaller, 3× faster |
| Memory-mapped model | Fast loading without copying |
| TFLite Interpreter | On-device inference |
| GPU/NPU delegate | Hardware acceleration |
| CameraX | Real-time camera feed |
| Representative dataset | Required for int8 calibration |
