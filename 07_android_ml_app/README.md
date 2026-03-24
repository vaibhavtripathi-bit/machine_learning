# Android ML App - On-Device Image Classification

Deploy trained ML models to Android for real-time, offline inference using TensorFlow Lite or ONNX Runtime.

## Features

- **Model Export**: Convert PyTorch/TensorFlow models to mobile formats
- **Quantization**: Int8 quantization for smaller, faster models
- **Benchmarking**: Measure inference latency on device
- **Camera Integration**: Real-time classification with CameraX
- **Offline**: No internet required after model deployment

## Quick Start

### 1. Export Model

```bash
# Install dependencies
pip install -r requirements.txt

# Export to TFLite
python model_export/export_to_tflite.py

# Run benchmarks
python model_export/benchmark.py
```

### 2. Android Integration

1. Copy `classifier.tflite` to `app/src/main/assets/`
2. Add TFLite dependencies to `build.gradle.kts`
3. Use `ImageClassifier.kt` for inference

## Project Structure

```
07_android_ml_app/
├── model_export/
│   ├── export_to_tflite.py   # Export models
│   ├── benchmark.py          # Performance benchmarks
│   ├── classifier.tflite     # Exported model
│   └── classifier.onnx       # Alternative format
├── android_app/
│   ├── ImageClassifier.kt    # Kotlin inference code
│   └── build.gradle.kts      # Example dependencies
├── benchmarks/               # Benchmark results
├── requirements.txt
└── README.md
```

## Model Formats

| Format | Size | Latency | Compatibility |
|--------|------|---------|---------------|
| TFLite (float32) | ~4 MB | ~50ms | Universal |
| TFLite (int8) | ~1 MB | ~20ms | Requires quantization |
| ONNX | ~4 MB | ~40ms | Via ONNX Runtime |

## TensorFlow Lite Usage

### Kotlin Integration

```kotlin
// Initialize classifier
val classifier = ImageClassifier(
    context = context,
    modelPath = "classifier.tflite",
    labels = listOf("cat", "dog")
)

// Classify image
val bitmap: Bitmap = // ... get image
val results = classifier.classify(bitmap)

// Get top prediction
val (label, confidence) = results.first()
println("$label: ${confidence * 100}%")

// Clean up
classifier.close()
```

### Dependencies (build.gradle.kts)

```kotlin
implementation("org.tensorflow:tensorflow-lite:2.14.0")
implementation("org.tensorflow:tensorflow-lite-support:0.4.4")
implementation("org.tensorflow:tensorflow-lite-gpu:2.14.0")  // GPU acceleration
```

## ONNX Runtime Usage

### Kotlin Integration

```kotlin
val session = OrtEnvironment.getEnvironment()
    .createSession(modelBytes, OrtSession.SessionOptions())

val inputTensor = OnnxTensor.createTensor(env, inputData)
val results = session.run(mapOf("input" to inputTensor))
```

### Dependencies

```kotlin
implementation("com.microsoft.onnxruntime:onnxruntime-android:1.16.0")
```

## Quantization

### Float32 vs Int8

| Aspect | Float32 | Int8 |
|--------|---------|------|
| Model Size | Baseline | ~4x smaller |
| Latency | Baseline | ~2-4x faster |
| Accuracy | Best | Slight drop (~1%) |
| Hardware | Any | Optimized for mobile |

### Quantization Code

```python
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.uint8
converter.inference_output_type = tf.uint8
```

## Benchmarking

Run benchmarks to measure performance:

```bash
python model_export/benchmark.py
```

Expected results on modern devices:
- Float32: 30-60ms per inference
- Int8: 10-25ms per inference
- With GPU delegate: 5-15ms per inference

## Key Concepts Learned

1. **Model Export**: Converting models to mobile-friendly formats
2. **Quantization**: Reducing model size and improving speed
3. **TFLite Interpreter**: Running inference on Android
4. **ONNX Runtime**: Cross-platform inference
5. **CameraX**: Real-time camera integration
6. **GPU Delegate**: Hardware acceleration on mobile

## CameraX Real-Time Inference

```kotlin
val imageAnalyzer = ImageAnalysis.Builder()
    .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
    .build()
    .also {
        it.setAnalyzer(executor) { imageProxy ->
            val bitmap = imageProxy.toBitmap()
            val result = classifier.classifyTop(bitmap)
            // Update UI with result
            imageProxy.close()
        }
    }
```

## Extending the Project

- Add GPU delegate for faster inference
- Implement batched inference
- Add model versioning and OTA updates
- Integrate with ML Kit for pre-built models
- Add object detection with bounding boxes

## License

MIT License
