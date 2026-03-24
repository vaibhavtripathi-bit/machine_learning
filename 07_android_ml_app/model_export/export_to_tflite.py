"""
Export trained models to TensorFlow Lite format for Android.
"""

import os
from pathlib import Path
import numpy as np


def create_simple_classifier():
    """Create a simple image classifier for demonstration."""
    import tensorflow as tf
    
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=(224, 224, 3)),
        tf.keras.layers.Conv2D(32, 3, activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(64, 3, activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(64, 3, activation='relu'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(2, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def convert_to_tflite(model, output_path: str, quantize: bool = True):
    """
    Convert Keras model to TFLite format.
    
    Args:
        model: Keras model
        output_path: Path to save TFLite model
        quantize: Whether to apply int8 quantization
        
    Returns:
        Path to saved model
    """
    import tensorflow as tf
    
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    if quantize:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        
        def representative_dataset():
            for _ in range(100):
                data = np.random.rand(1, 224, 224, 3).astype(np.float32)
                yield [data]
        
        converter.representative_dataset = representative_dataset
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.uint8
        converter.inference_output_type = tf.uint8
    
    tflite_model = converter.convert()
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'wb') as f:
        f.write(tflite_model)
    
    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"TFLite model saved to {output_path} ({size_mb:.2f} MB)")
    
    return output_path


def convert_pytorch_to_onnx(model, output_path: str, input_shape=(1, 3, 224, 224)):
    """
    Convert PyTorch model to ONNX format.
    
    Args:
        model: PyTorch model
        output_path: Path to save ONNX model
        input_shape: Input tensor shape
        
    Returns:
        Path to saved model
    """
    import torch
    
    model.eval()
    dummy_input = torch.randn(input_shape)
    
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
    
    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"ONNX model saved to {output_path} ({size_mb:.2f} MB)")
    
    return output_path


def verify_tflite_model(model_path: str):
    """Verify TFLite model works correctly."""
    import tensorflow as tf
    
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    print("\nModel Input:")
    print(f"  Shape: {input_details[0]['shape']}")
    print(f"  Type: {input_details[0]['dtype']}")
    
    print("\nModel Output:")
    print(f"  Shape: {output_details[0]['shape']}")
    print(f"  Type: {output_details[0]['dtype']}")
    
    input_shape = input_details[0]['shape']
    input_dtype = input_details[0]['dtype']
    
    if input_dtype == np.uint8:
        test_input = np.random.randint(0, 255, input_shape, dtype=np.uint8)
    else:
        test_input = np.random.rand(*input_shape).astype(np.float32)
    
    interpreter.set_tensor(input_details[0]['index'], test_input)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])
    
    print(f"\nTest inference successful!")
    print(f"  Output shape: {output.shape}")
    
    return True


def main():
    """Main export pipeline."""
    print("="*60)
    print("MODEL EXPORT FOR ANDROID")
    print("="*60)
    
    output_dir = Path(__file__).parent
    
    print("\n1. Creating simple classifier...")
    model = create_simple_classifier()
    model.summary()
    
    print("\n2. Exporting to TFLite (float32)...")
    tflite_path = str(output_dir / "classifier_float32.tflite")
    convert_to_tflite(model, tflite_path, quantize=False)
    
    print("\n3. Exporting to TFLite (int8 quantized)...")
    tflite_quant_path = str(output_dir / "classifier_int8.tflite")
    try:
        convert_to_tflite(model, tflite_quant_path, quantize=True)
    except Exception as e:
        print(f"   Quantization failed (expected on some systems): {e}")
    
    print("\n4. Verifying TFLite model...")
    verify_tflite_model(tflite_path)
    
    print("\n" + "="*60)
    print("Export complete!")
    print("="*60)
    print(f"\nModels saved in: {output_dir}")
    print("Copy .tflite files to Android app's assets folder")


if __name__ == "__main__":
    main()
