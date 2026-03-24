"""
Benchmark model inference performance.
"""

import time
import numpy as np
from pathlib import Path


def benchmark_tflite(model_path: str, num_runs: int = 100):
    """
    Benchmark TFLite model inference.
    
    Args:
        model_path: Path to TFLite model
        num_runs: Number of inference runs
        
    Returns:
        Dictionary of benchmark results
    """
    import tensorflow as tf
    
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    input_shape = input_details[0]['shape']
    input_dtype = input_details[0]['dtype']
    
    if input_dtype == np.uint8:
        test_input = np.random.randint(0, 255, input_shape, dtype=np.uint8)
    else:
        test_input = np.random.rand(*input_shape).astype(np.float32)
    
    for _ in range(10):
        interpreter.set_tensor(input_details[0]['index'], test_input)
        interpreter.invoke()
    
    latencies = []
    for _ in range(num_runs):
        start = time.perf_counter()
        interpreter.set_tensor(input_details[0]['index'], test_input)
        interpreter.invoke()
        end = time.perf_counter()
        latencies.append((end - start) * 1000)
    
    results = {
        'model': model_path,
        'num_runs': num_runs,
        'mean_latency_ms': np.mean(latencies),
        'std_latency_ms': np.std(latencies),
        'min_latency_ms': np.min(latencies),
        'max_latency_ms': np.max(latencies),
        'p50_latency_ms': np.percentile(latencies, 50),
        'p95_latency_ms': np.percentile(latencies, 95),
        'p99_latency_ms': np.percentile(latencies, 99),
    }
    
    return results


def benchmark_onnx(model_path: str, num_runs: int = 100):
    """
    Benchmark ONNX model inference.
    
    Args:
        model_path: Path to ONNX model
        num_runs: Number of inference runs
        
    Returns:
        Dictionary of benchmark results
    """
    import onnxruntime as ort
    
    session = ort.InferenceSession(model_path)
    input_name = session.get_inputs()[0].name
    input_shape = session.get_inputs()[0].shape
    
    input_shape = [1 if isinstance(d, str) else d for d in input_shape]
    test_input = np.random.rand(*input_shape).astype(np.float32)
    
    for _ in range(10):
        session.run(None, {input_name: test_input})
    
    latencies = []
    for _ in range(num_runs):
        start = time.perf_counter()
        session.run(None, {input_name: test_input})
        end = time.perf_counter()
        latencies.append((end - start) * 1000)
    
    results = {
        'model': model_path,
        'num_runs': num_runs,
        'mean_latency_ms': np.mean(latencies),
        'std_latency_ms': np.std(latencies),
        'min_latency_ms': np.min(latencies),
        'max_latency_ms': np.max(latencies),
        'p50_latency_ms': np.percentile(latencies, 50),
        'p95_latency_ms': np.percentile(latencies, 95),
        'p99_latency_ms': np.percentile(latencies, 99),
    }
    
    return results


def print_benchmark_results(results: dict):
    """Print benchmark results in a formatted table."""
    print(f"\nBenchmark Results: {Path(results['model']).name}")
    print("-" * 40)
    print(f"  Runs:          {results['num_runs']}")
    print(f"  Mean latency:  {results['mean_latency_ms']:.2f} ms")
    print(f"  Std latency:   {results['std_latency_ms']:.2f} ms")
    print(f"  Min latency:   {results['min_latency_ms']:.2f} ms")
    print(f"  Max latency:   {results['max_latency_ms']:.2f} ms")
    print(f"  P50 latency:   {results['p50_latency_ms']:.2f} ms")
    print(f"  P95 latency:   {results['p95_latency_ms']:.2f} ms")
    print(f"  P99 latency:   {results['p99_latency_ms']:.2f} ms")
    print(f"  FPS (mean):    {1000 / results['mean_latency_ms']:.1f}")


def main():
    """Run benchmarks on available models."""
    print("="*60)
    print("MODEL INFERENCE BENCHMARKS")
    print("="*60)
    
    model_dir = Path(__file__).parent
    
    tflite_models = list(model_dir.glob("*.tflite"))
    onnx_models = list(model_dir.glob("*.onnx"))
    
    for model_path in tflite_models:
        try:
            results = benchmark_tflite(str(model_path))
            print_benchmark_results(results)
        except Exception as e:
            print(f"\nError benchmarking {model_path.name}: {e}")
    
    for model_path in onnx_models:
        try:
            results = benchmark_onnx(str(model_path))
            print_benchmark_results(results)
        except Exception as e:
            print(f"\nError benchmarking {model_path.name}: {e}")
    
    if not tflite_models and not onnx_models:
        print("\nNo models found. Run export_to_tflite.py first.")


if __name__ == "__main__":
    main()
