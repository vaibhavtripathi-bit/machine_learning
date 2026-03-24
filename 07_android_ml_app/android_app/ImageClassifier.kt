/**
 * Image Classifier using TensorFlow Lite on Android.
 * 
 * This is a reference implementation showing how to use TFLite models
 * for on-device image classification.
 */
package com.example.mlapp

import android.content.Context
import android.graphics.Bitmap
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.support.common.FileUtil
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.image.ops.ResizeOp
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer
import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * TensorFlow Lite image classifier.
 * 
 * @param context Android context
 * @param modelPath Path to .tflite model in assets
 * @param labels List of class labels
 */
class ImageClassifier(
    private val context: Context,
    private val modelPath: String = "classifier.tflite",
    private val labels: List<String> = listOf("cat", "dog")
) {
    private var interpreter: Interpreter? = null
    private val imageSize = 224
    private val numChannels = 3
    private val numClasses = labels.size
    
    init {
        loadModel()
    }
    
    /**
     * Load the TFLite model from assets.
     */
    private fun loadModel() {
        try {
            val modelBuffer = FileUtil.loadMappedFile(context, modelPath)
            val options = Interpreter.Options().apply {
                setNumThreads(4)
            }
            interpreter = Interpreter(modelBuffer, options)
        } catch (e: Exception) {
            throw RuntimeException("Error loading model: ${e.message}")
        }
    }
    
    /**
     * Classify an image and return predictions.
     * 
     * @param bitmap Input image
     * @return List of (label, confidence) pairs sorted by confidence
     */
    fun classify(bitmap: Bitmap): List<Pair<String, Float>> {
        val interpreter = this.interpreter 
            ?: throw IllegalStateException("Model not loaded")
        
        // Preprocess image
        val tensorImage = preprocessImage(bitmap)
        
        // Run inference
        val outputBuffer = TensorBuffer.createFixedSize(
            intArrayOf(1, numClasses),
            org.tensorflow.lite.DataType.FLOAT32
        )
        
        interpreter.run(tensorImage.buffer, outputBuffer.buffer.rewind())
        
        // Process results
        val probabilities = outputBuffer.floatArray
        
        return labels.zip(probabilities.toList())
            .sortedByDescending { it.second }
    }
    
    /**
     * Preprocess image for model input.
     */
    private fun preprocessImage(bitmap: Bitmap): TensorImage {
        val imageProcessor = ImageProcessor.Builder()
            .add(ResizeOp(imageSize, imageSize, ResizeOp.ResizeMethod.BILINEAR))
            .build()
        
        var tensorImage = TensorImage(org.tensorflow.lite.DataType.FLOAT32)
        tensorImage.load(bitmap)
        tensorImage = imageProcessor.process(tensorImage)
        
        return tensorImage
    }
    
    /**
     * Get the top prediction.
     */
    fun classifyTop(bitmap: Bitmap): Pair<String, Float> {
        return classify(bitmap).first()
    }
    
    /**
     * Release resources.
     */
    fun close() {
        interpreter?.close()
        interpreter = null
    }
}

/**
 * Alternative implementation using ONNX Runtime.
 */
class OnnxImageClassifier(
    private val context: Context,
    private val modelPath: String = "classifier.onnx",
    private val labels: List<String> = listOf("cat", "dog")
) {
    // ONNX Runtime implementation would go here
    // Requires: implementation 'com.microsoft.onnxruntime:onnxruntime-android:latest'
    
    /**
     * Example usage with ONNX Runtime Android.
     * 
     * ```kotlin
     * val session = OrtEnvironment.getEnvironment()
     *     .createSession(modelBytes, OrtSession.SessionOptions())
     * 
     * val inputName = session.inputNames.first()
     * val inputTensor = OnnxTensor.createTensor(env, inputData)
     * 
     * val results = session.run(mapOf(inputName to inputTensor))
     * val output = results[0].value as Array<FloatArray>
     * ```
     */
}
