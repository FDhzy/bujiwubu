"""
Model loading utilities for TFLite and ONNX models.
Supports GPU acceleration on Android via delegates.
"""
import os
from typing import Optional, Tuple, List
import numpy as np

from utils.logger import get_logger
from utils.config import get_asset_path

# Try to import TensorFlow Lite
try:
    import tensorflow as tf
    TFLITE_AVAILABLE = True
except ImportError:
    TFLITE_AVAILABLE = False

# Try to import ONNX Runtime
try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False


class ModelLoader:
    """
    Utility class for loading and managing ML models.
    Supports TFLite and ONNX formats with optional GPU acceleration.
    """
    
    def __init__(self):
        self.logger = get_logger("model_loader")
        self._loaded_models = {}
    
    def load_tflite(
        self,
        model_path: str,
        use_gpu: bool = True,
        num_threads: int = 2
    ) -> Optional['TFLiteModel']:
        """
        Load a TensorFlow Lite model.
        
        Args:
            model_path: Path to .tflite file
            use_gpu: Whether to use GPU delegate (Android)
            num_threads: Number of CPU threads for inference
            
        Returns:
            TFLiteModel instance or None if loading failed
        """
        if not TFLITE_AVAILABLE:
            self.logger.error("TensorFlow Lite not available")
            return None
        
        # Resolve path
        full_path = get_asset_path(model_path)
        
        if not os.path.exists(full_path):
            self.logger.warning(f"Model not found: {full_path}")
            # Return a dummy model for development
            return DummyModel(model_path)
        
        try:
            model = TFLiteModel(full_path, use_gpu, num_threads)
            self._loaded_models[model_path] = model
            self.logger.info(f"Loaded TFLite model: {model_path}")
            return model
            
        except Exception as e:
            self.logger.error(f"Failed to load TFLite model: {e}")
            return DummyModel(model_path)
    
    def load_onnx(
        self,
        model_path: str,
        use_gpu: bool = True
    ) -> Optional['ONNXModel']:
        """
        Load an ONNX model.
        
        Args:
            model_path: Path to .onnx file
            use_gpu: Whether to use GPU execution provider
            
        Returns:
            ONNXModel instance or None if loading failed
        """
        if not ONNX_AVAILABLE:
            self.logger.error("ONNX Runtime not available")
            return None
        
        full_path = get_asset_path(model_path)
        
        if not os.path.exists(full_path):
            self.logger.warning(f"Model not found: {full_path}")
            return DummyModel(model_path)
        
        try:
            model = ONNXModel(full_path, use_gpu)
            self._loaded_models[model_path] = model
            self.logger.info(f"Loaded ONNX model: {model_path}")
            return model
            
        except Exception as e:
            self.logger.error(f"Failed to load ONNX model: {e}")
            return DummyModel(model_path)
    
    def release_all(self):
        """Release all loaded models."""
        for model in self._loaded_models.values():
            if hasattr(model, 'release'):
                model.release()
        self._loaded_models.clear()


class TFLiteModel:
    """Wrapper for TensorFlow Lite model inference."""
    
    def __init__(
        self,
        model_path: str,
        use_gpu: bool = True,
        num_threads: int = 2
    ):
        """
        Initialize TFLite interpreter.
        
        Args:
            model_path: Path to .tflite file
            use_gpu: Whether to use GPU delegate
            num_threads: Number of CPU threads
        """
        self.logger = get_logger("tflite")
        self.model_path = model_path
        
        # Normalize path for Windows
        model_path = os.path.normpath(model_path)
        
        # Verify file exists and is readable
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        file_size = os.path.getsize(model_path)
        self.logger.info(f"Loading model: {model_path} ({file_size/1024:.1f} KB)")
        
        # Create interpreter with optional GPU delegate
        delegates = []
        
        if use_gpu:
            try:
                # Try GPU delegate (Android/NNAPI)
                gpu_delegate = tf.lite.experimental.load_delegate('libdelegate.so')
                delegates.append(gpu_delegate)
                self.logger.info("GPU delegate loaded")
            except Exception:
                self.logger.info("GPU delegate not available, using CPU")
        
        try:
            # Try loading with delegates first
            if delegates:
                self.interpreter = tf.lite.Interpreter(
                    model_path=model_path,
                    num_threads=num_threads,
                    experimental_delegates=delegates
                )
            else:
                # Basic interpreter without delegates
                self.interpreter = tf.lite.Interpreter(
                    model_path=model_path,
                    num_threads=num_threads
                )
            
            self.interpreter.allocate_tensors()
            
        except Exception as e:
            self.logger.error(f"Failed to load model with error: {e}")
            # Try loading model bytes directly as fallback
            try:
                self.logger.info("Trying to load model from bytes...")
                with open(model_path, 'rb') as f:
                    model_content = f.read()
                self.interpreter = tf.lite.Interpreter(
                    model_content=model_content,
                    num_threads=num_threads
                )
                self.interpreter.allocate_tensors()
                self.logger.info("Model loaded from bytes successfully")
            except Exception as e2:
                raise RuntimeError(f"Failed to load TFLite model: {e2}")
        
        # Get input/output details
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        
        # Log model info
        input_shape = self.input_details[0]['shape']
        output_shape = self.output_details[0]['shape']
        self.logger.info(f"Model loaded: input {input_shape}, output {output_shape}")
    
    def get_input_shape(self) -> Tuple[int, ...]:
        """Get expected input shape (NHWC format)."""
        return tuple(self.input_details[0]['shape'])
    
    def get_output_shape(self) -> Tuple[int, ...]:
        """Get output shape."""
        return tuple(self.output_details[0]['shape'])
    
    def predict(self, input_data: np.ndarray) -> np.ndarray:
        """
        Run inference on input data.
        
        Args:
            input_data: Input tensor matching model's expected shape
            
        Returns:
            Model output tensor
        """
        # Ensure correct dtype
        input_dtype = self.input_details[0]['dtype']
        if input_data.dtype != input_dtype:
            input_data = input_data.astype(input_dtype)
        
        # Set input tensor
        self.interpreter.set_tensor(
            self.input_details[0]['index'],
            input_data
        )
        
        # Run inference
        self.interpreter.invoke()
        
        # Get output
        output = self.interpreter.get_tensor(self.output_details[0]['index'])
        
        return output
    
    def release(self):
        """Release model resources."""
        # TFLite interpreter doesn't need explicit cleanup
        pass


class ONNXModel:
    """Wrapper for ONNX Runtime model inference."""
    
    def __init__(self, model_path: str, use_gpu: bool = True):
        """
        Initialize ONNX Runtime session.
        
        Args:
            model_path: Path to .onnx file
            use_gpu: Whether to use GPU execution provider
        """
        self.logger = get_logger("onnx")
        self.model_path = model_path
        
        # Configure execution providers
        providers = ['CPUExecutionProvider']
        
        if use_gpu:
            if 'CUDAExecutionProvider' in ort.get_available_providers():
                providers.insert(0, 'CUDAExecutionProvider')
                self.logger.info("Using CUDA execution provider")
        
        # Create session
        self.session = ort.InferenceSession(model_path, providers=providers)
        
        # Get input/output info
        self.input_name = self.session.get_inputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape
        self.output_name = self.session.get_outputs()[0].name
        
        self.logger.info(f"ONNX model loaded: input shape {self.input_shape}")
    
    def get_input_shape(self) -> Tuple[int, ...]:
        """Get expected input shape."""
        return tuple(self.input_shape)
    
    def predict(self, input_data: np.ndarray) -> np.ndarray:
        """
        Run inference on input data.
        
        Args:
            input_data: Input tensor
            
        Returns:
            Model output tensor
        """
        outputs = self.session.run(
            [self.output_name],
            {self.input_name: input_data.astype(np.float32)}
        )
        return outputs[0]
    
    def release(self):
        """Release model resources."""
        self.session = None


class DummyModel:
    """
    Dummy model for development when real models aren't available.
    Returns random/mock outputs for testing the pipeline.
    """
    
    def __init__(self, model_path: str):
        self.logger = get_logger("dummy_model")
        self.model_path = model_path
        self.logger.warning(f"Using dummy model for: {model_path}")
        
        # Determine model type from path
        if 'deeplabv3' in model_path.lower() or 'water' in model_path.lower():
            self._model_type = 'segmentation'
            self._input_shape = (1, 256, 256, 3)
            self._output_shape = (1, 256, 256, 1)
        else:
            self._model_type = 'detection'
            self._input_shape = (1, 320, 320, 3)
            self._output_shape = (1, 25200, 85)  # YOLOv8 output format
    
    def get_input_shape(self) -> Tuple[int, ...]:
        return self._input_shape
    
    def get_output_shape(self) -> Tuple[int, ...]:
        return self._output_shape
    
    def predict(self, input_data: np.ndarray) -> np.ndarray:
        """Return mock predictions for testing."""
        if self._model_type == 'segmentation':
            # Return random segmentation mask (mostly zeros with some "puddles")
            output = np.zeros(self._output_shape, dtype=np.float32)
            # Add some random "puddle" regions
            h, w = self._output_shape[1:3]
            # Random puddle at bottom of frame
            y_start = int(h * 0.6)
            x_start = np.random.randint(0, w // 2)
            x_end = x_start + np.random.randint(w // 4, w // 2)
            output[0, y_start:, x_start:x_end, 0] = 0.8
            return output
        else:
            # Return empty detections (no obstacles for dummy)
            return np.zeros(self._output_shape, dtype=np.float32)
    
    def release(self):
        pass


# Global model loader instance
_loader: Optional[ModelLoader] = None


def get_model_loader() -> ModelLoader:
    """Get the global model loader instance."""
    global _loader
    if _loader is None:
        _loader = ModelLoader()
    return _loader
