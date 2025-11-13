import numpy as np
import cv2
import base64
import logging
from typing import List, Dict

logger = logging.getLogger(__name__)

def decode_image(image_base64: str) -> np.ndarray:
    """
    Decode base64 image to numpy array
    """
    # Decode base64
    image_bytes = base64.b64decode(image_base64)
    
    # Convert to numpy array
    np_array = np.frombuffer(image_bytes, dtype=np.uint8)
    
    # Decode image
    image = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
    
    if image is None:
        raise ValueError("Failed to decode image")
    
    return image

def preprocess_image(image: np.ndarray, target_size=(640, 640)) -> np.ndarray:
    """
    Preprocess image for ONNX inference
    - Resize to target size
    - Normalize to [0, 1]
    - Convert to float32
    - Transpose to CHW format (channels first)
    - Add batch dimension
    """
    # Resize
    resized = cv2.resize(image, target_size)
    
    # Normalize to [0, 1]
    normalized = resized.astype(np.float32) / 255.0
    
    # Transpose to CHW (channels first)
    transposed = np.transpose(normalized, (2, 0, 1))
    
    # Add batch dimension
    batched = np.expand_dims(transposed, axis=0)
    
    return batched

def run_field_detection(model_loader, image_base64: str) -> List[Dict]:
    """
    Run field detection (32 keypoints)
    Returns list of predictions in Roboflow format
    """
    # Get model
    model = model_loader.get_model('field')
    
    if model is None:
        raise ValueError("Field model not loaded")
    
    # Decode image
    image = decode_image(image_base64)
    h, w = image.shape[:2]
    
    logger.info(f"   - Image dimensions: {w}x{h}")
    
    # Preprocess
    input_tensor = preprocess_image(image)
    
    # Run inference
    input_name = model.get_inputs()[0].name
    outputs = model.run(None, {input_name: input_tensor})
    
    # Post-process (example - adapt to your model's output format)
    keypoints = outputs[0]  # Assuming first output is keypoints
    
    # Convert to Roboflow format
    predictions = []
    for i, kp in enumerate(keypoints[0]):  # Remove batch dimension
        predictions.append({
            "x": float(kp[0] * w),  # Scale to original image size
            "y": float(kp[1] * h),
            "class": f"keypoint_{i}",
            "confidence": float(kp[2]) if len(kp) > 2 else 0.95
        })
    
    return predictions

def run_player_detection(model_loader, image_base64: str) -> List[Dict]:
    """
    Run player detection (bounding boxes)
    Returns list of predictions in Roboflow format
    """
    # Get model
    model = model_loader.get_model('players')
    
    if model is None:
        raise ValueError("Players model not loaded")
    
    # Decode image
    image = decode_image(image_base64)
    h, w = image.shape[:2]
    
    logger.info(f"   - Image dimensions: {w}x{h}")
    
    # Preprocess
    input_tensor = preprocess_image(image)
    
    # Run inference
    input_name = model.get_inputs()[0].name
    outputs = model.run(None, {input_name: input_tensor})
    
    # Parse YOLO output (adapt to your model's format)
    detections_raw = outputs[0]
    
    # Convert to Roboflow format
    predictions = []
    for det in detections_raw[0]:  # Remove batch dimension
        if det[4] > 0.5:  # Confidence threshold
            predictions.append({
                "x": float(det[0] * w),
                "y": float(det[1] * h),
                "width": float(det[2] * w),
                "height": float(det[3] * h),
                "class": "player",
                "confidence": float(det[4])
            })
    
    return predictions

def run_ball_detection(model_loader, image_base64: str) -> List[Dict]:
    """
    Run ball detection (bounding boxes)
    Returns list of predictions in Roboflow format
    """
    # Get model
    model = model_loader.get_model('ball')
    
    if model is None:
        raise ValueError("Ball model not loaded")
    
    # Decode image
    image = decode_image(image_base64)
    h, w = image.shape[:2]
    
    logger.info(f"   - Image dimensions: {w}x{h}")
    
    # Preprocess
    input_tensor = preprocess_image(image)
    
    # Run inference
    input_name = model.get_inputs()[0].name
    outputs = model.run(None, {input_name: input_tensor})
    
    # Parse YOLO output (adapt to your model's format)
    detections_raw = outputs[0]
    
    # Convert to Roboflow format
    predictions = []
    for det in detections_raw[0]:  # Remove batch dimension
        if det[4] > 0.5:  # Confidence threshold
            predictions.append({
                "x": float(det[0] * w),
                "y": float(det[1] * h),
                "width": float(det[2] * w),
                "height": float(det[3] * h),
                "class": "ball",
                "confidence": float(det[4])
            })
    
    return predictions
