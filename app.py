from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import onnxruntime as ort
import base64
import time
import cv2
import os

app = Flask(__name__)
CORS(app)

# ============================================
# LOAD MODEL ON STARTUP
# ============================================
print("🔄 Loading ONNX model...")
MODEL_PATH = "field-detection.onnx"

if not os.path.exists(MODEL_PATH):
    print(f"❌ Model not found: {MODEL_PATH}")
    session = None
else:
    session = ort.InferenceSession(MODEL_PATH)
    print(f"✅ Model loaded: {MODEL_PATH}")
    print(f"   - Inputs: {[inp.name for inp in session.get_inputs()]}")
    print(f"   - Outputs: {[out.name for out in session.get_outputs()]}")

# ============================================
# HEALTH CHECK ENDPOINT
# ============================================
@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "message": "Football Tactics Python Service",
        "models_loaded": ["field"] if session else []
    })

# ============================================
# FIELD DETECTION ENDPOINT (32 KEYPOINTS)
# ============================================
@app.route('/detect/field', methods=['POST'])
def detect_field():
    """
    Detect football field keypoints from base64 image
    Returns 32 keypoints with x, y, confidence, class_name
    """
    try:
        start_time = time.time()
        
        if session is None:
            return jsonify({
                "success": False,
                "error": "Model not loaded"
            }), 500
        
        # Get base64 image from request
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({
                "success": False,
                "error": "No image data provided"
            }), 400
        
        base64_image = data['image']
        
        # Decode base64 to image
        try:
            image_bytes = base64.b64decode(base64_image)
            nparr = np.frombuffer(image_bytes, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if image is None:
                return jsonify({
                    "success": False,
                    "error": "Failed to decode image"
                }), 400
            
            h, w = image.shape[:2]
            print(f"✅ Image decoded: {w}x{h}")
                
        except Exception as e:
            print(f"❌ Image decoding error: {e}")
            return jsonify({
                "success": False,
                "error": f"Image decoding error: {str(e)}"
            }), 400
        
        # Preprocess image for ONNX
        print("🔍 Preprocessing image...")
        input_tensor = preprocess_image(image, target_size=(640, 640))
        
        # Run ONNX inference
        print("🔍 Running ONNX inference...")
        input_name = session.get_inputs()[0].name
        outputs = session.run(None, {input_name: input_tensor})
        
        print(f"✅ ONNX inference complete")
        print(f"   - Outputs: {len(outputs)}")
        print(f"   - Output shape: {outputs[0].shape}")
        
        # Parse predictions into 32 keypoints
        print("🔍 Parsing keypoints...")
        keypoints = parse_keypoint_predictions(outputs, w, h)
        print(f"✅ Parsed {len(keypoints)} keypoints")
        
        inference_time = (time.time() - start_time) * 1000  # Convert to ms
        
        return jsonify({
            "success": True,
            "keypoints": keypoints,
            "total_keypoints": len(keypoints),
            "inference_time": round(inference_time, 2)
        })
        
    except Exception as e:
        print(f"❌❌❌ CRITICAL ERROR in detect_field: {e}")
        import traceback
        traceback.print_exc()
        
        return jsonify({
            "success": False,
            "error": f"Server error: {str(e)}",
            "type": type(e).__name__
        }), 500

# ============================================
# HELPER: PREPROCESS IMAGE
# ============================================
def preprocess_image(image, target_size=(640, 640)):
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

# ============================================
# HELPER: PARSE KEYPOINT PREDICTIONS
# ============================================
def parse_keypoint_predictions(outputs, original_width, original_height):
    """
    Parse raw ONNX predictions into 32 keypoints
    
    YOLOv8 keypoint model output:
    - Shape: (1, 391, 8400) 
    - Format: [batch, features, detections]
    - Features (391): 
        [0-3]: bbox (x, y, w, h)
        [4]: objectness score
        [5-100]: class probabilities (96 classes)
        [101-390]: keypoint data (32 keypoints × 3 values each: x, y, confidence)
    """
    keypoints = []
    
    try:
        # Get the output tensor (should be shape: (1, 391, 8400))
        output = outputs[0]
        print(f"📊 Output shape: {output.shape}")
        
        # Transpose to (8400, 391) for easier processing
        output = output.transpose(0, 2, 1)  # (1, 8400, 391)
        output = output[0]  # Remove batch dimension -> (8400, 391)
        print(f"📊 Transposed shape: {output.shape}")
        
        # Extract keypoints from each detection
        # Keypoint data starts at index 101 (after bbox + objectness + class scores)
        KEYPOINT_START = 101
        NUM_KEYPOINTS = 32
        
        # Scale factors (from 640x640 to original image size)
        scale_x = original_width / 640.0
        scale_y = original_height / 640.0
        
        # Find detection with highest objectness
        best_detection_idx = -1
        best_objectness = 0.0
        
        for i in range(output.shape[0]):
            obj = float(output[i, 4])
            if obj > best_objectness:
                best_objectness = obj
                best_detection_idx = i
        
        print(f"   Best detection: idx={best_detection_idx}, objectness={best_objectness:.3f}")
        
        if best_detection_idx >= 0 and best_objectness > 0.3:
            detection = output[best_detection_idx]
            
            # Extract 32 keypoints
            for kp_idx in range(NUM_KEYPOINTS):
                base_idx = KEYPOINT_START + (kp_idx * 3)
                
                # Check bounds
                if base_idx + 2 >= len(detection):
                    break
                
                kp_x = float(detection[base_idx]) * scale_x
                kp_y = float(detection[base_idx + 1]) * scale_y
                kp_conf = float(detection[base_idx + 2])
                
                # Add all keypoints (even low confidence ones)
                keypoints.append({
                    "x": round(kp_x, 2),
                    "y": round(kp_y, 2),
                    "confidence": round(kp_conf, 3),
                    "class_name": get_keypoint_name(kp_idx)
                })
        
        print(f"✅ Extracted {len(keypoints)} keypoints")
        return keypoints
        
    except Exception as e:
        print(f"❌ Error parsing predictions: {e}")
        import traceback
        traceback.print_exc()
        return []

# ============================================
# HELPER: KEYPOINT NAMES
# ============================================
def get_keypoint_name(index):
    """Get human-readable name for keypoint index (0-31)"""
    names = [
        "top_left_corner", "top_right_corner", "bottom_left_corner", "bottom_right_corner",
        "left_penalty_top_left", "left_penalty_top_right", "left_penalty_bottom_left", "left_penalty_bottom_right",
        "right_penalty_top_left", "right_penalty_top_right", "right_penalty_bottom_left", "right_penalty_bottom_right",
        "left_six_yard_top_left", "left_six_yard_top_right", "left_six_yard_bottom_left", "left_six_yard_bottom_right",
        "right_six_yard_top_left", "right_six_yard_top_right", "right_six_yard_bottom_left", "right_six_yard_bottom_right",
        "center_circle_top", "center_circle_bottom", "center_circle_left", "center_circle_right",
        "halfway_line_top", "halfway_line_bottom",
        "left_goal_left_post", "left_goal_right_post", "right_goal_left_post", "right_goal_right_post",
        "penalty_spot_left", "penalty_spot_right"
    ]
    
    if index < len(names):
        return names[index]
    else:
        return f"keypoint_{index}"

# ============================================
# START SERVER
# ============================================
if __name__ == '__main__':
    port = 8000
    print(f"\n🚀 Starting Football Tactics Python Service on port {port}...")
    print(f"📡 Health check: http://localhost:{port}/health")
    print(f"🏟️  Field detection: http://localhost:{port}/detect/field")
    app.run(host='0.0.0.0', port=port)
