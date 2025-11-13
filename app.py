from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import base64
import time
import cv2
from inference import run_field_detection

app = Flask(__name__)
CORS(app)

# ============================================
# HEALTH CHECK ENDPOINT
# ============================================
@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "message": "Football Tactics Python Service",
        "models_loaded": ["field"]
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
            
            print(f"✅ Image decoded: {image.shape}")
                
        except Exception as e:
            print(f"❌ Image decoding error: {e}")
            return jsonify({
                "success": False,
                "error": f"Image decoding error: {str(e)}"
            }), 400
        
        # Run ONNX inference
        print("🔍 Running ONNX inference...")
        predictions = run_field_detection(image)
        print(f"✅ ONNX inference complete")
        print(f"   - Type: {type(predictions)}")
        print(f"   - Length: {len(predictions) if isinstance(predictions, list) else 'N/A'}")
        
        if isinstance(predictions, list) and len(predictions) > 0:
            print(f"   - First element type: {type(predictions[0])}")
            print(f"   - First element shape: {predictions[0].shape if hasattr(predictions[0], 'shape') else 'N/A'}")
        
        # Parse predictions into 32 keypoints
        print("🔍 Parsing keypoints...")
        keypoints = parse_keypoint_predictions(predictions)
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
# HELPER: PARSE KEYPOINT PREDICTIONS
# ============================================
def parse_keypoint_predictions(predictions):
    """
    Parse raw ONNX predictions into 32 keypoints
    
    Roboflow YOLOv8 keypoint model output:
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
        if not isinstance(predictions, list) or len(predictions) == 0:
            print("❌ Invalid predictions format")
            return []
        
        # Get the output tensor (should be shape: (1, 391, 8400))
        output = predictions[0]
        print(f"📊 Output shape: {output.shape}")
        
        # Transpose to (8400, 391) for easier processing
        output = output.T  # Now shape: (detections, features)
        print(f"📊 Transposed shape: {output.shape}")
        
        # Extract keypoints from each detection
        # Keypoint data starts at index 101 (after bbox + objectness + class scores)
        # Format: [kp1_x, kp1_y, kp1_conf, kp2_x, kp2_y, kp2_conf, ..., kp32_x, kp32_y, kp32_conf]
        
        KEYPOINT_START = 101
        NUM_KEYPOINTS = 32
        
        # Process each detection (row)
        for detection_idx in range(output.shape[0]):
            detection = output[detection_idx]
            
            # Get objectness score (index 4)
            objectness = float(detection[4])
            
            # Only process detections with high confidence
            if objectness < 0.3:
                continue
            
            print(f"   Detection {detection_idx}: objectness={objectness:.3f}")
            
            # Extract 32 keypoints
            for kp_idx in range(NUM_KEYPOINTS):
                base_idx = KEYPOINT_START + (kp_idx * 3)
                
                # Check bounds
                if base_idx + 2 >= len(detection):
                    break
                
                kp_x = float(detection[base_idx])
                kp_y = float(detection[base_idx + 1])
                kp_conf = float(detection[base_idx + 2])
                
                # Only add visible keypoints
                if kp_conf > 0.5:
                    keypoints.append({
                        "x": round(kp_x, 2),
                        "y": round(kp_y, 2),
                        "confidence": round(kp_conf, 3),
                        "class_name": get_keypoint_name(kp_idx)
                    })
            
            # If we found keypoints from this detection, stop (only process best detection)
            if len(keypoints) > 0:
                break
        
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
    # Standard football field keypoint names (32 total)
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
