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
                
        except Exception as e:
            return jsonify({
                "success": False,
                "error": f"Image decoding error: {str(e)}"
            }), 400
        
        # Run ONNX inference
        predictions = run_field_detection(image)
        
        # Parse predictions into 32 keypoints
        keypoints = parse_keypoint_predictions(predictions)
        
        inference_time = (time.time() - start_time) * 1000  # Convert to ms
        
        return jsonify({
            "success": True,
            "keypoints": keypoints,
            "total_keypoints": len(keypoints),
            "inference_time": round(inference_time, 2)
        })
        
    except Exception as e:
        return jsonify({
            "success": False,
            "error": f"Server error: {str(e)}"
        }), 500

# ============================================
# HELPER: PARSE KEYPOINT PREDICTIONS
# ============================================
def parse_keypoint_predictions(predictions):
    """
    Parse raw ONNX predictions into 32 keypoints
    
    Roboflow keypoint model output format:
    - predictions: array of shape (1, 391, 8400) or similar
    - Each detection has: [x, y, w, h, confidence, class_id, kp1_x, kp1_y, kp1_conf, ...]
    
    Returns: List of 32 keypoints with {x, y, confidence, class_name}
    """
    keypoints = []
    
    try:
        # predictions is a list with one array
        # Shape: (1, features, num_predictions)
        pred_array = predictions[0]  # Get first (and only) array
        
        # Transpose to (num_predictions, features)
        if len(pred_array.shape) == 2:
            pred_array = pred_array.T
        
        # Extract keypoint data
        # Assuming format: [x, y, w, h, obj_conf, class_conf, ...keypoint_data...]
        # Keypoints typically start after the first 6 values
        
        # For each prediction row
        for i in range(min(pred_array.shape[0], 100)):  # Check first 100 predictions
            row = pred_array[i]
            
            # Object confidence (usually at index 4)
            obj_conf = float(row[4]) if len(row) > 4 else 0.0
            
            # Only process predictions with confidence > 0.25
            if obj_conf < 0.25:
                continue
            
            # Keypoints usually start at index 6 (after bbox + confidences)
            # Format: [kp1_x, kp1_y, kp1_conf, kp2_x, kp2_y, kp2_conf, ...]
            keypoint_start_idx = 6
            num_keypoints_per_detection = (len(row) - keypoint_start_idx) // 3
            
            for kp_idx in range(num_keypoints_per_detection):
                base_idx = keypoint_start_idx + (kp_idx * 3)
                
                if base_idx + 2 >= len(row):
                    break
                
                kp_x = float(row[base_idx])
                kp_y = float(row[base_idx + 1])
                kp_conf = float(row[base_idx + 2])
                
                # Only add keypoints with confidence > 0.3
                if kp_conf > 0.3:
                    keypoints.append({
                        "x": round(kp_x, 2),
                        "y": round(kp_y, 2),
                        "confidence": round(kp_conf, 3),
                        "class_name": f"keypoint_{kp_idx}"
                    })
        
        # Sort by confidence and take top 32
        keypoints.sort(key=lambda k: k['confidence'], reverse=True)
        keypoints = keypoints[:32]
        
        return keypoints
        
    except Exception as e:
        print(f"❌ Error parsing predictions: {e}")
        # Return empty list if parsing fails
        return []

# ============================================
# START SERVER
# ============================================
if __name__ == '__main__':
    port = 8000
    print(f"\n🚀 Starting Football Tactics Python Service on port {port}...")
    print(f"📡 Health check: http://localhost:{port}/health")
    print(f"🏟️  Field detection: http://localhost:{port}/detect/field")
    app.run(host='0.0.0.0', port=port)
