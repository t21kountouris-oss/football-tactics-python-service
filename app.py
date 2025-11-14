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
    if session is None:
        return jsonify({
            "status": "error",
            "message": "Model not loaded"
        }), 500
    
    return jsonify({
        "status": "healthy",
        "model_loaded": True,
        "model_path": MODEL_PATH,
        "models_available": ["field-detection"]
    }), 200

# ============================================
# FIELD DETECTION ENDPOINT
# ============================================
@app.route('/detect-field', methods=['POST'])
def detect_field():
    """
    Detect football field keypoints from base64-encoded image
    
    Request body:
    {
        "image": "base64-encoded-image-data"
    }
    """
    if session is None:
        return jsonify({
            "success": False,
            "error": "Model not loaded"
        }), 500
    
    try:
        # Get image from request
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({
                "success": False,
                "error": "Missing 'image' field in request body"
            }), 400
        
        image_base64 = data['image']
        
        # Decode base64 image
        try:
            image_bytes = base64.b64decode(image_base64)
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
                "error": f"Invalid base64 image: {str(e)}"
            }), 400
        
        # Preprocess image for ONNX model
        # Resize to 640x640 (standard YOLO input size)
        original_height, original_width = image.shape[:2]
        input_size = 640
        
        # Resize with padding to maintain aspect ratio
        scale = min(input_size / original_width, input_size / original_height)
        new_width = int(original_width * scale)
        new_height = int(original_height * scale)
        
        resized = cv2.resize(image, (new_width, new_height))
        
        # Create padded image
        padded = np.full((input_size, input_size, 3), 114, dtype=np.uint8)
        
        # Calculate padding offsets
        pad_x = (input_size - new_width) // 2
        pad_y = (input_size - new_height) // 2
        
        # Place resized image in center
        padded[pad_y:pad_y+new_height, pad_x:pad_x+new_width] = resized
        
        # Convert BGR to RGB
        rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
        
        # Normalize to [0, 1] and transpose to CHW format
        input_tensor = rgb.astype(np.float32) / 255.0
        input_tensor = np.transpose(input_tensor, (2, 0, 1))  # HWC -> CHW
        input_tensor = np.expand_dims(input_tensor, axis=0)   # Add batch dimension
        
        # Run inference
        start_time = time.time()
        
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name
        
        predictions = session.run([output_name], {input_name: input_tensor})[0]
        
        inference_time = (time.time() - start_time) * 1000  # Convert to ms
        
        # Post-process predictions
        # predictions shape: [1, 39, 8400] for YOLO pose
        # 39 = 4 (bbox) + 1 (conf) + 32*2 (keypoints x,y) - but let's check actual shape
        
        print(f"   - Predictions shape: {predictions.shape}")
        
        # Extract keypoints (implementation depends on model output format)
        # For now, return raw predictions for debugging
        
        keypoints = []
        # This is a simplified example - actual implementation depends on model format
        # You'll need to parse the YOLO pose output format correctly
        
        return jsonify({
            "success": True,
            "inference_time": inference_time,
            "original_size": {
                "width": original_width,
                "height": original_height
            },
            "predictions_shape": predictions.shape,
            "message": "Field detection completed (raw output - needs post-processing)",
            "keypoints": keypoints  # Will be populated after proper post-processing
        }), 200
        
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print(f"❌ Error in detect_field: {error_trace}")
        
        return jsonify({
            "success": False,
            "error": str(e),
            "trace": error_trace
        }), 500

# ============================================
# START SERVER
# ============================================
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8000))
    print(f"🚀 Starting server on port {port}...")
    app.run(host='0.0.0.0', port=port, debug=False)
