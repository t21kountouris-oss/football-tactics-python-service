from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import onnxruntime as ort
import base64
import time
import cv2
import os
import logging

# ============================================
# CONFIGURE LOGGING (Gunicorn-compatible)
# ============================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# ============================================
# GLOBAL MODEL SESSION
# ============================================
model_session = None
MODEL_PATH = "models/field/field-detection.onnx"

# ============================================
# R2 MODEL DOWNLOADER
# ============================================
def download_model_from_r2():
    """
    Download field-detection.onnx from Cloudflare R2
    Returns True if successful or file already exists
    """
    try:
        # Check if model already exists
        if os.path.exists(MODEL_PATH):
            file_size_mb = os.path.getsize(MODEL_PATH) / 1024 / 1024
            logger.info(f"✅ Model already exists: {MODEL_PATH} ({file_size_mb:.2f} MB)")
            return True
        
        # Get R2 credentials from environment
        endpoint_url = os.environ.get('R2_ENDPOINT_URL')
        access_key = os.environ.get('R2_ACCESS_KEY_ID')
        secret_key = os.environ.get('R2_SECRET_ACCESS_KEY')
        bucket_name = os.environ.get('R2_BUCKET_NAME', 'football-tactics-models')
        
        # Check if credentials are configured
        if not all([endpoint_url, access_key, secret_key]):
            logger.warning("⚠️ R2 credentials not configured - cannot download model")
            return False
        
        logger.info("📥 Downloading model from R2...")
        
        # Import boto3 (S3-compatible client for R2)
        import boto3
        
        # Create S3 client
        s3_client = boto3.client(
            's3',
            endpoint_url=endpoint_url,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key
        )
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        
        # Download file
        r2_key = 'field/field-detection.onnx'
        s3_client.download_file(
            Bucket=bucket_name,
            Key=r2_key,
            Filename=MODEL_PATH
        )
        
        file_size_mb = os.path.getsize(MODEL_PATH) / 1024 / 1024
        logger.info(f"✅ Model downloaded: {MODEL_PATH} ({file_size_mb:.2f} MB)")
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to download model from R2: {e}")
        return False

# ============================================
# LOAD MODEL ON STARTUP
# ============================================
def load_model():
    """Load ONNX model into memory"""
    global model_session
    
    try:
        logger.info("🔄 Loading ONNX model...")
        
        # First, try to download from R2 if not exists
        download_success = download_model_from_r2()
        
        if not os.path.exists(MODEL_PATH):
            logger.error(f"❌ Model file not found: {MODEL_PATH}")
            logger.error("   - R2 download failed or model not in R2")
            logger.error("   - Model must be uploaded to R2 bucket")
            return False
        
        # Load model
        model_session = ort.InferenceSession(MODEL_PATH)
        
        # Log model info
        input_info = model_session.get_inputs()[0]
        output_info = model_session.get_outputs()[0]
        
        logger.info(f"✅ Model loaded successfully: {MODEL_PATH}")
        logger.info(f"   - Input: {input_info.name} {input_info.shape}")
        logger.info(f"   - Output: {output_info.name} {output_info.shape}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to load model: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

# Load model on startup
logger.info("🚀 Starting Flask application...")
load_model()

# ============================================
# HEALTH CHECK ENDPOINT
# ============================================
@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    if model_session is None:
        return jsonify({
            "success": False,
            "model_loaded": False,
            "model_path": MODEL_PATH,
            "file_exists": os.path.exists(MODEL_PATH),
            "message": "Model not loaded - check logs for errors"
        }), 500
    
    return jsonify({
        "success": True,
        "model_loaded": True,
        "model_path": MODEL_PATH,
        "models_available": ["field"],
        "message": "Python inference service is healthy"
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
    if model_session is None:
        return jsonify({
            "success": False,
            "error": "Model not loaded - service not ready"
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
        
        input_name = model_session.get_inputs()[0].name
        output_name = model_session.get_outputs()[0].name
        
        predictions = model_session.run([output_name], {input_name: input_tensor})[0]
        
        inference_time = (time.time() - start_time) * 1000  # Convert to ms
        
        logger.info(f"✅ Inference completed in {inference_time:.2f}ms")
        logger.info(f"   - Input: {original_width}x{original_height}")
        logger.info(f"   - Output shape: {predictions.shape}")
        
        # Post-process predictions
        keypoints = []
        # NOTE: This is simplified - actual keypoint parsing depends on model output format
        # You'll need to implement proper post-processing based on your YOLO pose model
        
        return jsonify({
            "success": True,
            "inference_time": inference_time,
            "original_size": {
                "width": original_width,
                "height": original_height
            },
            "predictions_shape": list(predictions.shape),
            "keypoints": keypoints,
            "message": "Field detection completed (post-processing needed)"
        }), 200
        
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        logger.error(f"❌ Error in detect_field: {error_trace}")
        
        return jsonify({
            "success": False,
            "error": str(e),
            "trace": error_trace
        }), 500

# ============================================
# START SERVER
# ============================================
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    logger.info(f"🚀 Starting server on port {port}...")
    app.run(host='0.0.0.0', port=port, debug=False)
