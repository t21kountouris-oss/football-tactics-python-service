import os
import io
import logging
import base64
import numpy as np
import onnxruntime as ort
from PIL import Image
from flask import Flask, request, jsonify
from flask_cors import CORS
import boto3
from botocore.client import Config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Global model storage
field_model = None
field_session = None

def list_r2_files():
    """List all files in R2 bucket for debugging"""
    try:
        logger.info("🔍 Listing R2 bucket contents...")
        
        # R2 credentials from environment
        r2_access_key = os.environ.get('R2_ACCESS_KEY_ID')
        r2_secret_key = os.environ.get('R2_SECRET_ACCESS_KEY')
        r2_endpoint = os.environ.get('R2_ENDPOINT_URL')
        r2_bucket = 'football-tactics-models'
        
        if not all([r2_access_key, r2_secret_key, r2_endpoint]):
            logger.error("❌ R2 credentials not configured")
            return []
        
        # Create S3 client for R2
        s3 = boto3.client(
            's3',
            endpoint_url=r2_endpoint,
            aws_access_key_id=r2_access_key,
            aws_secret_access_key=r2_secret_key,
            config=Config(signature_version='s3v4'),
            region_name='auto'
        )
        
        # List objects
        response = s3.list_objects_v2(Bucket=r2_bucket)
        
        if 'Contents' not in response:
            logger.warning("⚠️ R2 bucket is empty")
            return []
        
        files = []
        for obj in response['Contents']:
            file_info = {
                'key': obj['Key'],
                'size': obj['Size'] / (1024 * 1024),  # MB
                'last_modified': str(obj['LastModified'])
            }
            files.append(file_info)
            logger.info(f"   - Found: {obj['Key']} ({file_info['size']:.2f} MB)")
        
        return files
    except Exception as e:
        logger.error(f"❌ Failed to list R2 contents: {e}")
        return []

def download_model_from_r2():
    """Download ONNX model from Cloudflare R2"""
    try:
        logger.info("📥 Downloading model from R2...")
        
        # R2 credentials from environment
        r2_access_key = os.environ.get('R2_ACCESS_KEY_ID')
        r2_secret_key = os.environ.get('R2_SECRET_ACCESS_KEY')
        r2_endpoint = os.environ.get('R2_ENDPOINT_URL')
        r2_bucket = 'football-tactics-models'
        
        if not all([r2_access_key, r2_secret_key, r2_endpoint]):
            logger.error("❌ R2 credentials not configured")
            return None
        
        # Create S3 client for R2
        s3 = boto3.client(
            's3',
            endpoint_url=r2_endpoint,
            aws_access_key_id=r2_access_key,
            aws_secret_access_key=r2_secret_key,
            config=Config(signature_version='s3v4'),
            region_name='auto'
        )
        
        # List files first to find the model
        files = list_r2_files()
        
        # Try different possible paths
        possible_paths = [
            'field-detection.onnx',
            'models/field/field-detection.onnx',
            'field/field-detection.onnx',
            'onnx/field-detection.onnx'
        ]
        
        model_key = None
        for path in possible_paths:
            if any(f['key'] == path for f in files):
                model_key = path
                logger.info(f"✅ Found model at: {model_key}")
                break
        
        if not model_key:
            # If no exact match, try to find any .onnx file with 'field' in the name
            for f in files:
                if 'field' in f['key'].lower() and f['key'].endswith('.onnx'):
                    model_key = f['key']
                    logger.info(f"✅ Found field model at: {model_key}")
                    break
        
        if not model_key:
            logger.error("❌ No field detection model found in R2 bucket")
            logger.error(f"❌ Available files: {[f['key'] for f in files]}")
            return None
        
        # Download model
        local_path = 'models/field/field-detection.onnx'
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        
        logger.info(f"📥 Downloading {model_key} from R2...")
        s3.download_file(r2_bucket, model_key, local_path)
        
        file_size = os.path.getsize(local_path) / (1024 * 1024)
        logger.info(f"✅ Model downloaded: {local_path} ({file_size:.2f} MB)")
        
        return local_path
    except Exception as e:
        logger.error(f"❌ Failed to download model from R2: {e}")
        return None

def load_field_model():
    """Load field detection ONNX model"""
    global field_session
    
    try:
        logger.info("🔄 Loading ONNX model...")
        
        # Download from R2
        model_path = download_model_from_r2()
        if not model_path:
            raise Exception("Failed to download model")
        
        # Load ONNX model
        field_session = ort.InferenceSession(
            model_path,
            providers=['CPUExecutionProvider']
        )
        
        # Log model info
        input_info = field_session.get_inputs()[0]
        output_info = field_session.get_outputs()[0]
        
        logger.info(f"✅ Model loaded successfully: {model_path}")
        logger.info(f"   - Input: {input_info.name} {input_info.shape}")
        logger.info(f"   - Output: {output_info.name} {output_info.shape}")
        
        return True
    except Exception as e:
        logger.error(f"❌ Failed to load model: {e}")
        return False

def preprocess_image(image_data, target_size=(640, 640)):
    """Preprocess image for ONNX model"""
    try:
        # Decode base64
        if ',' in image_data:
            image_data = image_data.split(',')[1]
        
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        
        # Resize
        image = image.resize(target_size, Image.BILINEAR)
        
        # Convert to numpy array and normalize
        img_array = np.array(image).astype(np.float32) / 255.0
        
        # Transpose to CHW format
        img_array = img_array.transpose(2, 0, 1)
        
        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array
    except Exception as e:
        logger.error(f"❌ Image preprocessing failed: {e}")
        raise

def postprocess_field_detection(output, conf_threshold=0.5):
    """Post-process field detection output to extract keypoints"""
    try:
        # Output shape: [1, 101, 8400]
        # 101 = 4 (bbox) + 1 (confidence) + 96 (32 keypoints × 3: x, y, visibility)
        
        predictions = output[0]  # [101, 8400]
        
        # Get confidence scores (5th row)
        confidences = predictions[4, :]
        
        # Find detections above threshold
        mask = confidences > conf_threshold
        
        if not np.any(mask):
            return {"keypoints": [], "confidence": 0.0, "count": 0}
        
        # Get highest confidence detection
        max_idx = np.argmax(confidences)
        confidence = float(confidences[max_idx])
        
        # Extract keypoints (rows 5-100 contain 32 keypoints × 3 values)
        keypoint_data = predictions[5:101, max_idx]
        
        # Reshape to [32, 3] (32 keypoints, each with x, y, visibility)
        keypoints = keypoint_data.reshape(32, 3)
        
        # Convert to list of dicts with normalized coordinates
        keypoints_list = []
        for i, (x, y, visibility) in enumerate(keypoints):
            keypoints_list.append({
                "id": i,
                "x": float(x) / 640.0,  # Normalize to 0-1
                "y": float(y) / 640.0,
                "visibility": float(visibility)
            })
        
        return {
            "keypoints": keypoints_list,
            "confidence": confidence,
            "count": len(keypoints_list)
        }
    except Exception as e:
        logger.error(f"❌ Postprocessing failed: {e}")
        raise

# ========================================
# ROUTES
# ========================================

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "service": "football-tactics-python-inference",
        "models_loaded": ["field-detection"] if field_session else [],
        "version": "1.0.1"
    }), 200

@app.route('/debug/r2', methods=['GET'])
def debug_r2():
    """Debug endpoint to list R2 contents"""
    try:
        files = list_r2_files()
        return jsonify({
            "success": True,
            "bucket": "football-tactics-models",
            "file_count": len(files),
            "files": files
        }), 200
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/detect/field', methods=['POST'])
def detect_field():
    """Detect football field keypoints"""
    try:
        if not field_session:
            return jsonify({
                "success": False,
                "error": "Model not loaded"
            }), 503
        
        # Get image from request
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({
                "success": False,
                "error": "Missing 'image' in request body"
            }), 400
        
        image_data = data['image']
        logger.info(f"📸 Processing field detection (image size: {len(image_data) / 1024:.2f} KB)")
        
        # Preprocess image
        input_tensor = preprocess_image(image_data)
        
        # Run inference
        input_name = field_session.get_inputs()[0].name
        output_name = field_session.get_outputs()[0].name
        
        outputs = field_session.run([output_name], {input_name: input_tensor})
        
        # Postprocess results
        result = postprocess_field_detection(outputs[0])
        
        logger.info(f"✅ Field detection complete: {result['count']} keypoints, confidence: {result['confidence']:.2f}")
        
        return jsonify({
            "success": True,
            "data": result,
            "model": "field-detection.onnx"
        }), 200
        
    except Exception as e:
        logger.error(f"❌ Field detection failed: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/detect/players', methods=['POST'])
def detect_players():
    """Detect players (placeholder - requires players model)"""
    return jsonify({
        "success": False,
        "error": "Players detection model not yet loaded"
    }), 501

@app.route('/detect/ball', methods=['POST'])
def detect_ball():
    """Detect ball (placeholder - requires ball model)"""
    return jsonify({
        "success": False,
        "error": "Ball detection model not yet loaded"
    }), 501

@app.route('/', methods=['GET'])
def index():
    """Root endpoint"""
    return jsonify({
        "service": "football-tactics-python-inference",
        "status": "running",
        "models": ["field-detection"] if field_session else [],
        "endpoints": {
            "health": "/health",
            "debug_r2": "/debug/r2",
            "detect_field": "/detect/field",
            "detect_players": "/detect/players (coming soon)",
            "detect_ball": "/detect/ball (coming soon)"
        }
    }), 200

# ========================================
# STARTUP
# ========================================

# Load model on startup
logger.info("🚀 Starting Flask application...")
load_field_model()

if __name__ == '__main__':
    # Development server
    app.run(host='0.0.0.0', port=5000, debug=False)
