from flask import Flask, request, jsonify
from flask_cors import CORS
import logging
import os
from model_loader import ModelLoader
from inference import run_field_detection, run_player_detection, run_ball_detection
from r2_downloader import download_models_from_r2

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create Flask app
app = Flask(__name__)

# Enable CORS for all routes
CORS(app, resources={r"/*": {"origins": "*"}})

# Global model loader (singleton)
model_loader = None

# Initialize models at module level (Flask 3.0 compatible)
with app.app_context():
    global model_loader
    
    logger.info("🚀 Initializing Python inference service...")
    
    # Download models from R2 (if configured)
    try:
        logger.info("📥 Attempting to download models from R2...")
        success = download_models_from_r2()
        
        if success:
            logger.info("✅ Models downloaded from R2")
        else:
            logger.warning("⚠️ R2 download failed, will use local models if available")
    except Exception as e:
        logger.warning(f"⚠️ R2 download error: {e}")
    
    # Initialize model loader
    try:
        model_loader = ModelLoader()
        logger.info("✅ Model loader initialized")
    except Exception as e:
        logger.error(f"❌ Failed to initialize model loader: {e}")
        raise

@app.route('/health', methods=['GET'])
def health_check():
    """
    Health check endpoint
    Returns status and loaded models
    """
    if model_loader is None:
        return jsonify({
            "status": "initializing",
            "models_loaded": []
        }), 503
    
    loaded_models = model_loader.get_loaded_models()
    
    return jsonify({
        "status": "ok",
        "models_loaded": loaded_models,
        "message": f"{len(loaded_models)} models loaded"
    }), 200

@app.route('/detect/field', methods=['POST'])
def detect_field():
    """
    Detect field keypoints (32 points)
    Request body: { "image": "base64_encoded_image" }
    Response: { "predictions": [...] }
    """
    logger.info("🏟️ POST /detect/field called")
    
    try:
        # Parse request
        data = request.get_json()
        
        if not data or 'image' not in data:
            return jsonify({
                "success": False,
                "error": "Missing 'image' in request body"
            }), 400
        
        image_base64 = data['image']
        logger.info(f"   - Image size: {len(image_base64) / 1024:.2f} KB")
        
        # Run inference
        predictions = run_field_detection(model_loader, image_base64)
        
        logger.info(f"   ✅ Detected {len(predictions)} keypoints")
        
        return jsonify({
            "success": True,
            "predictions": predictions
        }), 200
    
    except Exception as e:
        logger.error(f"   ❌ Error in field detection: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/detect/players', methods=['POST'])
def detect_players():
    """
    Detect players (bounding boxes)
    Request body: { "image": "base64_encoded_image" }
    Response: { "predictions": [...] }
    """
    logger.info("👥 POST /detect/players called")
    
    try:
        # Parse request
        data = request.get_json()
        
        if not data or 'image' not in data:
            return jsonify({
                "success": False,
                "error": "Missing 'image' in request body"
            }), 400
        
        image_base64 = data['image']
        logger.info(f"   - Image size: {len(image_base64) / 1024:.2f} KB")
        
        # Run inference
        predictions = run_player_detection(model_loader, image_base64)
        
        logger.info(f"   ✅ Detected {len(predictions)} players")
        
        return jsonify({
            "success": True,
            "predictions": predictions
        }), 200
    
    except Exception as e:
        logger.error(f"   ❌ Error in player detection: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/detect/ball', methods=['POST'])
def detect_ball():
    """
    Detect ball
    Request body: { "image": "base64_encoded_image" }
    Response: { "detections": [...] }
    """
    logger.info("⚽ POST /detect/ball called")
    
    try:
        # Parse request
        data = request.get_json()
        
        if not data or 'image' not in data:
            return jsonify({
                "success": False,
                "error": "Missing 'image' in request body"
            }), 400
        
        image_base64 = data['image']
        logger.info(f"   - Image size: {len(image_base64) / 1024:.2f} KB")
        
        # Run inference
        detections = run_ball_detection(model_loader, image_base64)
        
        logger.info(f"   ✅ Detected {len(detections)} ball(s)")
        
        return jsonify({
            "success": True,
            "detections": detections
        }), 200
    
    except Exception as e:
        logger.error(f"   ❌ Error in ball detection: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

# Error handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Not found"}), 404

@app.errorhandler(500)
def internal_error(error):
    logger.error(f"Internal error: {error}")
    return jsonify({"error": "Internal server error"}), 500

# Run server
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('DEBUG', 'false').lower() == 'true'
    
    logger.info(f"🚀 Starting server on port {port}...")
    logger.info(f"   DEBUG mode: {debug}")
    
    app.run(host='0.0.0.0', port=port, debug=debug)
