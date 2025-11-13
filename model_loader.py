import onnxruntime as ort
import os
import logging

logger = logging.getLogger(__name__)

class ModelLoader:
    """
    Load and manage ONNX models
    Implements caching to avoid reloading models
    """
    
    def __init__(self):
        self.models = {}
        self.load_all_models()
    
    def load_all_models(self):
        """Load all available ONNX models"""
        # Model paths from environment or defaults
        field_path = os.environ.get('FIELD_MODEL_PATH', 'models/field/field-detection.onnx')
        players_path = os.environ.get('PLAYERS_MODEL_PATH', 'models/players/player-detection.onnx')
        ball_path = os.environ.get('BALL_MODEL_PATH', 'models/ball/ball-detection.onnx')
        
        # Load field model
        if os.path.exists(field_path):
            logger.info(f"📥 Loading field model: {field_path}")
            self.models['field'] = self._load_model(field_path)
            logger.info(f"   ✅ Field model loaded ({os.path.getsize(field_path) / 1024 / 1024:.2f} MB)")
        else:
            logger.warning(f"   ⚠️ Field model not found: {field_path}")
        
        # Load players model
        if os.path.exists(players_path):
            logger.info(f"📥 Loading players model: {players_path}")
            self.models['players'] = self._load_model(players_path)
            logger.info(f"   ✅ Players model loaded ({os.path.getsize(players_path) / 1024 / 1024:.2f} MB)")
        else:
            logger.warning(f"   ⚠️ Players model not found: {players_path}")
        
        # Load ball model
        if os.path.exists(ball_path):
            logger.info(f"📥 Loading ball model: {ball_path}")
            self.models['ball'] = self._load_model(ball_path)
            logger.info(f"   ✅ Ball model loaded ({os.path.getsize(ball_path) / 1024 / 1024:.2f} MB)")
        else:
            logger.warning(f"   ⚠️ Ball model not found: {ball_path}")
    
    def _load_model(self, path):
        """Load ONNX model from file"""
        try:
            # Configure ONNX Runtime session
            session_options = ort.SessionOptions()
            session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            
            # Create inference session
            session = ort.InferenceSession(path, session_options)
            
            return session
        except Exception as e:
            logger.error(f"❌ Failed to load model {path}: {e}")
            return None
    
    def get_model(self, model_name):
        """Get loaded model by name"""
        return self.models.get(model_name)
    
    def get_loaded_models(self):
        """Get list of loaded model names"""
        return list(self.models.keys())
