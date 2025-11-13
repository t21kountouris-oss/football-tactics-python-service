import boto3
import os
import logging

logger = logging.getLogger(__name__)

def download_models_from_r2():
    """
    Download ONNX models from Cloudflare R2
    Returns True if successful, False otherwise
    """
    try:
        # Get R2 credentials from environment
        endpoint_url = os.environ.get('R2_ENDPOINT_URL')
        access_key = os.environ.get('R2_ACCESS_KEY_ID')
        secret_key = os.environ.get('R2_SECRET_ACCESS_KEY')
        bucket_name = os.environ.get('R2_BUCKET_NAME', 'football-tactics-models')
        
        # Check if credentials are configured
        if not all([endpoint_url, access_key, secret_key]):
            logger.info("⚠️ R2 credentials not configured - skipping download")
            return False
        
        # Create S3 client (R2 is S3-compatible)
        s3_client = boto3.client(
            's3',
            endpoint_url=endpoint_url,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key
        )
        
        # List of models to download
        models_to_download = [
            {
                'key': 'field/field-detection.onnx',
                'local_path': 'models/field/field-detection.onnx'
            },
            {
                'key': 'players/player-detection.onnx',
                'local_path': 'models/players/player-detection.onnx'
            },
            {
                'key': 'ball/ball-detection.onnx',
                'local_path': 'models/ball/ball-detection.onnx'
            }
        ]
        
        downloaded = 0
        
        for model in models_to_download:
            try:
                # Create directory if it doesn't exist
                os.makedirs(os.path.dirname(model['local_path']), exist_ok=True)
                
                # Download file
                logger.info(f"📥 Downloading {model['key']}...")
                s3_client.download_file(
                    Bucket=bucket_name,
                    Key=model['key'],
                    Filename=model['local_path']
                )
                
                file_size = os.path.getsize(model['local_path'])
                logger.info(f"   ✅ Downloaded {model['key']} ({file_size / 1024 / 1024:.2f} MB)")
                downloaded += 1
                
            except Exception as e:
                logger.warning(f"   ⚠️ Failed to download {model['key']}: {e}")
        
        if downloaded > 0:
            logger.info(f"✅ Downloaded {downloaded}/{len(models_to_download)} models from R2")
            return True
        else:
            logger.warning("⚠️ No models downloaded from R2")
            return False
    
    except Exception as e:
        logger.error(f"❌ R2 download error: {e}")
        return False
