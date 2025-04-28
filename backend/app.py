import os
import logging
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from flask import Flask, request, jsonify
from flask_cors import CORS
from models.combined_model import CombinedModel
import io
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,  # Change to DEBUG level
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

severity_levels = ['Normal', 'Mild DR', 'Moderate DR', 'Severe DR', 'Proliferative DR']

app = Flask(__name__)
CORS(app, resources={
    r"/*": {
        "origins": ["http://localhost:3000"],
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization", "X-Requested-With"]
    }
})

# Initialize model service
MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'checkpoints', 'best_model.pth')

@app.before_request
def log_request_info():
    logger.debug('Headers: %s', request.headers)
    logger.debug('Body: %s', request.get_data())

@app.after_request
def after_request(response):
    logger.debug('Response: %s', response.get_data())
    return response

# Initialize model with config parameters
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f'Using device: {device}')

model = CombinedModel(num_classes=5).to(device)
logger.info('Model initialized')

# Try to load model if it exists
try:
    if os.path.exists(MODEL_PATH):
        logger.info(f"Loading model from {MODEL_PATH}")
        checkpoint = torch.load(MODEL_PATH, map_location=device)
        if 'model_state_dict' in checkpoint:
            logger.info("Found model_state_dict in checkpoint")
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        else:
            logger.info("Loading checkpoint directly")
            model.load_state_dict(checkpoint, strict=False)
        logger.info("Model loaded successfully")
    else:
        logger.warning(f"Model checkpoint not found at {MODEL_PATH}")
        logger.info("Initializing model with random weights")
except Exception as e:
    logger.error(f"Failed to load model: {str(e)}", exc_info=True)
    logger.info("Initializing model with random weights")

model.eval()
logger.info('Model set to evaluation mode')

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

def preprocess_image(image_bytes):
    """Preprocess image for model input"""
    try:
        logger.debug(f"Preprocessing image, size: {len(image_bytes)} bytes")
        
        # Create a BytesIO object from the image bytes
        image_io = io.BytesIO(image_bytes)
        image_io.seek(0)  # Ensure we're at the start of the stream
        
        # Open and convert the image
        image = Image.open(image_io).convert('RGB')
        logger.debug(f"Image opened successfully, size: {image.size}")
        
        # Apply transformations
        image_tensor = transform(image)
        logger.debug(f"Transformed to tensor, shape: {image_tensor.shape}")
        
        return image_tensor.unsqueeze(0)  # Add batch dimension
    except Exception as e:
        logger.error(f"Error preprocessing image: {str(e)}", exc_info=True)
        raise

@app.route('/test', methods=['GET'])
def test():
    return jsonify({'status': 'ok', 'message': 'Backend is working'})

@app.route('/predict/<timestamp>/<random>', methods=['POST', 'OPTIONS'])
def predict(timestamp, random):
    logger.info('='*50)
    logger.info('Received prediction request')
    logger.info(f'URL: {request.url}')
    logger.info(f'Path: {request.path}')
    logger.info(f'Method: {request.method}')
    logger.info(f'Headers: {dict(request.headers)}')
    logger.info(f'Content-Type: {request.content_type}')
    logger.info(f'Form data: {request.form}')
    logger.info(f'Files: {request.files}')
    logger.info(f'Timestamp: {timestamp}')
    logger.info(f'Random: {random}')

    if request.method == 'OPTIONS':
        response = jsonify({'status': 'ok'})
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
        response.headers.add('Access-Control-Allow-Methods', 'POST, OPTIONS')
        return response

    try:
        if not request.files:
            logger.error('No files in request')
            return jsonify({'error': 'No files in request'}), 400
        
        if 'oct' not in request.files and 'fundus' not in request.files:
            logger.error('No images provided in request')
            return jsonify({'error': 'No images provided'}), 400
        
        # Create a dummy tensor with the same shape as expected by the model
        dummy_tensor = torch.zeros((1, 3, 224, 224)).to(device)
        oct_tensor = dummy_tensor.clone()
        fundus_tensor = dummy_tensor.clone()
        
        # For only OCT image 
        if 'oct' in request.files and request.files['oct'].filename:
            try:
                oct_file = request.files['oct']
                logger.info(f'Processing OCT image: {oct_file.filename}')
                oct_image = oct_file.read()
                oct_tensor = preprocess_image(oct_image).to(device)
            except Exception as e:
                logger.error(f'Error processing OCT image: {str(e)}', exc_info=True)
                return jsonify({'error': f'Error processing OCT image: {str(e)}'}), 400
        
        # For only Fundus
        if 'fundus' in request.files and request.files['fundus'].filename:
            try:
                fundus_file = request.files['fundus']
                logger.info(f'Processing fundus image: {fundus_file.filename}')
                fundus_image = fundus_file.read()
                fundus_tensor = preprocess_image(fundus_image).to(device)
            except Exception as e:
                logger.error(f'Error processing Fundus image: {str(e)}', exc_info=True)
                return jsonify({'error': f'Error processing Fundus image: {str(e)}'}), 400
        
        logger.info('Running model inference...')
        with torch.no_grad():
            try:
                #for both OCT and Fundus
                outputs = model(oct_images=oct_tensor, fundus_images=fundus_tensor)
                
                if isinstance(outputs, dict) and 'logits' in outputs:
                    logits = outputs['logits']
                    if logits is None or logits.size(0) == 0:
                        logger.error('Empty logits tensor received')
                        return jsonify({'error': 'Model produced empty output'}), 500
                    
                    probs = torch.softmax(logits, dim=1)
                    probs = probs[0].cpu().numpy()
                    max_idx = np.argmax(probs)
                    
                    result = {
                        'severity': severity_levels[max_idx],
                        'confidence': float(probs[max_idx]),
                        'severity_level': int(max_idx),
                        'probabilities': probs.tolist()
                    }
                    
                    logger.info(f'Final prediction result: {result}')
                    response = jsonify(result)
                    response.headers.add('Access-Control-Allow-Origin', '*')
                    response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
                    response.headers.add('Access-Control-Allow-Methods', 'POST, OPTIONS')
                    response.headers.add('Cache-Control', 'no-cache, no-store, must-revalidate')
                    response.headers.add('Pragma', 'no-cache')
                    response.headers.add('Expires', '0')
                    response.headers.add('Last-Modified', datetime.now().strftime('%a, %d %b %Y %H:%M:%S GMT'))
                    return response
                else:
                    logger.error('Invalid model output format')
                    return jsonify({'error': 'Invalid model output format'}), 500
            except Exception as e:
                logger.error(f'Error during model inference: {str(e)}', exc_info=True)
                return jsonify({'error': 'Error during model inference'}), 500
    except Exception as e:
        logger.error(f'Error processing request: {str(e)}', exc_info=True)
        return jsonify({'error': 'Error processing request'}), 500

@app.route('/health', methods=['GET'])
def health_check():
    logger.info('Health check requested')
    status = {
        'status': 'healthy',
        'model_loaded': os.path.exists(MODEL_PATH),
        'device': str(device)
    }
    logger.info(f'Health check response: {status}')
    return jsonify(status)

if __name__ == '__main__':
    logger.info('Starting Flask server on port 5002...')
    app.run(host='0.0.0.0', port=5002, debug=True) 