import requests
import logging
import os
from PIL import Image
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_health_check():
    """Test the health check endpoint"""
    try:
        response = requests.get('http://localhost:5000/health')
        response.raise_for_status()
        data = response.json()
        logger.info(f"Health check response: {data}")
        return data['status'] == 'healthy'
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return False

def test_prediction():
    """Test the prediction endpoint with sample images"""
    try:
        # Create sample images (you should replace these with actual test images)
        oct_image = Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8))
        fundus_image = Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8))
        
        # Save temporary images
        oct_image.save('temp_oct.jpg')
        fundus_image.save('temp_fundus.jpg')
        
        # Prepare files for request
        files = {
            'oct': open('temp_oct.jpg', 'rb'),
            'fundus': open('temp_fundus.jpg', 'rb')
        }
        
        # Make request
        response = requests.post('http://localhost:5000/predict', files=files)
        response.raise_for_status()
        data = response.json()
        logger.info(f"Prediction response: {data}")
        
        # Clean up
        files['oct'].close()
        files['fundus'].close()
        os.remove('temp_oct.jpg')
        os.remove('temp_fundus.jpg')
        
        return 'severity' in data and 'confidence' in data
    except Exception as e:
        logger.error(f"Prediction test failed: {str(e)}")
        return False

if __name__ == '__main__':
    logger.info("Starting API tests...")
    
    # Test health check
    health_check_passed = test_health_check()
    logger.info(f"Health check test {'passed' if health_check_passed else 'failed'}")
    
    # Test prediction
    prediction_passed = test_prediction()
    logger.info(f"Prediction test {'passed' if prediction_passed else 'failed'}")
    
    # Summary
    all_tests_passed = health_check_passed and prediction_passed
    logger.info(f"All tests {'passed' if all_tests_passed else 'failed'}") 