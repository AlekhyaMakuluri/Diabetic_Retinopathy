import requests
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_connection():
    """Test the connection to the backend server"""
    try:
        # Test health check endpoint
        logger.info("Testing health check endpoint...")
        response = requests.get('http://localhost:5001/health')
        logger.info(f"Health check response status: {response.status_code}")
        logger.info(f"Health check response: {response.json()}")
        
        # Test prediction endpoint with dummy data
        logger.info("\nTesting prediction endpoint...")
        files = {
            'oct': ('test_oct.jpg', b'dummy_oct_data', 'image/jpeg'),
            'fundus': ('test_fundus.jpg', b'dummy_fundus_data', 'image/jpeg')
        }
        response = requests.post('http://localhost:5001/predict', files=files)
        logger.info(f"Prediction response status: {response.status_code}")
        try:
            logger.info(f"Prediction response: {response.json()}")
        except:
            logger.error(f"Error parsing response: {response.text}")
            
    except Exception as e:
        logger.error(f"Connection test failed: {str(e)}")

if __name__ == '__main__':
    test_connection() 