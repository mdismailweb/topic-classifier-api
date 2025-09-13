import requests
import json

# API endpoint
API_URL = "https://topic-classifier-api-1.onrender.com"

def test_health():
    """Test the health check endpoint"""
    response = requests.get(f"{API_URL}/health")
    print("\nHealth Check Response:")
    print(json.dumps(response.json(), indent=2))

def test_prediction():
    """Test the prediction endpoint with a sample paper"""
    # Sample paper
    data = {
        "title": "Deep Learning Applications in Quantum Computing",
        "abstract": "This paper explores the intersection of deep learning and quantum computing, focusing on neural network architectures for quantum state optimization and quantum circuit design. We demonstrate several applications in quantum error correction and quantum algorithm optimization.",
        "confidence_threshold": 0.3
    }

    # Make prediction request
    print("\nSending prediction request...")
    response = requests.post(f"{API_URL}/predict", json=data)
    
    print("\nPrediction Response:")
    print(json.dumps(response.json(), indent=2))

if __name__ == "__main__":
    print("Testing Topic Classification API at:", API_URL)
    print("-" * 50)
    
    # Test health endpoint
    test_health()
    
    # Test prediction endpoint
    test_prediction()