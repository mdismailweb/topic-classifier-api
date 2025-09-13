import requests
import json

def test_api():
    # Your deployed API URL
    API_URL = "https://topic-classifier-api-1.onrender.com"
    
    # Test case 1: Computer Science and Mathematics paper
    test_case_1 = {
        "title": "Deep Neural Networks for Quantum Circuit Optimization",
        "abstract": "This paper presents a novel approach to quantum circuit optimization using deep learning techniques. We demonstrate significant improvements in circuit depth and gate count using our neural network-based approach. The mathematical foundations of quantum computing are explored alongside practical implementations.",
        "confidence_threshold": 0.3
    }
    
    # Test case 2: Physics and Quantitative Finance paper
    test_case_2 = {
        "title": "Statistical Physics Models in Financial Market Analysis",
        "abstract": "We apply concepts from statistical physics to analyze financial market dynamics. Using entropy-based measures and phase transition analogies, we develop models for market crash prediction. The study combines theoretical physics with practical financial applications.",
        "confidence_threshold": 0.3
    }
    
    print("Testing Topic Classification API")
    print("-" * 50)
    
    # Test health endpoint
    print("\n1. Testing Health Endpoint...")
    try:
        response = requests.get(f"{API_URL}/health")
        print("Response:", json.dumps(response.json(), indent=2))
    except Exception as e:
        print(f"Error: {e}")
    
    # Test predictions
    print("\n2. Testing Prediction Endpoint...")
    
    print("\nTest Case 1 (Computer Science/Mathematics):")
    try:
        response = requests.post(f"{API_URL}/predict", json=test_case_1)
        print("Response:", json.dumps(response.json(), indent=2))
    except Exception as e:
        print(f"Error: {e}")
    
    print("\nTest Case 2 (Physics/Finance):")
    try:
        response = requests.post(f"{API_URL}/predict", json=test_case_2)
        print("Response:", json.dumps(response.json(), indent=2))
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_api()