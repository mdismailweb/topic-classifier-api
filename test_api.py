import requests
import json

def test_api():
    # API endpoint (local)
    url = "http://localhost:8000/predict"
    
    # Test cases
    test_papers = [
        {
            "title": "Deep Learning Applications in Quantum Computing",
            "abstract": "This paper explores the intersection of deep learning and quantum computing, focusing on optimization algorithms and neural network architectures suitable for quantum systems.",
        },
        {
            "title": "Statistical Analysis of Financial Markets",
            "abstract": "A comprehensive study of statistical methods applied to financial market data, including time series analysis and stochastic processes.",
        },
        {
            "title": "Mathematical Models in Biology",
            "abstract": "This research presents novel mathematical models for understanding biological systems, combining differential equations with statistical inference.",
        }
    ]
    
    print("Testing Topic Classification API...")
    print("-" * 50)
    
    # Test health endpoint
    try:
        health_response = requests.get("http://localhost:8000/health")
        print(f"\nHealth Check Status: {health_response.json()['status']}")
        print("-" * 50)
    except Exception as e:
        print(f"Error checking health: {e}")
        return
    
    # Test predictions
    for i, paper in enumerate(test_papers, 1):
        print(f"\nTest Case {i}:")
        print(f"Title: {paper['title']}")
        print(f"Abstract: {paper['abstract'][:100]}...")
        
        try:
            # Make prediction request
            response = requests.post(url, json=paper)
            result = response.json()
            
            # Print results
            print("\nPredicted Topics:")
            for pred in result['predictions']:
                print(f"- {pred['topic']}: {pred['confidence']:.3f}")
            
            print("\nMetadata:")
            for key, value in result['metadata'].items():
                print(f"- {key}: {value}")
            
        except Exception as e:
            print(f"Error: {e}")
        
        print("-" * 50)

if __name__ == "__main__":
    test_api()