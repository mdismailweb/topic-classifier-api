import requests
import json

def test_deployed_api(api_url):
    """Test the deployed API with sample papers"""
    print(f"Testing API at: {api_url}")
    print("-" * 50)
    
    # Test health endpoint
    health_url = f"{api_url}/health"
    print("\nTesting health endpoint...")
    try:
        response = requests.get(health_url)
        print(f"Health Status: {response.json()['status']}")
    except Exception as e:
        print(f"Error checking health: {e}")
        return
    
    # Test prediction endpoint
    predict_url = f"{api_url}/predict"
    
    test_papers = [
        {
            "title": "Deep Learning in Quantum Computing",
            "abstract": "This paper explores the applications of deep learning algorithms in quantum computing systems."
        },
        {
            "title": "Statistical Analysis of Financial Markets",
            "abstract": "A study of statistical methods applied to financial market data."
        }
    ]
    
    print("\nTesting predictions...")
    for i, paper in enumerate(test_papers, 1):
        print(f"\nTest Paper {i}:")
        print(f"Title: {paper['title']}")
        print("Predictions:")
        
        try:
            response = requests.post(predict_url, json=paper)
            result = response.json()
            
            if result.get('success'):
                for pred in result['predictions']:
                    print(f"- {pred['topic']}: {pred['confidence']:.3f}")
            else:
                print(f"Error: {result.get('error', 'Unknown error')}")
                
        except Exception as e:
            print(f"Error: {e}")
        
        print("-" * 30)

if __name__ == "__main__":
    # Replace this with your Render deployed URL
    api_url = input("Enter your Render API URL: ")
    # Remove trailing slash if present
    api_url = api_url.rstrip('/')
    test_deployed_api(api_url)