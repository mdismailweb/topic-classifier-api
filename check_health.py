import requests

def check_health():
    url = "https://topic-classifier-api-1.onrender.com/health"
    try:
        response = requests.get(url)
        print(f"Status Code: {response.status_code}")
        print("Response:")
        print(response.text)
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    check_health()