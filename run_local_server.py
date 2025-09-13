"""
Topic Classification API Service
Run this script to start the API server locally.
"""
from app import app
import webbrowser
import threading
import time

def open_browser():
    """Open browser after a short delay"""
    time.sleep(1.5)
    webbrowser.open('http://127.0.0.1:5000/docs')

def main():
    print("Starting Topic Classification API...")
    print("=" * 50)
    print("API Documentation will open in your browser")
    print("Available endpoints:")
    print("- Health check: http://127.0.0.1:5000/health")
    print("- Predictions: http://127.0.0.1:5000/predict")
    print("\nPress Ctrl+C to stop the server")
    print("=" * 50)
    
    # Open browser in a separate thread
    threading.Thread(target=open_browser).start()
    
    # Run the Flask app
    app.run(host='127.0.0.1', port=5000)

if __name__ == '__main__':
    main()