# Research Paper Topic Classification API

This API provides topic classification for research papers based on their titles and abstracts. It can predict multiple topics from six categories: Computer Science, Physics, Mathematics, Statistics, Quantitative Biology, and Quantitative Finance.

## API Endpoints

### Health Check
```
GET /health
```
Checks if the service is running and the model is loaded properly.

### Predict Topics
```
POST /predict
```

Request Body:
```json
{
    "title": "string",
    "abstract": "string",
    "confidence_threshold": float (optional, default=0.3)
}
```

Example Response:
```json
{
    "success": true,
    "predictions": [
        {
            "topic": "Computer Science",
            "confidence": 0.92
        },
        {
            "topic": "Mathematics",
            "confidence": 0.75
        }
    ],
    "metadata": {
        "input_length": 500,
        "num_predictions": 2,
        "avg_confidence": 0.835
    }
}
```

## Local Development

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the Flask app:
```bash
python app.py
```

## Docker Deployment

1. Build the Docker image:
```bash
docker build -t topic-classifier .
```

2. Run the container:
```bash
docker run -p 8000:8000 topic-classifier
```

## Heroku Deployment

1. Login to Heroku:
```bash
heroku login
```

2. Create a new Heroku app:
```bash
heroku create your-app-name
```

3. Push to Heroku:
```bash
git push heroku main
```

## Example Usage

```python
import requests
import json

# API endpoint
url = "https://your-app-name.herokuapp.com/predict"

# Example paper
data = {
    "title": "Deep Learning Applications in Quantum Computing",
    "abstract": "This paper explores the intersection of deep learning and quantum computing...",
    "confidence_threshold": 0.3
}

# Make prediction
response = requests.post(url, json=data)
predictions = response.json()

print(json.dumps(predictions, indent=2))
```