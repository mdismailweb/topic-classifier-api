# Use Python 3.9 slim image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy requirements and model files
COPY requirements.txt .
COPY app.py .
COPY model_artifacts.joblib .

# Install dependencies
RUN pip install -r requirements.txt

# Add gunicorn
RUN pip install gunicorn

# Expose port
EXPOSE 8000

# Run gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "app:app"]