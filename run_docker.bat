@echo off
echo Building Docker image...
docker build -t topic-classifier .

IF %ERRORLEVEL% NEQ 0 (
    echo Error building Docker image
    pause
    exit /b %ERRORLEVEL%
)

echo.
echo Starting Docker container...
echo The API will be available at http://localhost:8000
echo.
echo Available endpoints:
echo - Health check: http://localhost:8000/health
echo - Predictions: http://localhost:8000/predict
echo.
echo Press Ctrl+C to stop the container
echo.

docker run -p 8000:8000 topic-classifier