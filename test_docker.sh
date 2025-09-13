#!/bin/bash
echo "Testing Docker API deployment..."
echo "================================"

# Test health endpoint
echo "Testing health endpoint..."
curl http://localhost:8000/health
echo -e "\n"

# Test prediction endpoint
echo "Testing prediction endpoint..."
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "title": "Deep Learning in Quantum Computing",
    "abstract": "This paper explores the applications of deep learning algorithms in quantum computing systems, focusing on optimization and state prediction."
  }'