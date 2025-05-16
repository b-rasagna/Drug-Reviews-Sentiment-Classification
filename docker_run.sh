#!/bin/bash

# Variables
IMAGE_NAME="drug-review-app"
CONTAINER_NAME="drug-review-container"

# Step 1: Build the Docker image
echo "Building Docker image..."
docker build -t $IMAGE_NAME .

# Step 2: Stop and remove existing container if running
echo "Cleaning up old containers..."
docker stop $CONTAINER_NAME 2>/dev/null
docker rm $CONTAINER_NAME 2>/dev/null

# Step 3: Run the container
echo "Running Docker container..."
docker run --env-file .env \
           -p 8000:8000 \
           -p 8501:8501 \
           --name $CONTAINER_NAME \
           $IMAGE_NAME
