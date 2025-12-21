#!/bin/bash

# Set the Docker image name and tag
IMAGE_NAME="saliency-service"
TAG="latest"
REGISTRY="gpu-harbor.act.buaa.edu.cn/user-xuedh25/"
FULL_IMAGE_NAME="$REGISTRY$IMAGE_NAME:$TAG"

# Build the Docker image for the AI model function-service
DOCKER_BUILDKIT=1 docker build --platform linux/amd64 -t $IMAGE_NAME .

# Tag the Docker image
docker tag $IMAGE_NAME $FULL_IMAGE_NAME

# Push the Docker image to the remote Docker registry
docker push $FULL_IMAGE_NAME

# Deploy the service on the Kubernetes cluster
kubectl apply -f deployment.yaml