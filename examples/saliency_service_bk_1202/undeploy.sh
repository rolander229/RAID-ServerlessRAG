#!/bin/bash

# Set the Docker image name and tag
IMAGE_NAME="saliency-service"
TAG="latest"
REGISTRY="gpu-harbor.act.buaa.edu.cn/user-xuedh25/"
FULL_IMAGE_NAME="$REGISTRY$IMAGE_NAME:$TAG"

# Undeploy the service from the Kubernetes cluster
kubectl delete -f deployment.yaml

# Delete the built Docker images from the local machine
docker rmi $IMAGE_NAME $FULL_IMAGE_NAME

# Delete the Docker image from the remote Docker registry
docker push $REGISTRY$IMAGE_NAME:deleted-$TAG