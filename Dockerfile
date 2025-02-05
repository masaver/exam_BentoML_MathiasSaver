# Use an official Python base image
FROM python:3.8-slim

# Set environment variables
ENV BENTO_BUNDLE_PATH=/bento
WORKDIR $BENTO_BUNDLE_PATH

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install BentoML and other dependencies
RUN mkdir /project
COPY . /project
RUN pip install --no-cache-dir --upgrade pip \
    && pip install -r /project/requirements.txt

# Copy the Bento bundle into the container
COPY ./bento $BENTO_BUNDLE_PATH

# Compress the Bento bundle
RUN tar -czf /project/bento.tar.gz $BENTO_BUNDLE_PATH

# Add the model to the BentoML models store - `bentoml models import apparently expects a .tar or .tar.gz file`
RUN bentoml models import /project/bento.tar.gz

# Espose the 3000 port
EXPOSE 3000

# Set up BentoML entrypoint
CMD ["bentoml", "serve", "/bento"]
