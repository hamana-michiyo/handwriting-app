# syntax=docker/dockerfile:1
FROM python:3.11-slim

# Avoid warnings by switching to non-interactive
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies for OpenCV and Tesseract
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    libgl1-mesa-glx \
    libgomp1 \
    tesseract-ocr \
    tesseract-ocr-jpn \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /workspace

# Copy requirements and install Python dependencies
COPY requirements_api.txt /workspace/requirements_api.txt
RUN pip install --no-cache-dir -r requirements_api.txt

# Copy the rest of the workspace
COPY . /workspace

# Set environment variables
ENV PYTHONPATH=/workspace
ENV TESSDATA_PREFIX=/usr/share/tesseract-ocr/4.00/tessdata/
ENV TESSERACT_CMD=/usr/bin/tesseract

# Default to bash
CMD ["bash"]