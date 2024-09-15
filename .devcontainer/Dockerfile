# Base image with Python
FROM python:3.10

# Set working directory
WORKDIR /workspace

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --upgrade pip \
    && pip install torch==1.12.0+cpu \
    torchvision==0.13.0+cpu \
    torchaudio==0.12.0+cpu \
    pyg_lib \
    torch-scatter \
    torch-sparse \
    torch-cluster \
    torch-spline-conv \
    torch-geometric==2.1.0 \
    torch-geometric-temporal==0.54.0 \
    numpy<2.0 \
    -f https://download.pytorch.org/whl/torch_stable.html \
    -f https://data.pyg.org/whl/torch-1.12.0+cpu.html

# Set default shell to bash
SHELL ["/bin/bash", "-c"]
