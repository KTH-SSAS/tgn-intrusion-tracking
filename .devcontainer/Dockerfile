# Use the official Python image as the base image
FROM python:3.10

# Set the working directory in the container
WORKDIR /workspace

RUN useradd -ms /bin/bash vscode

# Update and install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and install torch separately to avoid conflicts
RUN pip install --upgrade pip \
    && pip install torch==1.12.0+cpu -f https://download.pytorch.org/whl/torch_stable.html

# Install the rest of the dependencies
RUN pip install \
    torchvision==0.13.0+cpu \
    torchaudio==0.12.0+cpu \
    pyg_lib \
    torch-scatter \
    torch-sparse \
    torch-cluster \
    torch-spline-conv \
    torch-geometric==2.1.0 \
    torch-geometric-temporal==0.54.0 \
    "numpy<2.0" \
    -f https://download.pytorch.org/whl/torch_stable.html \
    -f https://data.pyg.org/whl/torch-1.12.0+cpu.html

# Specify a default command to prevent the container from exiting immediately
CMD ["bash"]
