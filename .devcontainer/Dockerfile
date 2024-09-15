# Base image with Python
FROM python:3.10

# Set working directory
WORKDIR /workspace

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies from requirements.txt
COPY requirements.txt .
RUN pip install --upgrade pip \
    && pip install -r requirements.txt \
    -f https://download.pytorch.org/whl/torch_stable.html \
    -f https://data.pyg.org/whl/torch-1.12.0+cpu.html

# Set default shell to bash
SHELL ["/bin/bash", "-c"]

# Copy the rest of the application code to the container
COPY . /workspace

# Optionally specify the entry point or any other post-build commands if needed
# CMD ["python", "your_script.py"]
