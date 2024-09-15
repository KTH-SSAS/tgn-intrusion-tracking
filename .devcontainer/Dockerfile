# Use the official Python image as the base image
FROM python:3.6

# Set the working directory in the container
WORKDIR /workspace

RUN useradd -m vscode

# Update and install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and install torch separately to avoid conflicts
RUN pip install --upgrade pip \
    && pip install pandas==1.1.0 torch==1.6.0 scikit_learn==0.23.1 \
    -f https://download.pytorch.org/whl/torch_stable.html \
    -f https://data.pyg.org/whl/torch-1.6.0+cpu.html

# Specify a default command to prevent the container from exiting immediately
CMD ["bash"]
