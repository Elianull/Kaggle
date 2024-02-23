#MSYS_NO_PATHCONV=1 docker run --gpus all -v ${PWD}:/usr/src/app -v ${HOME}/.cache:/root/.cache sfr-embed python embed/embed.py data/train.csv
# Use an official NVIDIA CUDA image with CUDA 12.3 and cuDNN as a parent image
FROM nvidia/cuda:12.3.1-runtime-ubuntu20.04

# Set the working directory in the container
WORKDIR /usr/src/app

# Install Python and pip
RUN apt-get update && \
    apt-get install -y python3-pip python3-dev && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip3 install --upgrade pip

# Copy the requirements file into the container
COPY requirements.txt ./

# Install any needed packages specified in requirements.txt
RUN pip3 install --no-cache-dir -r requirements.txt