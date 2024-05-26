# Use an official NVIDIA runtime as a parent image
FROM nvidia/cuda:12.3-base
FROM python:3.10-slim
# Set the working directory in the container
WORKDIR /app

# Install Python and other necessary system utilities
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    build-essential

# Copy the current directory contents into the container at /app
COPY . /app 

# Install any needed packages specified in requirements.txt
RUN pip3 install -r requirements.txt
# RUN pip install transformers bitsandbytes
#RUN pip3 install --upgrade transformers huggingface-hub


# Make port 5000 available to the world outside this container
EXPOSE 5000

# Define environment variable
ENV FLASK_APP=server.py
ENV FLASK_RUN_HOST=0.0.0.0

# Run server.py when the container launches
CMD ["flask", "run"]
