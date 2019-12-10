# Use an official Python runtime as a parent image
# FROM nvcr.io/nvidia/pytorch:19.10-py3
# FROM python:3.6-slim
#FROM pytorch/pytorch
FROM python:3.6-slim

# Set the working directory to /app
WORKDIR /app

# Copy requirements first to enable build caching
COPY requirements.txt /

# Install any needed packages specified in requirements.txt
RUN apt-get update && apt-get install -y libgtk2.0-dev
RUN pip install --no-cache-dir --trusted-host pypi.python.org -r /requirements.txt

# Copy the current directory contents into the container at /app
COPY . /app

# Make port 80 available to the world outside this container
EXPOSE 80

# Run app.py when the container launches
CMD ["python3", "app.py"]