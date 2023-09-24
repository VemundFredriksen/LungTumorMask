# read the doc: https://huggingface.co/docs/hub/spaces-sdks-docker
# you will also find guides on how best to write your Dockerfile
#FROM pytorch/pytorch:1.10.0-cuda11.3-cudnn8-runtime
FROM python:3.8.10-slim

WORKDIR /code

RUN apt-get update -y
#RUN apt-get install -y python3 python3-pip
RUN apt install git --fix-missing -y
RUN apt install wget -y

# install dependencies
COPY ./demo/requirements.txt /code/demo/requirements.txt
RUN python -m pip install --no-cache-dir --upgrade -r /code/demo/requirements.txt

# Set up a new user named "user" with user ID 1000
RUN useradd -m -u 1000 user

# Switch to the "user" user
USER user

# Set home to the user's home directory
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

# Set the working directory to the user's home directory
WORKDIR $HOME/app

# Copy the current directory contents into the container at $HOME/app setting the owner to the user
COPY --chown=user . $HOME/app

# Download pretrained parenchyma model
#RUN wget "https://github.com/andreped/livermask/releases/download/trained-models-v1/model.h5"

# Download test sample
RUN wget https://github.com/VemundFredriksen/LungTumorMask/releases/download/0.0.1/lung_001.nii.gz

CMD ["python", "demo/app.py"]