# FROM pytorchlightning/pytorch_lightning:base-cuda-py3.10-torch2.0-cuda11.7.1
FROM python:3.10.1-buster 

## Install your dependencies here using apt install, etc.
RUN git clone https://github.com/isears/mvts_transformer
RUN pip install --editable ./mvts_transformer/

## DO NOT EDIT these 3 lines.
RUN mkdir /challenge
COPY ./ /challenge
WORKDIR /challenge

## Include the following line if you have a requirements.txt file.
RUN pip install -r requirements.txt
RUN pip install ./