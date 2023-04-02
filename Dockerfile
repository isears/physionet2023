FROM pytorchlightning/pytorch_lightning:base-cuda-py3.9-torch1.12-cuda11.6.1
# FROM python:3.10.1-buster

## Install your dependencies here using apt install, etc.
RUN git clone https://github.com/isears/mvts_transformer
RUN pip install --editable ./mvts_transformer/

# Do this earlier so that hopefully docker will cache it
COPY ./requirements.txt ./requirements.txt
RUN pip install -r requirements.txt

## DO NOT EDIT these 3 lines.
RUN mkdir /challenge
COPY ./ /challenge
WORKDIR /challenge

RUN pip install ./