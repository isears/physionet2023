FROM pytorchlightning/pytorch_lightning:base-cuda-py3.9-torch1.12-cuda11.6.1

## Install your dependencies here using apt install, etc.
RUN git clone https://github.com/isears/mvts_transformer
RUN pip install --editable ./mvts_transformer/

## DO NOT EDIT these 3 lines.
RUN mkdir /challenge
COPY ./ /challenge
WORKDIR /challenge

# Include the following line if you have a requirements.txt file.
RUN pip install -r requirements.txt
RUN pip install ./