FROM so77id/tensorflow-opencv-gpu-py3

MAINTAINER Stephen Pace <paceste1@gmail.com>

ENV DISPLAY=':0.0'

# Install necessary packages
RUN apt-get update
    apt-get install vim \
                    net-tools \
                    iputils-ping \

WORKDIR /home

ADD darkflow/
