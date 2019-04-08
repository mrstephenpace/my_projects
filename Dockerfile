FROM so77id/tensorflow-opencv-gpu-py3

MAINTAINER Stephen Pace <paceste1@gmail.com>

ENV DISPLAY=':0.0'

# Install necessary packages
RUN apt-get -y update
RUN apt-get -y install vim net-tools iputils-ping 

WORKDIR /home

# Copy darkflow files from repository
RUN git clone https://github.com/mrstephenpace/my_projects.git
RUN cp -r my_projects/darkflow .
RUN rm -r my_projects


# Copy yolo weights 
RUN cd darkflow/bin

ADD bin/yolo.weights darkflow/bin/

CMD ["/bin/bash"]
