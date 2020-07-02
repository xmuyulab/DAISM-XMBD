FROM pytorch/pytorch:1.5.1-cuda10.1-cudnn7-devel

ADD ./requirement.install.sh /opt/
RUN cd /opt/ && bash requirement.install.sh
