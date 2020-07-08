FROM pytorch/pytorch:1.5.1-cuda10.1-cudnn7-devel

ADD ./requirement.install.sh /opt/
RUN cd /opt/ && bash requirement.install.sh
ADD ./daism_dnn/daism_dnn.py /workspace/daism_dnn/
ADD ./daism_dnn/__init__.py /workspace/daism_dnn/
ADD ./daism_dnn/modules/ /workspace/daism_dnn/modules/
