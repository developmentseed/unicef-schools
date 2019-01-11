FROM geoyi/pixel_decoder_notebook

MAINTAINER nana@developmentseed.org

RUN apt-get update && apt-get install -y \
        wget \
        vim \
        bzip2

#Install ANACONDA Environment
RUN conda create -y -n sat_jupyter python=3.6 anaconda && \
         /opt/conda/envs/sat_jupyter/bin/pip install tensorflow \
         tensorflow-gpu \
         hyperopt \
         pygeotile \
         keras
