FROM tensorflow/tensorflow:latest-py3 

COPY build/requirements.txt ./

RUN pip install --no-cache-dir -r requirements.txt; \
    rm -r requirements.txt

COPY build /usr/src/app/build
COPY notebooks /usr/src/app/notebooks

COPY notebooks/jupyter_notebook_config.py /root/.jupyter/

RUN mkdir data

WORKDIR /usr/src/app

# Jupyter and Tensorboard ports
EXPOSE 9999 6006
