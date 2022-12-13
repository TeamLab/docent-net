FROM nvidia/cuda:11.7.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

SHELL ["/bin/bash", "-c"]
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    ccache \
    cmake \
    curl \
    git \
    libfreetype6-dev \
    libhdf5-serial-dev \
    libzmq3-dev \
    libjpeg-dev \
    libpng-dev \
    libsm6 \
    libxext6 \
    libxrender-dev \
    pkg-config \
    software-properties-common \
    ssh \
    sudo \
    unzip \
    vim \
    wget

RUN groupadd -g 999 eunbinpark
RUN useradd -r -u 999 -g eunbinpark eunbinpark

RUN rm -rf /var/lib/apt/lists/*

ENV LD_LIBRARY_PATH /usr/local/cuda-${CUDA}/targets/x86_64-linux/lib:/usr/local/cuda/extras/CUPTI/lib64:/usr/local/cuda/lib64:$LD_LIBRARY_PATH
RUN ln -s /usr/local/cuda/lib64/stubs/libcuda.so /usr/local/cuda/lib64/stubs/libcuda.so.1 && \
    echo "/usr/local/cuda/lib64/stubs" > /etc/ld.so.conf.d/z-cuda-stubs.conf && \
    ldconfig

# set timezone
ENV TZ=Asia/Seoul
RUN sudo ln -snf /usr/share/zoneinfo/$TZ /etc/localtime

ADD . /app
WORKDIR /app

# Install miniconda3
ENV LANG C.UTF-8
RUN curl -o /tmp/miniconda.sh -sSL http://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    chmod +x /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -bfp /usr/local && \
    rm /tmp/miniconda.sh && \
    conda init bash && \
    conda update -n base -c defaults conda


SHELL ["/bin/bash", "-c"]

ARG PYTHON_VERSION=3.8
ARG CONDA_ENV_NAME=

RUN conda env create -f /app/environment.yaml
ENV PATH /usr/local/envs/$CONDA_ENV_NAME/bin:$PATH


RUN echo "source activate ${CONDA_ENV_NAME}" >> ~/.bashrc
ENTRYPOINT ["sh", "run.sh"]