FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

LABEL maintainer="chrcarm@amazon.com"

# Avoid warnings by switching to noninteractive
ENV DEBIAN_FRONTEND=noninteractive

# Update packages
RUN apt-get update
RUN apt-get upgrade -y
# RUN apt-get install -y nvidia-cuda-toolkit

# Install some libraries
RUN apt-get install -y git wget unzip curl tree
RUN apt-get install -y awscli
RUN apt-get install -y zsh

# Install python3
RUN apt-get install -y python3-venv
RUN apt-get install -y python3-pip
RUN ln -sf /usr/bin/python3 /usr/bin/python
RUN ln -sf /usr/bin/pip3 /usr/bin/pip

# Upgrade pip
RUN pip --no-cache-dir install -U pip
# Install useful python modules
RUN pip install -U wheel setuptools pylint yapf

# For Sagemaker
RUN pip install -U sagemaker-training
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install Jax with CUDA support
RUN ln -s /usr/local/cuda /usr/local/cuda-12.1
ENV LD_LIBRARY_PATH "${LD_LIBRARY_PATH}:/usr/local/cuda/lib64"
ENV PATH "${PATH}:/usr/local/cuda/bin"
RUN pip --no-cache-dir install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# Add a non-root user
# https://code.visualstudio.com/remote/advancedcontainers/add-nonroot-user
ARG USERNAME=ubuntu
ARG USER_UID=1000
ARG USER_GID=$USER_UID
# Create the user
RUN groupadd --gid $USER_GID $USERNAME \
  && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME -s /usr/bin/zsh \
  #
  # [Optional] Add sudo support. Omit if you don't need to install software after connecting.
  && apt-get update \
  && apt-get install -y sudo \
  && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
  && chmod 0440 /etc/sudoers.d/$USERNAME
USER ubuntu
RUN wget https://github.com/robbyrussell/oh-my-zsh/raw/master/tools/install.sh -O - | zsh || true
ENV PATH "${PATH}:/home/ubuntu/.local/bin"
USER root

CMD ["/bin/bash"]
