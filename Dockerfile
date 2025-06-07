# 베이스 이미지 설정
FROM ubuntu:22.04

# 기본 패키지 설치
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-venv \
    python3.10-dev \
    python3-pip \
    git \
    vim \
    wget \
    curl \
    unzip \
    sudo \
    lsb-release \
    build-essential && \
    rm -rf /var/lib/apt/lists/*

# pip 업그레이드
RUN python3.10 -m pip install --upgrade pip

# Python 패키지 설치 (OpenVINO 및 관련 툴킬)
RUN pip install \
    openvino==2025.1 \
    openvino-genai==2025.1 \
    openvino-tokenizers==2025.1 \
    nncf==2.14.1 \
    optimum-intel==1.22.0 \
    transformers \
    accelerate \
    numpy \
    onnx==1.17.0 \
    torch \
    gradio

# CMake 3.23.0 설치
RUN wget https://github.com/Kitware/CMake/releases/download/v3.23.0/cmake-3.23.0-linux-x86_64.sh && \
    chmod +x cmake-3.23.0-linux-x86_64.sh && \
    ./cmake-3.23.0-linux-x86_64.sh --skip-license --prefix=/usr/local && \
    rm cmake-3.23.0-linux-x86_64.sh
