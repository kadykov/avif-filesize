# Use a Microsoft base image with common dev tools pre-installed
FROM mcr.microsoft.com/devcontainers/base:ubuntu-24.04

# Install system dependencies using apt-get
USER root

RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
    && apt-get install -y --no-install-recommends \
    git \
    file \
    shellcheck \
    python3 \
    python3-pip \
    pipx \
    pre-commit \
    nodejs \
    npm \
    build-essential \
    wget \
    ca-certificates \
    unzip \
    just \
    libavif-bin \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Switch back to vscode user provided by base image
USER vscode

RUN mkdir -p /home/vscode/.vscode-server/data/User/globalStorage
