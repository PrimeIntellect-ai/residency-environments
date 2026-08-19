FROM carlasim/carla:0.9.16

USER root
WORKDIR /workspace

ENV DEBIAN_FRONTEND=noninteractive
ENV CARLA_ROOT=/opt/carla-0.9.16
ENV NVIDIA_DRIVER_CAPABILITIES=all
ENV UV_PYTHON_INSTALL_DIR=/opt/uv-python

RUN mkdir -p /var/lib/apt/lists/partial && \
    apt-get update -y && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    docker.io \
    ffmpeg \
    fuse-overlayfs \
    git \
    gnupg2 \
    iproute2 \
    jq \
    less \
    libglib2.0-0 \
    libgl1 \
    libvulkan1 \
    mesa-vulkan-drivers \
    net-tools \
    openssh-server \
    procps \
    python3 \
    python3-dev \
    python3-pip \
    python3-venv \
    rsync \
    sudo \
    unzip \
    vim-tiny \
    vulkan-tools \
    wget && \
    rm -rf /var/lib/apt/lists/*

RUN curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
      gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg && \
    curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
      sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
      >/etc/apt/sources.list.d/nvidia-container-toolkit.list && \
    apt-get update -y && \
    apt-get install -y --no-install-recommends \
      nvidia-container-toolkit \
      nvidia-container-toolkit-base \
      libnvidia-container-tools \
      libnvidia-container1 && \
    nvidia-ctk runtime configure --runtime=docker && \
    rm -rf /var/lib/apt/lists/*

RUN curl -LsSf https://astral.sh/uv/install.sh | sh && \
    install -m 0755 /root/.local/bin/uv /usr/local/bin/uv && \
    install -m 0755 /root/.local/bin/uvx /usr/local/bin/uvx && \
    uv python install 3.10 3.12 && \
    ln -sf "$(uv python find 3.10)" /usr/local/bin/python3.10 && \
    ln -sf "$(uv python find 3.12)" /usr/local/bin/python3.12

RUN mkdir -p /var/run/sshd /root/.ssh /opt && \
    useradd -m -s /bin/bash ubuntu && \
    usermod -aG docker ubuntu && \
    echo 'ubuntu ALL=(ALL) NOPASSWD:ALL' >/etc/sudoers.d/ubuntu && \
    chmod 0440 /etc/sudoers.d/ubuntu

RUN test -x /workspace/CarlaUE4.sh && \
    ln -sfn /workspace "${CARLA_ROOT}" && \
    ln -sfn "${CARLA_ROOT}" /opt/carla

COPY docker/prime-template/entrypoint.sh /usr/local/bin/prime-template-entrypoint.sh
RUN chmod 0755 /usr/local/bin/prime-template-entrypoint.sh

EXPOSE 22 2000 2001 2002 8080 46435

ENTRYPOINT ["/usr/local/bin/prime-template-entrypoint.sh"]
