FROM carlasim/carla:0.10.0@sha256:b3ca7bfea1c58ba597b552887fc05209af8c69f00e20f208bc13ec6f5eaaf1fe

USER root

ENV DEBIAN_FRONTEND=noninteractive
ENV CARLA_ROOT=/home/carla
ENV NVIDIA_DRIVER_CAPABILITIES=compute,graphics,utility
ENV UV_PYTHON_INSTALL_DIR=/opt/uv-python
ENV UV_PYTHON=3.12

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

RUN curl -LsSf https://astral.sh/uv/0.8.15/install.sh | sh \
    && install -m 0755 /root/.local/bin/uv /usr/local/bin/uv \
    && install -m 0755 /root/.local/bin/uvx /usr/local/bin/uvx \
    && uv python install 3.12

COPY . /opt/carla-env-src

RUN uv venv --python 3.12 /opt/carla-env \
    && uv pip install --python /opt/carla-env/bin/python '/opt/carla-env-src[runtime]' \
    && rm -rf /opt/carla-env-src

ENV PATH=/opt/carla-env/bin:${PATH}

USER carla
WORKDIR /home/carla
