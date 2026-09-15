FROM physim-predictor:0.12.0
USER root
RUN pip install --no-cache-dir uv==0.12.8 && \
    mkdir -p /workspace /observations
COPY harness-deps.py /tmp/harness-deps.py
RUN uv sync --script /tmp/harness-deps.py --no-config && \
    UV_OFFLINE=true uv sync --script /tmp/harness-deps.py --no-config && \
    rm /tmp/harness-deps.py
ENV PYTHONPATH=/usr/local/lib/python3.11/site-packages UV_OFFLINE=true UV_PYTHON_DOWNLOADS=never PIP_NO_INDEX=1
WORKDIR /workspace
