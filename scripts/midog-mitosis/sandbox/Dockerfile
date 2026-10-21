# The sandbox every midog-mitosis task runs in: Python with Pillow and NumPy preinstalled, so a rollout installs nothing
# and needs no network. curl and CA certificates are what the verifiers harnesses use to fetch their own runtime.
# Build and publish: prime images push midog-mitosis-sandbox:<version> --context scripts/midog-mitosis/sandbox --public
FROM python:3.11-slim@sha256:922f47525757de33aff59f24cdfc85f412ac4a06aa8af7c7e9028d584b7bcdeb
RUN apt-get update -qq && apt-get install -y -qq --no-install-recommends curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir pillow==12.3.0 numpy==2.4.6
