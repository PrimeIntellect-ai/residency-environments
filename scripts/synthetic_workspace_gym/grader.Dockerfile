FROM python:3.12-slim@sha256:09f7da3bc104798d0afb40bc08d23ab2da20a76130cec1f2ef170848f5d85217

RUN mkdir -p /workspace /opt/swg-grade
WORKDIR /workspace

LABEL org.opencontainers.image.title="Synthetic Workspace Gym trusted grader"
