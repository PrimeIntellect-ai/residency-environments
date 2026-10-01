FROM python:3.12-slim@sha256:09f7da3bc104798d0afb40bc08d23ab2da20a76130cec1f2ef170848f5d85217

COPY workspaces/ /opt/swg/workspaces/
COPY task_ids.json /opt/swg/task_ids.json

RUN find /opt/swg/workspaces -type f -name '*.sh' -exec chmod +x {} +

LABEL org.opencontainers.image.title="Synthetic Workspace Gym agent split"
