# See here for image contents: https://github.com/microsoft/vscode-dev-containers/tree/v0.238.0/containers/python-3/.devcontainer/base.Dockerfile

# [Choice] Python version (use -bullseye variants on local arm64/Apple Silicon): 3, 3.10, 3.9, 3.8, 3.7, 3.6, 3-bullseye, 3.10-bullseye, 3.9-bullseye, 3.8-bullseye, 3.7-bullseye, 3.6-bullseye, 3-buster, 3.10-buster, 3.9-buster, 3.8-buster, 3.7-buster, 3.6-buster
ARG VARIANT="3.11-bullseye"
FROM mcr.microsoft.com/vscode/devcontainers/python:0-${VARIANT} AS base

LABEL org.opencontainers.image.source="https://github.com/LiquidAI-project/wasmiot-supervisor" 

WORKDIR /app

# Copy needed files to app root for installing deps beforehand.
COPY ["requirements.txt", "setup.py", "./"]

RUN pip3 --disable-pip-version-check --no-cache-dir install -r requirements.txt

# [Optional] Uncomment this section to install additional OS packages.
RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
    && apt-get -y install --no-install-recommends libgpiod2 \
    && rm -rf /var/lib/apt/lists/*

# Copy the rest.
COPY . .

FROM base AS run

ARG DEVICE_NAME
# NOTE: Uncomment this in order to fail if the ARG is not used. From
# https://stackoverflow.com/questions/38438933/how-to-make-a-build-arg-mandatory-during-docker-build
#RUN test -n "$DEVICE_NAME"

ENV FLASK_APP ${DEVICE_NAME}

# Move to correct directory for running the app (handles module paths imports).
WORKDIR /app/host_app
CMD ["python", "__main__.py"]

FROM base AS vscode-devcontainer

RUN su vscode -c "mkdir -p /home/vscode/.vscode-server/extensions"
