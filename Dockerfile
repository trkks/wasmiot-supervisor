# See here for image contents: https://github.com/microsoft/vscode-dev-containers/tree/v0.238.0/containers/python-3/.devcontainer/base.Dockerfile

# [Choice] Python version (use -bullseye variants on local arm64/Apple Silicon): 3, 3.10, 3.9, 3.8, 3.7, 3.6, 3-bullseye, 3.10-bullseye, 3.9-bullseye, 3.8-bullseye, 3.7-bullseye, 3.6-bullseye, 3-buster, 3.10-buster, 3.9-buster, 3.8-buster, 3.7-buster, 3.6-buster
ARG VARIANT="3.11-bullseye"
FROM mcr.microsoft.com/vscode/devcontainers/python:0-${VARIANT} AS base

LABEL org.opencontainers.image.source="https://github.com/LiquidAI-project/wasmiot-supervisor"

WORKDIR /app

ENV PYTHONUNBUFFERED 1

# [Optional] Uncomment this section to install additional OS packages.
RUN --mount=type=cache,target=/var/cache/apt \
    apt-get update && export DEBIAN_FRONTEND=noninteractive \
    && apt-get -y install --no-install-recommends libgpiod2

## Install requirements using pip. This is done before copying the app, so that
## requirements layer is cached. This way, if app code changes, only app code is
## copied, and requirements are not re-installed.
COPY requirements.txt /tmp/pip-tmp/
RUN --mount=type=cache,target=/root/.cache/pip \
    pip --disable-pip-version-check install -r /tmp/pip-tmp/requirements.txt && \
    rm -rf /tmp/pip-tmp

## Install self as editable (`-e`) module. In a long run it we should remove `COPY`
## and only install app as a package.
COPY . .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip --disable-pip-version-check install -v -e .

CMD ["python", "-m", "host_app"]

FROM base AS run

ARG DEVICE_NAME
# NOTE: Uncomment this in order to fail if the ARG is not used. From
# https://stackoverflow.com/questions/38438933/how-to-make-a-build-arg-mandatory-during-docker-build
#RUN test -n "$DEVICE_NAME"

ENV FLASK_APP ${DEVICE_NAME}

FROM base AS vscode-devcontainer

ARG SENTRY_ENVIRONMENT=devcontainer
ENV SENTRY_ENVIRONMENT=${SENTRY_ENVIRONMENT}

ENV PYTHONDONTWRITEBYTECODE 1

RUN  --mount=type=cache,target=/root/.cache/pip \
    pip --disable-pip-version-check install -v -e .[dev]

RUN su vscode -c "mkdir -p /home/vscode/.vscode-server/extensions"
