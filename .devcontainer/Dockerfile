# See here for image contents: https://github.com/microsoft/vscode-dev-containers/tree/v0.238.0/containers/python-3/.devcontainer/base.Dockerfile

# [Choice] Python version (use -bullseye variants on local arm64/Apple Silicon): 3, 3.10, 3.9, 3.8, 3.7, 3.6, 3-bullseye, 3.10-bullseye, 3.9-bullseye, 3.8-bullseye, 3.7-bullseye, 3.6-bullseye, 3-buster, 3.10-buster, 3.9-buster, 3.8-buster, 3.7-buster, 3.6-buster
ARG VARIANT="3.10-bullseye"
FROM mcr.microsoft.com/vscode/devcontainers/python:0-${VARIANT}

# [Choice] Node.js version: none, lts/*, 16, 14, 12, 10
ARG NODE_VERSION="none"
RUN if [ "${NODE_VERSION}" != "none" ]; then su vscode -c "umask 0002 && . /usr/local/share/nvm/nvm.sh && nvm install ${NODE_VERSION} 2>&1"; fi

RUN su vscode -c "mkdir -p /home/vscode/.vscode-server/extensions"

# Cache dependencies.
COPY requirements.txt requirements.txt

# __HORRIBLE__ (but funny) hack to allow fetching the dependencies even if the
# requirements.txt file contains "-e ." TODO Find out why it is even needed.
RUN cat .\requirements.txt | python -c "from sys import stdin; import os; print(os.linesep.join(filter(lambda x: x[:2] != '-e', stdin.read().splitlines())))" > requirements.txt

RUN pip3 --disable-pip-version-check --no-cache-dir install -r requirements.txt

WORKDIR /app

ENV FLASK_APP "thingi"

COPY . .

# [Optional] Uncomment this section to install additional OS packages.
RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
    && apt-get -y install --no-install-recommends libgpiod2

# Move to correct directory for running the app (handles module paths imports).
WORKDIR /app/host_app
CMD ["python", "__main__"]

# [Optional] Uncomment this line to install global node packages.
# RUN su vscode -c "source /usr/local/share/nvm/nvm.sh && npm install -g <your-package-here>" 2>&1
