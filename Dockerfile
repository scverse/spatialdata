ARG TARGETPLATFORM=linux/amd64

FROM --platform=$TARGETPLATFORM ubuntu:latest
LABEL authors="Luca Marconato"

ENV PYTHONUNBUFFERED=1

# Update and install system dependencies.
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    python3-venv \
    python3-dev \
    git \
    && rm -rf /var/lib/apt/lists/*
#
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --upgrade pip wheel
RUN pip install --no-cache-dir \
    spatialdata[torch] \
    spatialdata-io \
    spatialdata-plot
