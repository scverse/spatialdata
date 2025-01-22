ARG TARGETPLATFORM=linux/amd64

# Use the specified platform to pull the correct base image.
# Override TARGETPLATFORM during build for different architectures, such as linux/arm64 for Apple Silicon.
# For example, to build for ARM64 architecture (e.g., Apple Silicon),
# use the following command on the command line:
#
#     docker build --build-arg TARGETPLATFORM=linux/arm64 -t my-arm-image .
#
# Similarly, to build for the default x86_64 architecture, you can use:
#
#     docker build --build-arg TARGETPLATFORM=linux/amd64 -t my-amd64-image .
#
FROM --platform=$TARGETPLATFORM ubuntu:latest
LABEL authors="Luca Marconato"

ENV PYTHONUNBUFFERED=1

ARG SPATIALDATA_VERSION
ARG SPATIALDATA_IO_VERSION
ARG SPATIALDATA_PLOT_VERSION

# debugging
RUN echo "Target Platform: ${TARGETPLATFORM}" && \
    echo "spatialdata version: ${SPATIALDATA_VERSION}" && \
    echo "spatialdata-io version: ${SPATIALDATA_IO_VERSION}" && \
    echo "spatialdata-plot version: ${SPATIALDATA_PLOT_VERSION}"

# Update and install system dependencies.
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    python3-venv \
    python3-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

# setup python virtual environment
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --upgrade pip wheel

# Install the libraries with specific versions
RUN pip install --no-cache-dir \
    spatialdata[torch]==${SPATIALDATA_VERSION} \
    spatialdata-io==${SPATIALDATA_IO_VERSION} \
    spatialdata-plot==${SPATIALDATA_PLOT_VERSION}

LABEL spatialdata_version="${SPATIALDATA_VERSION}" \
      spatialdata_io_version="${SPATIALDATA_IO_VERSION}" \
      spatialdata_plot_version="${SPATIALDATA_PLOT_VERSION}"
