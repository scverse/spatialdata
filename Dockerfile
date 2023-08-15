FROM condaforge/mambaforge

RUN mamba install python=3.10

# complex dependencies that needs to be solved with conda
RUN mamba install -c conda-forge gcc libgdal gxx imagecodecs -y

# satellite spatialdata projects
RUN pip install --no-cache-dir \
    spatialdata \
    spatialdata-io \
    spatialdata-plot