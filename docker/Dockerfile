FROM fedora:38
LABEL maintainer="Jason Anthony Vander Heiden [vandej27@gene.com]" \
      description="SCimilarity"

# Bind points
VOLUME /data
VOLUME /models
VOLUME /workspace
VOLUME /scratch

# Tools
COPY start-notebook.sh /usr/local/bin/start-notebook

# Environment
ENV SCDATA_HOME=/data
ENV SCMODEL_HOME=/models

# Update and install required packages
RUN dnf -y update && dnf install -y \
    bzip2 \
    cmake \
    igraph \
    gcc-c++ \
    git \
    lz4 \
    pandoc \
    python3 \
    python3-aiohttp \
    python3-asciitree \
    python3-bash-kernel \
    python3-biopython \
    python3-cloudpickle \
    python3-Cython \
    python3-numcodecs \
    python3-dask \
    python3-dask+array \
    python3-fasteners \
    python3-GitPython \
    python3-h5py \
    python3-igraph \
    python3-jupyter-client \
    python3-jupyterlab_pygments \
    python3-matplotlib \
    python3-matplotlib-scalebar \
    python3-natsort \
    python3-nbconvert \
    python3-nbsphinx \
    python3-notebook \
    python3-numpy \
    python3-pandas \
    python3-pip \
    python3-pydantic \
    python3-pydata-sphinx-theme \
    python3-PyYAML \
    python3-seaborn \
    python3-setuptools \
    python3-scipy \
    python3-stdlib-list \
    python3-texttable \
    python3-toolz \
    python3-tqdm \
    python3-wrapt \
    python3-zarr \
    sudo \
    tar \
    wget \
    zstd \
    && dnf clean all

# Install python dependencies
RUN pip3 install \
    scikit-learn \
	scikit-misc \
	numba \
	tiledb \
	leidenalg \
	louvain \
	umap-learn \
	hnswlib \
	obonet \
	circlify \
	captum \
	torch \
	pytorch-lightning

# Install single-cell tools
RUN pip3 install \
    scanpy \
	pegasusio \
	pegasuspy

# Install SCimilarity API
# TODO: git-clone from remote once setup
#COPY scimilarity-src.tar.gz /scimilarity/scimilarity-src.tar.gz
#RUN mkdir -p /scimilarity \
#    && cd /scimilarity \
#    && tar -zxf scimilarity-src.tar.gz \
#    && rm scimilarity-src.tar.gz \
#    && cd /scimilarity/scimilarity-external \
#    && pip install .
COPY scimilarity-0.0.post1.dev30+g5e68fe1.tar.gz /tmp/scimilarity.tar.gz
RUN pip install /tmp/scimilarity.tar.gz \
    && rm /tmp/scimilarity.tar.gz

# Install SCimilarity models
# TODO: wget from remote once setup
# COPY model_2023_04_rep0.tar.gz /scimilarity/model_2023_04_rep0.tar.gz
# RUN cd /scimilarity \
#     && tar -zxf model_2023_04_rep0.tar.gz \
#     && rm model_2023_04_rep0.tar.gz

# Entry points
CMD ["start-notebook"]

# pip3 install jupyterlab
# jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root
# jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root --notebook-dir /scimilarity
