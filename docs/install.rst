.. _Installation:

Installation and Setup
================================================================================

Installing the SCimilarity API
--------------------------------------------------------------------------------

The latest API release can be installed quickly using ``pip`` in the
usual manner:

::

    pip install scimilarity

The SCimilarity API is under activate development. The latest development API
can be downloaded from `GitHub <https://github.com/genentech/scimilarity>`__
and installed as follows:

::

    git clone https://github.com/genentech/scimilarity.git
    cd scimilarity
    pip install -e .

.. warning::

    To enable rapid searches across tens of millions of cells, SCimilarity has very
    high memory requirements. To make queries, you will need at least 64 GB of
    system RAM.

.. warning::

    If your environment has sufficient memory but loading the model or making
    kNN queries crashes, that may be due to older versions of dependencies such
    as hnswlib or numpy. We recommend using either using the Docker container
    or Conda environment described below.

.. note::

    A GPU is not necessary for most applications, but model training will
    require GPU resources.

Downloading the pretrained models
--------------------------------------------------------------------------------

You can download the following pretrained models for use with SCimilarity from
Zenodo:
https://zenodo.org/records/10685499


Conda environment setup
--------------------------------------------------------------------------------

To install the SCimilarity API in a [Conda](https://docs.conda.io) environment
we recommend this environment setup:

:download:`Download environment file <_static/environment.yaml>`

.. literalinclude:: _static/environment.yaml
  :language: YAML

Followed by installing the ``scimilarity`` package via ``pip``, as above.

Using the SCimilarity Docker container
--------------------------------------------------------------------------------

A Docker container that includes the SCimilarity API is available from the
`GitHub Container Registry <https://ghcr.io/genentech/scimilarity>`__, which can
be pulled via:

::

    docker pull ghcr.io/genentech/scimilarity:latest

Models are not included in the Docker container and must be downloaded separately.

There are four preset bind points in the container:

* ``/models``
* ``/data``
* ``/workspace``
* ``/scratch``

We require binding ``/models`` to your local path storing SCimilarity models,
``/data`` to your repository of scRNA-seq data, and ``/workspace`` to your
notebook path.

You can initiate a Jupyter Notebook session rooted in ``/workspace`` using the
``start-notebook`` command as follows:

::

    docker run -it --platform linux/amd64 -p 8888:8888 \
        -v /path/to/workspace:/workspace \
        -v /path/to/data:/data \
        -v /path/to/models:/models \
        ghcr.io/genentech/scimilarity:latest start-notebook
