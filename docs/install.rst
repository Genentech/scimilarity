.. _Installation:

Installation and Setup
================================================================================

Installing the SCimilarity API
--------------------------------------------------------------------------------

The SCimilarity API is under activate development. The latest API release can be
downloaded from `GitHub <https://github.com/genentech/scimilarity>`__.
Installation is quick and performed using ``pip`` in the usual manner:

::

    git clone https://github.com/genentech/scimilarity.git
    cd scimilarity
    pip install -e .

.. warning::

    To enable rapid searches across tens of millions of cells, SCimilarity has very
    high memory requirements. To make queries, you will need at least 64 GB of
    system RAM.

.. note::

    A GPU is not necessary for most applications, but model training will
    require GPU resources.

Downloading the pretrained models
--------------------------------------------------------------------------------

You can download the following pretrained models for use with SCimilarity from
Zenodo:
https://zenodo.org/record/8240463

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
