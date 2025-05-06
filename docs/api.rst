.. _API:

API Reference
================================================================================

.. note::
    API documentation is under construction. Current documentation is focused
    on core functionality.

.. toctree::
    :maxdepth: 1
    :hidden:

    API Reference <self>

.. toctree::
    :maxdepth: 2
    :caption: Core Functionality
    :hidden:

    modules/cell_annotation
    modules/cell_embedding
    modules/cell_query
    modules/cell_search_knn
    modules/interpreter

.. toctree::
    :maxdepth: 2
    :caption: Model Training
    :hidden:

    modules/anndata_data_models
    modules/nn_models
    modules/tiledb_data_models
    modules/training_models
    modules/triplet_selector
    modules/zarr_data_models

.. toctree::
    :maxdepth: 2
    :caption: Utilities
    :hidden:

    modules/ontologies
    modules/utils
    modules/visualizations
    modules/zarr_dataset

Core Functionality
--------------------------------------------------------------------------------

These modules provide functionality for utilizing SCimilarity embeddings for a
variety of tasks, including cell type annotation, cell queries, and gene
attribution scoring.

* :mod:`scimilarity.cell_annotation`
* :mod:`scimilarity.cell_embedding`
* :mod:`scimilarity.cell_query`
* :mod:`scimilarity.cell_search_knn`
* :mod:`scimilarity.interpreter`

Model Training
--------------------------------------------------------------------------------

Training new SCimilarity models requires aggregated and curated training data.
This relies on specialized data loaders that are optimized for random cell access
across datasets, specialized variations of metric learning loss functions, and
procedures for cell ontology aware triplet mining. The following modules include
support for these training tasks.

* :mod:`scimilarity.anndata_data_models`
* :mod:`scimilarity.nn_models`
* :mod:`scimilarity.tiledb_data_models`
* :mod:`scimilarity.training_models`
* :mod:`scimilarity.triplet_selector`
* :mod:`scimilarity.zarr_data_models`
* :mod:`scimilarity.zarr_dataset`

Utilities
--------------------------------------------------------------------------------

SCimilarity uses specific visualizations, ontology interfaces, and data
preprocessing steps. These modules provide functionality useful for model
training as well as a variety of SCimilarity analyses.

* :mod:`scimilarity.ontologies`
* :mod:`scimilarity.utils`
* :mod:`scimilarity.visualizations`
