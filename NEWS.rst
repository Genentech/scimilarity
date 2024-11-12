Release Notes
================================================================================

Version 0.3.0 pre-release notes:  November 11, 2024
--------------------------------------------------------------------------------

General:
  + Various changes to utility functions to improve efficiency and flexibility.
  + Simplification of many class constructor parameters.
  + Tutorials have been updated with new download links and analyses.

Exhaustive queries:
  + Functionality to perform exhaustive queries has been added as new methods
    ``cell_query.search_exhaustive``, ``cell_query.search_centroid_exhaustive``,
    and ``cell_query.search_cluster_centroids_exhaustive``.
  + The kNN query method ``cell_query.search`` has been renamed to
    ``cell_query.search_nearest``.

Query result filtering and interpretation:
  + The ``cell_query.compile_sample_metadata`` method has been expanded to
    allow grouping by tissue and disease (in addition to study and sample).
  + The methods ``utils.subset_by_unique_values``,
    ``utils.subset_by_frequency``, and ``utils.categorize_and_sort_by_score``
    have been added to provide tools for filtering, sorting and summarizing
    query results.
  + The "query_stability" quality control metric has been renamed to
    "query_coherence" and is now deterministic (by setting a random seed).
  + Results from exhaustive queries can be constrained to specific
    metadata criteria (e.g., tissue, disease, in vitro vs in vivo, etc).
    using the ``metadata_filter`` argument to ``cell_query.search_exhaustive``.
  + Results from exhaustive queries can be constrained by distance-to-query
    using the ``max_dist`` argument to ``cell_query.search_exhaustive``.
  + The mappings in ``utils.clean_tissues`` and ``utils.clean_diseases`` have
    been expanded.

Optimizations to training:
  + The ASW and NMSE training evaluation metrics were added.
  + The ``triplet_selector.get_asw`` method was added to calculate ASW.
  + The ``ontologies.find_most_viable_parent`` method was added to help coarse
    grain cell type ontology labels.
  + Optimized sampling weights of study and cell type.

Version 0.2.0:  March 22, 2024
--------------------------------------------------------------------------------

+ Updated version requirements for multiple dependencies and removed
  the ``pegasuspy`` dependency.
+ Expanded API documentation and tutorials.
+ Simplified model file structure and model loader methods.
+ Added ``search_centroid`` method to class ``CellQuery`` for cell
  queries using custom centroids which provides quality control
  statistics to assess the consistency of query results.


Version 0.1.0:  August 13, 2023
--------------------------------------------------------------------------------

+ Initial public release.
