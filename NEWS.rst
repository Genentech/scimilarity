Release Notes
===============================================================================

Version 0.3.0 pre-release notes:  September 11, 2024
-------------------------------------------------------------------------------

Exhaustive queries: 
  + Functionality to perform exhaustive queries has been added as the new
    method `cell_query.search_exhaustive`.
  + The kNN query method `cell_query.search` has been renamed to
    `cell_query.search_nearest`.

Query result filtering and interpretation:
  + Results from exhaustive queries can be constrained to specific
    metadata criteria (e.g.  tissue, disease, in vitro vs in vivo, etc.)
    using the `metadata_filter` argument to `cell_query.search_exhaustive`.
  + Results from exhaustive queries can be constrained by distance-to-query
    using the `max_dist` argument to `cell_query.search_exhaustive`.
  + The `cell_query.compile_sample_metadata` method has been expanded to
    allow grouping by tissue and disease (in addition to study and sample).
  + The methods `utils.subset_by_unique_values`, `utils.subset_by_frequency`,
    and `utils.categorize_and_sort_by_score` have been added to provide
    tools for filtering, sorting and summarizing query results.

Optimizations to training:
  + The ASW and NMSE training evaluation metrics were added to multiple
    methods.
  + The `triplet_selector.get_asw` method was added to calculate ASW.
  + Optimized sampling weights of study and cell type.


Version 0.2.0:  March 22, 2024
-------------------------------------------------------------------------------

+ Updated version requirements for multiple dependencies and removed
  the ``pegasuspy`` dependency.
+ Expanded API documentation and tutorials.
+ Simplified model file structure and model loader methods.
+ Added ``search_centroid`` method to class ``CellQuery`` for cell
  queries using custom centroids which provides quality control
  statistics to assess the consistency of query results.


Version 0.1.0:  August 13, 2023
-------------------------------------------------------------------------------

+ Initial public release.
