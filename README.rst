SCimilarity
================================================================================

**SCimilarity** is a unifying representation of single cell expression profiles
that quantifies similarity between expression states and generalizes to
represent new studies without additional training.

This enables a novel cell search capability, which sifts through millions of
profiles to find cells similar to a query cell state and allows researchers to
quickly and systematically leverage massive public scRNA-seq atlases to learn
about a cell state of interest.

Documentation
--------------------------------------------------------------------------------

Tutorials and API documentation can be found at:
https://genentech.github.io/scimilarity/index.html

Download & Install
--------------------------------------------------------------------------------

The latest API release can be installed from PyPI::

    pip install scimilarity

Pretrained model weights, embeddings, kNN graphs, a single-cell metadata
can be downloaded from:
https://zenodo.org/records/10685499

A docker container with SCimilarity preinstalled can be pulled from:
https://ghcr.io/genentech/scimilarity

Citation
--------------------------------------------------------------------------------

To cite SCimilarity in publications please use:

**A cell atlas foundation model for scalable search of similar human cells.**
*Graham Heimberg\*, Tony Kuo\*, Daryle J. DePianto, Tobias Heigl,
Nathaniel Diamant, Omar Salem, Gabriele Scalia, Tommaso Biancalani,
Jason R. Rock, Shannon J. Turley, Héctor Corrada Bravo, Josh Kaminker\*\*,
Jason A. Vander Heiden\*\*, Aviv Regev\*\*.*
Nature (2024). https://doi.org/10.1038/s41586-024-08411-y