"""Active learning pipeline for PXR pEC50 prediction.

Submodules
----------
config
    Experiment configuration loading and validation (``ALConfig`` dataclass).
helpers
    Core active learning utilities: data splitting, featurization, committee
    training, acquisition strategies, and the main AL loop.
plots
    Plotly and Faerun visualization functions for learning curves, chemical
    space embeddings, and calibration diagnostics.
"""
