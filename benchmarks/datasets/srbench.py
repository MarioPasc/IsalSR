"""SRBench standardized benchmark loader.

Reference: La Cava et al. (2021). Contemporary symbolic regression methods
and their relative performance. NeurIPS Datasets & Benchmarks.

To implement:
    load_srbench_dataset(name: str) -> tuple[np.ndarray, np.ndarray]
        Load a dataset from the SRBench collection.

    list_srbench_datasets() -> list[str]
        List available SRBench dataset names.

    Note: SRBench datasets are available via the PMLB package or direct download.
"""
