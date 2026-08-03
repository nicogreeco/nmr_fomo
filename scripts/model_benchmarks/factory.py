"""Small lazy constructors for the four supported model families."""


def _model_name(name: str) -> str:
    return name.strip().lower().replace("_", "-")


def build_processor(model_name: str, **options):
    """Create one processor without importing the other model repositories."""

    name = _model_name(model_name)
    if name in {"nmrpeak", "nmrpeak-r"}:
        from .processors.nmrpeak import NMRPeakProcessor

        return NMRPeakProcessor(**options)
    if name == "nmrtrans":
        from .processors.nmrtrans import NMRTransProcessor

        return NMRTransProcessor(**options)
    if name == "ultranmr":
        from .processors.ultranmr import UltraNMRProcessor

        return UltraNMRProcessor(**options)
    if name in {"nmrsolver", "nmr-solver"}:
        from .processors.nmrsolver import NMRSolverProcessor

        return NMRSolverProcessor(**options)
    raise ValueError(
        f"unknown model {model_name!r}; choose nmrpeak, nmrtrans, "
        "ultranmr, or nmrsolver"
    )


def build_embedder(model_name: str, **options):
    """Create one embedder without importing the other model repositories."""

    name = _model_name(model_name)
    if name in {"nmrpeak", "nmrpeak-r"}:
        from .embedders.nmrpeak import NMRPeakEmbedder

        return NMRPeakEmbedder(**options)
    if name == "nmrtrans":
        from .embedders.nmrtrans import NMRTransEmbedder

        return NMRTransEmbedder(**options)
    if name == "ultranmr":
        from .embedders.ultranmr import UltraNMREmbedder

        return UltraNMREmbedder(**options)
    if name in {"nmrsolver", "nmr-solver"}:
        from .embedders.nmrsolver import NMRSolverEmbedder

        return NMRSolverEmbedder(**options)
    raise ValueError(
        f"unknown model {model_name!r}; choose nmrpeak, nmrtrans, "
        "ultranmr, or nmrsolver"
    )


__all__ = ["build_embedder", "build_processor"]

