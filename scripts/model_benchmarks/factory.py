"""Small lazy constructors for the supported model families."""


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
    if name in {"unimol2", "uni-mol2"}:
        from .processors.unimol2 import UniMol2Processor

        return UniMol2Processor(**options)
    if name == "morgan":
        from .processors.morgan import MorganProcessor

        return MorganProcessor(**options)
    if name == "fomonmr":
        from .processors.fomonmr import FoMoNMRProcessor

        return FoMoNMRProcessor(**options)
    raise ValueError(
        f"unknown model {model_name!r}; choose nmrpeak, nmrtrans, "
        "ultranmr, nmrsolver, unimol2, morgan, or fomonmr"
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
    if name in {"unimol2", "uni-mol2"}:
        from .embedders.unimol2 import UniMol2Embedder

        return UniMol2Embedder(**options)
    if name == "morgan":
        from .embedders.morgan import MorganEmbedder

        return MorganEmbedder(**options)
    if name == "fomonmr":
        from .embedders.fomonmr import FoMoNMREmbedder

        return FoMoNMREmbedder(**options)
    raise ValueError(
        f"unknown model {model_name!r}; choose nmrpeak, nmrtrans, "
        "ultranmr, nmrsolver, unimol2, morgan, or fomonmr"
    )


__all__ = ["build_embedder", "build_processor"]

