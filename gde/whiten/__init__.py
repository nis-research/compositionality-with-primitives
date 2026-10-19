from .euclidean import EuclideanWhiten
from .non_euc import SphericalWhiten, HyperbolicWhiten

DIM_RED_MODELS = {
    "euclidean": EuclideanWhiten,
    "spherical": SphericalWhiten,
    "hyperbolic": HyperbolicWhiten,
}


__all__ = ["DIM_RED_MODELS"]
