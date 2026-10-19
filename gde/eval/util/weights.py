import torch
import numpy as np

from gde.config import logger
from gde.dataset import CompositionDatasetEmbeddings

from .similarity import (
    GEOMETRIES,
    cosine_similarity,
    spherical_similarity,
    lorentzian_similarity,
)


def _normalize_within_pairs(weights: torch.Tensor, pairs: list) -> torch.Tensor:
    """Normalize weights so each unique pair's images sum to 1."""
    pairs_np = np.array(pairs, dtype=str)
    _, inverse = np.unique(pairs_np, axis=0, return_inverse=True)
    inverse = torch.tensor(inverse, dtype=torch.long)
    group_sums = torch.bincount(inverse, weights=weights).float()
    return weights / (group_sums[inverse] + 1e-8)


def build_weights(
    train_embs: torch.Tensor,
    pairs: list[tuple[str, str]],
    geometry: GEOMETRIES,
    text_embs=None,
    text_factorizer=None,
    use_im_text_sim: bool = False,
    temperature: float = 0.09,
    compute_robust_text_embeddings: bool = False,
) -> torch.Tensor:
    """Per-embedding weights: exp(sim(image, text_pair) / T) normalized within pair."""
    if use_im_text_sim:
        assert (
            text_embs is not None
        ), "Dataset is required to compute image-text similarity weights."
        if compute_robust_text_embeddings:
            assert (
                text_factorizer is not None
            ), "Text factorizer is required to compute robust text embeddings."
            text_embs = text_factorizer.compute_ideal_words_approximation(pairs)
        if geometry == "euclidean":
            similarity = cosine_similarity
        elif geometry == "spherical":
            similarity = spherical_similarity
        elif geometry == "hyperbolic":
            similarity = lorentzian_similarity
        else:
            raise ValueError(f"Unsupported geometry: {geometry}")

        sim = similarity(train_embs, text_embs)
        # Shift by max for numerical stability before exp.
        weights = torch.exp((sim - sim.max()) / temperature)
        logger.debug(
            "CLIP weights -- min: {:.4f}  max: {:.4f}  mean: {:.4f}",
            weights.min().item(),
            weights.max().item(),
            weights.mean().item(),
        )
    else:
        weights = torch.ones(len(pairs))
    norm_per_group = _normalize_within_pairs(weights, pairs)

    return (
        norm_per_group / norm_per_group.sum()
    )  # normalize to sum to 1 across all samples
