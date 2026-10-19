from typing import Dict, Any

import torch

from gde.dataset import CompositionDatasetEmbeddings


def hyperbolic_radius_from_spatial(x: torch.Tensor, eps=1e-8):
    """
    Compute hyperbolic radius from spatial coordinates x_{1:}
    instead of using x0 directly.
    """
    if x.dim() == 1:
        x = x.unsqueeze(0)

    spatial_norm = torch.norm(x[:, 1:], dim=-1)

    x0_reconstructed = torch.sqrt(1.0 + spatial_norm**2)

    x0_reconstructed = torch.clamp(x0_reconstructed, min=1.0 + eps)

    return torch.acosh(x0_reconstructed)


def radius_stratum(dataset_name, model_name, data: Dict[str, Any]):
    """
    Returns JSON-serializable structure with:
    - objs
    - attrs
    - radius values (as lists)
    """
    emb_ds_train = CompositionDatasetEmbeddings(
        dataset=dataset_name, phase="train", model_name=model_name
    )
    train_embs, train_pairs = emb_ds_train.load_all_image_embs()

    def compute(xs, labels):
        xs = torch.as_tensor(xs)

        radii = hyperbolic_radius_from_spatial(xs).detach().cpu().tolist()

        return [{"label": lab, "radius": r} for lab, r in zip(labels, radii)]

    return {
        "image": {
            "objs": compute(data["image"]["objs_iw"], data["image"]["objs"]),
            "attrs": compute(data["image"]["attrs_iw"], data["image"]["attrs"]),
            "pairs": compute(train_embs, train_pairs),
        },
    }
