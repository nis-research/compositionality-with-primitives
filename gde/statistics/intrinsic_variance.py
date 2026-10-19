from typing import Literal, Dict, Any
import torch

Geometry = Literal["euclidean", "spherical", "hyperbolic"]


# -------------------------
# Geometry distances
# -------------------------
def euclidean_dist(x: torch.Tensor, y: torch.Tensor):
    """
    x: (B, D) or (D,)
    y: (B, D) or (D,)
    """
    if x.dim() == 1:
        x = x.unsqueeze(0)
    if y.dim() == 1:
        y = y.unsqueeze(0)

    return torch.norm(x - y, p=2, dim=-1)


def spherical_dist(x: torch.Tensor, y: torch.Tensor, eps=1e-8):
    """
    Batch-safe spherical distance.
    """
    if x.dim() == 1:
        x = x.unsqueeze(0)
    if y.dim() == 1:
        y = y.unsqueeze(0)

    x = x / (x.norm(p=2, dim=-1, keepdim=True) + eps)
    y = y / (y.norm(p=2, dim=-1, keepdim=True) + eps)

    dot = torch.sum(x * y, dim=-1)
    dot = torch.clamp(dot, -1.0, 1.0)

    return torch.acos(dot)


def hyperbolic_dist(x: torch.Tensor, y: torch.Tensor, eps=1e-8):
    """
    Batch-safe Lorentz distance.
    """
    if x.dim() == 1:
        x = x.unsqueeze(0)
    if y.dim() == 1:
        y = y.unsqueeze(0)

    minkowski = -x[:, 0] * y[:, 0] + torch.sum(x[:, 1:] * y[:, 1:], dim=-1)

    return torch.acosh(torch.clamp(-minkowski, min=1.0 + eps))


def get_dist(geometry: Geometry):
    if geometry == "euclidean":
        return euclidean_dist
    elif geometry == "spherical":
        return spherical_dist
    elif geometry == "hyperbolic":
        return hyperbolic_dist
    else:
        raise ValueError(f"Unknown geometry: {geometry}")


# -------------------------
# Variance
# -------------------------


def geometric_variance(data: Dict[str, Any], geometry: Geometry):
    """
    Fréchet variance relative to data["all"]["iw"].
    """

    dist = get_dist(geometry)
    ref = data["origin"]

    def var(xs: torch.Tensor):
        if xs is None or len(xs) == 0:
            return torch.tensor(float("nan"))

        # squared geodesic distances
        dists = dist(xs, ref)
        return torch.mean(dists**2)

    return {
        "image": {
            "objs_iw": var(data["image"]["objs_iw"]).item(),
            "attrs_iw": var(data["image"]["attrs_iw"]).item(),
        },
    }
