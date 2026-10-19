from typing import Literal

import torch

GEOMETRIES = Literal["euclidean", "spherical", "hyperbolic"]


def pw_spherical_similarity(vec_1, vec_2):
    return vec_1 @ vec_2.T


def pw_lorentzian_similarity(vec_1, vec_2, curv=0.1):
    # vec_1: [N, D], vec_2: [M, D or D+1]  (space components or time + space)
    if vec_2.shape[-1] == vec_1.shape[-1] + 1:
        # If vec_2 has an extra dimension,
        # we assume it's already in (t, x) form and we ignore the time component for similarity.
        vec_2 = vec_2[:, 1:]

    time_1 = torch.sqrt(1.0 / curv + (vec_1**2).sum(dim=-1, keepdim=True))  # [N, 1]
    time_2 = torch.sqrt(1.0 / curv + (vec_2**2).sum(dim=-1, keepdim=True))  # [M, 1]

    space = vec_1 @ vec_2.T  # [N, M]
    time = time_1 @ time_2.T  # [N, M]

    return space - time


def pw_cosine_similarity(vec_1, vec_2):
    vec_1_norm = vec_1 / vec_1.norm(dim=1, keepdim=True)
    vec_2_norm = vec_2 / vec_2.norm(dim=1, keepdim=True)
    return vec_1_norm @ vec_2_norm.T


def spherical_similarity(vec_1, vec_2):
    return (vec_1 * vec_2).sum(dim=-1)


def lorentzian_similarity(vec_1, vec_2, curv=1.0):
    # vec_1: [N, D], vec_2: [M, D or D+1]  (space components or time + space)
    if vec_2.shape[-1] == vec_1.shape[-1] + 1:
        # If vec_2 has an extra dimension,
        # we assume it's already in (t, x) form and we ignore the time component for similarity.
        vec_2 = vec_2[:, 1:]

    time_1 = torch.sqrt(1.0 / curv + (vec_1**2).sum(dim=-1))  # [N]
    time_2 = torch.sqrt(1.0 / curv + (vec_2**2).sum(dim=-1))  # [N]

    space = (vec_1 * vec_2).sum(dim=-1)  # [N]

    return space - time_1 * time_2


def cosine_similarity(vec_1, vec_2):
    vec_1_norm = vec_1 / vec_1.norm(dim=-1, keepdim=True)
    vec_2_norm = vec_2 / vec_2.norm(dim=-1, keepdim=True)
    return (vec_1_norm * vec_2_norm).sum(dim=-1)
