import numpy as np
import torch
from scipy.linalg import null_space

EPS = 1e-7


def sphere_log_map(p: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Riemannian log map at base point p, evaluated at points x.

    p: (n,) unit vector.  x: (N, n) unit vectors.
    Returns (N, n) tangent vectors at p (i.e. orthogonal to p), whose norm
    equals the geodesic distance to x.
    """
    cos_theta = (x @ p).clamp(-1.0 + EPS, 1.0 - EPS)
    theta = torch.acos(cos_theta)
    sin_theta = torch.sin(theta)
    coeff = torch.where(
        sin_theta.abs() < EPS, torch.ones_like(theta), theta / sin_theta
    )
    v = coeff.unsqueeze(-1) * (x - cos_theta.unsqueeze(-1) * p.unsqueeze(0))
    return v


def sphere_exp_map(p: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Riemannian exp map at base point p, evaluated at tangent vectors v.

    p: (n,) unit vector.  v: (N, n) tangent vectors at p.
    Returns (N, n) points on the sphere.
    """
    norm_v = v.norm(dim=-1, keepdim=True)
    direction = v / norm_v.clamp_min(EPS)
    x = torch.cos(norm_v) * p.unsqueeze(0) + torch.sin(norm_v) * direction
    return x / x.norm(dim=-1, keepdim=True)  # renormalize away fp drift


def sphere_frechet_mean(
    x: torch.Tensor, n_iter: int = 200, tol: float = 1e-10
) -> torch.Tensor:
    """Karcher/Frechet mean on the sphere via iterative log/exp averaging."""
    mean = x.mean(dim=0)
    mean = mean / mean.norm()
    for _ in range(n_iter):
        v = sphere_log_map(mean, x)
        v_mean = v.mean(dim=0)
        if v_mean.norm() < tol:
            break
        mean = sphere_exp_map(mean, v_mean.unsqueeze(0)).squeeze(0)
    return mean


def sphere_tangent_basis(p: torch.Tensor) -> torch.Tensor:
    """Orthonormal (Euclidean) basis for {v : <v, p> = 0}, shape (n, n-1)."""
    p_np = p.detach().cpu().numpy().reshape(1, -1)
    basis = null_space(p_np)  # (n, n-1), already orthonormal
    return torch.from_numpy(basis).to(dtype=p.dtype, device=p.device)


def minkowski_dot(u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    return -u[..., 0] * v[..., 0] + (u[..., 1:] * v[..., 1:]).sum(dim=-1)


def lift_to_hyperboloid(x_spatial: torch.Tensor) -> torch.Tensor:
    """Reconstruct the time coordinate from spatial coords x_1..x_d via the
    constraint <x, x>_M = -1, i.e. x_0 = sqrt(1 + ||x_spatial||^2)."""
    time = torch.sqrt(1.0 + (x_spatial**2).sum(dim=-1, keepdim=True))
    return torch.cat([time, x_spatial], dim=-1)


def drop_time(x_full: torch.Tensor) -> torch.Tensor:
    return x_full[..., 1:]


def hyperboloid_log_map(p: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """p: (d+1,) point on the hyperboloid. x: (N, d+1). Returns (N, d+1)
    tangent vectors at p (Minkowski-orthogonal to p)."""
    alpha = (-minkowski_dot(p.unsqueeze(0), x)).clamp_min(1.0 + EPS)  # = cosh(dist)
    d = torch.acosh(alpha)
    sinh_d = torch.sinh(d)
    coeff = torch.where(d.abs() < EPS, torch.ones_like(d), d / sinh_d)
    u = x - alpha.unsqueeze(-1) * p.unsqueeze(0)
    return coeff.unsqueeze(-1) * u


def hyperboloid_exp_map(p: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """p: (d+1,). v: (N, d+1) tangent vectors at p. Returns (N, d+1) points
    on the hyperboloid."""
    norm_v = minkowski_dot(v, v).clamp_min(0.0).sqrt()
    direction = v / norm_v.clamp_min(EPS).unsqueeze(-1)
    x = (
        torch.cosh(norm_v).unsqueeze(-1) * p.unsqueeze(0)
        + torch.sinh(norm_v).unsqueeze(-1) * direction
    )
    return x


def hyperboloid_frechet_mean(
    x: torch.Tensor, n_iter: int = 200, tol: float = 1e-10
) -> torch.Tensor:
    """Frechet mean on the hyperboloid via iterative log/exp averaging.
    x: (N, d+1) full ambient points (with time coordinate)."""
    spatial_mean = x[:, 1:].mean(dim=0)
    mean = lift_to_hyperboloid(spatial_mean.unsqueeze(0)).squeeze(0)
    for _ in range(n_iter):
        v = hyperboloid_log_map(mean, x)
        v_mean = v.mean(dim=0)
        v_norm = minkowski_dot(v_mean, v_mean).clamp_min(0.0).sqrt()
        if v_norm < tol:
            break
        mean = hyperboloid_exp_map(mean, v_mean.unsqueeze(0)).squeeze(0)
    return mean


def hyperboloid_tangent_basis(p: torch.Tensor) -> torch.Tensor:
    """Orthonormal basis *w.r.t. the Minkowski form* for the tangent space
    {v : <v, p>_M = 0}, shape (d+1, d).

    The Minkowski form restricted to this subspace is positive-definite (the
    tangent space to the hyperboloid is spacelike), so an orthonormal basis
    w.r.t. it turns tangent-space coordinates into an ordinary Euclidean
    coordinate system, which is exactly what PCA/whitening expects.
    """
    p_np = p.detach().cpu().numpy()
    w = p_np.copy()
    w[0] = -w[0]  # <v, p>_M = w . v  (standard dot product)
    spanning = null_space(w.reshape(1, -1))  # (d+1, d), Euclidean-orthonormal

    def mdot(a, b):
        return -a[0] * b[0] + float(np.dot(a[1:], b[1:]))

    basis = []
    for i in range(spanning.shape[1]):
        v = spanning[:, i].copy()
        for e in basis:
            coeff = mdot(v, e)
            v = v - coeff * e
        norm = np.sqrt(max(mdot(v, v), EPS))
        basis.append(v / norm)
    e = np.stack(basis, axis=1)  # (d+1, d)
    return torch.from_numpy(e).to(dtype=p.dtype, device=p.device)
