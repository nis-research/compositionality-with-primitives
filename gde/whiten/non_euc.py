import math

import numpy as np
import torch

from .base import ManifoldWhiten

# ---------------------------------------------------------------------
# Common helpers
# ---------------------------------------------------------------------
STRENGTH_SPHERE = 0.5
STRENGTH_HYPERBOLOID = 1.0


def _to_tensor_preserve_dtype(x):
    """
    Same metadata convention as ManifoldWhiten._to_tensor, but does not
    silently convert NumPy float64 -> float32.
    """
    if isinstance(x, torch.Tensor):
        if not torch.is_floating_point(x):
            raise TypeError("Manifold whitening requires floating-point input.")

        return x, {
            "was_numpy": False,
            "device": x.device,
            "dtype": x.dtype,
        }

    if isinstance(x, np.ndarray):
        if not np.issubdtype(x.dtype, np.floating):
            raise TypeError("Manifold whitening requires floating-point input.")

        return torch.from_numpy(x), {
            "was_numpy": True,
            "device": None,
            "dtype": x.dtype,
        }

    raise TypeError(
        f"Expected a numpy.ndarray or torch.Tensor, " f"got {type(x).__name__}."
    )


def _work_dtype(x: torch.Tensor):
    """
    Use float64 on CPU for more stable:
      - Frechet mean estimation
      - eigendecomposition
      - spherical acos
      - hyperbolic acosh

    On GPU keep float32/float64, but promote half precision to float32.
    """
    if x.device.type == "cpu":
        return torch.float64

    if x.dtype in (torch.float16, torch.bfloat16):
        return torch.float32

    return x.dtype


def _fit_tangent_transform(
    z: torch.Tensor,
    *,
    strength: float = STRENGTH_SPHERE,
    shrinkage: float = 0.1,
    min_eigenvalue_ratio: float = 1e-4,
    target_rms=None,
):
    """
    Fit a regularized tangent-space shape normalization.

    Parameters
    ----------
    z:
        [N, d] tangent coordinates.

    strength:
        Whitening strength.

        0.0:
            no covariance-shape correction.

        0.5:
            partial whitening.

        1.0:
            full inverse-square-root whitening of the regularized
            covariance.

    shrinkage:
        Amount of isotropic covariance shrinkage.

        C_reg =
            (1 - shrinkage) C
            + shrinkage * mean_variance * I

    min_eigenvalue_ratio:
        Minimum eigenvalue relative to average variance.

    target_rms:
        Desired

            sqrt(E ||z_out||^2)

        If None, preserve the RMS geodesic radius of the input.

    Returns
    -------
    tangent_mean:
        [d]

    transform:
        [d, d]

    eigenvalues:
        Eigenvalues of the regularized covariance.

    effective_rms:
        Desired output RMS tangent/geodesic radius.
    """
    if z.ndim != 2:
        raise ValueError(f"Expected tangent data [N, d], got {tuple(z.shape)}.")

    if not 0.0 <= strength <= 1.0:
        raise ValueError("strength must be in [0, 1].")

    if not 0.0 <= shrinkage <= 1.0:
        raise ValueError("shrinkage must be in [0, 1].")

    if min_eigenvalue_ratio <= 0:
        raise ValueError("min_eigenvalue_ratio must be > 0.")

    n, d = z.shape

    if n < 2:
        raise ValueError("At least two samples are required.")

    if d < 1:
        raise ValueError("At least one tangent dimension is required.")

    tangent_mean = z.mean(dim=0)
    centered = z - tangent_mean

    covariance = centered.T @ centered / (n - 1)

    eps = torch.finfo(z.dtype).eps

    average_variance = (torch.trace(covariance) / d).clamp_min(eps)

    identity = torch.eye(
        d,
        device=z.device,
        dtype=z.dtype,
    )

    regularized_covariance = (
        1.0 - shrinkage
    ) * covariance + shrinkage * average_variance * identity

    eigenvalues, eigenvectors = torch.linalg.eigh(regularized_covariance)

    floor = (min_eigenvalue_ratio * average_variance).clamp_min(eps)

    safe_eigenvalues = eigenvalues.clamp_min(floor)

    # strength = 0 -> eigenvalues^0 = 1
    # strength = 1 -> eigenvalues^-1/2
    scales = safe_eigenvalues.pow(-0.5 * strength)

    transform = eigenvectors @ torch.diag(scales) @ eigenvectors.T

    shaped = centered @ transform

    input_rms = torch.sqrt(centered.square().sum(dim=-1).mean())

    shaped_rms = torch.sqrt(shaped.square().sum(dim=-1).mean()).clamp_min(eps)

    if target_rms is None:
        effective_rms = input_rms

    else:
        if target_rms < 0:
            raise ValueError("target_rms must be >= 0.")

        effective_rms = torch.as_tensor(
            target_rms,
            device=z.device,
            dtype=z.dtype,
        )

    # Preserve/choose overall geodesic scale after correcting
    # covariance shape.
    transform = transform * effective_rms / shaped_rms

    return (
        tangent_mean,
        transform,
        eigenvalues,
        effective_rms,
    )


# =====================================================================
# Sphere
# =====================================================================


class SphericalWhiten(ManifoldWhiten):
    """
    Geometry-aware whitening for the UNIT sphere.

    Input
    -----
    x : [N, D]
        Full ambient spherical coordinates.

        Input vectors are normalized internally to

            ||x|| = 1.

    Output
    ------
    [N, D]
        Full unit-sphere coordinates.

    Notes
    -----
    The intrinsic dimension is D - 1.

    Whitening is performed in tangent coordinates after moving the
    intrinsic mean to the canonical origin

        o = [1, 0, ..., 0].

    Default behavior intentionally does NOT force Cov = I.
    Instead it adjusts covariance shape while preserving the original
    RMS geodesic radius.
    """

    def __init__(
        self,
        max_iter: int = 100,
        tol: float = 1e-8,
        strength: float = STRENGTH_SPHERE,
        shrinkage: float = 0.1,
        min_eigenvalue_ratio: float = 1e-4,
        target_rms=None,
        antipode_tol: float = 1e-8,
        cut_locus_margin: float = 1e-3,
        clip_out_of_sample: bool = False,
    ):
        super().__init__()

        if cut_locus_margin <= 0:
            raise ValueError("cut_locus_margin must be > 0.")

        if cut_locus_margin >= math.pi:
            raise ValueError("cut_locus_margin must be < pi.")

        self.max_iter = max_iter
        self.tol = tol

        self.strength = strength
        self.shrinkage = shrinkage
        self.min_eigenvalue_ratio = min_eigenvalue_ratio
        self.target_rms = target_rms

        self.antipode_tol = antipode_tol
        self.cut_locus_margin = cut_locus_margin

        self.clip_out_of_sample = clip_out_of_sample

        self._mean = None
        self._centering_isometry = None

        self._tangent_mean = None
        self._tangent_transform = None

        self._eigenvalues = None
        self._effective_rms = None

    # -----------------------------------------------------------------
    # Sphere geometry
    # -----------------------------------------------------------------

    @staticmethod
    def _normalize(x):
        norm = torch.linalg.vector_norm(
            x,
            dim=-1,
            keepdim=True,
        )

        if torch.any(norm <= 0):
            raise ValueError("Zero vector cannot represent a point " "on the sphere.")

        return x / norm

    @staticmethod
    def _origin(
        ambient_dim,
        *,
        device,
        dtype,
    ):
        origin = torch.zeros(
            ambient_dim,
            device=device,
            dtype=dtype,
        )

        origin[0] = 1.0

        return origin

    def _log(self, base, x):
        """
        Principal spherical logarithmic map on the unit sphere.
        """
        cos_theta = (x * base).sum(dim=-1).clamp(-1.0, 1.0)

        if torch.any(cos_theta <= -1.0 + self.antipode_tol):
            raise ValueError(
                "Spherical logarithm encountered a point "
                "at or too close to the antipode of the "
                "current base point."
            )

        theta = torch.acos(cos_theta)

        sin_theta = torch.sin(theta)

        eps = torch.finfo(x.dtype).eps

        ratio = torch.where(
            theta.abs() < 1e-7,
            1.0 + theta.square() / 6.0,
            theta / sin_theta.clamp_min(eps),
        )

        tangent_direction = x - cos_theta[..., None] * base

        return ratio[..., None] * tangent_direction

    def _exp(self, base, v):
        """
        Spherical exponential map on the unit sphere.
        """
        theta = torch.linalg.vector_norm(
            v,
            dim=-1,
        )

        eps = torch.finfo(v.dtype).eps

        sinc = torch.where(
            theta.abs() < 1e-7,
            1.0 - theta.square() / 6.0,
            torch.sin(theta) / theta.clamp_min(eps),
        )

        result = torch.cos(theta)[..., None] * base + sinc[..., None] * v

        return self._normalize(result)

    # -----------------------------------------------------------------
    # Intrinsic mean
    # -----------------------------------------------------------------

    def _frechet_mean(self, x):
        """
        Karcher/Frechet mean using Riemannian gradient iterations
        with backtracking.
        """
        mean = x.mean(dim=0)

        if torch.linalg.vector_norm(mean) < 1e-12:
            mean = x[0].clone()
        else:
            mean = self._normalize(mean)

        for _ in range(self.max_iter):
            tangent = self._log(
                mean,
                x,
            )

            delta = tangent.mean(dim=0)

            if torch.linalg.vector_norm(delta).item() < self.tol:
                break

            old_objective = tangent.square().sum(dim=-1).mean()

            step = 1.0
            accepted = False

            for _ in range(12):
                candidate = self._exp(
                    mean,
                    step * delta,
                )

                try:
                    candidate_tangent = self._log(
                        candidate,
                        x,
                    )
                except ValueError:
                    step *= 0.5
                    continue

                new_objective = candidate_tangent.square().sum(dim=-1).mean()

                if new_objective <= old_objective:
                    mean = candidate
                    accepted = True
                    break

                step *= 0.5

            if not accepted:
                break

        return mean

    # -----------------------------------------------------------------
    # Exact sphere isometry mean -> canonical origin
    # -----------------------------------------------------------------

    @staticmethod
    def _make_centering_isometry(mean):
        """
        Householder orthogonal transformation H satisfying

            H mean = [1, 0, ..., 0].

        H is an exact isometry of the sphere.
        """
        ambient_dim = mean.numel()

        e0 = torch.zeros_like(mean)
        e0[0] = 1.0

        w = mean - e0

        denominator = w @ w

        if denominator.item() < 1e-14:
            return torch.eye(
                ambient_dim,
                device=mean.device,
                dtype=mean.dtype,
            )

        identity = torch.eye(
            ambient_dim,
            device=mean.device,
            dtype=mean.dtype,
        )

        return identity - 2.0 * torch.outer(w, w) / denominator

    # -----------------------------------------------------------------
    # Fit
    # -----------------------------------------------------------------

    def fit(self, x) -> "SphericalWhiten":
        self._is_fitted = False

        x_t, _ = _to_tensor_preserve_dtype(x)

        if x_t.ndim != 2:
            raise ValueError("SphericalWhiten.fit expects " "shape [N, D].")

        n, ambient_dim = x_t.shape

        if n < 2:
            raise ValueError("At least two samples are required.")

        if ambient_dim < 2:
            raise ValueError("Sphere requires at least two " "ambient dimensions.")

        self._components = ambient_dim

        X = x_t.detach().to(dtype=_work_dtype(x_t))

        if not torch.isfinite(X).all():
            raise ValueError("Input contains NaN or infinite values.")

        X = self._normalize(X)

        with torch.no_grad():
            # Intrinsic location.
            mean = self._frechet_mean(X)

            # Exact sphere isometry mean -> e0.
            H = self._make_centering_isometry(mean)

            # Logarithmic coordinates at mean.
            tangent = self._log(
                mean,
                X,
            )

            # Differential of sphere isometry.
            tangent_at_origin = tangent @ H.T

            # At e0, the first component of every tangent
            # vector is zero. Remaining D-1 coordinates form
            # an orthonormal tangent basis.
            z = tangent_at_origin[:, 1:]

            (
                tangent_mean,
                transform,
                eigenvalues,
                effective_rms,
            ) = _fit_tangent_transform(
                z,
                strength=self.strength,
                shrinkage=self.shrinkage,
                min_eigenvalue_ratio=(self.min_eigenvalue_ratio),
                target_rms=self.target_rms,
            )

            # ---------------------------------------------------------
            # Ensure training points remain inside the principal
            # log/exp region.
            # ---------------------------------------------------------

            z_out = (z - tangent_mean) @ transform

            max_radius = torch.linalg.vector_norm(
                z_out,
                dim=-1,
            ).max()

            max_allowed = math.pi - self.cut_locus_margin

            if max_radius.item() >= max_allowed:
                # Global scalar reduction:
                # preserves covariance SHAPE and avoids folding
                # through the antipode.
                correction = 0.999 * max_allowed / max_radius

                transform = transform * correction

                effective_rms = effective_rms * correction

            self._mean = mean.detach()

            self._centering_isometry = H.detach()

            self._tangent_mean = tangent_mean.detach()

            self._tangent_transform = transform.detach()

            self._eigenvalues = eigenvalues.detach()

            self._effective_rms = effective_rms.detach()

        self._is_fitted = True

        return self

    # -----------------------------------------------------------------
    # Project
    # -----------------------------------------------------------------

    def project(self, x):
        self._check_is_fitted()

        x_t, meta = _to_tensor_preserve_dtype(x)

        if x_t.shape[-1] != self._components:
            raise ValueError(
                f"Expected ambient dimension "
                f"{self._components}, got "
                f"{x_t.shape[-1]}."
            )

        X = x_t.to(dtype=_work_dtype(x_t))

        if not torch.isfinite(X).all():
            raise ValueError("Input contains NaN or infinite values.")

        X = self._normalize(X)

        mean = self._mean.to(X)

        H = self._centering_isometry.to(X)

        tangent_mean = self._tangent_mean.to(X)

        transform = self._tangent_transform.to(X)

        # x -> log_mu(x)
        tangent = self._log(
            mean,
            X,
        )

        # T_mu S -> T_o S
        tangent_at_origin = tangent @ H.T

        # Remove residual tangent mean and correct covariance shape.
        z = tangent_at_origin[..., 1:] - tangent_mean

        z_out = z @ transform

        # -------------------------------------------------------------
        # Sphere injectivity-radius handling for unseen samples
        # -------------------------------------------------------------

        radius = torch.linalg.vector_norm(
            z_out,
            dim=-1,
        )

        max_allowed = math.pi - self.cut_locus_margin

        outside = radius >= max_allowed

        if torch.any(outside):
            if not self.clip_out_of_sample:
                raise ValueError(
                    "An out-of-sample spherical point is "
                    "mapped outside the principal "
                    "injectivity region. Either reduce "
                    "`strength` / `target_rms`, or set "
                    "`clip_out_of_sample=True`."
                )

            eps = torch.finfo(z_out.dtype).eps

            scale = torch.clamp(
                (0.999 * max_allowed / radius.clamp_min(eps)),
                max=1.0,
            )

            z_out = z_out * scale[..., None]

        # Tangent coordinates -> ambient tangent vector.
        tangent_out = torch.zeros(
            *z_out.shape[:-1],
            self._components,
            device=z_out.device,
            dtype=z_out.dtype,
        )

        tangent_out[..., 1:] = z_out

        origin = self._origin(
            self._components,
            device=X.device,
            dtype=X.dtype,
        )

        result = self._exp(
            origin,
            tangent_out,
        )

        return self._to_output(
            result,
            meta,
        )

    # -----------------------------------------------------------------
    # Diagnostics
    # -----------------------------------------------------------------

    @property
    def effective_rms_(self):
        self._check_is_fitted()
        return self._effective_rms.item()

    @property
    def eigenvalues_(self):
        self._check_is_fitted()
        return self._eigenvalues.detach().cpu().numpy()


# =====================================================================
# Hyperboloid
# =====================================================================


class HyperbolicWhiten(ManifoldWhiten):
    """
    Geometry-aware whitening in the hyperboloid/Lorentz model.

    The manifold has sectional curvature

        -curvature

    where `curvature` is provided as a positive number.

    Input
    -----
    x : [N, d]
        SPATIAL coordinates only.

    The time coordinate is reconstructed internally:

        t = sqrt(R^2 + ||x||^2)

    where

        R = 1 / sqrt(curvature).

    Output
    ------
    [N, d]
        Spatial coordinates only.
    """

    def __init__(
        self,
        curvature: float = 0.1,
        max_iter: int = 100,
        tol: float = 1e-8,
        strength: float = STRENGTH_HYPERBOLOID,
        shrinkage: float = 0.1,
        min_eigenvalue_ratio: float = 1e-4,
        target_rms=None,
        mean_method: str = "lorentz",
    ):
        super().__init__()

        if curvature <= 0:
            raise ValueError("curvature must be > 0.")

        if mean_method not in {
            "lorentz",
            "frechet",
        }:
            raise ValueError("mean_method must be either " "'lorentz' or 'frechet'.")

        self.curvature = float(curvature)

        self.radius = 1.0 / math.sqrt(self.curvature)

        self.max_iter = max_iter
        self.tol = tol

        self.strength = strength
        self.shrinkage = shrinkage

        self.min_eigenvalue_ratio = min_eigenvalue_ratio

        self.target_rms = target_rms

        self.mean_method = mean_method

        self._mean = None
        self._centering_isometry = None

        self._tangent_mean = None
        self._tangent_transform = None

        self._eigenvalues = None
        self._effective_rms = None

    # -----------------------------------------------------------------
    # Lorentz geometry
    # -----------------------------------------------------------------

    @staticmethod
    def _lorentz_inner(x, y):
        """
        Lorentz metric with signature (-,+,+,...,+).
        """
        return -x[..., 0] * y[..., 0] + (x[..., 1:] * y[..., 1:]).sum(dim=-1)

    def _to_hyperboloid(self, spatial):
        """
        Spatial coordinates -> upper-sheet hyperboloid.
        """
        spatial_norm_squared = spatial.square().sum(
            dim=-1,
            keepdim=True,
        )

        time = torch.sqrt(self.radius**2 + spatial_norm_squared)

        return torch.cat(
            [time, spatial],
            dim=-1,
        )

    @staticmethod
    def _to_spatial(x):
        return x[..., 1:]

    def _origin(
        self,
        spatial_dim,
        *,
        device,
        dtype,
    ):
        origin = torch.zeros(
            spatial_dim + 1,
            device=device,
            dtype=dtype,
        )

        origin[0] = self.radius

        return origin

    def _log(self, base, x):
        """
        Hyperboloid logarithmic map.
        """
        alpha = (-self._lorentz_inner(base, x) / (self.radius**2)).clamp_min(1.0)

        theta = torch.acosh(alpha)

        sinh_theta = torch.sinh(theta)

        eps = torch.finfo(x.dtype).eps

        ratio = torch.where(
            theta.abs() < 1e-7,
            1.0 - theta.square() / 6.0,
            theta / sinh_theta.clamp_min(eps),
        )

        tangent_direction = x - alpha[..., None] * base

        return ratio[..., None] * tangent_direction

    def _exp(self, base, v):
        """
        Hyperboloid exponential map.
        """
        norm_squared = self._lorentz_inner(v, v).clamp_min(0.0)

        norm = torch.sqrt(norm_squared)

        theta = norm / self.radius

        eps = torch.finfo(v.dtype).eps

        sinhc = torch.where(
            theta.abs() < 1e-7,
            1.0 + theta.square() / 6.0,
            torch.sinh(theta) / theta.clamp_min(eps),
        )

        result = torch.cosh(theta)[..., None] * base + sinhc[..., None] * v

        if not torch.isfinite(result).all():
            raise FloatingPointError(
                "Hyperbolic exponential map overflowed. "
                "The tangent radius is too large. "
                "Reduce `target_rms` and/or `strength`."
            )

        # Re-project numerically to upper hyperboloid while
        # preserving spatial coordinates.
        spatial = result[..., 1:]

        time = torch.sqrt(
            self.radius**2
            + spatial.square().sum(
                dim=-1,
                keepdim=True,
            )
        )

        return torch.cat(
            [time, spatial],
            dim=-1,
        )

    def _tangent_norm(self, v):
        return torch.sqrt(self._lorentz_inner(v, v).clamp_min(0.0))

    # -----------------------------------------------------------------
    # Location estimation
    # -----------------------------------------------------------------

    def _lorentz_centroid(self, x):
        """
        Stable normalized ambient Lorentz centroid.

        This is the default because it tends to be substantially
        simpler and more stable than repeatedly running log/exp
        iterations.
        """
        mean = x.mean(dim=0)

        norm_squared = -self._lorentz_inner(
            mean,
            mean,
        )

        if norm_squared.item() <= 0:
            raise RuntimeError("Could not construct a valid " "Lorentz centroid.")

        mean = self.radius * mean / torch.sqrt(norm_squared)

        if mean[0].item() < 0:
            mean = -mean

        return mean

    def _frechet_mean(self, x):
        """
        Intrinsic Frechet/Karcher mean.

        Starts from the Lorentz centroid and performs Riemannian
        gradient iterations with backtracking.
        """
        mean = self._lorentz_centroid(x)

        for _ in range(self.max_iter):
            tangent = self._log(
                mean,
                x,
            )

            delta = tangent.mean(dim=0)

            delta_norm = self._tangent_norm(delta)

            if delta_norm.item() < self.tol:
                break

            old_objective = (
                self._lorentz_inner(
                    tangent,
                    tangent,
                )
                .clamp_min(0.0)
                .mean()
            )

            step = 1.0
            accepted = False

            for _ in range(12):
                candidate = self._exp(
                    mean,
                    step * delta,
                )

                candidate_tangent = self._log(
                    candidate,
                    x,
                )

                new_objective = (
                    self._lorentz_inner(
                        candidate_tangent,
                        candidate_tangent,
                    )
                    .clamp_min(0.0)
                    .mean()
                )

                if new_objective <= old_objective:
                    mean = candidate
                    accepted = True
                    break

                step *= 0.5

            if not accepted:
                break

        return mean

    # -----------------------------------------------------------------
    # Exact Lorentz boost mean -> canonical origin
    # -----------------------------------------------------------------

    def _make_centering_isometry(self, mean):
        """
        Lorentz boost B satisfying

            B mean = [R, 0, ..., 0]

        and

            B^T J B = J.

        Therefore this is an exact hyperbolic isometry.
        """
        time = mean[0]
        spatial = mean[1:]

        spatial_dim = spatial.numel()

        boost = torch.eye(
            spatial_dim + 1,
            device=mean.device,
            dtype=mean.dtype,
        )

        boost[0, 0] = time / self.radius

        boost[0, 1:] = -spatial / self.radius

        boost[1:, 0] = -spatial / self.radius

        denominator = self.radius * (time + self.radius)

        boost[1:, 1:] = (
            torch.eye(
                spatial_dim,
                device=mean.device,
                dtype=mean.dtype,
            )
            + torch.outer(
                spatial,
                spatial,
            )
            / denominator
        )

        return boost

    # -----------------------------------------------------------------
    # Fit
    # -----------------------------------------------------------------

    def fit(self, x) -> "HyperbolicWhiten":
        self._is_fitted = False

        x_t, _ = _to_tensor_preserve_dtype(x)

        if x_t.ndim != 2:
            raise ValueError(
                "HyperbolicWhiten.fit expects " "spatial coordinates [N, d]."
            )

        n, spatial_dim = x_t.shape

        if n < 2:
            raise ValueError("At least two samples are required.")

        if spatial_dim < 1:
            raise ValueError("At least one spatial dimension " "is required.")

        self._components = spatial_dim

        spatial = x_t.detach().to(dtype=_work_dtype(x_t))

        if not torch.isfinite(spatial).all():
            raise ValueError("Input contains NaN or infinite values.")

        with torch.no_grad():
            X = self._to_hyperboloid(spatial)

            if self.mean_method == "lorentz":
                mean = self._lorentz_centroid(X)
            else:
                mean = self._frechet_mean(X)

            boost = self._make_centering_isometry(mean)

            # Intrinsic log vectors at the chosen center.
            tangent = self._log(
                mean,
                X,
            )

            # Exact Lorentz isometry -> canonical origin.
            tangent_at_origin = tangent @ boost.T

            # At [R,0,...], tangent time coordinate is zero,
            # and spatial coordinates form an orthonormal
            # Riemannian frame.
            z = tangent_at_origin[:, 1:]

            (
                tangent_mean,
                transform,
                eigenvalues,
                effective_rms,
            ) = _fit_tangent_transform(
                z,
                strength=self.strength,
                shrinkage=self.shrinkage,
                min_eigenvalue_ratio=(self.min_eigenvalue_ratio),
                target_rms=self.target_rms,
            )

            self._mean = mean.detach()

            self._centering_isometry = boost.detach()

            self._tangent_mean = tangent_mean.detach()

            self._tangent_transform = transform.detach()

            self._eigenvalues = eigenvalues.detach()

            self._effective_rms = effective_rms.detach()

        self._is_fitted = True

        return self

    # -----------------------------------------------------------------
    # Project
    # -----------------------------------------------------------------

    def project(self, x):
        self._check_is_fitted()

        x_t, meta = _to_tensor_preserve_dtype(x)

        if x_t.shape[-1] != self._components:
            raise ValueError(
                f"Expected {self._components} "
                f"spatial dimensions, got "
                f"{x_t.shape[-1]}."
            )

        spatial = x_t.to(dtype=_work_dtype(x_t))

        if not torch.isfinite(spatial).all():
            raise ValueError("Input contains NaN or infinite values.")

        # Spatial representation -> hyperboloid.
        X = self._to_hyperboloid(spatial)

        mean = self._mean.to(X)

        boost = self._centering_isometry.to(X)

        tangent_mean = self._tangent_mean.to(X)

        transform = self._tangent_transform.to(X)

        # x -> log_mu(x)
        tangent = self._log(
            mean,
            X,
        )

        # T_mu H -> T_o H
        tangent_at_origin = tangent @ boost.T

        z = tangent_at_origin[..., 1:] - tangent_mean

        z_out = z @ transform

        # At canonical origin:
        #
        #     T_o H = {[0, z_1, ..., z_d]}
        tangent_out = torch.zeros(
            *z_out.shape[:-1],
            self._components + 1,
            device=z_out.device,
            dtype=z_out.dtype,
        )

        tangent_out[..., 1:] = z_out

        origin = self._origin(
            self._components,
            device=X.device,
            dtype=X.dtype,
        )

        result = self._exp(
            origin,
            tangent_out,
        )

        # User-facing representation remains spatial-only.
        result = self._to_spatial(result)

        return self._to_output(
            result,
            meta,
        )

    # -----------------------------------------------------------------
    # Diagnostics
    # -----------------------------------------------------------------

    @property
    def effective_rms_(self):
        self._check_is_fitted()
        return self._effective_rms.item()

    @property
    def eigenvalues_(self):
        self._check_is_fitted()
        return self._eigenvalues.detach().cpu().numpy()
