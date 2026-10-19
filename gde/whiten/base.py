from abc import ABC, abstractmethod

import numpy as np
import torch


class ManifoldWhiten(ABC):

    def __init__(self):
        self._is_fitted = False
        self._components = None

    @abstractmethod
    def fit(self, x) -> "ManifoldWhiten":
        raise NotImplementedError

    @abstractmethod
    def project(self, x):
        raise NotImplementedError

    def _check_is_fitted(self):
        if not self._is_fitted:
            raise RuntimeError(
                f"{self.__class__.__name__} instance is not fitted yet. "
                "Call `fit` before `project`."
            )

    def _work_dtype(self, x: torch.Tensor):
        if x.device.type == "cpu":
            return torch.float64

        if x.dtype in (torch.float16, torch.bfloat16):
            return torch.float32

        return x.dtype

    def _zca_whitener(self, z, relative_eigenvalue_floor=1e-10):
        n, d = z.shape

        if n <= 1:
            raise ValueError("At least two samples are required.")

        covariance = z.T @ z / (n - 1)

        eigenvalues, eigenvectors = torch.linalg.eigh(covariance)

        eps = torch.finfo(z.dtype).eps

        scale = eigenvalues.mean().clamp_min(eps)

        floor = relative_eigenvalue_floor * scale
        floor = floor.clamp_min(eps)

        safe_eigenvalues = eigenvalues.clamp_min(floor)

        whitening = eigenvectors @ torch.diag(safe_eigenvalues.rsqrt()) @ eigenvectors.T

        return whitening, eigenvalues

    @staticmethod
    def _to_tensor(x):
        if isinstance(x, torch.Tensor):
            return x, {"was_numpy": False, "device": x.device, "dtype": x.dtype}
        if isinstance(x, np.ndarray):
            meta = {"was_numpy": True, "device": None, "dtype": x.dtype}
            return torch.from_numpy(x), meta
        raise TypeError(
            f"Expected a numpy.ndarray or torch.Tensor, got {type(x).__name__}."
        )

    @staticmethod
    def _to_numpy(x):
        if isinstance(x, torch.Tensor):
            meta = {"was_numpy": False, "device": x.device, "dtype": x.dtype}
            return x.detach().cpu().numpy(), meta
        if isinstance(x, np.ndarray):
            meta = {"was_numpy": True, "device": None, "dtype": x.dtype}
            return x, meta
        raise TypeError(
            f"Expected a numpy.ndarray or torch.Tensor, got {type(x).__name__}."
        )

    @staticmethod
    def _to_output(result, meta: dict):
        if meta["was_numpy"]:
            if isinstance(result, torch.Tensor):
                result = result.detach().cpu().numpy()
            return result.astype(meta["dtype"])
        else:
            if isinstance(result, np.ndarray):
                result = torch.from_numpy(result)
            return result.to(device=meta["device"], dtype=meta["dtype"])
