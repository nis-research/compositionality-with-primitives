import numpy as np

from sklearn.cross_decomposition import CCA


class CrossCorrelationEstimator:

    def estimate(self, x: np.ndarray, y: np.ndarray) -> float:
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)

        if x.shape[0] != y.shape[0]:
            raise ValueError(
                f"X and Y must have the same number of rows (paired samples), "
                f"got {x.shape[0]} and {y.shape[0]}"
            )

        # Center each column (feature) to remove means, as with Pearson correlation.
        xc = x - x.mean(axis=0, keepdims=True)
        yc = y - y.mean(axis=0, keepdims=True)

        # Cross-covariance and within-set covariance matrices.
        sxy = xc.T @ yc  # (d_x, d_y)
        sxx = xc.T @ xc  # (d_x, d_x)
        syy = yc.T @ yc  # (d_y, d_y)

        numerator = np.trace(sxy @ sxy.T)
        denominator = np.sqrt(np.trace(sxx @ sxx) * np.trace(syy @ syy))

        if denominator == 0:
            return float("nan")

        return float(numerator / denominator)


class CCAEstimator:

    # reproduced from bench mi as bench mi had broken jax dependencies.

    def corr(self, x: np.ndarray, y: np.ndarray) -> float:
        return np.corrcoef(x, y)[0, 1]

    def mi_gauss(self, correlation: float) -> float:
        return -0.5 * np.log(1 - correlation**2 + 1e-9)

    def estimate(self, x: np.ndarray, y: np.ndarray) -> float:
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)

        if x.shape[0] != y.shape[0]:
            raise ValueError(
                f"X and Y must have the same number of rows (paired samples), "
                f"got {x.shape[0]} and {y.shape[0]}"
            )

        # sklearn's CCA needs at least 2 samples, and n_components can't
        # exceed the smaller of (n_samples, d_x, d_y).
        n_components = min(x.shape[0], x.shape[1], y.shape[1])
        if n_components < 1:
            return float("nan")

        cca = CCA(n_components=n_components)
        x_c, y_c = cca.fit_transform(x, y)

        # Per-component Pearson correlation between the canonical variates,
        # averaged into a single score.
        correlations = []
        for i in range(n_components):
            xi, yi = x_c[:, i], y_c[:, i]
            if xi.std() == 0 or yi.std() == 0:
                continue
            correlations.append(self.corr(xi, yi))

        return sum([self.mi_gauss(corr) for corr in correlations])
