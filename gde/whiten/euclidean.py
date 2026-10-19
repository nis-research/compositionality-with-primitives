from sklearn.decomposition import PCA

from .base import ManifoldWhiten


class EuclideanWhiten(ManifoldWhiten):

    def __init__(self):
        super().__init__()
        self._pca = None
        self._mean = None

    def find_ambient_mean(self, x):
        return x.mean(dim=(0,), keepdim=False)

    def log_map(self, mean, x):
        return x - mean.view(-1, x.shape[-1])

    def exp_map(self, mean, x):
        return x + mean.view(-1, x.shape[-1])

    def fit(self, x) -> "ManifoldWhiten":
        self._components = x.shape[-1]
        self._mean = self.find_ambient_mean(x)
        x_euc = self.log_map(self._mean, x)
        self._pca = PCA(self._components, whiten=True)
        x_np, _ = self._to_numpy(x_euc)
        self._pca.fit(x_euc)
        self._is_fitted = True
        return self

    def project(self, x):
        self._check_is_fitted()
        x_euc = self.log_map(self._mean, x)
        x_euc_np, meta = self._to_numpy(x_euc)
        x_whitened = self._pca.transform(x_euc_np)
        x_whitened_t = self._to_output(x_whitened, meta)
        x_exp = self.exp_map(self._mean, x_whitened_t)
        return x_exp
