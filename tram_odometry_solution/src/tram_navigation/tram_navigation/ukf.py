import numpy as np

class UnscentedKalmanFilter:
    """Универсальный фильтр Калмана (UKF) для аддитивного шума."""
    def __init__(
        self,
        dim_x: int,
        dim_z: int,
        alpha: float = 1.0,
        beta: float = 2.0,
        kappa: float = 0.0,
    ):
        self.n = int(dim_x)
        self.m = int(dim_z)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.kappa = float(kappa)

        self.lam = self.alpha * self.alpha * (self.n + self.kappa) - self.n
        self.x = np.zeros(self.n, dtype=float)
        self.P = np.eye(self.n, dtype=float)

        self.x_pred = self.x.copy()
        self.P_pred = self.P.copy()
        self._sigmas_f = None

        self.wm, self.wc = self._weights()

    def _weights(self):
        n = self.n
        lam = self.lam
        wm = np.full(2 * n + 1, 0.5 / (n + lam), dtype=float)
        wc = wm.copy()
        wm[0] = lam / (n + lam)
        wc[0] = lam / (n + lam) + (1.0 - self.alpha * self.alpha + self.beta)
        return wm, wc

    def _sigma_points(self, x, P):
        n = self.n
        A = (n + self.lam) * P
        A = 0.5 * (A + A.T)

        jitter = 1.0e-9
        L = None
        for _ in range(6):
            try:
                L = np.linalg.cholesky(A + jitter * np.eye(n))
                break
            except np.linalg.LinAlgError:
                jitter *= 10.0

        if L is None:
            eigval, eigvec = np.linalg.eigh(A)
            eigval = np.maximum(eigval, 1.0e-9)
            L = eigvec @ np.diag(np.sqrt(eigval))

        pts = np.zeros((2 * n + 1, n), dtype=float)
        pts[0] = x
        for i in range(n):
            pts[i + 1] = x + L[:, i]
            pts[n + i + 1] = x - L[:, i]
        return pts

    def predict(self, f, dt, u, Q):
        sigmas = self._sigma_points(self.x, self.P)
        self._sigmas_f = np.array([f(s, u, dt) for s in sigmas], dtype=float)
        self.x_pred = np.sum(self.wm[:, None] * self._sigmas_f, axis=0)

        P = np.zeros((self.n, self.n), dtype=float)
        for i in range(2 * self.n + 1):
            dx = self._sigmas_f[i] - self.x_pred
            P += self.wc[i] * np.outer(dx, dx)

        self.P_pred = P + Q
        self.P_pred = 0.5 * (self.P_pred + self.P_pred.T)
        self.x = self.x_pred.copy()
        self.P = self.P_pred.copy()

    def update(self, z, R, h):
        if self._sigmas_f is None:
            raise RuntimeError("predict() must be called before update().")

        sigmas_h = np.array([h(s) for s in self._sigmas_f], dtype=float)
        y_pred = np.sum(self.wm[:, None] * sigmas_h, axis=0)

        S = np.zeros((self.m, self.m), dtype=float)
        Pxy = np.zeros((self.n, self.m), dtype=float)

        for i in range(2 * self.n + 1):
            dy = sigmas_h[i] - y_pred
            S += self.wc[i] * np.outer(dy, dy)
            dx = self._sigmas_f[i] - self.x_pred
            Pxy += self.wc[i] * np.outer(dx, dy)

        S += R
        S = 0.5 * (S + S.T)
        z = np.asarray(z, dtype=float)
        innovation = z - y_pred

        try:
            K = np.linalg.solve(S.T, Pxy.T).T
        except np.linalg.LinAlgError:
            K = Pxy @ np.linalg.pinv(S)

        self.x = self.x_pred + K @ innovation
        self.P = self.P_pred - K @ S @ K.T
        self.P = 0.5 * (self.P + self.P.T)
        self.P += 1.0e-12 * np.eye(self.n)
        self._sigmas_f = None
