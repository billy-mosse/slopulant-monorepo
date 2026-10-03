"""LinUCB contextual bandit (disjoint linear model per arm, Li et al. 2010).

For each arm a we keep
    A_a = I*lambda + sum x x^T      (d x d)
    b_a = sum r x                   (d)
theta_a = A_a^{-1} b_a is the ridge-regression estimate of expected reward, and the score is
    p_a(x) = theta_a^T x + alpha * sqrt(x^T A_a^{-1} x)
The second term is the upper-confidence bonus that drives exploration.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

ARMS = ["free_shipping", "pct10_bedding", "bundle_discount", "none"]


class LinUCB:
    def __init__(self, n_features: int, arms: list[str] = ARMS, alpha: float = 0.5, ridge: float = 1.0):
        self.arms = list(arms)
        self.d = n_features
        self.alpha = alpha
        self.A = np.stack([np.eye(n_features) * ridge for _ in self.arms])     # (K, d, d)
        self.b = np.zeros((len(self.arms), n_features))                        # (K, d)
        self._A_inv = np.linalg.inv(self.A)

    # --- learning -------------------------------------------------------------
    def update(self, X: np.ndarray, arm_idx: np.ndarray, rewards: np.ndarray) -> None:
        """Batch update from logged (context, arm, reward) triples."""
        for k in range(len(self.arms)):
            mask = arm_idx == k
            if not mask.any():
                continue
            Xk = X[mask]
            self.A[k] += Xk.T @ Xk
            self.b[k] += Xk.T @ rewards[mask]
        self._A_inv = np.linalg.inv(self.A)

    @property
    def theta(self) -> np.ndarray:
        return np.einsum("kij,kj->ki", self._A_inv, self.b)                    # (K, d)

    # --- scoring --------------------------------------------------------------
    def scores(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return (mean estimate, ucb) arrays of shape (n, K)."""
        mean = X @ self.theta.T
        # x^T A^{-1} x for every row and arm
        width = np.sqrt(np.einsum("ni,kij,nj->nk", X, self._A_inv, X).clip(min=0))
        return mean, mean + self.alpha * width

    def choose(self, X: np.ndarray) -> np.ndarray:
        _, ucb = self.scores(X)
        return ucb.argmax(axis=1)

    def save(self, path: str) -> None:
        np.savez(path, A=self.A, b=self.b, alpha=self.alpha, arms=np.array(self.arms))

    @classmethod
    def load(cls, path: str) -> "LinUCB":
        z = np.load(path, allow_pickle=False)
        m = cls(z["A"].shape[1], list(z["arms"]), float(z["alpha"]))
        m.A, m.b = z["A"], z["b"]
        m._A_inv = np.linalg.inv(m.A)
        return m


def build_context(engine) -> pd.DataFrame:
    """Context = customer vector + one-hot segment + predicted uplift per offer, plus bias."""
    emb = pd.read_sql(
        "SELECT * FROM features.customer_embeddings "
        "WHERE snapshot_date = (SELECT MAX(snapshot_date) FROM features.customer_embeddings)",
        engine,
    ).set_index("customer_id")
    emb = emb[[c for c in emb.columns if c.startswith("emb_")]]

    seg = pd.read_sql("SELECT customer_id, segment_name FROM marketing.segments", engine).set_index("customer_id")
    seg_oh = pd.get_dummies(seg["segment_name"], prefix="seg", dtype=float)

    upl = pd.read_sql("SELECT customer_id, promo_type, predicted_uplift FROM marketing.promo_uplift", engine)
    upl = upl.pivot_table(index="customer_id", columns="promo_type", values="predicted_uplift", aggfunc="mean")
    upl.columns = [f"uplift_{c}" for c in upl.columns]

    ctx = emb.join(seg_oh, how="left").join(upl, how="left").fillna(0.0)
    # embeddings are already unit-scale; normalise rows so the UCB width is comparable across customers
    ctx = ctx.div(np.linalg.norm(ctx.to_numpy(), axis=1).clip(min=1e-9), axis=0)
    ctx["bias"] = 1.0
    return ctx
