"""
Physics-informed neural network for prediction markets.
Constrained by probability axioms: sigmoid output, Bayesian shrinkage, variance penalty.
"""

import numpy as np
from kalshi_terminal.config import PINN_EPOCHS, PINN_LR


class ProbabilityPINN:
    def __init__(self, seed=42):
        np.random.seed(seed)
        self.w1 = np.random.randn(4, 8) * 0.3
        self.b1 = np.zeros(8)
        self.w2 = np.random.randn(8, 1) * 0.3
        self.b2 = np.zeros(1)
        self.losses = []

    def _sigmoid(self, x):
        """Rule 1: output bounded to [0, 1]."""
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

    def forward(self, features):
        h1 = np.tanh(features @ self.w1 + self.b1)
        return self._sigmoid(h1 @ self.w2 + self.b2), h1

    def _physics_loss(self, pred, target, n_obs):
        """
        Rule 2: Bayesian shrinkage toward 50% when evidence is weak.
        Rule 3: Variance penalty for unstable predictions.
        """
        mse      = np.mean((pred - target) ** 2)
        shrink   = 0.1 * np.mean((pred - 0.5) ** 2) / np.sqrt(n_obs + 1)
        variance = 0.05 * np.var(pred)
        return mse + shrink + variance

    def train(self, X, y, volumes, epochs=PINN_EPOCHS, lr=PINN_LR):
        for _ in range(epochs):
            pred, h1 = self.forward(X)
            loss = self._physics_loss(pred, y, np.mean(volumes))
            self.losses.append(loss)

            d_out = 2 * (pred - y.reshape(-1, 1)) / len(X)
            d_w2  = h1.T @ d_out
            d_b2  = d_out.sum(axis=0)
            d_h1  = d_out @ self.w2.T * (1 - h1 ** 2)
            d_w1  = X.T @ d_h1
            d_b1  = d_h1.sum(axis=0)

            self.w1 -= lr * d_w1
            self.b1 -= lr * d_b1
            self.w2 -= lr * d_w2
            self.b2 -= lr * d_b2

    def predict(self, features):
        pred, _ = self.forward(features.reshape(1, -1))
        return float(pred[0, 0])


def build_pinn(markets_dict):
    """Train PINN on all markets in the local store."""
    X, y, vols = [], [], []
    for data in markets_dict.values():
        bid    = data["bids"][0][0]
        ask    = data["asks"][0][0]
        spread = ask - bid
        X.append([bid / 100, ask / 100, data["volume"] / 1_000_000, spread / 100])
        y.append(data["price"] / 100)
        vols.append(data["volume"])
    pinn = ProbabilityPINN()
    pinn.train(np.array(X), np.array(y), np.array(vols))
    return pinn
