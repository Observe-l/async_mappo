"""RUL prediction utilities for the scheduling environment.

This module previously contained a TensorFlow GCPATr model implementation.
It has been replaced with a lightweight PyTorch Transformer predictor that
loads the trained CMAPSS FD001 checkpoint produced by `train_cmapss_transformer.py`.

Public API intentionally preserved:
  - class `predictor` with method `predict(obs)`

`obs` can be either:
- a list of per-timestep observations, each shaped (F,) or (1, F)
- a numpy array shaped (T, F)

The Transformer model trained in this repo expects input shaped (B, T, F)
with T=lookback_window (default 40) and F=18.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional, Sequence, Union

import numpy as np


def _default_artifact_dir() -> str:
    # /onpolicy/envs/transformer_schedule/rul_gen.py -> repo root is 3 levels up
    repo_root = Path(__file__).resolve().parents[3]

    # Prefer an explicit env var.
    env = os.environ.get("CMAPSS_FD001_ARTIFACT_DIR")
    if env:
        p = Path(env)
        if not p.is_absolute():
            p = repo_root / p
        return str(p)

    # Common local defaults.
    candidates = [
        repo_root / "artifacts/cmapss_fd001_transformer",
        repo_root / "artifacts/_smoke_fd001_default_env",
    ]
    for c in candidates:
        if Path(c).exists():
            return str(c)
    return str(repo_root / "artifacts/cmapss_fd001_transformer")


class predictor:
    """PyTorch Transformer-based FD001 RUL predictor."""

    def __init__(
        self,
        lookback_window: int = 40,
        artifact_dir: Optional[str] = None,
        device: Optional[str] = None,
        apply_scaler: Optional[bool] = None,
    ) -> None:
        self.lw = int(lookback_window)
        self.apply_scaler = apply_scaler
        if artifact_dir is None:
            self.artifact_dir = _default_artifact_dir()
        else:
            repo_root = Path(__file__).resolve().parents[3]
            p = Path(artifact_dir)
            if not p.is_absolute():
                p = repo_root / p
            self.artifact_dir = str(p)

        try:
            import torch
        except Exception as e:  # pragma: no cover
            raise RuntimeError(
                "PyTorch is required for the FD001 predictor. "
                "Install torch or switch predictor implementation."
            ) from e

        self.torch = torch
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        # Load model + scaler from our training artifacts.
        try:
            # Prefer a package-relative import to avoid PYTHONPATH/cwd issues.
            from .load_cmapss_transformer import load_checkpoint
        except Exception as e:  # pragma: no cover
            raise RuntimeError(
                "Could not import load_cmapss_transformer. "
                "Ensure the 'onpolicy.envs.transformer_schedule' package is importable."
            ) from e

        self.model, self.scaler, self.meta = load_checkpoint(self.artifact_dir, device=self.device)
        # Training feature dimension (usually 18)
        try:
            self.input_dim = int(len(self.meta.get("feature_columns", [])))
        except Exception:
            self.input_dim = 18
        if self.input_dim <= 0:
            self.input_dim = 18

    def _to_window(self, obs: Union[Sequence[np.ndarray], np.ndarray]) -> np.ndarray:
        # Normalize input into an array of shape (T, F)
        if isinstance(obs, np.ndarray):
            x = obs
        else:
            rows = []
            for o in obs:
                a = np.asarray(o)
                if a.ndim == 2 and a.shape[0] == 1:
                    a = a[0]
                rows.append(a)
            x = np.stack(rows, axis=0)

        x = np.asarray(x, dtype=np.float32)
        if x.ndim != 2:
            raise ValueError(f"obs must be (T,F), got shape={x.shape}")
        if x.shape[1] != self.input_dim:
            raise ValueError(
                f"obs feature dim mismatch: got F={x.shape[1]}, expected F={self.input_dim}. "
                "Make sure engine observations use the same 18 features as training."
            )

        # Take the last lw steps (same as training window length).
        if x.shape[0] < self.lw:
            return x
        return x[-self.lw :]

    def predict(self, obs) -> float:
        """Return scalar RUL prediction. If not enough history, returns 125.0."""
        if obs is None or len(obs) < self.lw:
            return 125.0

        x = self._to_window(obs)
        if x.shape[0] < self.lw:
            return 125.0

        # Apply the same normalization as training.
        # In this project, truck engine observations are often already min-max scaled to ~[0, 1].
        # Applying MinMaxScaler again would distort them and can cause pathological predictions.
        if self.apply_scaler is None:
            already_scaled = (x.min() >= -1e-3) and (x.max() <= 1.0 + 1e-3)
        else:
            already_scaled = not self.apply_scaler

        if already_scaled:
            x_scaled = x.astype(np.float32)
        else:
            x_scaled = self.scaler.transform(x.astype(np.float32))

        xb = self.torch.from_numpy(x_scaled).unsqueeze(0).float().to(self.device)
        with self.torch.no_grad():
            y = self.model(xb).detach().cpu().numpy()[0, 0]
        return float(y)