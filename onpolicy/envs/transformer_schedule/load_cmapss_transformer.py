import json
import pickle
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

from onpolicy.envs.transformer_schedule.cmapss_transformer_model import CmapssTransformerConfig, CmapssTransformerRegressor


def load_checkpoint(artifact_dir: str, device: str = "cpu") -> Tuple[CmapssTransformerRegressor, object, Dict]:
    artifact_path = Path(artifact_dir).expanduser().resolve()

    if not artifact_path.exists():
        raise FileNotFoundError(
            f"CMAPSS artifact_dir not found: {artifact_path}. "
            "Set CMAPSS_FD001_ARTIFACT_DIR or pass artifact_dir explicitly."
        )

    required = ["model.pt", "scaler.pkl", "meta.json"]
    missing = [name for name in required if not (artifact_path / name).exists()]
    if missing:
        raise FileNotFoundError(
            f"CMAPSS artifacts missing in {artifact_path}: {missing}. "
            "Expected files produced by the transformer training script."
        )

    ckpt = torch.load(artifact_path / "model.pt", map_location=device)
    cfg = CmapssTransformerConfig(**ckpt["config"])
    model = CmapssTransformerRegressor(cfg)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device)
    model.eval()

    with open(artifact_path / "scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    with open(artifact_path / "meta.json", "r", encoding="utf-8") as f:
        meta = json.load(f)

    return model, scaler, meta


@torch.no_grad()
def predict_rul(
    model: CmapssTransformerRegressor,
    scaler,
    feature_columns: List[str],
    window: np.ndarray,
    device: str = "cpu",
) -> float:
    """window: (seq_len, num_features) in the *unscaled* feature space."""
    x = scaler.transform(window.astype(np.float32))
    xb = torch.from_numpy(x).unsqueeze(0).float().to(device)
    y = model(xb).cpu().numpy()[0, 0]
    return float(y)
