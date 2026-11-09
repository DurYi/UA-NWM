"""Recover the released action normalizer without inventing training statistics."""
import json
from pathlib import Path
import torch

root = Path(__file__).resolve().parents[2]
destination = root / "pretrained/ua-nwm/action_stats.json"
if destination.exists():
    print(f"Keeping existing {destination}")
else:
    checkpoint = torch.load(root / "pretrained/ua-nwm/ua-nwm_best.pt", map_location="cpu", weights_only=False)
    stats = checkpoint.get("action_stats")
    if not isinstance(stats, dict) or not all(k in stats for k in ("mean", "std")):
        raise RuntimeError("Checkpoint has no action statistics; obtain the original training statistics.")
    destination.write_text(json.dumps(stats, indent=2) + "\n")
    print(f"Recovered checkpoint action_stats to {destination}")
