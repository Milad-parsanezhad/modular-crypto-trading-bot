from __future__ import annotations

from pathlib import Path
import json

import numpy as np
from PIL import Image

from research_bot.vision_ict_v22 import CHANNEL_NAMES


def export_channel_preview(tensor: np.ndarray, metadata: dict, output_dir: str | Path, scale: int = 3) -> dict:
    """Export a human-audit preview without changing the model tensor.

    The preview is explicitly outside the ML input path. Channel names are stored
    in JSON, not burned into pixels, so this utility cannot accidentally create
    a text-bearing training image.
    """
    if tensor.ndim != 3 or tensor.shape[0] != len(CHANNEL_NAMES):
        raise ValueError(f"expected [{len(CHANNEL_NAMES)},H,W], got {tensor.shape}")
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    channel_paths = []
    images = []
    for i, name in enumerate(CHANNEL_NAMES):
        arr = np.clip(tensor[i], 0.0, 1.0)
        img = Image.fromarray(np.rint(arr * 255).astype(np.uint8), mode="L")
        if scale != 1:
            img = img.resize((img.width * scale, img.height * scale), resample=Image.Resampling.NEAREST)
        path = out / f"channel_{i:02d}_{name}.png"
        img.save(path)
        channel_paths.append(path.name)
        images.append(img)

    rows, cols = 2, 4
    w, h = images[0].size
    montage = Image.new("L", (cols * w, rows * h), color=0)
    for i, img in enumerate(images):
        montage.paste(img, ((i % cols) * w, (i // cols) * h))
    montage_path = out / "multichannel_montage.png"
    montage.save(montage_path)

    manifest = {
        "model_input_unchanged": True,
        "preview_is_not_training_input": True,
        "montage_order": list(CHANNEL_NAMES),
        "channel_files": channel_paths,
        "montage_file": montage_path.name,
        "source_metadata": metadata,
    }
    (out / "preview_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest
