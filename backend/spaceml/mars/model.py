from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], np.float32)


def build_model(encoder_weights: str | None = "imagenet", encoder: str = "resnet50"):
    """DeepLabV3+ with a ResNet encoder. With ImageNet weights requested but not downloadable
    (offline, blocked mirror) training falls back to a random init instead of failing."""
    import segmentation_models_pytorch as smp

    try:
        return smp.DeepLabV3Plus(encoder_name=encoder, encoder_weights=encoder_weights, classes=4)
    except Exception as e:  # noqa: BLE001
        if encoder_weights is None:
            raise
        print(f"ImageNet weights for {encoder} could not be downloaded ({type(e).__name__}: {str(e)[:160]}); "
              "training from a random initialisation instead (lower accuracy).", flush=True)
        return smp.DeepLabV3Plus(encoder_name=encoder, encoder_weights=None, classes=4)


def local_encoder_weights(encoder: str) -> bool:
    """True if ImageNet weights for this encoder are already on disk (torch hub / HF cache, or a release
    bundle), so pretrained training needs no network. Never downloads."""
    import os
    roots = [Path(os.environ.get("TORCH_HOME", Path.home() / ".cache" / "torch")),
             Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))]
    for r in roots:
        if r.exists() and any(encoder in p.name for p in r.rglob("*") if p.suffix in (".pth", ".safetensors", ".bin")
                              or p.name.startswith(f"models--smp-hub--{encoder}")):
            return True
    return False


def to_tensor(image: Image.Image, size: int) -> torch.Tensor:
    img = image.convert("RGB").resize((size, size), Image.BILINEAR)
    a = (np.asarray(img, np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(a.transpose(2, 0, 1))


def load(version_dir: Path):
    ckpt = torch.load(version_dir / "model.pt", map_location="cpu", weights_only=False)
    model = build_model(encoder_weights=None, encoder=str(ckpt.get("encoder", "resnet50")))
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, int(ckpt.get("img_size", 512))


@torch.no_grad()
def segment(model, image: Image.Image, size: int, device: str = "cpu") -> np.ndarray:
    """Return class ids (H, W) at the original image resolution."""
    x = to_tensor(image, size)[None].to(device)
    logits = model(x)
    logits = F.interpolate(logits, size=(image.height, image.width), mode="bilinear", align_corners=False)
    return logits.argmax(1)[0].cpu().numpy().astype(np.uint8)
