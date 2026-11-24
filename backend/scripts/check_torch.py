"""Fail loudly if torch/torchvision are not a working pair (run at image build time and by the audit).

Checks: versions and build type (CPU/CUDA local tag) of torch and torchvision match, torchvision's compiled
operators are registered (torchvision.ops.nms on CPU), and segmentation_models_pytorch/timm import.
"""
import json
import sys


def check() -> dict:
    out: dict = {"ok": False}
    import torch
    import torchvision
    out.update(torch=torch.__version__, torchvision=torchvision.__version__, cuda_build=torch.version.cuda)
    tag = lambda v: v.split("+", 1)[1] if "+" in v else "pypi"  # noqa: E731
    out["build_types"] = [tag(torch.__version__), tag(torchvision.__version__)]
    if out["build_types"][0] != out["build_types"][1]:
        out["error"] = f"torch is {out['build_types'][0]} but torchvision is {out['build_types'][1]} - mixed builds"
        return out
    keep = torchvision.ops.nms(torch.tensor([[0.0, 0.0, 1.0, 1.0]]), torch.tensor([1.0]), 0.5)
    out["nms"] = keep.tolist()
    import segmentation_models_pytorch as smp
    import timm
    out.update(timm=timm.__version__, smp=smp.__version__)
    m = smp.DeepLabV3Plus(encoder_name="resnet18", encoder_weights=None, classes=4).eval()
    with torch.no_grad():
        y = m(torch.zeros(1, 3, 64, 64))
    out["smp_forward"] = list(y.shape)
    out["ok"] = keep.tolist() == [0] and list(y.shape) == [1, 4, 64, 64]
    return out


if __name__ == "__main__":
    try:
        r = check()
    except Exception as e:  # noqa: BLE001
        r = {"ok": False, "error": f"{type(e).__name__}: {e}"}
    print(json.dumps(r))
    sys.exit(0 if r.get("ok") else 1)
