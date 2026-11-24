"""Export a trained checkpoint to ONNX for the browser app, with Grad-CAM computed in-graph.

ONNX Runtime Web can't backpropagate, so we compute exact Grad-CAM analytically inside the
forward pass. Outputs are probs [1,5] and cams [1,5,h,w], with each cam normalized to [0,1].
  * MobileNetV3 head (pool -> 1x1 conv_head -> hardswish -> linear): the gradient of each logit
    with respect to the pooled features is  (W_cls[k] * hardswish'(z)) @ W_head, which is exact.
  * Pool -> linear heads (EfficientNet, ResNet, DenseNet, ...): Grad-CAM reduces to classic CAM.

    python export/export_onnx.py --ckpt ../chexlite/checkpoints/best.pt --out app/model/chexlite.onnx
"""
import argparse
import json
from pathlib import Path

import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
REPO = HERE.parent


class CamModel(nn.Module):
    def __init__(self, m: nn.Module):
        super().__init__()
        self.m = m
        self.kind = "mbv3" if type(m).__name__ == "MobileNetV3" else "cam"

    def forward(self, x):
        A = self.m.forward_features(x)  # [B, C, h, w]
        p = A.mean(dim=(2, 3))
        if self.kind == "mbv3":
            Wh = self.m.conv_head.weight.flatten(1)  # [D, C]
            z = p @ Wh.t() + self.m.conv_head.bias
            logits = self.m.classifier(F.hardswish(z))
            dh = torch.where(z < -3, torch.zeros_like(z), torch.where(z > 3, torch.ones_like(z), (2 * z + 3) / 6))
            g = (self.m.classifier.weight.unsqueeze(0) * dh.unsqueeze(1)) @ Wh  # [B, K, C]
        else:
            fc = self.m.get_classifier()
            logits = fc(p)
            g = fc.weight.unsqueeze(0).expand(x.shape[0], -1, -1)
        cam = torch.relu(torch.einsum("bkc,bchw->bkhw", g, A))
        cam = cam / (cam.amax(dim=(2, 3), keepdim=True) + 1e-6)
        return torch.sigmoid(logits), cam


def gradcam_reference(m, x, k):
    """Autograd Grad-CAM, used only to verify the analytic version."""
    A = m.forward_features(x).detach().requires_grad_(True)
    m.forward_head(A)[0, k].backward()
    w = A.grad.mean(dim=(2, 3), keepdim=True)
    cam = torch.relu((w * A).sum(1))[0]
    return (cam / (cam.max() + 1e-6)).detach()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(REPO.parent / "chexlite" / "checkpoints" / "best.pt"))
    ap.add_argument("--out", default=str(REPO / "app" / "model" / "chexlite.onnx"))
    a = ap.parse_args()

    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    cfg = ck["cfg"]
    labels = [l for l in cfg["uncertain_policy"]]
    m = timm.create_model(cfg["arch"], pretrained=False, num_classes=len(labels))
    m.load_state_dict(ck["state_dict"])
    m.eval()
    w = CamModel(m).eval()

    s = cfg["img_size"]
    x = torch.randn(1, 3, s, s)
    with torch.no_grad():
        probs, cams = w(x)
        ref = torch.sigmoid(m(x))
    assert torch.allclose(probs, ref, atol=1e-5), "wrapper logits differ from model; unsupported head"
    for k in range(len(labels)):
        r = gradcam_reference(m, x, k)
        assert torch.allclose(cams[0, k], r, atol=1e-4), f"analytic Grad-CAM mismatch for {labels[k]}"
    print(f"verified: probs match model, analytic Grad-CAM matches autograd for {len(labels)} labels ({w.kind})")

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(w, (x,), str(out), input_names=["image"], output_names=["probs", "cams"],
                      opset_version=18, dynamo=False)

    import onnxruntime as ort
    sess = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
    op, oc = sess.run(None, {"image": x.numpy()})
    print(f"onnx max |dprob| {np.abs(op - probs.numpy()).max():.2e}  max |dcam| {np.abs(oc - cams.numpy()).max():.2e}")
    assert np.abs(op - probs.numpy()).max() < 1e-4

    meta_path = out.parent / "meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    meta.update({"labels": labels, "img_size": s, "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225],
                 "arch": cfg["arch"], "cam_size": list(oc.shape[2:]), "onnx_bytes": out.stat().st_size})
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB) and {meta_path}")


if __name__ == "__main__":
    main()
