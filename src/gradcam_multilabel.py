import argparse
import os

import numpy as np
import torch
from PIL import Image
import matplotlib.cm as cm

from data.chexpert_multilabel import build_transforms
from data.lung_mask import LungMasker
from models.resnet18_multilabel import build_resnet18_multilabel, load_checkpoint


def _select_device(device_arg: str) -> torch.device:
    if device_arg != "auto":
        return torch.device(device_arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _overlay_cam(img: Image.Image, cam: np.ndarray, alpha: float = 0.4) -> Image.Image:
    cam = np.clip(cam, 0, 1)
    heatmap = cm.jet(cam)[..., :3]
    heatmap = (heatmap * 255).astype(np.uint8)
    heatmap = Image.fromarray(heatmap).resize(img.size, resample=Image.BILINEAR)
    heatmap = heatmap.convert("RGB")
    return Image.blend(img.convert("RGB"), heatmap, alpha=alpha)


def main() -> None:
    parser = argparse.ArgumentParser(description="Grad-CAM for CheXpert multi-label model.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--label", default=None)
    parser.add_argument("--label_index", type=int, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--lungmask_model", default=None)
    parser.add_argument("--lungmask_device", default="cpu")
    args = parser.parse_args()

    device = _select_device(args.device)
    model = build_resnet18_multilabel(num_labels=14, pretrained=False).to(device)
    checkpoint = load_checkpoint(model, args.checkpoint, device)
    label_names = checkpoint.get("label_names")
    if label_names is None:
        raise ValueError("Checkpoint missing label_names; retrain or update checkpoint.")
    if len(label_names) != model.fc.out_features:
        model = build_resnet18_multilabel(num_labels=len(label_names), pretrained=False).to(device)
        load_checkpoint(model, args.checkpoint, device)
    model.eval()

    if args.label_index is not None:
        label_index = args.label_index
    elif args.label is not None:
        if args.label not in label_names:
            raise ValueError(f"Label '{args.label}' not found in label_names.")
        label_index = label_names.index(args.label)
    else:
        raise ValueError("Provide --label or --label_index.")

    target_layer = model.layer4
    activations = []
    gradients = []

    def _forward_hook(_, __, output):
        activations.append(output)

    def _backward_hook(_, grad_in, grad_out):
        gradients.append(grad_out[0])

    handle_fwd = target_layer.register_forward_hook(_forward_hook)
    handle_bwd = target_layer.register_full_backward_hook(_backward_hook)

    img = Image.open(args.image).convert("L")
    masker = LungMasker(model_path=args.lungmask_model, device=args.lungmask_device)
    img = masker.apply(img)
    original = img.copy().convert("RGB")
    x = build_transforms()(img).unsqueeze(0).to(device)
    logits = model(x)
    score = logits[0, label_index]
    model.zero_grad()
    score.backward()

    handle_fwd.remove()
    handle_bwd.remove()

    act = activations[0]
    grad = gradients[0]
    weights = grad.mean(dim=(2, 3), keepdim=True)
    cam = torch.relu((weights * act).sum(dim=1)).squeeze(0)
    cam = cam.detach().cpu().numpy()
    cam = cam - cam.min()
    cam = cam / (cam.max() + 1e-6)

    overlay = _overlay_cam(original, cam)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    overlay.save(args.out)
    print(f"Saved Grad-CAM to {args.out}")


if __name__ == "__main__":
    main()
