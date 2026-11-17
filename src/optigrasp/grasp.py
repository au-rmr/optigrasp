"""OptiGrasp inference: one RGB crop + its segmentation mask in, a suction grasp pixel and angles out.

Port of the inference wrapper that ran on the AURMR workcell (hardware_inference_comms.py, Soofiyan Atar, 2024; the
empty-mask fallback added Feb 2026 by Paolo). Behaviour is kept, including what is surprising; see docs/porting_notes.md:
- the image is resized to 224 x 224 without keeping its aspect ratio; the grasp pixel is scaled back per axis;
- the grasp pixel is the argmax of the affordance map (the original also ran DBSCAN but discarded its result);
- the mask selects the smallest non-zero label; an all-zero mask selects label 0, i.e. the whole crop.
"""
from dataclasses import dataclass

import numpy as np
import torch
from torchvision import transforms

from .model import DepthAnything

NET_SIZE = 224
# The released weights (hardware_comms_30_c): DINOv2 ViT-S/14 backbone.
CONFIG_30_C = {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384], "use_bn": False,
               "use_clstoken": False}
ROLL_RANGE_DEG, PITCH_RANGE_DEG = 40.0, 20.0   # map value 0..1 -> -range/2..+range/2


@dataclass
class Grasp:
    pixel: np.ndarray          # (x, y) in the input crop, int, as the workcell used it
    pixel_net: np.ndarray      # (x, y) in the 224 x 224 network image
    roll: float                # rad
    pitch: float               # rad
    affordance: np.ndarray     # 224 x 224, 0..1, zero outside the mask
    roll_map: np.ndarray       # 224 x 224, 0..1
    pitch_map: np.ndarray      # 224 x 224, 0..1


def load_model(weights, device="cuda", config=CONFIG_30_C):
    """weights: a state_dict file (torch.load(weights_only=True) safe), e.g. hardware_comms_30_c.state_dict.pth."""
    model = DepthAnything(config)
    model.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True), strict=True)
    return model.to(device).eval()


# antialias=False: torchvision >= 0.17 antialiases tensor resizes by default; the model was trained and validated
# with 0.14 (no antialiasing) and the grasp pixel moves by up to 129 px with it on.
_resize = transforms.Resize((NET_SIZE, NET_SIZE), interpolation=transforms.InterpolationMode.BICUBIC, antialias=False)
_normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


def preprocess(bgr, mask, device="cuda"):
    """bgr: H x W x 3 uint8 (OpenCV order). mask: H x W integer labels. Returns the network image and mask tensors.
    The image is resized on the CPU and the mask on `device`, as in the original (bicubic differs slightly by device)."""
    rgb = np.ascontiguousarray(bgr[..., ::-1]).astype(np.uint8)
    image = _resize(_normalize(torch.from_numpy(rgb).permute(2, 0, 1).float().div(255)))
    labels = np.unique(mask)
    label = labels[1] if len(labels) > 1 else labels[0]
    m = (torch.from_numpy(np.asarray(mask)).to(device) == int(label)).unsqueeze(0).float()
    return image.unsqueeze(0).to(device), _resize(m).unsqueeze(0)


@torch.no_grad()
def predict(model, bgr, mask, device="cuda"):
    """One grasp for one crop. `model` from load_model()."""
    image, m = preprocess(bgr, mask, device)
    _, affordance, roll_map, pitch_map = model(image, m)
    affordance, roll_map, pitch_map = (t[0, 0].cpu().numpy() for t in (affordance, roll_map, pitch_map))
    y, x = np.unravel_index(affordance.argmax(), affordance.shape)
    roll = np.deg2rad((roll_map[y, x] - 0.5) * ROLL_RANGE_DEG)
    pitch = np.deg2rad((pitch_map[y, x] - 0.5) * PITCH_RANGE_DEG)
    h, w = mask.shape[:2]
    pixel = np.array([int(x * w / NET_SIZE), int(y * h / NET_SIZE)])
    return Grasp(pixel=pixel, pixel_net=np.array([x, y]), roll=float(roll), pitch=float(pitch),
                 affordance=affordance, roll_map=roll_map, pitch_map=pitch_map)
