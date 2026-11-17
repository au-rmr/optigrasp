# OptiGrasp

Suction grasp detection from a single RGB image, for picking items out of warehouse bins. Given a colour crop of a
bin and a segmentation mask of the target item, the model predicts a per-pixel grasp affordance plus roll and pitch
maps; the grasp is the affordance maximum.

Paper: Soofiyan Atar, Yi Li, Markus Grotz, Michael Wolf, Dieter Fox, Joshua Smith.
*OptiGrasp: Optimized Grasp Pose Detection Using RGB Images for Warehouse Picking Robots*, 2024.
[arXiv:2409.19494](https://arxiv.org/abs/2409.19494). See `CITATION.cff`.

This repo is the inference code, as it runs on the UW au-rmr workcell. Training: `docs/training.md`.

## Use
```python
import cv2, optigrasp
model = optigrasp.load_model("hardware_comms_30_c.state_dict.pth")      # CUDA by default
g = optigrasp.predict(model, cv2.imread("crop.png"), cv2.imread("mask.png", cv2.IMREAD_UNCHANGED))
g.pixel, g.roll, g.pitch          # grasp pixel (x, y) in the crop; angles in rad
```
- Input: BGR uint8 crop (OpenCV order) and an integer label mask of the same size. The smallest non-zero label is the
  target; an all-zero mask makes the whole crop the target.
- Output: `Grasp` with `pixel` (crop coordinates), `pixel_net` (224 x 224 network coordinates), `roll`, `pitch` and
  the three 224 x 224 maps. Turning the pixel into a 3D pose (camera calibration, depth) is up to the robot code.
- Install: `pip install .` (needs PyTorch with CUDA; checked with torch 2.5.1, torchvision 0.20.1).

## Weights
`hardware_comms_30_c` (DINOv2 ViT-S/14 backbone, ~100 MB), as a state_dict:
`sha256 0da4738de666148a893f0c08d0f3aaeaf5fbe2a81d4e450d8985d688379360b3` (`hardware_comms_30_c.state_dict.pth`).
Download: [release v0.1.0](https://github.com/au-rmr/optigrasp/releases/tag/v0.1.0)
(`hardware_comms_30_c.state_dict.pth`). Check it with `sha256sum`. The weights are under the same MIT licence.

## Tests
`OPTIGRASP_WEIGHTS=<state_dict path> pytest tests` on a CUDA machine: the model must reproduce outputs recorded from
the original code on the workcell (4 crops, one with an empty mask). With torch 2.5.1 on an RTX-class GPU the grasp
pixels are identical and the maps differ by at most 0.003.

## Credits
- Model, training and the original inference wrapper: Soofiyan Atar
  (from [Soofiyan/NBV-GP](https://github.com/Soofiyan/NBV-GP) @ 0906287; model code unchanged since 82e2c19).
- Empty-mask fallback in the inference wrapper (2026): Paolo.
- Port to current PyTorch and packaging (2026): UW au-rmr project contributors. What changed and why: `docs/porting_notes.md`.

## Licence
MIT (`LICENSE`). Included third-party code keeps its licence:
- `src/optigrasp/model.py`, `blocks.py`: built on [Depth-Anything](https://github.com/LiheYoung/Depth-Anything),
  Apache-2.0 (`src/optigrasp/LICENSE.depth_anything`).
- `src/optigrasp/_dinov2/`: [DINOv2](https://github.com/facebookresearch/dinov2) @ 81b2b64, Apache-2.0
  (`src/optigrasp/_dinov2/LICENSE`).
