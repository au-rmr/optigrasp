# Porting notes (2026)

Source: the inference wrapper `hardware_inference_comms.py` that ran on the AURMR workcell (never committed to
NBV-GP; the Feb 2026 copy, = Soofiyan's 2024 version + the empty-mask fallback) and NBV-GP's
`huggingface/Depth-Anything/depth_anything/{dpt_angle_fine,blocks}.py`.

Checked: on the 16 crops recorded on the workcell on 2026-03-24, the port (torch 2.5.1, torchvision 0.20.1) gives the
same grasp pixel as the original (torch 1.13.1, torchvision 0.14.1) on all 16; maps differ by < 0.008, angles by
< 0.03 deg. Four of them are in `tests/data`.

## Kept on purpose (behaviour the workcell relied on)
- The crop is resized to 224 x 224 without keeping its aspect ratio; the pixel is scaled back per axis and truncated
  to int, as the workcell code did.
- Grasp = argmax of the affordance map. The original also ran DBSCAN clustering on the map, but its result was
  overwritten by the argmax right after, so it never had an effect; it is gone (and so is scikit-learn).
- Mask label: the smallest non-zero label; an all-zero mask selects label 0 = everything (the Feb 2026 fallback;
  before it, an empty mask raised).
- The image is resized on the CPU, the mask on the GPU (bicubic results differ slightly between the two).

## Changed
- torchvision >= 0.17 antialiases tensor resizes by default; 0.14 did not. `antialias=False` restores the trained
  behaviour; with antialiasing the grasp pixel moved by up to 129 px on the recorded crops.
- Weights load from a state_dict (`torch.load(weights_only=True)`), not a pickled whole model; strict load, the same
  numbers. The state_dict was saved from the original pickle `hardware_comms_30_c.pth`.
- DINOv2 comes from upstream facebookresearch/dinov2 @ 81b2b64 (Apache-2.0), vendored, instead of NBV-GP's copy. The
  copy is upstream @ 10d4201 (CC BY-NC); 10d4201 -> 81b2b64 changes only licence headers and xFormers switches, and
  xFormers is never used here (it was not installed on the workcell either).
- Removed: hard-coded workstation paths, debug image writes (/tmp and a "Paper evaluation" folder), prints, the
  Hugging Face Hub mixin, torch.hub loading.
