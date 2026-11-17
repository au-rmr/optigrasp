# Training (not ported yet)

This repo holds inference only: it is what runs and is tested on the robot. Training is not ported because nobody has
retrained since 2024 and untested training code would break silently. Port it when someone retrains, and test it then.

Where it is: [Soofiyan/NBV-GP](https://github.com/Soofiyan/NBV-GP) @ 0906287, `huggingface/Depth-Anything/`:
- `hugging_face_training_angle_fine.py` (multi-GPU, DistributedDataParallel, 200 epochs, Adam lr 1e-3) and
  `hugging_face_training_aws_angle_fine.py` (the AWS variant) train this model (`depth_anything/dpt_angle_fine.py`).
  Which run produced `hardware_comms_30_c` is not recorded.
- Data layout they expect: `<images_root>/run_ID_*/3F/<folder>/rgb*.jpg` with matching label files under
  `<labels_root>`. Paths in the scripts are hard-coded to the original machines.
- Where the training data is now: unknown (ask the lab; possibly on the lab's data server).

Porting it: same model code as here (`optigrasp.model`); apply the same version fixes as `docs/porting_notes.md`
(antialias=False on tensor resizes, weights_only loading), make the paths arguments, and add a short training smoke
test on a few images.
