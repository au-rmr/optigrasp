"""The model reproduces the workcell's recorded outputs.

tests/data/<run>/: rgb.png + mask.png = the crop and mask the workcell sent (2026-03-24), expected.npz = what the
original code produced on them (torch 1.13.1, torchvision 0.14.1, RTX 3090). 24_12_16_33 has an empty mask.
Needs a CUDA GPU and the weights: OPTIGRASP_WEIGHTS=<path to hardware_comms_30_c.state_dict.pth> pytest tests
Tolerances were fixed before the first comparison: outputs are not bit-identical across torch versions or GPUs.
"""
import os
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

import optigrasp

DATA = Path(__file__).parent / "data"
RUNS = sorted(p.name for p in DATA.iterdir() if p.is_dir())
WEIGHTS = os.environ.get("OPTIGRASP_WEIGHTS")
MAP_MAXABS, GRASP_PX, ANGLE_DEG = 0.02, 3.0, 1.0

pytestmark = pytest.mark.skipif(not (WEIGHTS and torch.cuda.is_available()),
                                reason="set OPTIGRASP_WEIGHTS and run on a CUDA GPU")


@pytest.fixture(scope="module")
def model():
    return optigrasp.load_model(WEIGHTS)


@pytest.mark.parametrize("run", RUNS)
def test_matches_recorded_output(model, run):
    bgr = cv2.imread(str(DATA / run / "rgb.png"), cv2.IMREAD_UNCHANGED)
    mask = cv2.imread(str(DATA / run / "mask.png"), cv2.IMREAD_UNCHANGED)
    want = np.load(DATA / run / "expected.npz")
    g = optigrasp.predict(model, bgr, mask)
    for got, key in ((g.affordance, "affordance"), (g.roll_map, "roll"), (g.pitch_map, "pitch")):
        assert np.abs(got - want[key]).max() <= MAP_MAXABS, key
    assert np.linalg.norm(g.pixel_net - want["grasp_point"]) <= GRASP_PX
    assert np.degrees(np.abs(np.array([g.roll, g.pitch]) - want["grasp_angle"])).max() <= ANGLE_DEG


def test_pixel_is_in_crop_coordinates(model):
    run = RUNS[0]
    bgr = cv2.imread(str(DATA / run / "rgb.png"), cv2.IMREAD_UNCHANGED)
    mask = cv2.imread(str(DATA / run / "mask.png"), cv2.IMREAD_UNCHANGED)
    g = optigrasp.predict(model, bgr, mask)
    h, w = mask.shape
    assert list(g.pixel) == [int(g.pixel_net[0] * w / 224), int(g.pixel_net[1] * h / 224)]
