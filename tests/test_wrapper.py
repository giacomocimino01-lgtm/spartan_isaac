import tempfile
import os
from pathlib import Path

import torch
from PIL import Image
import torchvision.transforms as T

from m_dVrk.hrl.tcc import XIRLResnet18
from m_dVrk.hrl.wrapper import DVRKVisionHRLWrapper


class DummyEnv:
    def __init__(self):
        self.num_envs = 1
        self.device = torch.device("cpu")


class DummySM:
    pass


def _make_temp_dataset(tmpdir: str, n_videos: int = 2):
    root = Path(tmpdir)
    for i in range(n_videos):
        vd = root / f"video_{i}"
        vd.mkdir(parents=True, exist_ok=True)
        # create a simple RGB png
        img = Image.new("RGB", (128, 128), color=(i * 40, i * 80, i * 120))
        img.save(vd / "frame_000.png")
        img.save(vd / "frame_001.png")
    return str(root)


def test_compute_and_set_goal_embedding_smoke(tmp_path):
    dataset_path = _make_temp_dataset(str(tmp_path), n_videos=3)

    tcc = XIRLResnet18(embedding_size=32)
    proc = T.Compose([T.Resize((112, 112))])

    # Use the unbound method on a minimal dummy container to avoid creating
    # a full IsaacLab environment instance.
    class DummyWrapperObj:
        def __init__(self):
            self.env = DummyEnv()
            self.emb_dim = 32
            self.goal_embedding = None

    dw = DummyWrapperObj()
    # call the unbound function with our dummy self
    DVRKVisionHRLWrapper.compute_and_set_goal_embedding(dw, tcc, proc, dataset_path, raise_on_error=True)

    assert hasattr(dw, "goal_embedding")
    ge = dw.goal_embedding
    assert isinstance(ge, torch.Tensor)
    assert ge.shape == (32,)
    assert ge.device.type == "cpu"


def test_sanitize_arm_command_enforces_sequence_order():
    wrapper = object.__new__(DVRKVisionHRLWrapper)
    wrapper.num_envs = 1
    wrapper.sm = type(
        "DummySM",
        (),
        {
            "last_target_r": [None],
            "last_target_l": [None],
            "attached_target_r": [None],
            "attached_target_l": [None],
            "ring_support_peg": {"ring_red": [None], "ring_green": [None]},
            "get_top_rings_on_peg": lambda self, env_id, peg_name, num_rings=1: ["ring_red"],
        },
    )()

    # No reach before grasp is invalid.
    assert wrapper._sanitize_arm_command(0, "right_arm", "grasp", "ring_red") == ("idle", "None", True)

    # After reaching the ring, grasp becomes valid.
    wrapper.sm.last_target_r[0] = "ring_red"
    assert wrapper._sanitize_arm_command(0, "right_arm", "grasp", "ring_red") == ("grasp", "ring_red", False)

    # Release without reaching the peg is invalid.
    wrapper.sm.attached_target_r[0] = "ring_red"
    assert wrapper._sanitize_arm_command(0, "right_arm", "release", "peg_green") == ("idle", "None", True)

    # Release is valid only after a reach to the target peg.
    wrapper.sm.last_target_r[0] = "peg_green"
    assert wrapper._sanitize_arm_command(0, "right_arm", "release", "peg_green") == ("release", "peg_green", False)
