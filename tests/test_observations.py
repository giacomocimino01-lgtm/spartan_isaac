import numpy as np
import torch

from m_dVrk.hrl.constants import STACK_SIZE, EMB_DIM, AUX_DIM, GEOM_DIM, OBS_DIM
from m_dVrk.hrl.observations import build_obs


def test_build_obs_shape_and_dtype():
    num_envs = 2
    emb_buffer = torch.randn((num_envs, STACK_SIZE, EMB_DIM), dtype=torch.float32)
    aux_obs = torch.randn((num_envs, AUX_DIM), dtype=torch.float32)
    geom_obs = torch.randn((num_envs, GEOM_DIM), dtype=torch.float32)

    obs = build_obs(emb_buffer, aux_obs, geom_obs, num_envs)

    assert isinstance(obs, np.ndarray)
    assert obs.shape == (num_envs, OBS_DIM)
    assert obs.dtype == np.float32
