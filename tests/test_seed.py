"""Tests for SDEdit / partial-diffusion seeding.

Covers ``ProductSpaceFlowMatcher.seed_state`` -- the SDEdit forward marginal that replaces
pure-noise init in ``search/fk_steering.py``. Pure flow-matching math, no model weights and
no config files required (a minimal two-channel flow matcher is built inline).

The encoder path (``utils.pdb_utils.encode_seed``) and the ``partial_simulation`` start_step
integration require model weights and are exercised by a live run, not here.
"""

import torch
from omegaconf import OmegaConf

from proteinfoundation.flow_matching.product_space_flow_matcher import ProductSpaceFlowMatcher


def _fm():
    """Minimal two-channel (bb_ca + local_latents) flow matcher, no weights."""
    cfg = OmegaConf.create(
        {
            "product_flowmatcher": {
                "bb_ca": {"zero_com_noise": True, "guidance_enabled": False, "dim": 3},
                "local_latents": {"zero_com_noise": False, "guidance_enabled": False, "dim": 8},
            }
        }
    )
    fm = ProductSpaceFlowMatcher(cfg)
    fm.eval()
    return fm


def _ts(nsteps):
    # ts[0] = 0, ts[-1] = 1; seed_state only indexes ts[dm][start_step], so a linear
    # schedule suffices to test the interpolant math independent of get_schedule internals.
    sched = torch.linspace(0.0, 1.0, nsteps + 1)
    return {"bb_ca": sched, "local_latents": sched}


def _clean(nsamples, n):
    return {
        "bb_ca": torch.randn(nsamples, n, 3),
        "local_latents": torch.randn(nsamples, n, 8),
    }


def test_seed_state_t1_returns_clean():
    """start_step == nsteps (t = 1): the seeded state is exactly the clean input (no noise)."""
    nsamples, n, nsteps = 2, 10, 100
    fm, ts = _fm(), _ts(100)
    mask = torch.ones(nsamples, n, dtype=torch.bool)
    clean = _clean(nsamples, n)
    s = fm.seed_state(clean=clean, mask=mask, ts=ts, start_step=nsteps)
    assert torch.allclose(s["bb_ca"], clean["bb_ca"], atol=1e-5)
    assert torch.allclose(s["local_latents"], clean["local_latents"], atol=1e-5)


def test_seed_state_t0_is_pure_noise():
    """start_step == 0 (t = 0): the seeded state is pure noise, independent of the clean input."""
    nsamples, n, nsteps = 2, 10, 100
    fm, ts = _fm(), _ts(nsteps)
    mask = torch.ones(nsamples, n, dtype=torch.bool)
    clean = _clean(nsamples, n)
    clean = {k: v + 100.0 for k, v in clean.items()}  # push clean far away
    s = fm.seed_state(clean=clean, mask=mask, ts=ts, start_step=0)
    assert not torch.allclose(s["bb_ca"], clean["bb_ca"], atol=1.0)
    assert not torch.allclose(s["local_latents"], clean["local_latents"], atol=1.0)


def test_seed_state_larger_start_stays_closer():
    """Larger start_step (t -> 1) stays closer to the clean input than a smaller one."""
    nsamples, n, nsteps = 2, 10, 200
    fm, ts = _fm(), _ts(nsteps)
    mask = torch.ones(nsamples, n, dtype=torch.bool)
    clean = _clean(nsamples, n)
    torch.manual_seed(0)
    near = (fm.seed_state(clean=clean, mask=mask, ts=ts, start_step=180)["local_latents"] - clean["local_latents"]).norm()
    torch.manual_seed(0)
    far = (fm.seed_state(clean=clean, mask=mask, ts=ts, start_step=40)["local_latents"] - clean["local_latents"]).norm()
    assert near < far


def test_seed_state_preserves_shape_and_modes():
    """Output has both channels with the clean shapes (contract for fk_steering init)."""
    nsamples, n, nsteps = 3, 12, 50
    fm, ts = _fm(), _ts(nsteps)
    mask = torch.ones(nsamples, n, dtype=torch.bool)
    clean = _clean(nsamples, n)
    s = fm.seed_state(clean=clean, mask=mask, ts=ts, start_step=25)
    assert set(s) == {"bb_ca", "local_latents"}
    assert s["bb_ca"].shape == (nsamples, n, 3)
    assert s["local_latents"].shape == (nsamples, n, 8)
