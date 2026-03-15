from typing import Literal

import torch
from jaxtyping import Bool, Float
from torch import Tensor

from proteinfoundation.flow_matching.base_flow_matcher import BaseFlowMatcher
from proteinfoundation.utils.align_utils import mean_w_mask


class RDNFlowMatcher(BaseFlowMatcher):
    """
    Flow matching on (R^d)^n, where n is for the number of elements
    per sample (e.g. number of residues) and d is the dimensionality
    for ach element.

    We include the option of using (R_0^d)^n (centering happening for each {1, ..., d} across n dimension).
    """

    def __init__(
        self,
        zero_com_noise: bool,
        guidance_enabled: bool,
        dim: int,
        stochastic_centering_mode: Literal["none", "x_0", "x_1", "x_t"] = "none",
        stochastic_centering_scale: float = None,
    ):
        super().__init__(
            guidance_enabled=guidance_enabled,
            dim=dim,
        )
        self.zero_com_noise = zero_com_noise
        self.stochastic_centering_scale = stochastic_centering_scale
        self.stochastic_centering_mode = stochastic_centering_mode

    def _force_zero_com(
        self, x: Float[Tensor, "* n d"], mask: Bool[Tensor, "* n"] | None = None
    ) -> Float[Tensor, "* n d"]:
        """
        Centers tensor over n (residue) dimension.

        Args:
            x: Tensor of shape [*, n, d]
            mask (optional): Binary mask of shape [*, n]

        Returns:
            Centered x = x - mean(x, dim=-2), shape [*, n, d].
        """
        if mask is None:
            x = x - torch.mean(x, dim=-2, keepdim=True)
        else:
            x = (x - mean_w_mask(x, mask, keepdim=True)) * mask[..., None]
        return x

    def _apply_mask(self, x: Float[Tensor, "* n d"], mask: Bool[Tensor, "* n"] | None = None) -> Float[Tensor, "* n d"]:
        """
        Applies mask to x. Sets masked elements to zero.

        Args:
            x: Tensor of shape [*, n, d]
            mask (optional): Binary mask of shape [*, n]

        Returns:
            Masked x of shape [*, n, d]
        """
        if mask is None:
            return x
        return x * mask[..., None]  # [*, n, d]

    def sample_noise(
        self,
        n: int,
        device: torch.device,
        shape: tuple = tuple(),
        mask: Bool[Tensor, "* n"] | None = None,
        training: bool = True,
    ) -> Float[Tensor, "* n d"]:
        """
        Samples reference distribution std Gaussian (possibly centered).

        Args:
            n: number of frames in a single sample, int
            device: torch device
            shape: tuple (if empty then single sample)
            mask (optional): Binary mask of shape [*, n]
            training (optional): whether in training mode

        Returns:
            Samples from refenrece [N(0, I_d)]^n shape [*shape, n, d]
        """
        noise = torch.randn(
            shape + (n, self.dim),
            device=device,
        )
        noise = self._apply_mask(noise, mask)
        if self.zero_com_noise:
            noise = self._force_zero_com(noise, mask)
        if self.stochastic_centering_mode == "x_0" and not training:
            noise = self.add_stochastic_noise(noise, mask)
        return noise

    def add_stochastic_noise(
        self,
        x: Float[Tensor, "* n d"],
        mask: Bool[Tensor, "* n"] | None = None,
    ) -> Float[Tensor, "* n d"]:
        """
        Adds stochastic noise to x.
        """
        x = self._force_zero_com(x, mask)
        noise = torch.randn(
            *x.shape[:-2],
            x.shape[-1],
            device=x.device,
        )
        x = x + self.stochastic_centering_scale * noise.unsqueeze(-2)
        if mask is not None:
            x = x * mask[..., None]
        return x

    def interpolate(
        self,
        x_0: Float[Tensor, "* n d"],
        x_1: Float[Tensor, "* n d"],
        t: Float[Tensor, "*"],
        mask: Bool[Tensor, "* n"] | None = None,
    ) -> Float[Tensor, "* n d"]:
        """
        Interpolates between rigids x_0 (base) and x_1 (data) using t.

        Args:
            x_0: Tensor sampled from reference, shape [*, n, d]
            x_1: Tensor sampled from target, shape [*, n, d]
            t: Interpolation times, shape [*]
            mask (optional): Binary mask, shape [*, n]

        Returns:
            x_t: Interpolated tensor, shape [*, n, d]
        """
        # x_0 masked, x_1 depends on dataloader
        # x_0 maybe centered (depends on input arg zero_com_noise), x_1 depends on dataloader
        x_0, x_1 = map(lambda args: self._apply_mask(*args), ((x_0, mask), (x_1, mask)))
        t = t[..., None, None]
        if self.stochastic_centering_mode == "x_0":
            x_0 = self.add_stochastic_noise(x_0, mask)
        elif self.stochastic_centering_mode == "x_1":
            x_1 = self.add_stochastic_noise(x_1, mask)
        # No maks since x_0 and x_1 are masked
        x_t = (1.0 - t) * x_0 + t * x_1
        if self.stochastic_centering_mode == "x_t":
            x_t = self.add_stochastic_noise(x_t, mask)
        return x_t

    def nn_out_add_clean_sample_prediction(
        self,
        x_t: Float[Tensor, "* n d"],
        t: Float[Tensor, "*"],
        mask: Bool[Tensor, "* n"],
        nn_out: dict[str, torch.Tensor],
    ) -> dict[str, Float[Tensor, "* n d"]]:
        """
        Computes predicted clean sample given nn output.

        Args:
            x_0: noise sample, shape [*, n, d]
            x_1: clean sample, shape [*, n, d]
            x_t: interpolated sample, shape [*, n, d]
            t: time sampled, shape [*]
            nn_out: output of neural network, Dict[str, torch.Tensor]

        Returns:
            The nn_out dictionary updated with clean sample prediction.
        """
        t = t[..., None, None]  # [*, 1, 1]
        if "x_1" in nn_out:
            pass
        elif "v" in nn_out:
            nn_out["x_1"] = x_t + (1.0 - t) * nn_out["v"]
        else:
            raise OSError(f"Cannot compute clean sample prediction from keys {[k for k in nn_out]}")
        nn_out["x_1"] = nn_out["x_1"] * mask[..., None]
        return nn_out

    def nn_out_add_simulation_tensor(
        self,
        x_t: Float[Tensor, "* n d"],
        t: Float[Tensor, "*"],
        mask: Bool[Tensor, "* n"],
        nn_out: dict[str, torch.Tensor],
    ) -> dict[str, Float[Tensor, "* n d"]]:
        """
        Computes vector field v given nn output.

        Args:
            x_0: noise sample, shape [*, n, d]
            x_1: clean sample, shape [*, n, d]
            x_t: interpolated sample, shape [*, n, d]
            t: time sampled, shape [*]
            nn_out: output of neural network, Dict[str, torch.Tensor]

        Returns:
            The nn_out dictionary updated with v.
        """
        t = t[..., None, None]  # [*, 1, 1]
        if "v" in nn_out:
            pass
        elif "x_1" in nn_out:
            num = nn_out["x_1"] - x_t  # [*, n, d]
            den = 1.0 - t  # [*, 1, 1]
            nn_out["v"] = num / (den + 1e-5)  # [*, n, d]
        else:
            raise OSError(f"Cannot compute simulation tensor (v) from keys {[k for k in nn_out]}")
        nn_out["v"] = nn_out["v"] * mask[..., None]
        return nn_out

    def compute_fm_loss(
        self,
        x_0: Float[Tensor, "* n d"],
        x_1: Float[Tensor, "* n d"],
        x_t: Float[Tensor, "* n d"],
        mask: Bool[Tensor, "* n"],
        t: Float[Tensor, "*"],
        nn_out: dict[str, Float[Tensor, "* n d"]],
    ) -> Float[Tensor, "*"]:
        """
        Computes flow matching loss per element in the batch.

        Args:
            x_0: noise sample, shape [*, n, d]
            x_1: clean sample, shape [*, n, d]
            x_t: interpolated sample, shape [*, n, d]
            mask (optional): Binary mask, shape [*, n]
            t: time sampled, shape [*]
            x_1_pred: predicted clean sample, shape [*, n, d]

        Returns:
            Loss per element in the batch, shape [*]
        """
        nn_out = self.nn_out_add_clean_sample_prediction(
            x_t=x_t,
            t=t,
            mask=mask,
            nn_out=nn_out,
        )
        nres = torch.sum(mask, dim=-1)  # [*]
        err = (x_1 - nn_out["x_1"]) * mask[..., None]  # [*, n, d]
        loss = torch.sum(err**2, dim=(-1, -2)) / nres  # [*]
        total_loss_w = 1.0 / ((1.0 - t) ** 2 + 1e-5)
        loss = loss * total_loss_w  # [*]
        return loss

    def nn_out_add_guided_simulation_tensor(
        self,
        nn_out: dict[str, torch.Tensor],
        nn_out_ag: dict[str, torch.Tensor] | None,
        nn_out_ucond: dict[str, torch.Tensor] | None,
        guidance_w: float,
        ag_ratio: float,
    ) -> dict[str, torch.Tensor]:
        """
        Computes predicted clean sample given nn output. Note this assumes the nn_out
        dictionaries contain the simulation tensor (v, score, ...) for each corresponding
        base flow matcher.

        Args:
            nn_out: output of neural network from full model, Dict[str, Tensor]
            nn_out_ag: output of neural network from autoguidance model, Dict[str, torch.Tensor] or None
            nn_out_ucond: output of neural network from unconditional model, Dict[str, torch.Tensor] or None
            guidance_w: guidance weight, float
            ag_ratio: autoguidance ratio, float

        Returns:
            The nn_out dictionary updated with guided v.
        """
        assert "v" in nn_out, "`v` should be a key in the nn_out dict"
        if not self.guidance_enabled:
            return nn_out

        v = nn_out["v"]
        v_ag = torch.zeros_like(v) if nn_out_ag is None else nn_out_ag["v"]
        v_ucond = torch.zeros_like(v) if nn_out_ucond is None else nn_out_ucond["v"]

        nn_out["v_guided"] = guidance_w * v + (1 - guidance_w) * (ag_ratio * v_ag + (1 - ag_ratio) * v_ucond)
        return nn_out

    def simulation_step(
        self,
        x_t: Float[Tensor, "* n d"],
        nn_out: dict[str, Float[Tensor, "* n d"]],
        t: Float[Tensor, "*"],
        dt: float,
        gt: float,
        mask: Bool[Tensor, "* n"],
        simulation_step_params: dict,
    ) -> Float[Tensor, "* n d"]:
        r"""
        Single integration step of ODE

        eq. (1): d x_t = v(x_t, t) dt

        or SDE

        eq. (2): d x_t = [v(x_t, t) + g(t) s(x_t, t)] dt + \sqrt{2g(t)} dw_t

        using Euler integration scheme.

        For our interpolation scheme (i.e. stochastic interpolant) we can obtain
        the score as a function of the vector field from

        v(x_t, t) = (1 / t) (x_t + scale_ref ** 2 * (1 - t) * s(x_t, t)),

        or equivalently,

        s(x_t, t) = (t * v(x_t, t) - x_t) / (scale_ref ** 2 * (1 - t)).

        We add a few additional parameters to the SDE to control noise/score scale and
        perform stochastic and low temperature sampling:

        eq. (3): d x_t = [v(x_t, t) + g(t) * sc_score_scale * s(x_t, t) * sc_score_scale] dt + \sqrt{2 * g(t) * sc_noise_scale} dw_t.

        At the moment we do not scale the vector field v.

        Args:
            x_t: Current value, shape [*, n, d]
            nn_out: Dictionary with all available predictions, should include "v" and possibly guided "v_guided".
            May include "x_1", etc as well. All shape [*, n, d]
            t: Current time, shape [*]
            dt: Step-size, float
            gt: Noise injection, float
            mask: Binary mask of shape [*, n]
            simulation_step_params: parameters for the simulation step, keys
                sampling_mode, sc_scale_noise, sc_scale_score, t_lim_ode

        Returns:
            Updated values for x_t after an Euler integration step, shape [*, n, d].
        """
        sampling_mode = simulation_step_params["sampling_mode"]
        sc_scale_noise = simulation_step_params["sc_scale_noise"]
        sc_scale_score = simulation_step_params["sc_scale_score"]
        t_lim_ode = simulation_step_params["t_lim_ode"]
        t_lim_ode_below = simulation_step_params["t_lim_ode_below"]
        center_every_step = simulation_step_params["center_every_step"]
        tsr_k = simulation_step_params.get("tsr_k", 1.0)
        tsr_sigma = simulation_step_params.get("tsr_sigma", 1.0)
        # TODO: we should make all of these have a safe gettter to prevent errors
        # In case we want a default value for this parameter or other

        assert sampling_mode in [
            "vf",
            "sc",
            "vf_ss",
            "vf_ss_sc_sn",
            "vf_tsr",
        ], f"Invalid sampling mode {sampling_mode}, should be `vf`, `sc`, `vf_ss`, `vf_ss_sc_sn`, or `vf_tsr`"
        assert sc_scale_noise >= 0, f"Scale noise for sampling should be >= 0, got {sc_scale_noise}"
        assert sc_scale_score >= 0, f"Scale score for sampling should be >= 0, got {sc_scale_score}"
        assert gt >= 0, f"gt for sampling should be >= 0, got {gt}"
        t_element = t.flatten()[0]
        assert torch.all(t_element == t), "Sampling only implemented for same time for all samples"

        if self.guidance_enabled:
            v = nn_out["v_guided"]
        else:
            v = nn_out["v"]

        sc_scale_score_def = 1.5  # used when very close to 1 for sc mode, since we don't want to add more noise, we switch to low temp ODE with this scale for the score
        sc_scale_noise_def = 0.3  # used when very close to 0 for vf_ss mode, since we can't use low temp ODE then, we switch to low temp SDE with this scale for the noise

        # if sampling_mode == "vf" or t_element > t_lim_ode:
        #     delta_x = v * dt
        if sampling_mode == "vf":  # ODE mode
            delta_x = v * dt

        elif sampling_mode == "vf_ss":  # ODE with score scaling
            if t_element < t_lim_ode_below:  # Close to zero cannot do score_to_vf, switch to SDE with noise scaling
                score = vf_to_score(x_t, v, t)  # get score from v, [*, dim]
                eps = torch.randn(x_t.shape, dtype=x_t.dtype, device=x_t.device)  # [*, dim]
                std_eps = torch.sqrt(2 * gt * sc_scale_noise_def * dt)
                delta_x = (v + gt * score) * dt + std_eps * eps
            else:
                score = vf_to_score(x_t, v, t)
                scaled_score = score * sc_scale_score
                v_scaled = score_to_vf(x_t, scaled_score, t)
                delta_x = v_scaled * dt

        elif sampling_mode == "vf_tsr":  # ODE with time-scale-ratio scaling
            snr = t_element**2 / (1 - t_element) ** 2
            tsr_ratio = (snr * tsr_sigma**2 + 1) / (snr * tsr_sigma**2 / tsr_k + 1)
            if t_element == 0.0:
                ### Minimal trick: When skip t=0
                v_scaled = torch.zeros_like(v).to(v.device)
            else:
                score = vf_to_score(x_t, v, t)
                scaled_score = score * tsr_ratio
                v_scaled = score_to_vf(x_t, scaled_score, t)
            delta_x = v_scaled * dt

        elif sampling_mode == "sc":  # SDE with noise scaling
            if t_element > t_lim_ode:  # close to 1 stop SDE, switch to low temp ODE
                score = vf_to_score(x_t, v, t)
                scaled_score = score * sc_scale_score_def
                v_scaled = score_to_vf(x_t, scaled_score, t)
                delta_x = v_scaled * dt
            else:
                score = vf_to_score(x_t, v, t)  # get score from v, [*, dim]
                eps = torch.randn(x_t.shape, dtype=x_t.dtype, device=x_t.device)  # [*, dim]
                std_eps = torch.sqrt(2 * gt * sc_scale_noise * dt)
                delta_x = (v + gt * score) * dt + std_eps * eps

        elif sampling_mode == "vf_ss_sc_sn":  # SDE with score scaling (in ODE) and noise scaling (in Langevin term)
            if t_element > t_lim_ode:  # close to 1 stop SDE, switch to plain low temp ODE
                score = vf_to_score(x_t, v, t)
                scaled_score = score * sc_scale_score_def
                v_scaled = score_to_vf(x_t, scaled_score, t)
                delta_x = v_scaled * dt
            elif t_element < t_lim_ode_below:  # close to 0 switch to SDE with noise scaling
                score = vf_to_score(x_t, v, t)  # get score from v, [*, dim]
                eps = torch.randn(x_t.shape, dtype=x_t.dtype, device=x_t.device)  # [*, dim]
                std_eps = torch.sqrt(2 * gt * sc_scale_noise_def * dt)
                delta_x = (v + gt * score) * dt + std_eps * eps
            else:  # ODE scaled with SDE scaled
                score = vf_to_score(x_t, v, t)
                scaled_score = score * sc_scale_score
                v_scaled = score_to_vf(x_t, scaled_score, t)
                eps = torch.randn(x_t.shape, dtype=x_t.dtype, device=x_t.device)  # [*, dim]
                std_eps = torch.sqrt(2 * gt * sc_scale_noise * dt)
                delta_x = (v_scaled + gt * score) * dt + std_eps * eps

        else:
            raise ValueError(f"Invalid sampling mode {sampling_mode}")

        x_next = x_t + delta_x

        # Mask and potentially center
        x_next = self._apply_mask(x_next, mask)
        if center_every_step:
            x_next = self._force_zero_com(x_next, mask)
        return x_next


def vf_to_score(
    x_t: Float[Tensor, "* n d"],
    v: Float[Tensor, "* n d"],
    t: Float[Tensor, "*"],
) -> Float[Tensor, "* n d"]:
    """
    Computes score of noisy density given the vector field learned by flow matching. With
    our interpolation scheme these are related by

    v(x_t, t) = (1 / t) (x_t + scale_ref ** 2 * (1 - t) * s(x_t, t)),

    or equivalently,

    s(x_t, t) = (t * v(x_t, t) - x_t) / (scale_ref ** 2 * (1 - t)).

    Args:
        x_t: Noisy sample, shape [*, dim]
        v: Vector field, shape [*, dim]
        t: Interpolation time, shape [*] (must be < 1)

    Returns:
        Score of intermediate density, shape [*, dim].
    """
    assert torch.all(t < 1.0), "vf_to_score requires t < 1 (strict)"
    num = t[..., None, None] * v - x_t  # [*, dim]
    den = (1.0 - t)[..., None, None]  # [*, 1]
    score = num / den
    return score  # [*, dim]


def score_to_vf(
    x_t: Float[Tensor, "* n d"],
    score: Float[Tensor, "* n d"],
    t: Float[Tensor, "*"],
) -> Float[Tensor, "* n d"]:
    """
    Computes vector field from score. See `vf_to_score` function for equations.

    Args:
        x_t: Noisy sample, shape [*, n, dim]
        score: Vector field, shape [*, n, dim]
        t: Interpolation time, shape [*] (must be > 0)

    Returns:
        Vector field, shape [*, n, dim].
    """
    assert torch.all(t > 0.0), "score_to_vf requires t > 0 (strict)"
    t = t[..., None, None]  # [*, 1, 1]
    return (x_t + (1.0 - t) * score) / t  # [*, n, dim]
