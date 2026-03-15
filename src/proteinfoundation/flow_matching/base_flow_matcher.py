from abc import abstractmethod

import torch
from jaxtyping import Bool
from torch import Tensor

ABS_CLASS_ERR_MSG = "Method not implemented in abstract class"


class BaseFlowMatcher:
    """
    Base class for flow matchers. All flow matching methods in the `base_flow_matching`
    directory should inherit from this class (and implement most -see below- methods).

    Some details:

    - Attributes:
        - `zero_com` specifies whether it should center in that modality
        (see `mask_n_zero_com` method below).

        - `guidance_enabled` specifies whether the underlying data modality
        admits guidance or not.

    - Methods:
        - `nn_out_to_clean`: [x_t, t, nn_out] -> x_1_pred (+ some stuff).

        - `interpolate`: [x_0, x_1, t] -> x_t

        - `compute_loss` In principle, we use the v-loss || v_pred - v_true ||^2
        for all modalities. We can add a config to change this.

        - `mask_n_zero_com`. Should be used carefully, since centering multiple modalities
        independently yields incorrect behavior. For now, centering can only be used for a
        single modality. For some modalities centering may not make sense, for these the
        `mask_n_zero_com` method just masks and does not center.

        - `sample_noise`. Samples the reference distribution.

        - `simulation_step`. Takes a simulation step.

        - `extract_clean_sample_from_batch`. Interfaces with our dataloader's batch.
    """

    def __init__(self, guidance_enabled: bool, dim: int):
        self.guidance_enabled = guidance_enabled
        self.dim = dim

    @abstractmethod
    def mask_n_zero_com(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Masks sample and fixes center of mass to zero (if applicable).

        Args:
            x: sample to mask and zero com
            mask: mask, shape [*, n]

        Returns:
            sample x masked and with zero com
        """
        raise NotImplementedError(ABS_CLASS_ERR_MSG)

    @abstractmethod
    def sample_noise(
        self,
        n: int,
        device: torch.device,
        shape: tuple = tuple(),
        mask: Bool[Tensor, "* n"] | None = None,
    ) -> torch.Tensor:
        """
        Samples reference distribution (possibly centered).

        Args:
            n: number of residues in a single sample (i.e. protein length), int
            mask_n_zero_com
            shape: tuple (if empty then single sample)
            mask (optional): Binary mask of shape [*, n]

        Returns:
            Samples from refenrece with shape [*shape, n, ...]
        """
        raise NotImplementedError(ABS_CLASS_ERR_MSG)

    @abstractmethod
    def interpolate(
        self,
        x_0: torch.Tensor,
        x_1: torch.Tensor,
        t: torch.Tensor,
        mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Interpolates between x_0 (base) and x_1 (data) using t.

        Args:
            x_0: Samples from reference with batch shape *
            x_1: Sampels from target with batch shape *
            t: Interpolation times, shape [*]
            mask (optional): Binary mask, shape [*, n]

        Returns:
            x_t: Interpolated samples, same shape as x_0 and x_1
        """
        raise NotImplementedError(ABS_CLASS_ERR_MSG)

    @abstractmethod
    def extract_clean_sample_from_batch(self, batch: dict) -> torch.Tensor:
        """
        Extracts clean sample from the batch coming from the dataloader.

        Args:
            batch: batch from dataloader.

        Returns:
            Clean sample x_1, a tensor with batch shape [*]
        """
        raise NotImplementedError(ABS_CLASS_ERR_MSG)

    @abstractmethod
    def nn_out_add_clean_sample_prediction(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        mask: torch.Tensor,
        nn_out: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """
        Computes predicted clean sample given nn output, and adds it to the nn output
        (if not there due to parameterization used).

        Args:
            x_0: noise sample, shape [*, n, 3]
            x_1: clean sample, shape [*, n, 3]
            x_t: interpolated sample, shape [*, n, 3]
            t: time sampled, shape [*]
            nn_out: output of neural network for this flow matcher, Dict[str, torch.Tensor]

        Returns:
            The nn_out dictionary updated with clean sample prediction (key "x_1").
        """
        raise NotImplementedError(ABS_CLASS_ERR_MSG)

    @abstractmethod
    def nn_out_add_simulation_tensor(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        mask: torch.Tensor,
        nn_out: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """
        Computes simulation tensor (v or score, depending on base flow matcher) given nn output, and
        adds it to the nn output if not there.

        Args:
            x_0: noise sample, shape [*, n, 3]
            x_1: clean sample, shape [*, n, 3]
            x_t: interpolated sample, shape [*, n, 3]
            t: time sampled, shape [*]
            nn_out: output of neural network, Dict[str, torch.Tensor]

        Returns:
            The nn_out dictionary updated with simulation tensor (key "v" or "score").
        """
        raise NotImplementedError(ABS_CLASS_ERR_MSG)

    @abstractmethod
    def compute_fm_loss(
        self,
        x_0: torch.Tensor,
        x_1: torch.Tensor,
        x_t: torch.Tensor,
        mask: torch.Tensor,
        t: torch.Tensor,
        x_1_pred: torch.Tensor,
    ) -> torch.Tensor:
        """
        Computes flow matching loss per element in the batch.

        Args:
            x_0: noise sample, shape [b, ...]
            x_1: clean sample, shape [b, ...]
            x_t: interpolated sample, shape [b, ...]
            mask (optional): Binary mask, shape [*, n]
            t: time sampled, shape [b]
            x_1_pred: predicted clean sample, shape [b, ...]

        Returns:
            Loss per batch element, shape [b]
        """
        raise NotImplementedError(ABS_CLASS_ERR_MSG)

    @abstractmethod
    def nn_out_add_guided_simulation_tensor(
        self,
        nn_out: dict[str, torch.Tensor],
        nn_out_ag: dict[str, torch.Tensor] | None,
        nn_out_ucond: dict[str, torch.Tensor] | None,
        guidance_w: float,
        ag_ratio: float,
    ) -> dict[str, torch.Tensor]:
        """
        Guidance logic, assumes the nn_out stuff contain the corresponding
        simulation tensors. See `R3NFlowMatcher` for an example.

        Args:
            nn_out: output of neural network from full model, Dict[str, torch.Tensor]
            nn_out_ag: output of neural network from autoguidance model, Dict[str, torch.Tensor] or None
            nn_out_ucond: output of neural network from unconditional model, Dict[str, torch.Tensor] or None
            guidance_w: guidance weight, float
            ag_ratio: autoguidance ratio, float

        Returns:
            The nn_out dictionary updated with guided  simulation tensor ("v" or "score", or whatever is needed).
        """
        raise NotImplementedError(ABS_CLASS_ERR_MSG)

    @abstractmethod
    def simulation_step(
        self,
        x_t: torch.Tensor,
        nn_out: dict[str, torch.Tensor],
        t: torch.Tensor,
        dt: float,
        gt: float,
        mask: torch.Tensor,
        simulation_step_params: dict,
    ):
        r"""
        Single integration step of ODE \dot{x_t} = v(x_t, t) using Euler integration scheme.

        Args:
            x_t: Current value, batch shape *
            nn_out: Dictionary with all available predictions, should include "v" and possibly guided "v_guided".
            May include "x_1", etc as well. All batch shape *
            t: Current time, shape [*]
            dt: Step-size, float
            gt: Noise injection, float
            mask: Binary mask of shape [*, n]
            simulation_step_params: parameters for the simulation step, depends on
            data mode.

        Returns:
            Updated x_t after integration step, same shape as input
        """
        raise NotImplementedError(ABS_CLASS_ERR_MSG)
