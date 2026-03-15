"""AF2 Reward Model for protein design optimization.

This module provides a reward model based on AlphaFold2 for evaluating
and optimizing protein sequences and structures.
"""

import logging
import random
from typing import Any

logger = logging.getLogger(__name__)

import jax
import jax.dlpack
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.dlpack
from colabdesign import mk_afdesign_model
from omegaconf import DictConfig

from proteinfoundation.rewards.alphafold2_reward_utils import (
    add_alignment_bb_ca_loss,
    add_helix_binder_loss,
    add_i_ptm_energy_loss,
    add_i_ptm_loss,
    add_rg_loss,
    add_termini_distance_loss,
)
from proteinfoundation.rewards.base_reward import REWARD_KEY, TOTAL_REWARD_KEY, BaseRewardModel, standardize_reward
from proteinfoundation.utils.pdb_utils import from_pdb_file


class AF2RewardModel(BaseRewardModel):
    """AlphaFold2-based reward model for protein sequence optimization.

    This class implements a reward model that uses AlphaFold2 to evaluate
    protein sequences and provide gradients for optimization. It supports
    various reward components including pLDDT, PAE, and structural metrics.
    """

    IS_FOLDING_MODEL = True
    SUPPORTS_GRAD = True
    SUPPORTS_SAVE_PDB = True

    reward_options = {
        "binder": (
            "plddt",
            "pae",
            "exp_res",
            "con",
            "i_con",
            "i_pae",
            "rg",
            "i_ptm",
            "i_ptm_energy",
            "nc_termini",
            "helix_binder",
            "alignment_bb_ca_binder",
            "dgram_cce",
            "min_ipae",
            "min_ipsae",
            "avg_ipsae",
            "max_ipsae",
            "min_ipsae_10",
            "max_ipsae_10",
            "avg_ipsae_10",
        ),
        "hallucination": ("plddt", "pae", "exp_res", "con", "helix", "alignment_bb_ca"),
    }

    def __init__(
        self,
        protocol: str,
        af_params_dir: str,
        reward_weights: dict[str, float],
        use_multimer: bool = False,
        model_nums: list[int] | None = None,
        num_recycles: int = 3,
        use_initial_guess: bool = False,
        use_initial_atom_pos: bool = False,
        seed: int = 0,
        device_id: int | None = None,
    ) -> None:
        """Initialize the AF2RewardModel.

        Args:
            protocol: Protocol to use for the model.
            af_params_dir: Directory containing AlphaFold2 parameters.
            reward_weights: Dictionary of weights for different loss components.
            use_multimer: Whether to use multimer model.
            model_nums: List of model numbers to use.
            num_recycles: Number of recycles for validation.
            use_initial_guess: Whether to use initial guess.
            use_initial_atom_pos: Whether to use initial atom positions.
            seed: Random seed for reproducibility.
            device_id: GPU device ID to use. If None, auto-detects current CUDA device.
        """
        if device_id is None:
            device_id = torch.cuda.current_device() if torch.cuda.is_available() else 0

        self.protocol = protocol
        assert protocol in [
            "hallucination",
            "binder",
        ], f"Protocol must be either 'hallucination' or 'binder', but got {protocol}"
        self.num_recycles = num_recycles
        self.af_params_dir = af_params_dir
        self.use_multimer = use_multimer
        self.model_nums = model_nums if model_nums is not None else [0, 1, 2, 3, 4]
        # Convert DictConfig to dict if necessary
        if isinstance(reward_weights, DictConfig):
            self.reward_weights = dict(reward_weights)
        else:
            self.reward_weights = reward_weights
        self.use_initial_guess = use_initial_guess
        self.use_initial_atom_pos = use_initial_atom_pos
        for reward_name in reward_weights:
            assert reward_name in self.reward_options[protocol], (
                f"Invalid reward name: {reward_name} for protocol {protocol}"
            )
        for reward_name in self.reward_options[protocol]:
            if reward_name not in self.reward_weights:
                self.reward_weights[reward_name] = 0.0
        self.device = jax.devices("gpu")[device_id]
        self.seed = seed
        self.rng = random.Random(seed)

        # Initialize the AF2 model
        self.model = mk_afdesign_model(
            protocol=protocol,
            use_multimer=use_multimer,
            num_recycles=num_recycles,
            data_dir=af_params_dir,
            use_initial_guess=use_initial_guess,
            use_initial_atom_pos=use_initial_atom_pos,
            learning_rate=1.0,  # No learning rate is used in the reward model
            device=self.device,
        )

        if reward_weights.get("alignment_bb_ca", 0) != 0:
            add_alignment_bb_ca_loss(self.model, reward_weights["alignment_bb_ca"])

        if reward_weights.get("alignment_bb_ca_binder", 0) != 0:
            add_alignment_bb_ca_loss(self.model, reward_weights["alignment_bb_ca_binder"], binder_only=True)

        if reward_weights.get("helix_binder", 0) != 0:
            add_helix_binder_loss(self.model, reward_weights["helix_binder"])

        if reward_weights.get("rg", 0) != 0:
            add_rg_loss(self.model, reward_weights["rg"])

        if reward_weights.get("nc_termini", 0) != 0:
            add_termini_distance_loss(self.model, reward_weights["nc_termini"])

        if reward_weights.get("i_ptm", 0) != 0:
            add_i_ptm_loss(self.model, reward_weights["i_ptm"])

        if reward_weights.get("i_ptm_energy", 0) != 0:
            add_i_ptm_energy_loss(self.model, reward_weights["i_ptm_energy"])

    def score(
        self,
        pdb_path: str,
        requires_grad: bool = False,
        sequence: torch.Tensor | None = None,
        structure: torch.Tensor | None = None,
        binder_chain: str | None = None,
        target_chain: str | None = None,
        save_pdb: bool | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        """Calculate reward and gradients for a given sequence and structure.

        Args:
            pdb_path: Path to PDB file (required)
            requires_grad: Whether to calculate gradients
            sequence: Optional input sequence tensor (logits or one-hot encoded), shape (L, 20).
                If protocol is "hallucination", this is the whole sequence.
                If protocol is "binder", this is the binder sequence.
                If None and protocol is "binder", will extract from PDB file.
            structure: Optional generated coordinates, shape (L, 37, 3). Given in ang.
                If protocol is "binder", this is the binder structure.
                If None and protocol is "binder", will extract from PDB file.
            binder_chain: Optional binder chain ID (required for binder protocol when extracting from PDB)
            target_chain: Optional target chain ID (required for binder protocol)
            save_pdb: Optional flag to save refolded PDB file (supported for AF2)

        Returns:
            Dictionary with reward components and gradients:
                reward: Dict[str, float]  # Reward values for each reward function
                grad: Dict[str, torch.Tensor]  # Gradients for sequence and structure
                plddt: torch.Tensor, shape (L,)  # pLDDT values
                pae: torch.Tensor, shape (L, L)  # PAE values
                ptm: torch.Tensor, shape (1,)  # pTM values
                total_reward: torch.Tensor, shape (1,)  # total reward

        Raises:
            Exception: Re-raises any exception after cleaning up JAX state.
        """
        # Handle binder protocol: extract from PDB if sequence/structure not provided
        if self.protocol == "binder":
            assert target_chain is not None, "target_chain is required for binder protocol"

            # If sequence is not provided, extract from PDB
            if sequence is None:
                assert binder_chain is not None, "binder_chain is required when extracting sequence from PDB"
                assert not requires_grad, "Binder sequence logits are required for gradient calculation"

                binder = from_pdb_file(pdb_path, chain_id=binder_chain)
                seq = F.one_hot(torch.tensor(binder.aatype), num_classes=20).float()
                seq = seq * 1e9

                # Extract structure if not provided
                if structure is None:
                    struct = binder.atom_positions
                else:
                    struct = structure.detach().cpu().numpy()
            else:
                seq = sequence
                struct = structure.detach().cpu().numpy() if structure is not None else None
        else:
            # Hallucination protocol
            assert sequence is not None, "sequence is required for hallucination protocol"
            seq = sequence
            struct = structure.detach().cpu().numpy() if structure is not None else None

        try:
            # Prepare model inputs
            if self.protocol == "hallucination":
                self.model.prep_inputs(
                    length=len(seq),
                    seq=seq,  #  need to pass it here, see `_prep_binder` in `af/prep.py`
                    struct=struct,
                    seed=self.rng.randint(0, 2**32 - 1),
                )
            else:
                # `prep_inputs` will call `_prep_binder` in `af/prep.py`
                # `_prep_binder` will call `_prep_model` in `af/prep.py`
                # `_prep_model` will call `restart` in `af/design.py`
                # `restart` will set opt, seq, and weights
                prep_kwargs = {
                    "pdb_filename": pdb_path,
                    "target_chain": target_chain,
                    "binder_len": len(seq),
                    "seq": seq,  #  need to pass it here, see `_prep_binder` in `af/prep.py`
                    "struct": struct,
                    "seed": self.rng.randint(0, 2**32 - 1),
                    "rm_target": False,
                    "rm_target_seq": False,
                    "rm_target_sc": False,  # remove target coordinates, sequence, and sidechain info from template, hardcoded here
                    "hotspot": None,  # not supported now
                }

                # If extracting from PDB, use binder template
                if binder_chain is not None and sequence is None:
                    prep_kwargs["binder_chain"] = binder_chain
                    prep_kwargs["use_binder_template"] = self.use_initial_atom_pos or self.use_initial_guess
                    prep_kwargs["rm_template_ic"] = self.use_initial_atom_pos or self.use_initial_guess
                else:
                    # When sequence is provided, don't use binder template
                    prep_kwargs["use_binder_template"] = False
                    prep_kwargs["rm_template_ic"] = False

                self.model.prep_inputs(**prep_kwargs)

            self.model.set_opt(
                hard=False,
                soft=True,
                temp=1.0,
                dropout=False,
                pssm_hard=False,
                weights=self.reward_weights,
            )  # transform logits to probs inide, see `soft_seq` in `shared/model.py`

            # Run prediction or optimization
            # sample_models - whether to randomly choose a model for prediction
            # we enable to randomly choose a model for prediction when requires_grad is True, i.e., in the training mode.
            sample_models = requires_grad if sequence is not None else False
            self.model.run(
                num_recycles=self.num_recycles,
                sample_models=sample_models,
                models=self.model_nums,
                backprop=requires_grad,
            )

            if save_pdb:
                save_pdb_filename = kwargs.get("output_pdb_path", pdb_path.replace(".pdb", "_refolded.pdb"))
                self.model.save_pdb(save_pdb_filename)

            # Extract results
            reward_dict = self.extract_results(self.model.aux)

            # Log scalar results
            logger.info("AF2RewardModel score results:")
            for k, v in reward_dict[REWARD_KEY].items():
                if isinstance(v, torch.Tensor) and v.numel() == 1:
                    logger.info("  %s: %.4f", k, v.item())
            logger.info("total_reward: %.4f", reward_dict[TOTAL_REWARD_KEY].item())
            logger.info("--------------------------------")
            return reward_dict

        except Exception:
            self._cleanup_jax_state()
            raise
        finally:
            # Clear model state
            self._clear_model_state()

    def extract_results(self, aux: dict[str, Any]) -> dict[str, Any]:
        """Extract reward dictionary and scores from model auxiliary output.

        Args:
            aux: Auxiliary output from the AF2 model.

        Returns:
            Dictionary with reward components and gradients
        """
        reward_components = {}
        for key in aux["losses"]:
            reward_components[key] = torch.tensor(aux["losses"][key], dtype=torch.float32)

        total_reward = torch.tensor(0.0, dtype=torch.float32)
        for key, weight in self.reward_weights.items():
            total_reward += reward_components.get(key, 0.0) * weight

        # Append raw confidence scores from aux["log"] (always populated by
        # ColabDesign) with a _log suffix so they are saved alongside
        # losses but clearly not part of the reward computation.
        # There is a difference between the loss/reward and the actual score from the log
        log = aux.get("log", {})
        for metric_key in log.keys():
            log_value = log[metric_key]
            if isinstance(log_value, torch.Tensor):
                log_value = log_value.item()
            elif isinstance(log_value, list):
                log_value = log_value[0]
            elif isinstance(log_value, np.ndarray):
                log_value = log_value.item() if log_value.ndim == 0 else float(log_value.mean())
            else:
                log_value = float(log_value)
            reward_components[f"{metric_key}_log"] = torch.tensor(log_value, dtype=torch.float32)
        grad_dict = {}
        if "grad" in aux:
            jax_grad_seq = aux["grad"].get("seq", None)
            if jax_grad_seq is not None:
                if isinstance(jax_grad_seq, jax.Array):
                    grad_dict["sequence"] = torch.utils.dlpack.from_dlpack(jax.dlpack.to_dlpack(jax_grad_seq))
                else:
                    grad_dict["sequence"] = torch.from_numpy(jax_grad_seq)

            jax_grad_struct = aux["grad"].get("struct", None)
            if jax_grad_struct is not None:
                if isinstance(jax_grad_struct, jax.Array):
                    grad_dict["structure"] = torch.utils.dlpack.from_dlpack(jax.dlpack.to_dlpack(jax_grad_struct))
                else:
                    grad_dict["structure"] = torch.from_numpy(jax_grad_struct)

        return standardize_reward(
            reward=reward_components,
            grad=grad_dict,
            total_reward=total_reward,
            plddt=torch.from_numpy(aux["plddt"]),
            pae=torch.from_numpy(aux["pae"]),
            ptm=torch.tensor(aux["ptm"], dtype=torch.float32),
        )

    def _clear_model_state(self) -> None:
        """Clear internal model state dictionaries."""
        if hasattr(self, "model"):
            if hasattr(self.model, "_inputs"):
                self.model._inputs.clear()
            if hasattr(self.model, "aux"):
                self.model.aux.clear()
            if hasattr(self.model, "_tmp"):
                self.model._tmp.clear()

    def _cleanup_jax_state(self) -> None:
        """Clean up JAX caches and backends."""
        # TODO: I don't know but this gives me an error.
        jax.clear_caches()
        jax.clear_backends()

    def cleanup(self) -> None:
        """Explicit cleanup of JAX and model memory.

        Call this method to free up GPU memory and clear JAX state.
        """
        self._clear_model_state()
        # self._cleanup_jax_state()
        # gc.collect()
