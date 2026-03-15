"""Interface energy reward model using TMol for protein structure optimization.

This module provides a flexible reward model based on TMol's energy scoring
for evaluating and optimizing protein structures with focus on interchain
interactions. Supports both hydrogen bond and electrostatic energy terms
individually or in combination, making it particularly useful for:

- Protein-protein interface optimization
- Salt bridge analysis and optimization
- Combined interaction scoring for protein complexes
- Structure-based drug design applications

The main class `TmolRewardModel` allows users to:
- Enable/disable hydrogen bond (hbond) and electrostatic (fa_elec) terms
- Set custom weights for each energy type
- Get detailed energy breakdowns and gradients
- Analyze interface interactions with utility functions

Example usage:
    # H-bond only (backward compatible)
    model = TmolRewardModel(enable_hbond=True, enable_elec=False)

    # Electrostatic only (useful for salt bridges)
    model = TmolRewardModel(enable_hbond=False, enable_elec=True)

    # Combined with custom weights
    model = TmolRewardModel(
        enable_hbond=True, enable_elec=True,
        hbond_weight=1.0, elec_weight=0.5
    )
"""

import logging

from dotenv import load_dotenv

load_dotenv()
import gc
from typing import Any

import numpy as np
import tmol
import torch

logger = logging.getLogger(__name__)
from tmol.extern.openfold import residue_constants as rc
from tmol.io.chain_deduction import chain_inds_for_pose_stack
from tmol.score.score_function import ScoreFunction
from tmol.score.score_types import ScoreType

from proteinfoundation.rewards.base_reward import REWARD_KEY, BaseRewardModel, standardize_reward


def make_atom14_masks(protein):
    """Construct denser atom positions (14 dimensions instead of 37)."""
    restype_atom14_to_atom37 = []
    restype_atom37_to_atom14 = []
    restype_atom14_mask = []

    for rt in rc.restypes:
        atom_names = rc.restype_name_to_atom14_names[rc.restype_1to3[rt]]
        restype_atom14_to_atom37.append([(rc.atom_order.get(name, 0)) for name in atom_names])
        atom_name_to_idx14 = {name: i for i, name in enumerate(atom_names)}
        restype_atom37_to_atom14.append([(atom_name_to_idx14.get(name, 0)) for name in rc.atom_types])

        restype_atom14_mask.append([(1.0 if name else 0.0) for name in atom_names])

    # Add dummy mapping for restype 'UNK'
    restype_atom14_to_atom37.append([0] * 14)
    restype_atom37_to_atom14.append([0] * 37)
    restype_atom14_mask.append([0.0] * 14)

    restype_atom14_to_atom37 = torch.tensor(
        restype_atom14_to_atom37,
        dtype=torch.int32,
        device=protein["aatype"].device,
    )
    restype_atom37_to_atom14 = torch.tensor(
        restype_atom37_to_atom14,
        dtype=torch.int32,
        device=protein["aatype"].device,
    )
    restype_atom14_mask = torch.tensor(
        restype_atom14_mask,
        dtype=torch.float32,
        device=protein["aatype"].device,
    )
    protein_aatype = protein["aatype"].to(torch.long)

    # create the mapping for (residx, atom14) --> atom37, i.e. an array
    # with shape (num_res, 14) containing the atom37 indices for this protein
    residx_atom14_to_atom37 = restype_atom14_to_atom37[protein_aatype]
    residx_atom14_mask = restype_atom14_mask[protein_aatype]

    protein["atom14_atom_exists"] = residx_atom14_mask
    protein["residx_atom14_to_atom37"] = residx_atom14_to_atom37.long()

    # create the gather indices for mapping back
    residx_atom37_to_atom14 = restype_atom37_to_atom14[protein_aatype]
    protein["residx_atom37_to_atom14"] = residx_atom37_to_atom14.long()

    # create the corresponding mask
    restype_atom37_mask = torch.zeros([21, 37], dtype=torch.float32, device=protein["aatype"].device)
    for restype, restype_letter in enumerate(rc.restypes):
        restype_name = rc.restype_1to3[restype_letter]
        atom_names = rc.residue_atoms[restype_name]
        for atom_name in atom_names:
            atom_type = rc.atom_order[atom_name]
            restype_atom37_mask[restype, atom_type] = 1

    residx_atom37_mask = restype_atom37_mask[protein_aatype]
    protein["atom37_atom_exists"] = residx_atom37_mask

    return protein


def convert_atom37_to_atom14_openfold(positions, aatype):
    """Convert atom37 positions to atom14 using OpenFold's approach"""

    # Step 1: Create a minimal protein dict to use make_atom14_masks
    protein_dict = {
        "aatype": aatype,  # [batch_size, n_residues]
        "all_atom_positions": positions,  # [batch_size, n_residues, 37, 3]
        "all_atom_mask": torch.ones_like(positions[..., 0]),  # [batch_size, n_residues, 37]
    }

    # Step 2: Generate the atom14 mappings dynamically
    protein_dict = make_atom14_masks(protein_dict)

    # Step 3: Use the generated mappings to convert positions
    residx_atom14_to_atom37 = protein_dict["residx_atom14_to_atom37"]
    atom14_atom_exists = protein_dict["atom14_atom_exists"]

    # Convert atom37 to atom14 format
    positions_atom14 = torch.gather(
        positions,
        dim=2,
        index=residx_atom14_to_atom37.unsqueeze(-1).expand(-1, -1, -1, 3),
    )

    # Apply the existence mask
    mask = atom14_atom_exists.unsqueeze(-1).expand(-1, -1, -1, 3)
    positions_atom14 = positions_atom14 * mask

    return positions_atom14


class TmolRewardModel(BaseRewardModel):
    """TMol-based interface energy reward model for protein structure optimization.

    This class implements a reward model that uses TMol to evaluate
    interface energies including hydrogen bonds and electrostatic interactions,
    providing gradients for optimization with specific focus on interchain
    interactions (useful for protein-protein interfaces, salt bridges, etc.).
    """

    IS_FOLDING_MODEL = False
    SUPPORTS_GRAD = True
    SUPPORTS_SAVE_PDB = False

    def __init__(
        self,
        device: torch.device | None = None,
        enable_hbond: bool = True,
        enable_elec: bool = False,
        hbond_weight: float = 1.0,
        elec_weight: float = 1.0,
        reward_type: str = "energy_sum",  # "energy_sum" or "interaction_count"
        energy_threshold: float = -0.6,  # Threshold for counting significant interactions
        normalize_by_interface_size: bool = False,
        structure_source: str | None = None,
    ) -> None:
        """Initialize the Interface Energy reward model.

        Args:
            device: Device to use for computations.
            enable_hbond: Whether to include hydrogen bond energy term.
            enable_elec: Whether to include electrostatic energy term (fa_elec).
            hbond_weight: Weight for the hydrogen bond term.
            elec_weight: Weight for the electrostatic term.
            reward_type: Type of reward - "energy_sum" (sum of interaction energies) or
                        "interaction_count" (number of significant interactions)
            energy_threshold: Energy threshold for counting significant interactions (kcal/mol)
            normalize_by_interface_size: Whether to normalize by number of interface residue pairs
            structure_source: Optional folding model key whose predicted structure should be
                used instead of the generated PDB (resolved by CompositeRewardModel).
        """
        if device is None:
            device = torch.device("cuda")
        self.device = device
        self.structure_source = structure_source
        self.enable_hbond = enable_hbond
        self.enable_elec = enable_elec
        self.hbond_weight = hbond_weight
        self.elec_weight = elec_weight
        self.reward_type = reward_type
        self.energy_threshold = energy_threshold
        self.normalize_by_interface_size = normalize_by_interface_size

        # Validate inputs
        assert reward_type in [
            "energy_sum",
            "interaction_count",
        ], f"reward_type must be 'energy_sum' or 'interaction_count', got {reward_type}"

        if not enable_hbond and not enable_elec:
            raise ValueError("At least one of enable_hbond or enable_elec must be True")

        # Initialize TMol components
        self.param_db = tmol.ParameterDatabase.get_default()
        self.pose_stack = None
        self.energy_sfxn = None
        self.block_pair_scorer = None

        # Setup score function with requested energy terms
        self._setup_energy_scorefunction()

    def _setup_energy_scorefunction(self) -> None:
        """Setup TMol score function with requested energy terms."""
        self.energy_sfxn = ScoreFunction(self.param_db, self.device)

        # Set weights for requested energy terms
        if self.enable_hbond:
            self.energy_sfxn.set_weight(ScoreType.hbond, self.hbond_weight)

        if self.enable_elec:
            self.energy_sfxn.set_weight(ScoreType.fa_elec, self.elec_weight)

        # All other weights remain 0.0 by default

    def _prep_inputs(
        self,
        pdb_path: str = None,
        openfold_dict: dict[str, torch.Tensor] = None,
        coords: torch.Tensor = None,
        require_grad: bool = True,
    ) -> None:
        """Prepare inputs for H-bond scoring.

        Args:
            pdb_path: Path to PDB file to load
            openfold_dict: Dictionary containing OpenFold format data with keys:
                          - positions: atom positions [batch_size, n_residues, n_atoms, 3]
                          - aatype: amino acid types [batch_size, n_residues]
                          - chain_index: chain indices [batch_size, n_residues]
            coords: Optional coordinates tensor to use instead (for updating existing pose_stack)
        """
        # Validate input arguments
        input_count = sum([pdb_path is not None, openfold_dict is not None])
        if input_count != 1:
            raise ValueError("Must provide exactly one of: pdb_path or openfold_dict")

        # Create pose_stack from PDB file
        if pdb_path is not None:
            self.pose_stack = tmol.pose_stack_from_pdb(pdb_path, device=self.device)

        # Create pose_stack from OpenFold dictionary
        elif openfold_dict is not None:
            # Ensure all tensors are on the correct device
            openfold_dict_device = {}
            for key, value in openfold_dict.items():
                if isinstance(value, torch.Tensor):
                    openfold_dict_device[key] = value.to(self.device)
                else:
                    openfold_dict_device[key] = value

            self.pose_stack = tmol.pose_stack_from_openfold(openfold_dict_device)

        # Override coordinates if provided
        if coords is not None:
            coords = coords.to(self.device)
            if self.pose_stack is not None:
                self.pose_stack.coords = coords
            else:
                raise ValueError(
                    "Must provide pdb_path or openfold_dict if coords is provided without existing pose_stack"
                )

        # Ensure coordinates require gradients
        if require_grad:
            self.pose_stack.coords.requires_grad_(True)
        else:
            self.pose_stack.coords.requires_grad_(False)
        if self.pose_stack.coords.grad is not None:
            self.pose_stack.coords.grad.zero_()

        # Create block-pair scoring module for residue-residue energies
        self.block_pair_scorer = self.energy_sfxn.render_block_pair_scoring_module(self.pose_stack)

    def _get_interchain_mask(self, pose_idx: int = 0) -> torch.Tensor:
        """Create mask for interchain residue pairs.

        Args:
            pose_idx: Index of pose in pose stack

        Returns:
            Boolean mask where True indicates interchain residue pairs
        """
        # Get chain assignments for each residue
        chain_indices = chain_inds_for_pose_stack(self.pose_stack)
        chains = torch.as_tensor(chain_indices[pose_idx]).to(self.device)

        # Get dimensions for the mask
        # n_residues = len(chains)
        # inter_chain_mask = torch.zeros(
        #     (n_residues, n_residues),
        #     dtype=torch.bool,
        #     device=self.device
        # )

        # Create interchain mask - True where residues are from different chains
        inter_chain_mask = (chains[:, None] != chains[None, :]) & (chains[:, None] >= 0) & (chains[None, :] >= 0)
        # for i in range(n_residues):
        #     for j in range(n_residues):
        #         if (i != j and
        #             chains[i] >= 0 and chains[j] >= 0 and  # Both real residues
        #             chains[i] != chains[j]):  # Different chains
        #             inter_chain_mask[i, j] = True

        return inter_chain_mask

    def _compute_interface_energies(self) -> dict[str, torch.Tensor]:
        """Compute block-pair interface energies for enabled terms.

        Returns:
            Dictionary with energy components:
                - total: Combined weighted energies (n_poses x max_blocks x max_blocks)
                - hbond: H-bond energies (if enabled)
                - elec: Electrostatic energies (if enabled)
                - unweighted_scores: Raw unweighted scores from TMol
        """
        # Get all block-pair energies for enabled terms
        # Shape: n_poses x n_terms x max_blocks x max_blocks
        block_pair_energies = self.block_pair_scorer.unweighted_scores(self.pose_stack.coords)
        # Initialize result dictionary
        energy_dict = {"unweighted_scores": block_pair_energies.detach().cpu()}
        # Extract individual energy terms based on what's enabled
        term_idx = 0
        individual_energies = []

        if self.enable_elec:
            elec_energies = block_pair_energies[term_idx, ...] * self.elec_weight
            energy_dict["elec"] = elec_energies.squeeze(0)
            individual_energies.append(elec_energies)
            term_idx += 1

        if self.enable_hbond:
            hbond_energies = block_pair_energies[term_idx, ...] * self.hbond_weight
            energy_dict["hbond"] = hbond_energies.squeeze(0)
            individual_energies.append(hbond_energies)
            term_idx += 1

        # Combine all enabled energy terms
        if len(individual_energies) == 1:
            total_energies = individual_energies[0].squeeze(0)
        else:
            total_energies = torch.stack(individual_energies, dim=0).sum(dim=0).squeeze(0)

        energy_dict["total"] = total_energies

        return energy_dict

    def _compute_reward(
        self,
        energy_dict: dict[str, torch.Tensor],
        inter_chain_mask: torch.Tensor,
        pose_idx: int = 0,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Compute reward based on interchain interface energies.

        Args:
            energy_dict: Dictionary containing energy tensors for different types
            inter_chain_mask: Interchain mask
            pose_idx: Index of pose

        Returns:
            Tuple of (total_reward, reward_components_dict)
        """
        # Apply mask to get only inter-chain energies
        total_energies = energy_dict["total"]
        interface_total_energies = total_energies * inter_chain_mask

        # Compute reward based on specified type
        if self.reward_type == "energy_sum":
            # Sum of interface energies (more negative = better for both H-bonds and elec)
            # We negate because we want to maximize favorable interactions (minimize positive reward)
            total_reward = -torch.sum(interface_total_energies)
        elif self.reward_type == "interaction_count":
            # Count of significant interactions
            total_reward = torch.sum((interface_total_energies < self.energy_threshold).float())

        # Normalize by interface size if requested
        if self.normalize_by_interface_size:
            n_interface_pairs = torch.sum(inter_chain_mask.float())
            if n_interface_pairs > 0:
                total_reward = total_reward / n_interface_pairs

        # Compute reward components for each energy type
        reward_components = {
            "total_interface_energy": torch.sum(interface_total_energies),
            "n_interface_interactions": torch.sum((interface_total_energies < self.energy_threshold).float()),
            "n_interface_pairs": torch.sum(inter_chain_mask.float()),
            "mean_interface_energy": (
                torch.mean(interface_total_energies[inter_chain_mask])
                if torch.sum(inter_chain_mask) > 0
                else torch.tensor(0.0)
            ),
            "strongest_interface_interaction": (
                torch.min(interface_total_energies[inter_chain_mask])
                if torch.sum(inter_chain_mask) > 0
                else torch.tensor(0.0)
            ),
        }

        # Add individual energy type components if enabled
        if self.enable_hbond and "hbond" in energy_dict:
            interface_hbond_energies = energy_dict["hbond"] * inter_chain_mask
            reward_components.update(
                {
                    "total_interface_hbond_energy": torch.sum(interface_hbond_energies),
                    "n_interface_hbonds": torch.sum((interface_hbond_energies < self.energy_threshold).float()),
                    "mean_interface_hbond_energy": (
                        torch.mean(interface_hbond_energies[inter_chain_mask])
                        if torch.sum(inter_chain_mask) > 0
                        else torch.tensor(0.0)
                    ),
                    "strongest_interface_hbond": (
                        torch.min(interface_hbond_energies[inter_chain_mask])
                        if torch.sum(inter_chain_mask) > 0
                        else torch.tensor(0.0)
                    ),
                }
            )

        if self.enable_elec and "elec" in energy_dict:
            interface_elec_energies = energy_dict["elec"] * inter_chain_mask
            reward_components.update(
                {
                    "total_interface_elec_energy": torch.sum(interface_elec_energies),
                    "n_interface_elec_interactions": torch.sum(
                        (interface_elec_energies < self.energy_threshold).float()
                    ),
                    "mean_interface_elec_energy": (
                        torch.mean(interface_elec_energies[inter_chain_mask])
                        if torch.sum(inter_chain_mask) > 0
                        else torch.tensor(0.0)
                    ),
                    "strongest_interface_elec": (
                        torch.min(interface_elec_energies[inter_chain_mask])
                        if torch.sum(inter_chain_mask) > 0
                        else torch.tensor(0.0)
                    ),
                }
            )

        return total_reward, reward_components

    def score(
        self,
        pdb_path: str,
        requires_grad: bool = False,
        sequence: torch.Tensor | None = None,
        structure: torch.Tensor | None = None,
        binder_chain: str | None = None,
        target_chain: str | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        """Calculate interface energy reward and gradients for a given structure.

        This method evaluates the structure using the enabled energy terms (H-bonds and/or
        electrostatics) and returns detailed energy breakdowns and gradients for optimization.

        Args:
            pdb_path: Path to PDB file (required)
            requires_grad: Whether to calculate gradients
            sequence: Not used (required by interface)
            structure: Optional coordinates tensor to override existing pose_stack coordinates (alias for coords)
            binder_chain: Not used (required by interface)
            target_chain: Not used (required by interface)
            **kwargs: Additional arguments:
                - openfold_dict: Dictionary containing OpenFold format data with keys:
                  - positions: atom positions [batch_size, n_residues, n_atoms, 3]
                  - aatype: amino acid types [batch_size, n_residues]
                  - chain_index: chain indices [batch_size, n_residues]
                - coords: Optional coordinates tensor (alias for structure)

        Returns:
            Dictionary with reward components and gradients:
                reward: Dict[str, torch.Tensor]  # Individual reward metrics for each energy type
                total_reward: torch.Tensor  # Combined total reward value
                grad: Dict[str, torch.Tensor]  # Gradients w.r.t. coordinates
                energy_dict: Dict[str, torch.Tensor]  # Full energy matrices by type
                interface_energies: Dict[str, torch.Tensor]  # Interchain energies by type
                inter_chain_mask: torch.Tensor  # Boolean mask for interchain pairs
        """
        try:
            # Support both structure and coords for backward compatibility
            coords = structure if structure is not None else kwargs.get("coords")
            openfold_dict = kwargs.get("openfold_dict")

            # _prep_inputs requires exactly one of pdb_path / openfold_dict.
            # Prefer openfold_dict when provided (avoids re-parsing the PDB).
            if openfold_dict is not None:
                self._prep_inputs(
                    openfold_dict=openfold_dict,
                    coords=coords,
                    require_grad=requires_grad,
                )
            else:
                self._prep_inputs(pdb_path=pdb_path, coords=coords, require_grad=requires_grad)

            # Compute interface energies for enabled terms
            energy_dict = self._compute_interface_energies()

            # Get interchain mask
            inter_chain_mask = self._get_interchain_mask(pose_idx=0)

            # Compute reward
            total_reward, reward_components = self._compute_reward(energy_dict, inter_chain_mask, pose_idx=0)

            # Compute gradients if requested
            grad_dict = {}
            if requires_grad:
                total_reward.backward()
                grad_dict["structure"] = self.pose_stack.coords.grad.clone().detach().cpu()

            # Prepare interface energies breakdown
            interface_energies = {}
            for energy_type, energies in energy_dict.items():
                if energy_type != "unweighted_scores":
                    interface_energies[f"interface_{energy_type}"] = (energies[0] * inter_chain_mask).detach().cpu()

            # Log scalar results
            logger.info("TmolRewardModel score results:")
            # logger.info("  total_reward: %.4f", total_reward.item())
            for k, v in reward_components.items():
                if isinstance(v, torch.Tensor) and v.numel() == 1:
                    logger.info("  %s: %.4f", k, v.item())
            logger.info("total_reward: %.4f", total_reward.detach().cpu().item())
            logger.info("--------------------------------")
            return standardize_reward(
                reward={k: v.detach().cpu() for k, v in reward_components.items()},
                grad=grad_dict,
                total_reward=total_reward.detach().cpu(),
                energy_dict={k: v.detach().cpu() for k, v in energy_dict.items()},
                interface_energies=interface_energies,
                inter_chain_mask=inter_chain_mask.cpu(),
            )

        except Exception as e:
            # Ensure cleanup even on error
            self._cleanup_state()
            raise e
        finally:
            # Clear model state
            self._clear_model_state()

    def _clear_model_state(self) -> None:
        """Clear internal model state dictionaries."""
        if hasattr(self, "pose_stack") and self.pose_stack is not None:
            if hasattr(self.pose_stack, "coords") and self.pose_stack.coords is not None:
                if self.pose_stack.coords.grad is not None:
                    self.pose_stack.coords.grad = None

    def _cleanup_state(self) -> None:
        """Clean up model state and memory."""
        self._clear_model_state()
        gc.collect()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

    def cleanup(self) -> None:
        """Explicit cleanup of model memory.

        Call this method to free up GPU memory and clear state.
        """
        self._cleanup_state()


# Utility functions for interface energy analysis
def analyze_interface_interactions(
    pdb_path: str = None,
    openfold_dict: dict[str, torch.Tensor] = None,
    energy_threshold: float = -0.1,
    device: torch.device | None = None,
    enable_hbond: bool = True,
    enable_elec: bool = False,
    hbond_weight: float = 1.0,
    elec_weight: float = 1.0,
) -> dict[str, Any]:
    """Analyze interface interactions (H-bonds and/or electrostatics) in a protein complex.

    Args:
        pdb_path: Path to PDB file (mutually exclusive with openfold_dict)
        openfold_dict: OpenFold format dictionary (mutually exclusive with pdb_path)
        energy_threshold: Energy threshold for significant interactions
        device: Device for computation
        enable_hbond: Whether to include hydrogen bond analysis
        enable_elec: Whether to include electrostatic analysis
        hbond_weight: Weight for H-bond term
        elec_weight: Weight for electrostatic term

    Returns:
        Dictionary with interface interaction analysis results
    """
    if device is None:
        device = torch.device("cuda")
    energy_model = TmolRewardModel(
        device=device,
        enable_hbond=enable_hbond,
        enable_elec=enable_elec,
        hbond_weight=hbond_weight,
        elec_weight=elec_weight,
        reward_type="energy_sum",
        energy_threshold=energy_threshold,
    )

    result = energy_model.score(pdb_path=pdb_path, openfold_dict=openfold_dict, require_grad=False)

    analysis = {
        "total_interface_energy": result[REWARD_KEY]["total_interface_energy"],
        "n_interface_interactions": result[REWARD_KEY]["n_interface_interactions"],
        "n_interface_pairs": result[REWARD_KEY]["n_interface_pairs"],
        "mean_interface_energy": result[REWARD_KEY]["mean_interface_energy"],
        "strongest_interaction": result[REWARD_KEY]["strongest_interface_interaction"],
    }

    if enable_hbond and "total_interface_hbond_energy" in result[REWARD_KEY]:
        hbond_analysis = _analyze_energy_type(
            result["interface_energies"]["interface_hbond"],
            result["inter_chain_mask"],
            energy_threshold,
            "hbond",
        )
        analysis.update(
            {
                "total_interface_hbond_energy": result[REWARD_KEY]["total_interface_hbond_energy"],
                "n_interface_hbonds": result[REWARD_KEY]["n_interface_hbonds"],
                "mean_interface_hbond_energy": result[REWARD_KEY]["mean_interface_hbond_energy"],
                "strongest_hbonds": hbond_analysis["strongest_interactions"],
            }
        )

    if enable_elec and "total_interface_elec_energy" in result[REWARD_KEY]:
        elec_analysis = _analyze_energy_type(
            result["interface_energies"]["interface_elec"],
            result["inter_chain_mask"],
            energy_threshold,
            "elec",
        )
        analysis.update(
            {
                "total_interface_elec_energy": result[REWARD_KEY]["total_interface_elec_energy"],
                "n_interface_elec_interactions": result[REWARD_KEY]["n_interface_elec_interactions"],
                "mean_interface_elec_energy": result[REWARD_KEY]["mean_interface_elec_energy"],
                "strongest_elec_interactions": elec_analysis["strongest_interactions"],
            }
        )

    return analysis


def _analyze_energy_type(
    interface_energies: torch.Tensor,
    inter_chain_mask: torch.Tensor,
    energy_threshold: float,
    energy_type: str,
) -> dict[str, Any]:
    """Analyze strongest interactions for a specific energy type.

    Args:
        interface_energies: Interface energy matrix for specific type
        inter_chain_mask: Boolean mask for interchain pairs
        energy_threshold: Threshold for significant interactions
        energy_type: Name of energy type for labeling

    Returns:
        Dictionary with analysis results including strongest interactions
    """
    # Find strongest interactions
    interface_energies_np = interface_energies.numpy()
    nonzero_indices = np.nonzero(interface_energies_np)

    strongest_interactions = []
    if len(nonzero_indices[0]) > 0:
        energies = interface_energies_np[nonzero_indices]
        sorted_indices = np.argsort(energies)[:10]  # Top 10 strongest

        for idx in sorted_indices:
            i, j = nonzero_indices[0][idx], nonzero_indices[1][idx]
            energy = energies[idx]
            strongest_interactions.append(
                {
                    "residue_i": i,
                    "residue_j": j,
                    "energy": energy,
                    "energy_type": energy_type,
                }
            )

    return {
        "strongest_interactions": strongest_interactions,
        "n_significant": len([e for e in strongest_interactions if e["energy"] < energy_threshold]),
    }


# Backward compatibility function
def analyze_interface_hbonds(
    pdb_path: str = None,
    openfold_dict: dict[str, torch.Tensor] = None,
    energy_threshold: float = -0.6,
    device: torch.device | None = None,
) -> dict[str, Any]:
    """Analyze interface hydrogen bonds in a protein complex.

    This function is maintained for backward compatibility. For new code,
    use analyze_interface_interactions() which supports both H-bonds and electrostatics.

    Args:
        pdb_path: Path to PDB file (mutually exclusive with openfold_dict)
        openfold_dict: OpenFold format dictionary (mutually exclusive with pdb_path)
        energy_threshold: Energy threshold for significant H-bonds
        device: Device for computation

    Returns:
        Dictionary with H-bond analysis results
    """
    if device is None:
        device = torch.device("cuda")
    return analyze_interface_interactions(
        pdb_path=pdb_path,
        openfold_dict=openfold_dict,
        energy_threshold=energy_threshold,
        device=device,
        enable_hbond=True,
        enable_elec=False,
    )


# Example usage
# def example_usage():
#     """Example of how to use the TmolRewardModel."""

#     print("=== TmolRewardModel Examples ===\n")

#     # Example 1: H-bond only model (backward compatible)
#     print("Example 1: H-bond only model")
#     hbond_model = TmolRewardModel(
#         device=torch.device("cuda"),
#         enable_hbond=True,
#         enable_elec=False,
#         hbond_weight=1.0,
#         reward_type="energy_sum",
#         energy_threshold=-0.6
#     )

#     # Example 2: Electrostatic only model
#     print("\nExample 2: Electrostatic only model")
#     elec_model = TmolRewardModel(
#         device=torch.device("cuda"),
#         enable_hbond=False,
#         enable_elec=True,
#         elec_weight=1.0,
#         reward_type="energy_sum",
#         energy_threshold=-0.6
#     )

#     # Example 3: Combined H-bond + electrostatic model
#     print("\nExample 3: Combined H-bond + electrostatic model")
#     combined_model = TmolRewardModel(
#         device=torch.device("cuda"),
#         enable_hbond=True,
#         enable_elec=True,
#         hbond_weight=1.0,
#         elec_weight=0.5,  # Different weight for electrostatics
#         reward_type="energy_sum",
#         energy_threshold=-0.6
#     )

#     # Score existing structure from PDB (if available)
#     pdb_file = "path/to/your/test.pdb"
#     try:
#         result = combined_model.score(
#             pdb_path=pdb_file,
#             require_grad=True
#         )

#         print(f"\nPDB Results (Combined Model):")
#         print(f"Total interface reward: {result['total_reward']:.4f}")
#         print(f"Combined interface energy: {result['reward']['total_interface_energy']:.4f}")
#         print(f"Interface H-bond energy: {result['reward'].get('total_interface_hbond_energy', 'N/A')}")
#         print(f"Interface elec energy: {result['reward'].get('total_interface_elec_energy', 'N/A')}")
#         print(f"Number of interface interactions: {result['reward']['n_interface_interactions']}")
#         print(f"Gradient shape: {result['grad']['structure'].shape if 'structure' in result['grad'] else None}")
#     except FileNotFoundError:
#         print(f"PDB file {pdb_file} not found, skipping PDB example")


#     # Dataset example using OpenFold format
#     config_path = "configs/dataset/afdb_preprocessed/"
#     config_name = "afdb_cathdimer"
#     try:
#         with hydra.initialize(config_path, version_base=hydra.__version__):
#             cfg = hydra.compose(config_name=config_name)

#         with open_dict(cfg):
#             cfg.datamodule.transforms = [
#                 {
#                     '_target_': 'proteinfoundation.datasets.transforms.TEDDimerTransform',
#                     'enable_CAT': False,
#                     'chunk_size': 15_000_000,
#                 },
#             ]

#         print(f"\nDataset transforms: {cfg.datamodule.transforms}")

#         random.seed(0)
#         torch.manual_seed(0)
#         cfg.datamodule.datasplitter.train_val_test = [1, 0.0, 0.0]
#         cfg.datamodule.num_workers = 0
#         sampling_mode = cfg.datamodule.sampling_mode
#         datamodule = hydra.utils.instantiate(cfg.datamodule)
#         datamodule.prepare_data()
#         datamodule.setup("fit")
#         train_ds = datamodule.train_ds
#         split = "train"
#         ds = datamodule._get_dataset(split)
#         sequence_id_to_idx = ds.protein_to_idx
#         clusterid_to_seqid_mapping = datamodule.clusterid_to_seqid_mappings[split]
#         cluster_names = list(clusterid_to_seqid_mapping.keys())

#         idx = 183677
#         cluster_name = cluster_names[idx]
#         if sampling_mode == "cluster-reps":
#             seq_id = cluster_name
#         elif sampling_mode == "cluster-random":
#             seq_id = random.choice(clusterid_to_seqid_mapping[cluster_name])
#         full_dimer_id = seq_id
#         monomer_id = full_dimer_id.split("_", 1)[1]
#         ds_index = sequence_id_to_idx[monomer_id]
#         full_idx = (ds_index, full_dimer_id)
#         print(f"\nDataset sample: {full_idx}")

#         sample = ds[full_idx]
#         atom_37_pos = sample["coords"].unsqueeze(0)
#         atom_14_pos = convert_atom37_to_atom14_openfold(atom_37_pos, sample["residue_type"].unsqueeze(0))
#         openfold_dict = {
#             "positions": atom_14_pos.unsqueeze(0),
#             "aatype": sample["residue_type"].unsqueeze(0),
#             "chain_index": sample["chains"].unsqueeze(0)
#         }

#         # Test all three models on the same structure
#         models = {
#             "H-bond only": hbond_model,
#             "Electrostatic only": elec_model,
#             "Combined": combined_model
#         }

#         for model_name, model in models.items():
#             result = model.score(
#                 openfold_dict=openfold_dict,
#                 require_grad=True
#             )

#             print(f"\n{model_name} Results:")
#             print(f"  Total reward: {result['total_reward']:.4f}")
#             print(f"  Combined interface energy: {result['reward']['total_interface_energy']:.4f}")
#             if 'total_interface_hbond_energy' in result['reward']:
#                 print(f"  H-bond energy: {result['reward']['total_interface_hbond_energy']:.4f}")
#                 print(f"  Number of H-bonds: {result['reward']['n_interface_hbonds']}")
#             if 'total_interface_elec_energy' in result['reward']:
#                 print(f"  Electrostatic energy: {result['reward']['total_interface_elec_energy']:.4f}")
#                 print(f"  Number of elec interactions: {result['reward']['n_interface_elec_interactions']}")
#             print(f"  Gradient shape: {result['grad']['structure'].shape if 'structure' in result['grad'] else None}")

#         # Demonstrate analysis function
#         print(f"\n=== Interface Analysis Example ===")
#         analysis = analyze_interface_interactions(
#             openfold_dict=openfold_dict,
#             enable_hbond=True,
#             enable_elec=True,
#             hbond_weight=1.0,
#             elec_weight=0.5
#         )

#         print(f"Analysis results:")
#         print(f"  Total interface pairs: {analysis['n_interface_pairs']}")
#         print(f"  Total interactions: {analysis['n_interface_interactions']}")
#         if 'n_interface_hbonds' in analysis:
#             print(f"  H-bonds: {analysis['n_interface_hbonds']}")
#         if 'n_interface_elec_interactions' in analysis:
#             print(f"  Electrostatic interactions: {analysis['n_interface_elec_interactions']}")

#     except Exception as e:
#         print(f"Dataset example failed: {e}")
#         print("This is expected if the dataset configuration is not available")


# if __name__ == "__main__":

#     example_usage()
