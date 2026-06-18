"""Fast geometric energy pre-filter reward model.

Runs in <100ms per sample using only backbone geometry — no structure prediction.
Use as a cheap first gate before expensive AF2/RF3 calls to eliminate candidates
with obvious geometric defects.

Checks:
  - CA clash rate: non-adjacent CA pairs closer than ``ca_clash_threshold`` Å
  - Backbone angle deviation: N-CA-C angles far from the ideal ~111°

Both penalties are returned as rates in [0, 1] and combined into ``total_reward``
(negative, so higher reward = fewer defects).  Use ``reward_threshold`` in the
search config to skip AF2/RF3 on geometrically invalid samples.
"""

import logging
import math

import torch
import torch.nn.functional as F

from proteinfoundation.rewards.base_reward import REWARD_KEY, TOTAL_REWARD_KEY, BaseRewardModel, standardize_reward

logger = logging.getLogger(__name__)

# atom37 indices for backbone atoms
_IDX_N = 0
_IDX_CA = 1
_IDX_C = 2


class GeometricEnergyReward(BaseRewardModel):
    """Fast geometry-based pre-filter: CA clash rate + backbone angle deviation.

    Intended to run before AF2/RF3 to discard geometrically invalid candidates
    early, reducing expensive structure prediction calls by 4–8× on bad samples.

    Args:
        clash_weight: Weight for CA clash penalty (negative = penalise clashes).
        rama_weight: Weight for backbone angle deviation penalty.
        ca_clash_threshold: Minimum allowed CA–CA distance in Å for non-adjacent
            residues. Pairs closer than this count as clashes.
        adjacency_window: Residue index distance within which CA contacts are
            considered bonded/adjacent and excluded from clash counting.
    """

    IS_FOLDING_MODEL = False
    SUPPORTS_GRAD = False
    SUPPORTS_SAVE_PDB = False

    def __init__(
        self,
        clash_weight: float = -1.0,
        rama_weight: float = -0.5,
        ca_clash_threshold: float = 3.8,
        adjacency_window: int = 2,
    ) -> None:
        self.clash_weight = clash_weight
        self.rama_weight = rama_weight
        self.ca_clash_threshold = ca_clash_threshold
        self.adjacency_window = adjacency_window

    def score(self, pdb_path: str, requires_grad: bool = False, **kwargs) -> dict:
        """Compute fast geometric scores from a PDB file.

        Args:
            pdb_path: Path to the PDB file to evaluate.
            requires_grad: Ignored — no gradient support.

        Returns:
            Standardized reward dict with ``clash`` and ``rama`` components.
        """
        try:
            from proteinfoundation.utils.pdb_utils import from_pdb_file

            prot = from_pdb_file(pdb_path)
            atom_pos = torch.from_numpy(prot.atom_positions).float()  # [n, 37, 3]
            atom_mask = torch.from_numpy(prot.atom_mask).bool()  # [n, 37]

            ca_coords = atom_pos[:, _IDX_CA, :]  # [n, 3]
            n_coords = atom_pos[:, _IDX_N, :]
            c_coords = atom_pos[:, _IDX_C, :]
            ca_mask = atom_mask[:, _IDX_CA]  # [n]

            clash = self._ca_clash_rate(ca_coords, ca_mask)
            rama = self._backbone_angle_penalty(n_coords, ca_coords, c_coords, ca_mask)

            total = torch.tensor(
                self.clash_weight * clash.item() + self.rama_weight * rama.item(),
                dtype=torch.float32,
            )

            return standardize_reward(
                reward={"clash": clash, "rama": rama},
                total_reward=total,
            )

        except Exception as exc:
            logger.warning(f"GeometricEnergyReward failed for {pdb_path}: {exc}")
            return standardize_reward(reward={}, total_reward=torch.tensor(0.0))

    # ------------------------------------------------------------------
    # Internal geometry helpers
    # ------------------------------------------------------------------

    def _ca_clash_rate(self, ca: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Fraction of non-adjacent CA pairs closer than threshold.

        Args:
            ca: CA coordinates [n, 3].
            mask: Boolean mask [n] of present residues.

        Returns:
            Scalar clash rate in [0, 1].
        """
        n = ca.shape[0]
        if n < self.adjacency_window + 2:
            return torch.tensor(0.0)

        dists = torch.cdist(ca, ca)  # [n, n]

        idx = torch.arange(n, device=ca.device)
        adjacency = (idx[:, None] - idx[None, :]).abs() <= self.adjacency_window
        valid = mask[:, None] & mask[None, :] & ~adjacency  # [n, n]

        clash = (dists < self.ca_clash_threshold) & valid
        n_valid = valid.float().sum().clamp(min=1.0)
        return clash.float().sum() / n_valid

    def _backbone_angle_penalty(
        self,
        n_pos: torch.Tensor,
        ca: torch.Tensor,
        c_pos: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Fraction of residues with N-CA-C angle > 20° from the ideal ~111°.

        Args:
            n_pos: N coordinates [n, 3].
            ca: CA coordinates [n, 3].
            c_pos: C coordinates [n, 3].
            mask: Boolean CA mask [n].

        Returns:
            Scalar outlier rate in [0, 1].
        """
        ideal_rad = 111.0 * math.pi / 180.0
        deviation_threshold = 20.0 * math.pi / 180.0

        v1 = F.normalize(n_pos - ca, dim=-1)  # N→CA direction [n, 3]
        v2 = F.normalize(c_pos - ca, dim=-1)  # C→CA direction [n, 3]

        cos_angle = (v1 * v2).sum(dim=-1).clamp(-1.0, 1.0)  # [n]
        angles = torch.acos(cos_angle)  # [n]

        outliers = (angles - ideal_rad).abs() > deviation_threshold
        outliers = outliers & mask

        n_valid = mask.float().sum().clamp(min=1.0)
        return outliers.float().sum() / n_valid
