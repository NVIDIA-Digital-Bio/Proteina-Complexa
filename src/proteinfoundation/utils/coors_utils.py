# MIT License

# Copyright (c) Microsoft Corporation.

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE

# (We extract the function to adjust oxygen position from FrameFlow)


from math import prod

import torch
from jaxtyping import Bool, Float
from openfold.np.residue_constants import (
    restype_atom14_mask,
    restype_atom14_rigid_group_positions,
    restype_atom14_to_rigid_group,
    restype_rigid_group_default_frame,
)
from openfold.utils.all_atom_multimer import atom14_to_atom37
from openfold.utils.feats import frames_and_literature_positions_to_atom14_pos, torsion_angles_to_frames
from openfold.utils.rigid_utils import Rigid, Rotation
from scipy.spatial.transform import Rotation as Scipy_Rotation
from torch import Tensor

nm_to_ang_scale = 10.0
ang_to_nm = lambda trans: trans / nm_to_ang_scale
nm_to_ang = lambda trans: trans * nm_to_ang_scale


def get_atom37_ca_mask(n: int, device: torch.device) -> Bool[Tensor, "n 37"]:
    """
    Returns an atom37 mask that only keeps CA.

    Args:
        n: number of residues

    Returns:
        Boolean mask of shape [n, 37] where all zeros except [:, 1] which
        is ones (indicates CA).
    """
    mask = torch.zeros((n, 37), device=device, dtype=torch.bool)  # [n, 37]
    mask[:, 1] = True
    return mask.bool()


def get_atom37_bb3_mask(n: int, device: torch.device) -> Bool[Tensor, "n 37"]:
    """
    Returns an atom37 mask that only keeps [N CA C].

    Args:
        n: number of residues

    Returns:
        Boolean mask of shape [n, 37] where all zeros except [:, [0, 1, 2]] which
        is ones (indicates [N CA C]).
    """
    mask = torch.zeros((n, 37), device=device, dtype=torch.bool)  # [n, 37]
    mask[:, 0] = True
    mask[:, 1] = True
    mask[:, 2] = True
    return mask.bool()


def get_atom37_bb3o_mask(n: int, device: torch.device) -> Bool[Tensor, "n 37"]:
    """
    Returns an atom37 mask that only keeps [N CA C O].

    Args:
        n: number of residues

    Returns:
        Boolean mask of shape [n, 37] where all zeros except [:, [0, 1, 2, 4]] which
        is ones (indicates [N CA C O]).
    """
    mask = torch.zeros((n, 37), device=device, dtype=torch.bool)  # [n, 37]
    mask[:, 0] = True
    mask[:, 1] = True
    mask[:, 2] = True
    mask[:, 4] = True
    return mask.bool()


def trans_nm_and_rot_to_atom37(trans, rot, impute_ox=False):
    """
    Converts rigids composed by rotations and translations (in nm) to atom37 representation.

    We rely on openfold's functionality.

    Args:
        trans: Translations in nm, shape [*, N, 3]
        rot: Rotations, shape [*, N, N, 3]
        impute_ox: Whether to impuite oxygen position or not

    Returns:
        Coordinates in atom37 representation
    """
    return trans_ang_and_rot_to_atom37(nm_to_ang(trans), rot, impute_ox=impute_ox)


def trans_ang_and_rot_to_atom37(trans, rot, impute_ox=False):
    """
    Converts rigids composed by rotations and translations (in Å) to atom37 representation.

    We rely on openfold's functionality.

    Args:
        trans: Translations in Å, shape [*, N, 3]
        rot: Rotations, shape [*, N, N, 3]
        impute_ox: Whether to impuite oxygen position or not

    Returns:
        Coordinates in atom37 representation
    """
    return openfold_bb_frames_to_atom37(Rigid(Rotation(rot_mats=rot, quats=None), trans), impute_ox=impute_ox)


def trans_nm_to_atom37(ca_coors_nm):
    """
    Converts CA positions (in nm) into atom37 representation (in Å).

    We rely on openfold's functionality.

    Args:
        ca_coors: CA coordinates in nm, shape [*, N, 3]

    Returns:
        Coordinates in atom37 representation (in Å)
    """
    return trans_ang_to_atom37(nm_to_ang(ca_coors_nm))


def trans_ang_to_atom37(ca_coors):
    """
    Converts CA positions (in Å) into atom37 representation.

    We rely on openfold's functionality.

    Args:
        ca_coors: CA coordinates in Å, shape [*, N, 3]

    Returns:
        Coordinates in atom37 representation
    """
    original_shape = ca_coors.shape  # [*, N, 3]
    atom37_shape = list(original_shape[:-1]) + [37, original_shape[-1]]  # [*, N, 37, 3]
    ca_coors_atom37 = torch.zeros(atom37_shape, dtype=ca_coors.dtype, device=ca_coors.device)  # [*, N, 37, 3]
    ca_coors_atom37[..., 1, :] = ca_coors  # Sets correct positions for CA [*, N, 37, 3]
    return ca_coors_atom37


def openfold_bb_frames_to_atom37(frames, impute_ox=False):
    """Converts backbone frames to coordinates of backbone atoms.

    Note: This computes sidechains for some random angles (not predicted). We only need this
    to use openfold's functionality, we produce these but do not use them.

    Note: The atom ordering in atom14 and atom37 representations is given by
    atom14 is [N CA C O ...]
    atom37 is [N CA C CB O ...]

    Args:
        frames: Batch of openfold rigids, shape [*, N]
        impute_ox: Whether to impute oxygen coordinates as done by FrameFlow

    Returms:
        Coordinates in atom37 representation
    """
    # Load residue constants
    default_frames = torch.tensor(
        restype_rigid_group_default_frame,
        dtype=frames.dtype,
        device=frames.device,
        requires_grad=False,
    )
    group_idx = torch.tensor(restype_atom14_to_rigid_group, device=frames.device, requires_grad=False)
    atom_mask = torch.tensor(
        restype_atom14_mask,
        dtype=frames.dtype,
        device=frames.device,
        requires_grad=False,
    )
    lit_positions = torch.tensor(
        restype_atom14_rigid_group_positions,
        dtype=frames.dtype,
        device=frames.device,
        requires_grad=False,
    )

    # print(default_frames.shape)  # [21, 8, 4, 4] 21 aatypes, 8 frames (bb, torsion x 3, sidechain x 4)
    # print(group_idx.shape)  # [21, 14] 21 aatypes, 14 from atom representation
    # print(atom_mask.shape)  # [21, 14] 21 aatypes, 14 from atom representation
    # print(lit_positions.shape)  # [21, 14, 3] 21 aatypes, 14 from atom representation, 3 coords

    # Transform rigid to rotations, done by openfold, no clear reason...
    backb_to_global = Rigid(
        Rotation(rot_mats=frames.get_rots().get_rot_mats(), quats=None),
        frames.get_trans(),
    )

    # Define angles which we are not predicting
    angles = torch.randn(frames.shape + (7, 2), device=frames.device) * 0.001 + 1.0

    # Define aatype for each residue, set all to 1 as does not affect backbone
    aatype = torch.ones(frames.shape).long()  # aatype indexing, all type 1 to try

    all_frames_to_global = torsion_angles_to_frames(backb_to_global, angles, aatype, default_frames)
    coords_atom14 = frames_and_literature_positions_to_atom14_pos(
        all_frames_to_global,
        aatype,
        default_frames,
        group_idx,
        atom_mask,
        lit_positions,
    )
    coords_atom37 = atom14_to_atom37(coords_atom14, torch.zeros(frames.shape, device=frames.device).int())[0]
    if not impute_ox:
        return coords_atom37
    return batch_adjust_oxygen_pos(coords_atom37.clone())  # Changes in place otherwise


def batch_adjust_oxygen_pos(atom_37):
    assert atom_37.ndim == 4  # [b, n, 37, 3]
    return torch.stack([adjust_oxygen_pos(atom_37[i]) for i in range(atom_37.shape[0])])


# Extracted from frameflow
# https://github.com/microsoft/protein-frame-flow
def adjust_oxygen_pos(atom_37: torch.Tensor, pos_is_known=None) -> torch.Tensor:
    """
    Imputes the position of the oxygen atom on the backbone by using adjacent frame information.
    Specifically, we say that the oxygen atom is in the plane created by the Calpha and C from the
    current frame and the nitrogen of the next frame. The oxygen is then placed c_o_bond_length Angstrom
    away from the C in the current frame in the direction away from the Ca-C-N triangle.

    For cases where the next frame is not available, for example we are at the C-terminus or the
    next frame is not available in the data then we place the oxygen in the same plane as the
    N-Ca-C of the current frame and pointing in the same direction as the average of the
    Ca->C and Ca->N vectors.

    Args:
        atom_37 (torch.Tensor): (N, 37, 3) tensor of positions of the backbone atoms in atom_37 ordering
                                which is ['N', 'CA', 'C', 'CB', 'O', ...]
        pos_is_known (torch.Tensor): (N,) mask for known residues.
    """

    N = atom_37.shape[0]
    assert atom_37.shape == (N, 37, 3)

    # Get vectors to Carbonly from Carbon alpha and N of next residue. (N-1, 3)
    # Note that the (N,) ordering is from N-terminal to C-terminal.

    # Calpha to carbonyl both in the current frame.
    calpha_to_carbonyl: torch.Tensor = (atom_37[:-1, 2, :] - atom_37[:-1, 1, :]) / (
        torch.norm(atom_37[:-1, 2, :] - atom_37[:-1, 1, :], keepdim=True, dim=1) + 1e-7
    )
    # For masked positions, they are all 0 and so we add 1e-7 to avoid division by 0.
    # The positions are in Angstroms and so are on the order ~1 so 1e-7 is an insignificant change.

    # Nitrogen of the next frame to carbonyl of the current frame.
    nitrogen_to_carbonyl: torch.Tensor = (atom_37[:-1, 2, :] - atom_37[1:, 0, :]) / (
        torch.norm(atom_37[:-1, 2, :] - atom_37[1:, 0, :], keepdim=True, dim=1) + 1e-7
    )

    carbonyl_to_oxygen: torch.Tensor = calpha_to_carbonyl + nitrogen_to_carbonyl  # (N-1, 3)
    carbonyl_to_oxygen = carbonyl_to_oxygen / (torch.norm(carbonyl_to_oxygen, dim=1, keepdim=True) + 1e-7)

    atom_37[:-1, 4, :] = atom_37[:-1, 2, :] + carbonyl_to_oxygen * 1.23

    # Now we deal with frames for which there is no next frame available.

    # Calpha to carbonyl both in the current frame. (N, 3)
    calpha_to_carbonyl_term: torch.Tensor = (atom_37[:, 2, :] - atom_37[:, 1, :]) / (
        torch.norm(atom_37[:, 2, :] - atom_37[:, 1, :], keepdim=True, dim=1) + 1e-7
    )
    # Calpha to nitrogen both in the current frame. (N, 3)
    calpha_to_nitrogen_term: torch.Tensor = (atom_37[:, 0, :] - atom_37[:, 1, :]) / (
        torch.norm(atom_37[:, 0, :] - atom_37[:, 1, :], keepdim=True, dim=1) + 1e-7
    )
    carbonyl_to_oxygen_term: torch.Tensor = calpha_to_carbonyl_term + calpha_to_nitrogen_term  # (N, 3)
    carbonyl_to_oxygen_term = carbonyl_to_oxygen_term / (
        torch.norm(carbonyl_to_oxygen_term, dim=1, keepdim=True) + 1e-7
    )

    # Create a mask that is 1 when the next residue is not available either
    # due to this frame being the C-terminus or the next residue is not
    # known due to pos_is_known being false.

    if pos_is_known is None:
        pos_is_known = torch.ones((atom_37.shape[0],), dtype=torch.int64, device=atom_37.device)

    next_res_gone: torch.Tensor = ~pos_is_known.bool()  # (N,)
    next_res_gone = torch.cat([next_res_gone, torch.ones((1,), device=pos_is_known.device).bool()], dim=0)  # (N+1, )
    next_res_gone = next_res_gone[1:]  # (N,)

    atom_37[next_res_gone, 4, :] = atom_37[next_res_gone, 2, :] + carbonyl_to_oxygen_term[next_res_gone, :] * 1.23

    return atom_37


def sample_uniform_rotation(shape=tuple(), dtype=None, device=None) -> Float[Tensor, "*batch 3 3"]:
    """
    Samples rotations distributed uniformly. Adapted from FrameFlow's code.
    https://github.com/microsoft/protein-frame-flow/blob/main/data/so3_utils.py

    Args:
        shape: tuple (if empty then samples single rotation)
        dtype: used for samples
        device: torch.device

    Returns:
        Uniformly samples rotation matrices [*shape, 3, 3]
    """
    return torch.tensor(
        Scipy_Rotation.random(prod(shape)).as_matrix(),
        device=device,
        dtype=dtype,
    ).reshape(*shape, 3, 3)
