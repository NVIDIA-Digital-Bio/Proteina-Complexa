import torch


# TODO: can remove after replacing seq pair rep with feature factory
def distance(p, eps=1e-10):
    """
    Compute distances between pairs of Euclidean coordinates.

    Args:
        p:
            [*, 2, 3] Input tensor where the last two dimensions have
            a shape of [2, 3], representing a pair of coordinates in
            the Euclidean space.

    Returns:
        [*] Output tensor of distances, where each distance is computed
        between the pair of Euclidean coordinates in the last two
        dimensions of the input tensor p.
    """
    return (eps + torch.sum((p[..., 0, :] - p[..., 1, :]) ** 2, dim=-1)) ** 0.5


def compute_frenet_frames(coords, chains, mask, eps=1e-10):
    """
    Construct Frenet-Serret frames based on a sequence of coordinates.

    Since the Frenet-Serret frame is constructed based on three consecutive
    residues, for each chain, the rotational component of its first residue
    is assigned with the rotational component of its second residue; and the
    rotational component of its last residue is assigned with the rotational
    component of its second last residue.

    Args:
        coords:
            [B, N, 3] Per-residue atom positions.
        chains:
            [B, N] Per-residue chain indices.
        mask:
            [B, N] Residue mask.
        eps:
            Epsilon for computational stability. Default to 1e-10.

    Returns:
        rots:
            [B, N, 3, 3] Rotational components for the constructed frames.
    """

    # [B, N-1, 3]
    t = coords[:, 1:] - coords[:, :-1]
    t_norm = torch.sqrt(eps + torch.sum(t**2, dim=-1))
    t = t / t_norm.unsqueeze(-1)

    # [B, N-2, 3]
    b = torch.linalg.cross(t[:, :-1], t[:, 1:], dim=-1)
    b_norm = torch.sqrt(eps + torch.sum(b**2, dim=-1))
    b = b / b_norm.unsqueeze(-1)

    # [B, N-2, 3]
    n = torch.linalg.cross(b, t[:, 1:], dim=-1)

    # [B, N-2, 3, 3]
    tbn = torch.stack([t[:, 1:], b, n], dim=-1)

    # Construct rotation matrices
    rots = []
    for i in range(mask.shape[0]):
        rots_ = torch.eye(3).unsqueeze(0).repeat(mask.shape[1], 1, 1)
        length = torch.sum(mask[i]).int()
        rots_[1 : length - 1] = tbn[i, : length - 2]

        # Handle start of chain
        for j in range(length):
            if j == 0 or chains[i][j] != chains[i][j - 1]:
                rots_[j] = rots_[j + 1]

        # Handle end of chain
        for j in range(length):
            if j == length - 1 or chains[i][j] != chains[i][j + 1]:
                rots_[j] = rots_[j - 1]

        # Update
        rots.append(rots_)

    # [B, N, 3, 3]
    rots = torch.stack(rots, dim=0).to(coords.device)

    return rots
