import torch


def normalize_last_dim(v):
    norm = torch.linalg.norm(v, dim=-1, keepdim=True)
    return v / torch.clamp(norm, min=1e-8, max=None)


def bond_angles(a, b, c):
    """
    Computes bond angles for the 3 points a, b, c.
    Since torch.linalg.cross and torch.linalg.cross  support
    broadcasting, this supports broadcasting.

    Args:
        a, b, c: Each is a tensor of shape [*, 3]

    Returns:
        Angle between 0 and pi, shape [*]
    """
    b0 = b - a  # [*, 3]
    b1 = c - a  # [*, 3]
    b0, b1 = map(normalize_last_dim, (b0, b1))  # [*, 3] each
    cos_angle = torch.linalg.vecdot(b0, b1, dim=-1)  # [*]
    cross = torch.linalg.cross(b0, b1, dim=-1)  # [*, 3]
    sin_angle = torch.linalg.norm(cross, dim=-1)  # [*]
    return torch.atan2(sin_angle, cos_angle)  # [*]


def signed_dihedral_angle(a, b, c, d):
    """
    Compputes the signed angle for the 4 points a, b, c, d.
    Since torch.linalg.cross and torch.linalg.cross  support
    broadcasting, this supports broadcasting.

    Args:
        a, b, c, d: Each is a tensor of shape [*, 3]

    Returns:
        Angle between -pi and pi (signed), shape [*]
    """
    b0 = b - a  # [*, 3]
    b1 = c - b  # [*, 3]
    b2 = d - c  # [*, 3]
    n1 = torch.linalg.cross(b0, b1)  # [*, 3]
    n2 = torch.linalg.cross(b1, b2)  # [*, 3]
    n1, n2 = map(normalize_last_dim, (n1, n2))  # Each [*, 3]

    cos_angle = torch.linalg.vecdot(n1, n2, dim=-1)  # [*]
    n1_cross_n2 = torch.linalg.cross(n1, n2, dim=-1)
    sin_angle_magnitude = torch.linalg.norm(n1_cross_n2, dim=-1)  # [*]
    sign = torch.sign(torch.linalg.vecdot(n1_cross_n2, b1))  # [*]
    return torch.atan2(sign * sin_angle_magnitude, cos_angle)  # [*]
