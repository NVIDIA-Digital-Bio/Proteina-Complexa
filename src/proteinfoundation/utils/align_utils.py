# MIT License

# Copyright (c) 2023 Gray Lab

# Permission is hereby granted, free of charge, to any person obtaining a copy of this
# software and associated documentation files (the "Software"), to deal in the Software
# without restriction, including without limitation the rights to use, copy, modify, merge,
# publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons
# to whom the Software is furnished to do so, subject to the following conditions:

# # The above copyright notice and this permission notice shall be included in all copies or
# substantial portions of the Software.

# # THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
# PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE
# FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
# OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.


import einops
import torch


def mean_w_mask(a, mask, keepdim=True):
    """
    Computes the mean of point cloud a accounting for the mask.

    Args:
        a: Input point cloud of shape [*, n, d]
        mask: Input mask of shape [*, n] of boolean values
        keepdim: whether to keep the dimension across which we're computing the mean
            like normal pytorch mean

    Returns:
        Masked mean of a across dimension -2 (or n)
    """
    mask = mask[..., None]  # [*, n, 1]
    num_elements = torch.sum(mask, dim=-2, keepdim=True)  # [*, 1, 1]
    num_elements = torch.where(num_elements == 0, torch.tensor(1.0), num_elements)  # [*, 1, 1]
    a_masked = torch.masked_fill(a, ~mask, 0.0)  # [*, n, d]
    mean = torch.sum(a_masked, dim=-2, keepdim=True) / num_elements  # [*, 1, d]
    mean = torch.masked_fill(mean, num_elements == 0, 0.0)  # [*, 1, d]
    if not keepdim:
        mean = einops.rearrange(mean, "... () d -> ... d")
    return mean


def kabsch_align_ind(mobile, target, mask=None, ret_both=False):
    """
    Aligns mobile to target.

    Args:
        mobile: Torch tensor of shape [n, 3] -- Point Cloud to Align (source)
        target: Torch tensor of shape [n, 3] -- Reference Point Cloud (target)
        mask: Torch tensor of bools shape [n] -- if not None
        ret_both: Whether to return both pointclouds or just the mobile

    Returns:
        mobile_aligned: mobile point cloud aligned to target, shape [n, 3]

    TODO: Change _find_rot_alignment so that it works with any batch shape
        so that we don't require these two different functions.
    """
    if mask is None:
        mask = torch.ones(mobile.shape[:-1]).bool()

    mobile, target = mobile[None, ...], target[None, ...]  # [1, n, 3]
    mobile_aligned = kabsch_align(mobile, target)  # [1, n, 3]

    if ret_both:
        return mobile_aligned[0], target[0]  # [n, 3]
    return mobile_aligned[0]  # [n, 3]


def kabsch_align(mobile, target, mask=None):
    """
    Aligns mobile to target.

    Args:
        mobile: Torch tensor of shape [b, n, 3] -- Point Cloud to Align (source)
        target: Torch tensor of shape [b, n, 3] -- Reference Point Cloud (target)
        mask: Torch tensor of bools shape [b, n] -- if not None

    Returns:
        mobile_aligned: mobile point cloud aligned to target, shape [b, n, 3]
    """
    if mask is None:
        mask = torch.ones(mobile.shape[:-1]).bool()  # [b, n] all True

    mean_mobile = mean_w_mask(mobile, mask, keepdim=True)
    mean_target = mean_w_mask(target, mask, keepdim=True)

    mobile_centered = mobile - mean_mobile
    target_ceneterd = target - mean_target
    # These two operations make masked positions non-zero

    mobile_centered = torch.masked_fill(mobile_centered, ~mask[..., None], 0.0)  # Fill masked positions with 0
    target_ceneterd = torch.masked_fill(target_ceneterd, ~mask[..., None], 0.0)  # Fill masked positions with 0

    R = _find_rot_alignment(mobile_centered, target_ceneterd, mask)

    mobile_aligned = (
        torch.matmul(
            R,
            mobile_centered.transpose(-2, -1),
        ).transpose(-2, -1)
        + mean_target
    )  # [b, n, 3]

    mobile_aligned = torch.masked_fill(mobile_aligned, ~mask[..., None], 0.0)  # Fill masked positions with 0
    return mobile_aligned


def kabsch_align_ligand(mobile, target, mask=None):
    """
    Aligns mobile to target.

    Args:
        mobile: Torch tensor of shape [b, n, 3] -- Point Cloud to Align (source)
        target: Torch tensor of shape [b, n, 3] -- Reference Point Cloud (target)
        mask: Torch tensor of bools shape [b, n] -- if not None

    Returns:
        mobile_aligned: mobile point cloud aligned to target, shape [b, n, 3]
    """
    if mask is None:
        mask = torch.ones(mobile.shape[:-1]).bool()  # [b, n] all True

    mean_mobile = mean_w_mask(mobile, mask, keepdim=True)
    mean_target = mean_w_mask(target, mask, keepdim=True)

    mobile_centered = mobile - mean_mobile
    target_ceneterd = target - mean_target

    mobile_cenetered_orig = mobile_centered.clone()
    target_ceneterd_orig = target_ceneterd.clone()
    # These two operations make masked positions non-zero

    mobile_centered = torch.masked_fill(mobile_centered, ~mask[..., None], 0.0)  # Fill masked positions with 0
    target_ceneterd = torch.masked_fill(target_ceneterd, ~mask[..., None], 0.0)  # Fill masked positions with 0

    R = _find_rot_alignment(mobile_centered, target_ceneterd, mask)

    mobile_aligned = (
        torch.matmul(
            R,
            mobile_cenetered_orig.transpose(-2, -1),
        ).transpose(-2, -1)
        # + mean_target
    )  # [b, n, 3]

    # mobile_aligned = torch.masked_fill(
    #     mobile_aligned, ~mask[..., None], 0.0
    # )  # Fill masked positions with 0
    return mobile_aligned, target_ceneterd_orig


# This function was adapted from GeoDock's code (MIT License)
# https://github.com/Graylab/GeoDock/blob/main/geodock/utils/metrics.py#L103
# We pulled out translation stuff, vecotrize it and added masks
def _find_rot_alignment(A, B, mask=None):
    """
    Finds rotation that alignes two point clouds with zero center of mass.

    The mask functionality is simple. Once we center the point clouds,
    we zero-out the masked elements. Since the point clouds are centered,
    the masked elements will coincide at the origin for all rotations,
    so they do not affect the rotation found.

    Args:
        A: Torch tensor of shape [b, n, 3] -- point cloud to align (source) / mobile
        B: Torch tensor of shape [b, n, 3] -- reference point cloud (target) / target
        mask: Torch tensor of bools shape [b, n] -- if not None

    Returns:
        R: optimal rotations that best aligns A towards B, shape [b, 3, 3]
    """
    if mask is None:
        mask = torch.ones(A.shape[:-1]).bool()  # [b, n] all True

    # Confirm pointclouds are centered
    sh = mean_w_mask(A, mask, keepdim=True).shape
    assert torch.allclose(
        mean_w_mask(A, mask, keepdim=True),
        torch.zeros(sh, device=A.device),
        atol=1e-4,
        rtol=1e-4,
    )
    assert torch.allclose(
        mean_w_mask(B, mask, keepdim=True),
        torch.zeros(sh, device=B.device),
        atol=1e-4,
        rtol=1e-4,
    )
    assert A.shape == B.shape

    mask = mask[..., None]  # [b, n, 1]
    A = torch.masked_fill(A, ~mask, 0.0)
    B = torch.masked_fill(B, ~mask, 0.0)

    # Covariance matrix and SVD
    H = torch.matmul(A.transpose(-2, -1), B)

    # To float32, batched SVD not implemented otherwise
    # U, S, Vt = torch.linalg.svd(H, full_matrices=True)
    U, S, Vt = torch.linalg.svd(H.to(torch.float32), full_matrices=True)  # Breaks with mixed precision

    R = torch.matmul(
        Vt.transpose(-2, -1),
        U.transpose(-2, -1),
    )  # [b, 3, 3]

    # Handle the special reflection case
    # det_R = torch.linalg.det(R)  # [b]
    det_R = torch.linalg.det(R.to(torch.float32))  # [b], breaks with mixed precision
    # det_R = torch.linalg.det(R)  # [b]
    det_R = torch.linalg.det(R.to(torch.float32))  # [b], breaks with mixed precision
    SS = torch.eye(3, device=R.device).repeat(A.shape[0], 1, 1)  # Shape [b, 3, 3]
    SS[:, -1, -1] = torch.where(
        det_R < 0,
        torch.tensor(-1.0, device=R.device),
        torch.tensor(1.0, device=R.device),
    )  # [b, 3, 3]
    R_aux = torch.matmul(Vt.transpose(-2, -1), SS)
    # This stuff with SS is essentially changing the sign of the last column of Vt^T for the cases where the det < 0
    # Calculation as before but with this change in sign
    R = torch.matmul(R_aux, U.transpose(-2, -1))  # [b, 3, 3]

    return R
