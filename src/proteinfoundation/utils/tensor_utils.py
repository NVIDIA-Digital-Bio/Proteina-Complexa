import torch


def concat_padded_tensor(
    a: torch.Tensor,
    b: torch.Tensor,
    mask_a: torch.Tensor,
    mask_b: torch.Tensor,
):
    """
    Given two padded tensors of shape [b, n, *], merge them into one big padded tensor.

    Args:
        a: Tensor of shape [b, n, *]
        b: Tensor of shape [b, m, *]
        mask_a: Mask of shape [b, n]
        mask_b: Mask of shape [b, m]

    Returns:
        x: Tensor of shape [b, pad_len, *]
        mask: Mask of shape [b, pad_len]
    """
    len_a = mask_a.int().sum(dim=-1)
    len_b = mask_b.int().sum(dim=-1)
    assert a.shape[0] == b.shape[0], "Batch size of a and b to be the same"
    assert a.shape[:2] == mask_a.shape[:2], "Mask should be compatible with the tensor"
    assert b.shape[:2] == mask_b.shape[:2], "Mask should be compatible with the tensor"
    bs = a.shape[0]
    device = a.device
    n = (len_a + len_b).max().item()

    index_a = torch.arange(a.shape[1], device=device)[None, :].expand(bs, -1)
    index_b = torch.arange(b.shape[1], device=device)[None, :].expand(bs, -1) + len_a[:, None]
    batch_id_a = torch.arange(bs, device=device)[:, None].expand(-1, a.shape[1])
    batch_id_b = torch.arange(bs, device=device)[:, None].expand(-1, b.shape[1])

    mask = torch.zeros(bs, n, dtype=torch.bool, device=a.device)
    mask[batch_id_a[mask_a], index_a[mask_a]] = True
    mask[batch_id_b[mask_b], index_b[mask_b]] = True

    x = torch.zeros(bs, n, *a.shape[2:], dtype=a.dtype, device=a.device)
    x[batch_id_a[mask_a], index_a[mask_a]] = a[mask_a]
    x[batch_id_b[mask_b], index_b[mask_b]] = b[mask_b]

    return x, mask


def concat_dict_tensors(dict_list, dim: int = 0):
    """
    Concatenate a list of dictionaries of tensors along the specified dimension.

    Args:
        dict_list: List of dictionaries, each containing tensors with the same keys
        dim: Dimension along which to concatenate (default: 0)

    Returns:
        Dictionary with concatenated tensors

    Example:
        dict_list = [
            {'key1': tensor([1, 2]), 'key2': tensor([3, 4])},
            {'key1': tensor([5, 6]), 'key2': tensor([7, 8])}
        ]
        result = concat_dict_tensors(dict_list, dim=0)
        # result = {'key1': tensor([1, 2, 5, 6]), 'key2': tensor([3, 4, 7, 8])}
    """
    if not dict_list:
        return {}

    result = {}
    for key in dict_list[0]:
        values = [d[key] for d in dict_list if key in d]
        if not values:
            continue
        if isinstance(values[0], (list, tuple)):
            # Extend lists
            merged = []
            for v in values:
                merged.extend(v)
            result[key] = merged
        elif isinstance(values[0], dict):
            # Merge dicts of tensors (e.g. rewards with total_reward + components)
            all_keys = set()
            for v in values:
                all_keys.update(v.keys())
            result[key] = {}
            for k in all_keys:
                tensors = []
                for v in values:
                    if k in v:
                        tensors.append(v[k])
                    else:
                        # Missing key: fill with NaN. Use ref for dtype/device/trailing dims,
                        # but derive batch size from a sibling tensor in *this* dict so
                        # dim-0 matches when batch sizes differ across dicts.
                        ref = next(
                            (other[k] for other in values if k in other and isinstance(other[k], torch.Tensor)),
                            None,
                        )
                        if ref is not None:
                            batch_ref = next(
                                (v[kk] for kk in v if isinstance(v[kk], torch.Tensor)),
                                None,
                            )
                            batch_size = batch_ref.shape[0] if batch_ref is not None else ref.shape[0]
                            fill_shape = (batch_size, *ref.shape[1:])
                            tensors.append(
                                torch.full(
                                    fill_shape,
                                    float("nan"),
                                    device=ref.device,
                                    dtype=torch.float32,
                                )
                            )
                if tensors:
                    result[key][k] = torch.cat(tensors, dim=dim)
        else:
            result[key] = torch.cat(values, dim=dim)

    return result
