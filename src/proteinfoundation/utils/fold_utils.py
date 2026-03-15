from typing import Literal


def extract_cath_code_from_batch(batch):
    cath_code = batch.get("cath_code", None)
    if cath_code:
        # Remove the additional tuple layer introduced during collate
        _cath_code = []
        for codes in cath_code:
            _cath_code.append([code[0] if isinstance(code, tuple) else code for code in codes])
        cath_code = _cath_code
    return cath_code


def extract_cath_code_by_level(cath_code: str, level: Literal["C", "A", "T", "H"]) -> str:
    """Extract cath_code at certain level.

    Args:
      cath_code: CATH code.
      level: Level to be extracted

    Returns:
      CATH code at the corresponding level.
    """
    mapping = {"H": 0, "T": 1, "A": 2, "C": 3}
    return cath_code.rsplit(".", mapping[level])[0]


def mask_cath_code_by_level(cath_code: list[str], level: Literal["C", "A", "T", "H"]) -> str:
    """Mask cath_code at certain level.

    Args:
      cath_code: List of CATH code.
      level: Level to be extracted

    Returns:
      List of CATH code. Replace the corresponding level with unknown token 'x'.
    """
    mapping = {"H": 3, "T": 2, "A": 1, "C": 0}
    _cath_code = []
    for code in cath_code:
        code = code.split(".")
        code[mapping[level]] = "x"
        _cath_code.append(".".join(code))
    return _cath_code


def transform_global_percentage_to_mask_dropout(fold_label_sample_ratio):
    assert len(fold_label_sample_ratio) == 4, "Length of fold_label_sample_ratio should be 4"
    assert sum(fold_label_sample_ratio) == 1.0, "Sum of fold_label_sample_ratio should be 1.0"
    mask_T_prob = sum(fold_label_sample_ratio[:3]) / sum(
        fold_label_sample_ratio
    )  # Among all samples, how many T-level labels are dropped?       (null + C + CA) / (null + C + CA + CAT)
    mask_A_prob = sum(fold_label_sample_ratio[:2]) / (
        sum(fold_label_sample_ratio[:3]) + 1e-10
    )  # Among samples with T labels dropped, how many A-level labels are dropped?    (null + C) / (null + C + CA)
    mask_C_prob = sum(fold_label_sample_ratio[:1]) / (
        sum(fold_label_sample_ratio[:2]) + 1e-10
    )  # Among samples with A and T labels dropped, how many C-level labels are dropped?     null / (null + C)
    return mask_T_prob, mask_A_prob, mask_C_prob
