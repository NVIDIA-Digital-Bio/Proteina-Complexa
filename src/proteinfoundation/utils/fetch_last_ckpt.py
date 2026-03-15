import os
import re


def get_version_number(fname: str) -> int | None:
    """
    Gets version numnber of a last checkpoint, or None if not a last checkpoint.

    Args:
        fname: name of the file

    Returns:
        version number if in the format last-v<X>.ckpt, with X and integer > 0,
        or last.ckpt yields 0. If not the right naming returns None.
    """
    match = re.search(r"last-v(\d+).ckpt", fname)
    if match:
        return int(match.group(1))
    elif fname == "last.ckpt":
        return 0  # last.ckpt is base version
    return None  # ignore if does not match


def fetch_last_ckpt(ckpt_dir: str) -> str | None:
    """
    Returns the name of the latest last-v<X>.ckpt where X is an integer. Defaults to last.ckpt if just that's the only one.
    If no last ckpt then returns None.

    Args:
        ckpt_dir: directory where checkpoints are stored.

    Returns:
        Name of the latest checkpoint, None if no such checkpoint present.
    """
    if not os.path.exists(ckpt_dir):
        return None
    last_ckpts = [
        f for f in os.listdir(ckpt_dir) if "last" in f and f.endswith(".ckpt") and get_version_number(f) is not None
    ]
    if len(last_ckpts) == 0:
        return None
    sorted_files = sorted(last_ckpts, key=get_version_number, reverse=True)  # fort by version #, highest first
    return sorted_files[0]
