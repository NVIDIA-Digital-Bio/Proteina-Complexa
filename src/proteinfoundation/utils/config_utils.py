"""Utilities for configuration handling and logging."""

from typing import Any

from omegaconf import DictConfig, OmegaConf

# Keys to filter out when logging configs (e.g., large dictionaries that clutter logs)
KEYS_TO_FILTER_FOR_LOGGING = [
    "target_dict_cfg",
    "dataset_target_dict_cfg",
    "generation_target_dict_cfg",
]


def filter_config_for_logging(
    cfg: DictConfig | dict,
    keys_to_filter: list[str] = None,
) -> dict[str, Any]:
    """Filter out large/verbose keys from config for cleaner logging.

    Args:
        cfg: Configuration to filter (DictConfig or dict)
        keys_to_filter: List of keys to exclude from output.
            Defaults to KEYS_TO_FILTER_FOR_LOGGING.

    Returns:
        Dictionary with filtered keys replaced by placeholder string.

    Example:
        >>> cfg = {"model": {...}, "target_dict_cfg": {large_dict}}
        >>> filtered = filter_config_for_logging(cfg)
        >>> # filtered["target_dict_cfg"] = "<filtered: 50 targets>"
    """
    if keys_to_filter is None:
        keys_to_filter = KEYS_TO_FILTER_FOR_LOGGING

    # Convert to dict if OmegaConf
    if isinstance(cfg, DictConfig):
        cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    else:
        cfg_dict = dict(cfg) if cfg else {}

    def _filter_recursive(d: dict, parent_key: str = "") -> dict:
        """Recursively filter keys from nested dictionaries."""
        filtered = {}
        for key, value in d.items():
            full_key = f"{parent_key}.{key}" if parent_key else key

            # Check if this key should be filtered
            if key in keys_to_filter:
                # Replace with placeholder showing the size
                if isinstance(value, dict):
                    filtered[key] = f"<filtered: {len(value)} entries>"
                elif isinstance(value, (list, tuple)):
                    filtered[key] = f"<filtered: {len(value)} items>"
                else:
                    filtered[key] = "<filtered>"
            elif isinstance(value, dict):
                # Recurse into nested dicts
                filtered[key] = _filter_recursive(value, full_key)
            else:
                filtered[key] = value

        return filtered

    return _filter_recursive(cfg_dict)


def log_config_summary(cfg: DictConfig | dict, logger) -> None:
    """Log a summary of the config with verbose keys filtered.

    Args:
        cfg: Configuration to log
        logger: Logger instance to use
    """
    filtered = filter_config_for_logging(cfg)
    logger.info(f"Config: {filtered}")
