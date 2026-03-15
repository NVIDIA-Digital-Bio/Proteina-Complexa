"""
ESM Pseudo-Perplexity Evaluation.

Computes ESM2 pseudo-perplexity scores for protein sequences.
Lower values indicate more "natural" sequences according to the language model.

The model is cached globally after first load to avoid repeated loading overhead.
Set HF_HUB_OFFLINE=1 environment variable to force fully offline mode.
"""

import os

import numpy as np
import pandas as pd
import torch
from loguru import logger

# =============================================================================
# Safe Imports
# =============================================================================

ESM_AVAILABLE = False
try:
    from transformers import AutoModelForMaskedLM, AutoTokenizer

    ESM_AVAILABLE = True
except ImportError as e:
    logger.warning(f"ESM/transformers import failed: {e}. ESM metrics will return NaN.")

# =============================================================================
# Constants
# =============================================================================

DEFAULT_ESM_MODEL = "facebook/esm2_t33_650M_UR50D"

ESM_METRIC_COLS = [
    "esm_pseudo_perplexity",
    "esm_log_likelihood",
]

# =============================================================================
# Global Model Cache
# =============================================================================

_ESM_MODEL_CACHE = {
    "model": None,
    "tokenizer": None,
    "model_name": None,
    "device": None,
}


# =============================================================================
# Core Computation
# =============================================================================


def compute_pseudo_perplexity(
    model,
    tokenizer,
    sequence: str,
    device: str = "cuda",
) -> tuple[float, float]:
    """Compute pseudo-perplexity for a single sequence."""
    if not sequence or len(sequence) == 0:
        return np.nan, np.nan

    try:
        inputs = tokenizer(sequence, return_tensors="pt").to(device)
        input_ids = inputs["input_ids"]
        seq_length = input_ids.size(1) - 2  # Exclude BOS and EOS tokens

        log_probs = []
        for i in range(1, seq_length + 1):
            masked_input = input_ids.clone()
            masked_input[0, i] = tokenizer.mask_token_id

            with torch.no_grad():
                outputs = model(masked_input)
                logits = outputs.logits
                log_prob = torch.log_softmax(logits[0, i], dim=-1)
                true_token = input_ids[0, i]
                log_probs.append(log_prob[true_token].item())

        avg_log_likelihood = sum(log_probs) / len(log_probs)
        pseudo_ppl = np.exp(-avg_log_likelihood)

        return pseudo_ppl, avg_log_likelihood

    except Exception as e:
        logger.error(f"ESM computation failed: {e}")
        return np.nan, np.nan


def _resolve_esm_dir() -> str | None:
    """Resolve ESM_DIR as a direct local model path.

    ESM_DIR should point to a HuggingFace hub cache directory that already
    contains the downloaded model (e.g. ``community_models/ckpts/ESM2``).
    Returns None if ESM_DIR is not set or the directory doesn't exist.
    """
    esm_dir = os.environ.get("ESM_DIR")
    if esm_dir:
        esm_dir = os.path.expanduser(esm_dir)
        if os.path.isdir(esm_dir):
            logger.debug(f"Using ESM_DIR from environment: {esm_dir}")
            return esm_dir
    return None


def _resolve_cache_dir() -> str:
    """Resolve the HuggingFace cache directory for downloading ESM models.

    Priority:
    1. CACHE_DIR environment variable
    2. ~/.cache (default)

    Note: ESM_DIR is handled separately as a local model path, not as a
    download cache. This prevents HuggingFace from downloading model files
    into the project tree.
    """
    cache_dir = os.environ.get("CACHE_DIR")
    if cache_dir:
        cache_dir = os.path.expanduser(cache_dir)
        logger.debug(f"Using CACHE_DIR from environment: {cache_dir}")
        return cache_dir

    default_cache = os.path.expanduser("~/.cache")
    logger.debug(f"Using default cache: {default_cache}")
    return default_cache


def get_esm_model(
    model_name: str = DEFAULT_ESM_MODEL,
    device: str | None = None,
    force_offline: bool = True,
):
    """Get or load the ESM model and tokenizer (cached globally).

    This function caches the model globally so it's only loaded once per session.
    Subsequent calls return the cached model immediately.

    Args:
        model_name: HuggingFace model name (default: facebook/esm2_t33_650M_UR50D)
        device: Device to load model on (default: auto-detect cuda/cpu)
        force_offline: If True, set HF_HUB_OFFLINE=1 to prevent network requests

    Returns:
        Tuple of (model, tokenizer, device)
    """
    global _ESM_MODEL_CACHE

    if not ESM_AVAILABLE:
        raise RuntimeError("ESM/transformers not available. Install with: pip install transformers")

    # Check if already cached with same model
    if _ESM_MODEL_CACHE["model"] is not None and _ESM_MODEL_CACHE["model_name"] == model_name:
        logger.debug(f"Using cached ESM model: {model_name}")
        return (
            _ESM_MODEL_CACHE["model"],
            _ESM_MODEL_CACHE["tokenizer"],
            _ESM_MODEL_CACHE["device"],
        )

    # Set offline mode to prevent network requests
    if force_offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
        logger.debug("Set HF_HUB_OFFLINE=1 to force offline mode")

    esm_dir = _resolve_esm_dir()
    cache_dir = _resolve_cache_dir()

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Try ESM_DIR first (pre-downloaded local copy), then HF cache
    load_locations = []
    if esm_dir:
        load_locations.append(("ESM_DIR", esm_dir))
    load_locations.append(("cache", cache_dir))

    model = None
    tokenizer = None

    for label, loc in load_locations:
        logger.info(f"Loading ESM model: {model_name} ({label}: {loc})")
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                cache_dir=loc,
                local_files_only=True,
            )
            model = AutoModelForMaskedLM.from_pretrained(
                model_name,
                cache_dir=loc,
                local_files_only=True,
            )
            logger.info(f"Loaded ESM model from {label} (offline)")
            break
        except Exception:
            logger.debug(f"ESM model not found in {label}: {loc}")
            continue

    if model is None:
        if force_offline:
            search_paths = ", ".join(loc for _, loc in load_locations)
            logger.error(
                f"Failed to load ESM model from local paths: {search_paths}\n"
                f"The model may not be downloaded yet. To download, run:\n"
                f'  python -c "from transformers import AutoTokenizer, AutoModelForMaskedLM; '
                f"AutoTokenizer.from_pretrained('{model_name}', cache_dir='{cache_dir}'); "
                f"AutoModelForMaskedLM.from_pretrained('{model_name}', cache_dir='{cache_dir}')\""
            )
            raise RuntimeError(f"ESM model not found in local paths: {search_paths}")

        # If not forcing offline, download to cache_dir (not ESM_DIR)
        logger.info(f"Downloading ESM model from HuggingFace to {cache_dir}...")
        os.environ.pop("HF_HUB_OFFLINE", None)
        tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
        model = AutoModelForMaskedLM.from_pretrained(model_name, cache_dir=cache_dir)

    model = model.to(device)
    model.eval()

    # Cache the model
    _ESM_MODEL_CACHE["model"] = model
    _ESM_MODEL_CACHE["tokenizer"] = tokenizer
    _ESM_MODEL_CACHE["model_name"] = model_name
    _ESM_MODEL_CACHE["device"] = device

    logger.info(f"ESM model loaded on {device} and cached for reuse")

    return model, tokenizer, device


def clear_esm_cache():
    """Clear the cached ESM model to free GPU memory."""
    global _ESM_MODEL_CACHE

    if _ESM_MODEL_CACHE["model"] is not None:
        del _ESM_MODEL_CACHE["model"]
        del _ESM_MODEL_CACHE["tokenizer"]
        _ESM_MODEL_CACHE = {
            "model": None,
            "tokenizer": None,
            "model_name": None,
            "device": None,
        }
        torch.cuda.empty_cache()
        logger.info("Cleared ESM model cache")


def compute_esm_ppl_for_sequences(
    sequences: list[str],
    model_name: str = DEFAULT_ESM_MODEL,
    force_offline: bool = True,
) -> pd.DataFrame:
    """Compute ESM pseudo-perplexity for a list of sequences.

    Args:
        sequences: List of protein sequences
        model_name: HuggingFace model name
        force_offline: If True, only load from local cache (no network requests)

    Returns:
        DataFrame with sequences and their ESM metrics
    """
    if not ESM_AVAILABLE:
        logger.error("ESM/transformers not available. Install with: pip install transformers")
        return pd.DataFrame(columns=["sequence"] + ESM_METRIC_COLS)

    if not sequences:
        logger.warning("No sequences provided for ESM evaluation")
        return pd.DataFrame(columns=["sequence"] + ESM_METRIC_COLS)

    logger.info(f"Computing ESM pseudo-perplexity for {len(sequences)} sequences")

    # Get cached model (loads once, reuses thereafter)
    try:
        model, tokenizer, device = get_esm_model(model_name, force_offline=force_offline)
    except RuntimeError as e:
        logger.error(f"Failed to load ESM model: {e}")
        # Return NaN results
        return pd.DataFrame(
            [
                {
                    "sequence": seq,
                    "esm_pseudo_perplexity": np.nan,
                    "esm_log_likelihood": np.nan,
                }
                for seq in sequences
            ]
        )

    results = []
    for idx, seq in enumerate(sequences):
        pppl, log_ll = compute_pseudo_perplexity(model, tokenizer, seq, device)
        results.append(
            {
                "sequence": seq,
                "esm_pseudo_perplexity": pppl,
                "esm_log_likelihood": log_ll,
            }
        )

    logger.info(f"ESM evaluation complete for {len(sequences)} sequences")
    return pd.DataFrame(results)


def compute_esm_ppl_for_pdbs(
    pdb_paths: list[str],
    protein_type: str = "binder",
    model_name: str = DEFAULT_ESM_MODEL,
    force_offline: bool = True,
) -> pd.DataFrame:
    """
    Compute ESM pseudo-perplexity for sequences extracted from PDB files.

    Args:
        pdb_paths: List of PDB file paths
        protein_type: "binder" (last chain) or "monomer" (all chains)
        model_name: ESM model to use
        force_offline: If True, only load from local cache (no network requests)

    Returns:
        DataFrame with pdb_path, sequence, and ESM metrics
    """
    from proteinfoundation.evaluation.binder_eval_utils import get_binder_chain_from_complex
    from proteinfoundation.utils.pdb_utils import extract_seq_from_pdb

    if not ESM_AVAILABLE:
        logger.error("ESM/transformers not available. Install with: pip install transformers")
        return pd.DataFrame(columns=["pdb_path", "sequence"] + ESM_METRIC_COLS)

    if not pdb_paths:
        logger.warning("No PDB paths provided for ESM evaluation")
        return pd.DataFrame(columns=["pdb_path", "sequence"] + ESM_METRIC_COLS)

    logger.info(f"Computing ESM pseudo-perplexity for {len(pdb_paths)} PDB files (type: {protein_type})")

    # Get cached model (loads once, reuses thereafter)
    try:
        model, tokenizer, device = get_esm_model(model_name, force_offline=force_offline)
    except RuntimeError as e:
        logger.error(f"Failed to load ESM model: {e}")
        # Return NaN results
        return pd.DataFrame(
            [
                {
                    "pdb_path": p,
                    "sequence": None,
                    "esm_pseudo_perplexity": np.nan,
                    "esm_log_likelihood": np.nan,
                }
                for p in pdb_paths
            ]
        )

    results = []
    failed_count = 0
    for idx, pdb_path in enumerate(pdb_paths):
        try:
            if protein_type == "binder":
                binder_chain, _ = get_binder_chain_from_complex(pdb_path)
                sequence = extract_seq_from_pdb(pdb_path, chain_id=binder_chain)
            else:  # monomer
                sequence = extract_seq_from_pdb(pdb_path, chain_id=None)

            pppl, log_ll = compute_pseudo_perplexity(model, tokenizer, sequence, device)

            results.append(
                {
                    "pdb_path": pdb_path,
                    "sequence": sequence,
                    "esm_pseudo_perplexity": pppl,
                    "esm_log_likelihood": log_ll,
                }
            )

        except Exception as e:
            logger.warning(f"Failed to process {pdb_path}: {e}")
            failed_count += 1
            results.append(
                {
                    "pdb_path": pdb_path,
                    "sequence": "FAIL",
                    "esm_pseudo_perplexity": np.nan,
                    "esm_log_likelihood": np.nan,
                }
            )

    if failed_count > 0:
        logger.warning(f"ESM evaluation failed for {failed_count}/{len(pdb_paths)} PDB files")
    logger.info(f"ESM evaluation complete for {len(pdb_paths) - failed_count}/{len(pdb_paths)} PDB files")

    return pd.DataFrame(results)
