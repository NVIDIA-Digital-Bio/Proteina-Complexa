"""ColabDesign utilities for AF2-based binder evaluation.

Used by the evaluation pipeline (binder_metrics.py) to refold generated
binders with AF2-Multimer and extract confidence metrics (pLDDT, pTM,
iPTM, pAE, etc.).

Loss helper functions (add_rg_loss, add_i_ptm_loss, etc.) live in
``proteinfoundation.rewards.alphafold2_reward_utils`` — import from
there instead of this module.

Performance note
----------------
``mk_afdesign_model`` reloads ~3.7 GB of AF2-Multimer parameters from
``AF2_DIR`` (typically a network filesystem) and rebuilds JAX-compiled
forward functions every time it is called.  Historically ``run_af_eval``
invoked it once per *sample*, costing ~18 s/sample of pure setup overhead
on top of the ~2 s actual refold.  We now cache the constructed model
process-globally, keyed on the architecture-affecting settings, and reuse
it across all samples in an evaluation job.  ``prep_inputs`` is still
called per-call (cheap; rebuilds templates for the new design PDB).

Use :func:`clear_af2_binder_model_cache` to release the cached model
(e.g. in tests or before switching to a very different AF2 config).
"""

import os
import pathlib
import re
from typing import Any, Literal

from colabdesign import clear_mem, mk_afdesign_model
from colabdesign.shared.utils import copy_dict
from loguru import logger


def get_af2_advanced_settings():
    """Return default advanced settings for AF2 evaluation.

    Reads ``AF2_DIR`` and ``DSSP_EXEC`` from the environment (set via
    ``.env``), falling back to ``$DATA_PATH/tools/...`` paths.
    """
    data_path = os.environ.get("DATA_PATH")
    advanced_settings = {
        "sample_models": True,
        "rm_template_seq_predict": False,
        "rm_template_sc_predict": False,
        "predict_initial_guess": True,
        "predict_bigbang": False,
        "num_recycles_validation": 3,
        "af_params_dir": os.getenv("AF2_DIR", f"{data_path}/tools/AF2" if data_path else None),
        "dssp_path": os.getenv("DSSP_EXEC", f"{data_path}/tools/dssp" if data_path else None),
    }
    return advanced_settings


# ---------------------------------------------------------------------------
# Persistent AF2-Multimer binder model cache
# ---------------------------------------------------------------------------
#
# Keyed on the subset of ``advanced_settings`` that ``mk_afdesign_model``
# actually consumes — anything else (e.g. per-sample sequence) is
# irrelevant to whether two builds would be equivalent.
_AF2_BINDER_MODEL_CACHE: dict[tuple, Any] = {}


def _af2_binder_cache_key(advanced_settings: dict, multimer_validation: bool = True) -> tuple:
    """Build a hashable cache key from the architecture-affecting settings.

    Includes only fields that ``mk_afdesign_model`` consumes at build
    time.  Per-sample inputs (PDB, sequence, chain ids) are NOT part of
    the key — those vary per call but reuse the same compiled model.
    """
    return (
        advanced_settings["af_params_dir"],
        int(advanced_settings["num_recycles_validation"]),
        bool(multimer_validation),
        bool(advanced_settings["predict_initial_guess"]),
        bool(advanced_settings["predict_bigbang"]),
    )


def _get_or_build_af2_binder_model(
    advanced_settings: dict, multimer_validation: bool = True
) -> Any:
    """Return a cached AF2 binder model or build + cache a new one.

    Safe to call concurrently from a single thread per process (no
    locking needed — Python dict get/set is atomic in CPython for our
    use).  Cross-process callers each maintain their own cache.

    The returned model is shared by reference; do not call ``clear_mem``
    on it externally.  ``prep_inputs`` and ``predict`` are safe to call
    repeatedly because ColabDesign's ``prep_inputs`` flow goes through
    ``_prep_model`` → ``restart()`` which fully resets ``_inputs``,
    ``aux``, and ``_tmp`` on every call.
    """
    key = _af2_binder_cache_key(advanced_settings, multimer_validation)
    cached = _AF2_BINDER_MODEL_CACHE.get(key)
    if cached is not None:
        logger.debug(f"AF2 binder model cache HIT (key={key})")
        return cached

    logger.info(
        "AF2 binder model cache MISS — building model "
        f"(num_recycles={key[1]}, use_multimer={key[2]}, "
        f"use_initial_guess={key[3]}, use_initial_atom_pos={key[4]}, "
        f"data_dir={key[0]})"
    )
    model = mk_afdesign_model(
        protocol="binder",
        num_recycles=advanced_settings["num_recycles_validation"],
        data_dir=advanced_settings["af_params_dir"],
        use_multimer=multimer_validation,
        use_initial_guess=advanced_settings["predict_initial_guess"],
        use_initial_atom_pos=advanced_settings["predict_bigbang"],
    )
    _AF2_BINDER_MODEL_CACHE[key] = model
    return model


def clear_af2_binder_model_cache() -> None:
    """Drop all cached AF2 binder models and free their GPU memory.

    Intended for tests, explicit lifecycle management, or when switching
    between distinct AF2 configurations in the same process.  Idempotent.
    """
    if not _AF2_BINDER_MODEL_CACHE:
        return
    logger.info(f"Clearing AF2 binder model cache ({len(_AF2_BINDER_MODEL_CACHE)} entries)")
    _AF2_BINDER_MODEL_CACHE.clear()
    # ColabDesign's clear_mem deletes every live JAX buffer on the GPU.
    # Safe to call now because we've already evicted all cached models;
    # any caller still holding a reference is responsible for not using
    # it after this call (documented contract).
    try:
        clear_mem()
    except Exception as e:
        logger.warning(f"clear_mem failed during AF2 cache reset: {e}")


def _evict_af2_binder_model(advanced_settings: dict, multimer_validation: bool = True) -> None:
    """Drop a single cached AF2 binder model entry (does NOT call ``clear_mem``).

    Used in the exception path of :func:`run_af_eval` so a corrupted
    partial state on the cached model can't poison subsequent samples.
    Other cached entries (e.g. different settings) survive.
    """
    key = _af2_binder_cache_key(advanced_settings, multimer_validation)
    if _AF2_BINDER_MODEL_CACHE.pop(key, None) is not None:
        logger.warning(f"Evicted AF2 binder model from cache after error (key={key})")


def _clear_af2_binder_model_state(model: Any) -> None:
    """Best-effort empty of the model's per-call mutable dicts.

    Mirrors :meth:`AF2RewardModel._clear_model_state` so the
    eval-time refold path has the same defensive cleanup the reward
    path has been using in production.  Empties ``_inputs``, ``aux``,
    and ``_tmp`` in place — the next ``prep_inputs`` call replaces
    them anyway, but this guarantees no stale data is observable if
    something later short-circuits ``prep_inputs``.

    Does **not** touch ``_model``, ``_params``, or any compiled JAX
    function — those are the cached model itself and must persist.
    """
    if model is None:
        return
    for attr in ("_inputs", "aux", "_tmp"):
        d = getattr(model, attr, None)
        if isinstance(d, dict):
            d.clear()


def run_af_eval(
    trajectory_pdb: pathlib.Path,
    binder_sequences: list[dict],
    design_name: str,
    output_path: pathlib.Path,
    target_settings: dict,
    advanced_settings: dict,
    binder_length: int,
    binder_chain: str = "B",
    sequence_type_list: list[Literal["mpnn", "mpnn_fixed", "self"]] | None = None,
):
    """Run AF2-Multimer refolding evaluation for generated binders.

    For each sequence in *binder_sequences*, predicts the complex
    structure with AF2 and returns per-sequence confidence statistics.

    The underlying AF2 model is cached across calls (keyed on
    architecture-affecting settings in *advanced_settings*).  The first
    call pays a one-time ~10–20 s model-load + JAX compile cost;
    subsequent calls reuse the cached model and only pay for
    ``prep_inputs`` (~1 s) plus the actual forward pass.

    The mutable per-call state of the cached model (``_inputs``, ``aux``,
    ``_tmp``) is reset in the ``finally`` block, mirroring
    :meth:`AF2RewardModel._clear_model_state`.  On exception the cache
    entry is evicted so a corrupted partial state cannot poison the next
    sample; other cached entries (different settings) survive.
    """
    multimer_validation = True

    complex_prediction_model = _get_or_build_af2_binder_model(
        advanced_settings,
        multimer_validation=multimer_validation,
    )

    try:
        if advanced_settings["predict_initial_guess"] or advanced_settings["predict_bigbang"]:
            complex_prediction_model.prep_inputs(
                pdb_filename=trajectory_pdb,
                chain=target_settings["chains"],
                binder_chain=binder_chain,
                binder_len=binder_length,
                use_binder_template=True,
                rm_target_seq=advanced_settings["rm_template_seq_predict"],
                rm_target_sc=advanced_settings["rm_template_sc_predict"],
                rm_template_ic=True,
            )
        else:
            complex_prediction_model.prep_inputs(
                pdb_filename=target_settings["starting_pdb"],
                chain=target_settings["chains"],
                binder_len=binder_length,
                rm_target_seq=advanced_settings["rm_template_seq_predict"],
                rm_target_sc=advanced_settings["rm_template_sc_predict"],
            )

        save_location = "AF2"
        complex_pdb_path = os.path.join(output_path, save_location)
        design_paths = {save_location: complex_pdb_path}
        os.makedirs(complex_pdb_path, exist_ok=True)

        mpnn_complex_statistics = []
        output_complex_pdb_paths = []
        for seq_num, mpnn_sequence in enumerate(binder_sequences):
            logger.info(f"Predicting complex for sequence {seq_num + 1} of {len(binder_sequences)}")
            if sequence_type_list:
                mpnn_sample_name = f"{design_name}_{sequence_type_list[seq_num]}_seq_{seq_num}"
            else:
                mpnn_sample_name = f"{design_name}_seq_{seq_num}"

            complex_statistics = predict_binder_complex(
                prediction_model=complex_prediction_model,
                binder_sequence=mpnn_sequence["seq"],
                mpnn_design_name=mpnn_sample_name,
                advanced_settings=advanced_settings,
                design_paths=design_paths,
            )
            logger.info(f"Complex PDB path for seq_{seq_num + 1}: {complex_statistics['complex_pdb_path']}")
            mpnn_complex_statistics.append({f"seq_{seq_num + 1}": complex_statistics})
            output_complex_pdb_paths.append(complex_statistics["complex_pdb_path"])

        return mpnn_complex_statistics, output_complex_pdb_paths

    except Exception:
        # Evict the (possibly corrupted) cached model entry — but keep
        # other entries with different settings intact.  Next call to
        # run_af_eval with these settings will rebuild from scratch.
        _evict_af2_binder_model(advanced_settings, multimer_validation=multimer_validation)
        raise

    finally:
        # Defensive cleanup of per-call mutable state on the cached
        # model.  Matches AF2RewardModel.score's finally block.
        _clear_af2_binder_model_state(complex_prediction_model)


def predict_binder_complex(
    prediction_model,
    binder_sequence,
    mpnn_design_name,
    advanced_settings,
    design_paths,
):
    """Predict a binder–target complex with AF2 and extract confidence scores."""
    binder_sequence = re.sub("[^A-Z]", "", binder_sequence.upper())

    model_num = 0
    save_location = "AF2"
    complex_pdb = os.path.join(design_paths[save_location], f"{mpnn_design_name}_model{model_num + 1}.pdb")
    prediction_model.predict(
        seq=binder_sequence,
        models=[model_num],
        num_recycles=advanced_settings["num_recycles_validation"],
        verbose=False,
    )
    prediction_model.save_pdb(complex_pdb)
    prediction_metrics = copy_dict(prediction_model.aux["log"])

    stats = {
        "pLDDT": round(prediction_metrics["plddt"], 3),
        "pTM": round(prediction_metrics["ptm"], 3),
        "i_pTM": round(prediction_metrics["i_ptm"], 3),
        "pAE": round(prediction_metrics["pae"], 3),
        "i_pAE": round(prediction_metrics["i_pae"], 3),
        "min_ipAE": round(prediction_metrics["min_ipae"], 4),
        "min_ipSAE": round(prediction_metrics["min_ipsae"], 4),
        "max_ipSAE": round(prediction_metrics["max_ipsae"], 4),
        "avg_ipSAE": round(prediction_metrics["avg_ipsae"], 4),
        "min_ipSAE_10": round(prediction_metrics.get("min_ipsae_10", 0.0), 4),
        "max_ipSAE_10": round(prediction_metrics.get("max_ipsae_10", 0.0), 4),
        "avg_ipSAE_10": round(prediction_metrics.get("avg_ipsae_10", 0.0), 4),
        "complex_pdb_path": complex_pdb,
    }
    return stats
