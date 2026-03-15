"""ColabDesign utilities for AF2-based binder evaluation.

Used by the evaluation pipeline (binder_metrics.py) to refold generated
binders with AF2-Multimer and extract confidence metrics (pLDDT, pTM,
iPTM, pAE, etc.).

Loss helper functions (add_rg_loss, add_i_ptm_loss, etc.) live in
``proteinfoundation.rewards.alphafold2_reward_utils`` — import from
there instead of this module.
"""

import os
import pathlib
import re
from typing import Literal

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
    """
    multimer_validation = True

    clear_mem()
    complex_prediction_model = mk_afdesign_model(
        protocol="binder",
        num_recycles=advanced_settings["num_recycles_validation"],
        data_dir=advanced_settings["af_params_dir"],
        use_multimer=multimer_validation,
        use_initial_guess=advanced_settings["predict_initial_guess"],
        use_initial_atom_pos=advanced_settings["predict_bigbang"],
    )
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
