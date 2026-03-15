# RF3 reward runner for structure prediction evaluation

import json
import os
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import torch
from loguru import logger

from proteinfoundation.metrics.ipsae import complex_ipSAE
from proteinfoundation.rewards.base_reward import BaseRewardModel, ensure_tensor, standardize_reward
from proteinfoundation.utils.pdb_utils import extract_seq_from_pdb


class RF3RewardRunner(BaseRewardModel):
    """RF3 reward runner for structure prediction evaluation."""

    IS_FOLDING_MODEL = True
    SUPPORTS_GRAD = False
    SUPPORTS_SAVE_PDB = False

    reward_options = (
        "plddt",
        "ipAE",
        "min_ipAE",
        "mean_min_ipAE",
        "mean_ipAE",
        "min_mean_ipAE",
        "pAE",
        "ipTM",
        "pTM",
        "ranking_score",
        "has_clash",
        "min_ipSAE",
        "max_ipSAE",
        "avg_ipSAE",  # default AF3 iPSAE 10 threshold
    )

    PAE_KEYS = {
        "ipAE",
        "min_ipAE",
        "mean_min_ipAE",
        "mean_ipAE",
        "min_mean_ipAE",
        "pAE",
    }

    def __init__(
        self,
        ckpt_path: str,
        dump_dir: str = "./rf3_outputs",
        rf3_path: str = "rf3",
        verbose: bool = False,
        reward_weights: dict[str, float] | None = None,
        normalize_pae: bool = True,
    ):
        """Initialize RF3 reward runner.

        Args:
            ckpt_path: Path to RF3 checkpoint file
            dump_dir: Directory to dump results
            rf3_path: Path to RF3 executable
            verbose: If True, print debug info (file listings, JSON contents)
            reward_weights: Dict mapping metric names to scalar weights for computing
                total_reward in score(). Keys must be from reward_options. Missing keys
                default to 0.0. When None, defaults to {"min_ipAE": -1.0}.
            normalize_pae: If True, PAE-family metrics (ipAE, min_ipAE, mean_min_ipAE,
                mean_ipAE, min_mean_ipAE, pAE) are divided by 31.0 in score() before
                applying weights. This normalizes them to ~0-1 range so weights are
                on the same scale as pLDDT/ipTM/pTM. Does not affect predict_* methods.
        """
        assert ckpt_path is not None, "ckpt_path is required — set RF3_CKPT_PATH env var or pass explicitly"
        assert os.path.exists(ckpt_path), f"RF3 checkpoint file {ckpt_path} does not exist"
        self.ckpt_path = ckpt_path
        self.dump_dir = dump_dir
        self.rf3_path = rf3_path
        self.verbose = verbose
        self.normalize_pae = normalize_pae

        if reward_weights is None:
            denom = 31.0 if normalize_pae else 1.0
            reward_weights = {"min_ipAE": -1.0 / denom}
        for key in reward_weights:
            assert key in self.reward_options, (
                f"Invalid reward weight key '{key}'. Must be one of {self.reward_options}"
            )
        self.reward_weights = dict.fromkeys(self.reward_options, 0.0)
        self.reward_weights.update(reward_weights)

        # Create output directory
        os.makedirs(self.dump_dir, exist_ok=True)
        logger.info(f"RF3 initialized with checkpoint: {self.ckpt_path}")
        logger.info(f"RF3 reward_weights: {self.reward_weights}")
        logger.info(f"RF3 normalize_pae: {self.normalize_pae}")

    def reset_dump_dir(self, new_dump_dir: str):
        """Reset the dump directory."""
        self.dump_dir = new_dump_dir
        os.makedirs(self.dump_dir, exist_ok=True)

    def predict_batch_from_files_boltz_compatible(
        self,
        input_files: list[str],
        template_selection: str | None = None,
        ground_truth_conformer_selection: str | None = None,
        out_dir: str | None = None,
        smiles: str = None,
        add_missing_atoms: bool | None = None,
    ) -> list[dict[str, Any]]:
        """Run RF3 prediction on multiple files efficiently.

        Args:
            input_files: List of paths to PDB/CIF files
            template_selection: Template selection string (like ColabFold templates)
            ground_truth_conformer_selection: Ground truth conformer selection string
            out_dir: Output directory (defaults to self.dump_dir)

        Returns:
            List of dictionaries containing prediction results for each input
        """
        if out_dir is None:
            out_dir = self.dump_dir

        logger.info(f"Running RF3 batch prediction on {len(input_files)} inputs...")
        logger.info(f"Converting PDB to CIF for {len(input_files)} inputs...")
        logger.info(f"smiles: {smiles}")
        new_input_files = []
        # for input_file in input_files:
        #  # ground_truth_conformer_selection = None
        #     atom_array = load_any(input_file)[0]
        #     new_input_file = input_file.replace('.pdb', '.cif')
        #     to_cif_file(atom_array, new_input_file)
        #     new_input_files.append(new_input_file)
        #! TODO try cif path in json instead of cif entry
        for input_file in input_files:
            new_input_file = input_file.replace(".pdb", ".json")
            sequence = extract_seq_from_pdb(input_file, chain_id="B")
            json_data = {
                "name": Path(input_file).stem,
                "components": [
                    {"seq": sequence, "chain_id": "B"},
                    {
                        "smiles": smiles,
                        # "chain_id": "A"
                    },
                ],
            }
            logger.info(f"Writing JSON data to {new_input_file}")
            logger.info(f"JSON data: {json_data}")
            with open(new_input_file, "w") as f:
                json.dump(json_data, f)
            new_input_files.append(new_input_file)
        input_files = new_input_files
        try:
            # Try batch processing with list of input paths (RF3 supports this format)
            if len(input_files) > 1:
                try:
                    # Create comma-separated list of input paths for RF3
                    input_list = f"[{','.join(input_files)}]"

                    # Try batch processing with list of input paths
                    cmd = [
                        self.rf3_path,
                        "fold",
                        f"inputs={input_list}",  # Comma-separated list of paths
                        f"ckpt_path={self.ckpt_path}",
                        f"out_dir={out_dir}",
                        "early_stopping_plddt_threshold=0",
                    ]

                    if add_missing_atoms is not None:
                        cmd.append(f"add_missing_atoms={add_missing_atoms}")

                    if template_selection:
                        template_selection = f"[{template_selection}]"
                        cmd.append(f"template_selection={template_selection}")

                    if ground_truth_conformer_selection:
                        if not isinstance(ground_truth_conformer_selection, list):
                            # Only add brackets if they're not already present
                            if not (
                                ground_truth_conformer_selection.startswith("[")
                                and ground_truth_conformer_selection.endswith("]")
                            ):
                                ground_truth_conformer_selection = f"[{ground_truth_conformer_selection}]"
                        else:
                            ground_truth_conformer_selection = "[" + ",".join(ground_truth_conformer_selection) + "]"
                        cmd.append(f"ground_truth_conformer_selection={ground_truth_conformer_selection}")

                    logger.info(f"Running RF3 batch command: {' '.join(cmd)}")
                    logger.info(f"Processing {len(input_files)} input files in batch")
                    try:
                        subprocess.run(
                            cmd,
                            timeout=1500,  # Longer timeout for batch processing
                            check=True,
                        )

                        logger.info("RF3 batch prediction completed successfully")

                        # Parse all results
                        predictions = []
                        for input_file in input_files:
                            pred = self._parse_rf3_output_from_file(input_file, out_dir)
                            predictions.append(pred)

                        return predictions

                    except (
                        subprocess.CalledProcessError,
                        subprocess.TimeoutExpired,
                    ) as e:
                        logger.warning(f"Batch processing failed ({e}), falling back to individual processing")

                except Exception as e:
                    logger.warning(f"Error preparing batch processing ({e}), falling back to individual processing")

            # Fallback to individual processing or if only one file
            logger.info("Processing files individually...")
            predictions = []
            for i, input_file in enumerate(input_files):
                try:
                    logger.info(f"Processing file {i + 1}/{len(input_files)}: {os.path.basename(input_file)}")
                    pred = self.predict_from_file(
                        input_file=input_file,
                        template_selection=template_selection,
                        ground_truth_conformer_selection=ground_truth_conformer_selection,
                        out_dir=out_dir,
                    )
                    predictions.append(pred)
                except Exception as e:
                    logger.error(f"Failed to process {input_file}: {e}")
                    predictions.append(self._empty_prediction())

            return predictions
        finally:
            for json_file in new_input_files:
                try:
                    os.remove(json_file)
                except OSError:
                    pass

    def predict_from_file(
        self,
        input_file: str,
        ground_truth_conformer_selection: str | None = None,
        template_selection: str | None = None,
        out_dir: str | None = None,
        add_missing_atoms: bool | None = None,
    ) -> dict[str, Any]:
        """Run RF3 prediction directly from PDB/CIF file.

        Args:
            input_file: Path to PDB/CIF file
            ground_truth_conformer_selection: Ground truth conformer selection string
            template_selection: Template selection string (like ColabFold templates)
            out_dir: Output directory (defaults to self.dump_dir)

        Returns:
            Dictionary containing prediction results
        """
        if out_dir is None:
            out_dir = self.dump_dir

        # RF3 command with user-specified format
        cmd = [
            self.rf3_path,
            "fold",
            f"inputs={input_file}",
            f"ckpt_path={self.ckpt_path}",
            f"out_dir={out_dir}",
            "early_stopping_plddt_threshold=0",
        ]

        if add_missing_atoms is not None:
            cmd.append(f"add_missing_atoms={add_missing_atoms}")

        # Add ground truth conformer selection if specified
        if ground_truth_conformer_selection is not None:
            cmd.append(f"ground_truth_conformer_selection={ground_truth_conformer_selection}")

        # Add template selection if specified
        if template_selection is not None:
            cmd.append(f"template_selection={template_selection}")

        logger.info(f"Running RF3 command: {' '.join(cmd)}")
        try:
            # Run RF3
            result = subprocess.run(
                cmd,
                timeout=500,  # 500 seconds timeout
                capture_output=True,
                text=True,
            )

            if result.returncode != 0:
                logger.error(f"RF3 failed with return code {result.returncode}")
                logger.error(f"RF3 stderr: {result.stderr}")
                raise RuntimeError(f"RF3 failed with return code {result.returncode}: {result.stderr}")

            logger.info("RF3 prediction completed successfully")

            # List actual output files recursively for debugging (verbose only)
            if self.verbose:
                if os.path.exists(out_dir):
                    logger.info("RF3 output directory contents (recursive):")
                    json_files = []
                    for root, dirs, files in os.walk(out_dir):
                        level = root.replace(out_dir, "").count(os.sep)
                        indent = "  " * level
                        logger.info(f"{indent}{os.path.basename(root)}/")
                        subindent = "  " * (level + 1)
                        for file in files:
                            logger.info(f"{subindent}{file}")
                            if file.endswith("_summary_confidences.json"):
                                json_files.append(os.path.join(root, file))
                    # Print first summary confidences JSON only
                    if json_files:
                        json_file = json_files[0]
                        try:
                            with open(json_file) as f:
                                content = json.load(f)
                            logger.info(f"Contents of {os.path.basename(json_file)}:")
                            logger.info(json.dumps(content, indent=2))
                        except Exception as e:
                            logger.warning(f"Could not read {json_file}: {e}")
                else:
                    logger.warning(f"RF3 output directory does not exist: {out_dir}")

            # Parse results
            prediction = self._parse_rf3_output_from_file(input_file, out_dir)

            return prediction

        except subprocess.CalledProcessError as e:
            logger.error(f"RF3 prediction failed: {e.stderr}")
            raise RuntimeError(f"RF3 prediction failed: {e.stderr}")
        except subprocess.TimeoutExpired:
            logger.error("RF3 prediction timed out")
            raise RuntimeError("RF3 prediction timed out")

    def score(
        self,
        pdb_path: str,
        requires_grad: bool = False,
        sequence: torch.Tensor | None = None,
        structure: torch.Tensor | None = None,
        binder_chain: str | None = None,
        target_chain: str | None = None,
        save_pdb: bool | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        """Calculate reward using RF3 model.

        Args:
            pdb_path: Path to PDB file (required)
            requires_grad: Whether to calculate gradients (not supported, raises error if True)
            sequence: Not used for RF3 (uses PDB file)
            structure: Not used for RF3 (uses PDB file)
            binder_chain: Not used (extracted from PDB)
            target_chain: Not used (extracted from PDB)
            save_pdb: Not supported for RF3 (raises error if True)
            **kwargs: Additional arguments

        Returns:
            Dictionary with reward components:
                reward: Dict[str, torch.Tensor]  # Reward values per metric
                total_reward: torch.Tensor  # Weighted sum of reward components
                plddt: torch.Tensor  # pLDDT value
                pae: torch.Tensor  # Overall PAE value
        """
        self._check_capabilities(requires_grad=requires_grad, save_pdb=bool(save_pdb))

        ground_truth_conformer_selection = kwargs.get("ground_truth_conformer_selection")
        template_selection = kwargs.get("template_selection")

        prediction = self.predict_from_file(
            input_file=pdb_path,
            ground_truth_conformer_selection=ground_truth_conformer_selection,
            template_selection=template_selection,
        )

        # Extract all metric values from prediction
        if prediction and "summary_confidence" in prediction and len(prediction["summary_confidence"]) > 0:
            conf = prediction["summary_confidence"][0]
        else:
            conf = {}

        # Build reward_components: one scalar tensor per metric in reward_options.
        # - has_clash: boolean -> 1.0/0.0 so it can be penalized via a negative weight
        # - PAE-family keys (ipAE, min_ipAE, ...): raw values are in Angstroms (0-31);
        #   divided by 31 when normalize_pae=True so weights are on the same ~0-1 scale
        #   as pLDDT/ipTM/pTM. Defaults: 0.0 for higher-is-better, 100.0 for lower-is-better.
        reward_components = {}
        for key in self.reward_options:
            if key == "has_clash":
                val = 1.0 if conf.get("has_clash", False) else 0.0
            else:
                if key in (
                    "plddt",
                    "ipTM",
                    "pTM",
                    "ranking_score",
                    "min_ipSAE",
                    "max_ipSAE",
                    "avg_ipSAE",
                ):
                    default = 0.0
                else:
                    default = 100.0
                val = conf.get(key, default)
                val = val.item() if isinstance(val, torch.Tensor) else float(val)
                if self.normalize_pae and key in self.PAE_KEYS:
                    val = val / 31.0
            reward_components[key] = ensure_tensor(val)

        # total_reward = sum(component_value * weight) over all metrics
        total_reward = ensure_tensor(
            sum(reward_components[k].item() * self.reward_weights[k] for k in self.reward_weights)
        )

        plddt = reward_components["plddt"]
        pae = reward_components["pAE"]

        return standardize_reward(
            reward=reward_components,
            total_reward=total_reward,
            plddt=plddt,
            pae=pae,
        )

    def predict(
        self,
        input_file: str,
        require_grad: bool = False,
        ground_truth_conformer_selection: str | None = None,
    ) -> dict[str, Any]:
        """Run RF3 prediction (legacy method - now calls predict_from_file).

        Args:
            input_file: Path to PDB/CIF file
            require_grad: Whether gradients are required (not used for RF3)
            ground_truth_conformer_selection: Ground truth conformer selection

        Returns:
            Dictionary containing prediction results
        """
        # Delegate to the simplified method
        return self.predict_from_file(
            input_file=input_file,
            ground_truth_conformer_selection=ground_truth_conformer_selection,
        )

    def predict_batch_from_files(
        self,
        input_files: list[str],
        template_selection: str | None = None,
        ground_truth_conformer_selection: str | None = None,
        out_dir: str | None = None,
        smiles: str = None,
        add_missing_atoms: bool | None = None,
    ) -> list[dict[str, Any]]:
        """Run RF3 prediction on multiple files efficiently.

        Args:
            input_files: List of paths to PDB/CIF files
            template_selection: Template selection string (like ColabFold templates)
            ground_truth_conformer_selection: Ground truth conformer selection string
            out_dir: Output directory (defaults to self.dump_dir)

        Returns:
            List of dictionaries containing prediction results for each input
        """
        if smiles is not None:
            return self.predict_batch_from_files_boltz_compatible(
                input_files=input_files,
                template_selection=template_selection,
                ground_truth_conformer_selection=ground_truth_conformer_selection,
                out_dir=out_dir,
                smiles=smiles,
                add_missing_atoms=add_missing_atoms,
            )

        if out_dir is None:
            out_dir = self.dump_dir

        logger.info(f"Running RF3 batch prediction on {len(input_files)} inputs...")

        # Try batch processing with list of input paths (RF3 supports this format)
        if len(input_files) > 1:
            try:
                # Create comma-separated list of input paths for RF3
                input_list = f"[{','.join(input_files)}]"

                # Try batch processing with list of input paths
                cmd = [
                    self.rf3_path,
                    "fold",
                    f"inputs={input_list}",  # Comma-separated list of paths
                    f"ckpt_path={self.ckpt_path}",
                    f"out_dir={out_dir}",
                    "early_stopping_plddt_threshold=0",
                ]

                if add_missing_atoms is not None:
                    cmd.append(f"add_missing_atoms={add_missing_atoms}")

                if template_selection:
                    template_selection = f"[{template_selection}]"
                    cmd.append(f"template_selection={template_selection}")

                if ground_truth_conformer_selection:
                    if not isinstance(ground_truth_conformer_selection, list):
                        # Only add brackets if they're not already present
                        if not (
                            ground_truth_conformer_selection.startswith("[")
                            and ground_truth_conformer_selection.endswith("]")
                        ):
                            ground_truth_conformer_selection = f"[{ground_truth_conformer_selection}]"
                    else:
                        ground_truth_conformer_selection = "[" + ",".join(ground_truth_conformer_selection) + "]"
                    cmd.append(f"ground_truth_conformer_selection={ground_truth_conformer_selection}")

                logger.info(f"Running RF3 batch command: {' '.join(cmd)}")
                logger.info(f"Processing {len(input_files)} input files in batch")
                try:
                    result = subprocess.run(
                        cmd,
                        timeout=1500,  # Longer timeout for batch processing
                        check=True,
                    )

                    logger.info("RF3 batch prediction completed successfully")

                    # Parse all results
                    predictions = []
                    for input_file in input_files:
                        pred = self._parse_rf3_output_from_file(input_file, out_dir)
                        predictions.append(pred)

                    return predictions

                except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
                    logger.warning(f"Batch processing failed ({e}), falling back to individual processing")

            except Exception as e:
                logger.warning(f"Error preparing batch processing ({e}), falling back to individual processing")

        # Fallback to individual processing or if only one file
        logger.info("Processing files individually...")
        predictions = []
        for i, input_file in enumerate(input_files):
            try:
                logger.info(f"Processing file {i + 1}/{len(input_files)}: {os.path.basename(input_file)}")
                pred = self.predict_from_file(
                    input_file=input_file,
                    template_selection=template_selection,
                    ground_truth_conformer_selection=ground_truth_conformer_selection,
                    out_dir=out_dir,
                    add_missing_atoms=add_missing_atoms,
                )
                predictions.append(pred)
            except Exception as e:
                logger.error(f"Failed to process {input_file}: {e}")
                predictions.append(self._empty_prediction())

        return predictions

    def _empty_prediction(self) -> dict[str, Any]:
        """Return empty prediction for failed cases."""
        return {
            "summary_confidence": [
                {
                    "plddt": torch.tensor(0.0),
                    "ipAE": torch.tensor(100.0),  # Mean PAE across binder interfaces
                    "min_ipAE": torch.tensor(100.0),  # Best binder interface PAE (min)
                    "pAE": torch.tensor(100.0),
                    "ipTM": torch.tensor(0.0),
                    "pTM": torch.tensor(0.0),
                    "ranking_score": torch.tensor(0.0),
                    "has_clash": False,
                    "min_ipSAE": torch.tensor(0.0),
                    "max_ipSAE": torch.tensor(0.0),
                    "avg_ipSAE": torch.tensor(0.0),
                }
            ],
            "output_cif_path": None,
        }

    def _parse_rf3_output_from_file(self, input_file: str, out_dir: str) -> dict[str, Any]:
        """Parse RF3 output files from direct file input.

        Args:
            input_file: Path to input PDB/CIF file
            out_dir: Output directory where RF3 results are saved

        Returns:
            Dictionary containing parsed results
        """
        # Extract name from input file (without extension)
        name = Path(input_file).stem
        if name.endswith(".cif"):
            name = name[:-4]  # Remove .cif if present
        elif name.endswith(".pdb"):
            name = name[:-4]  # Remove .pdb if present

        # RF3 outputs to a subdirectory named after the input file
        # Files: {name}_summary_confidences.json, {name}_model.cif, {name}_confidences.json, {name}_ranking_scores.csv
        sample_out_dir = os.path.join(out_dir, name)

        # Parse summary_confidences.json for all metrics
        summary_conf_file = os.path.join(sample_out_dir, f"{name}_summary_confidences.json")
        if os.path.exists(summary_conf_file):
            with open(summary_conf_file) as f:
                summary_conf = json.load(f)

            # Extract metrics from RF3 summary_confidences.json
            # RF3 JSON structure:
            # - overall_plddt: float (0-1 scale, multiply by 100 for 0-100)
            # - overall_pae: float (PAE in Angstroms)
            # - ptm: float (0-1)
            # - iptm: float (0-1) - interface pTM
            # - ranking_score: float
            # - has_clash: bool
            # - chain_pair_pae_min: 2D array with min interface PAE between chain pairs
            # - chain_pair_pae: 2D array with mean interface PAE between chain pairs

            # pLDDT (convert from 0-1 to 0-100 scale)
            plddt = summary_conf.get("overall_plddt", 0.0)  # * 100.0

            # pTM and ipTM
            ptm_value = summary_conf.get("ptm", 0.0)
            iptm_value = summary_conf.get("iptm", 0.0)

            # Overall PAE
            pae = summary_conf.get("overall_pae", 100.0)

            # Interface PAE - last chain is binder, extract last column (binder interfaces)
            # Convert to numpy with nan for None, then use nanmin/nanmean on last column
            chain_pair_pae_min = summary_conf.get("chain_pair_pae_min", [])
            chain_pair_pae = summary_conf.get("chain_pair_pae", [])

            if chain_pair_pae_min and len(chain_pair_pae_min) > 1:
                arr = np.array(chain_pair_pae_min, dtype=float)  # None -> nan
                last_col = arr[:-1, -1]  # Last column, excluding diagonal
                min_ipae = float(np.nanmin(last_col)) if not np.all(np.isnan(last_col)) else 100.0
                mean_min_ipae = float(np.nanmean(last_col)) if not np.all(np.isnan(last_col)) else 100.0
            else:
                min_ipae = 100.0
                mean_min_ipae = 100.0

            if chain_pair_pae and len(chain_pair_pae) > 1:
                arr = np.array(chain_pair_pae, dtype=float)
                last_col = arr[:-1, -1]
                mean_ipae = float(np.nanmean(last_col)) if not np.all(np.isnan(last_col)) else 100.0
                min_mean_ipae = float(np.nanmin(last_col)) if not np.all(np.isnan(last_col)) else 100.0
            else:
                mean_ipae = 100.0
                min_mean_ipae = 100.0

            ranking_score = summary_conf.get("ranking_score", 0.0)
            has_clash = summary_conf.get("has_clash", False)

            logger.info(
                f"RF3 metrics: pLDDT={plddt:.1f}, pTM={ptm_value:.3f}, ipTM={iptm_value:.3f}, "
                f"pAE={pae:.2f}, min_ipAE={min_ipae:.2f}, mean_min_ipAE={mean_min_ipae:.2f}, mean_ipAE={mean_ipae:.2f}, min_mean_ipAE={min_mean_ipae:.2f}, "
                f"ranking_score={ranking_score:.3f}, has_clash={has_clash}"
            )
        else:
            logger.warning(f"Summary confidences file not found: {summary_conf_file}")
            plddt = 0.0
            pae = 100.0
            min_ipae = 100.0
            mean_ipae = 100.0
            mean_min_ipae = 100.0
            min_mean_ipae = 100.0
            iptm_value = 0.0
            ptm_value = 0.0
            ranking_score = 0.0
            has_clash = False

        # Find the CIF file
        cif_file = os.path.join(sample_out_dir, f"{name}_model.cif.gz")
        if not os.path.exists(cif_file):
            cif_file = os.path.join(sample_out_dir, f"{name}_model.cif")

        # Compute ipSAE from the full PAE matrix + output structure,
        # using complex_ipSAE which filters to spatially interacting chain pairs
        min_ipsae, max_ipsae, avg_ipsae = 0.0, 0.0, 0.0
        conf_file = os.path.join(sample_out_dir, f"{name}_confidences.json")
        if os.path.exists(conf_file) and os.path.exists(cif_file):
            try:
                with open(conf_file) as f:
                    full_conf = json.load(f)
                pae_raw = full_conf.get("pae")
                if pae_raw is not None:
                    pae_matrix = torch.tensor(pae_raw, dtype=torch.float32)
                    ipsae_result = complex_ipSAE(
                        pae_matrix,
                        cif_file,
                        interaction_cutoff=8.0,
                        pae_cutoff=10.0,
                    )
                    min_ipsae = float(ipsae_result["min"])
                    max_ipsae = float(ipsae_result["max"])
                    avg_ipsae = float(ipsae_result["avg"])
                    logger.info(f"RF3 ipSAE: min={min_ipsae:.4f}, max={max_ipsae:.4f}, avg={avg_ipsae:.4f}")
                else:
                    logger.warning(f"PAE matrix missing in {conf_file}")
            except Exception as e:
                logger.warning(f"Could not compute ipSAE from {conf_file}: {e}")
        else:
            logger.debug(f"Confidences or CIF file not found (ipSAE unavailable): {conf_file}")

        prediction = {
            "summary_confidence": [
                {
                    "plddt": torch.tensor(plddt if plddt else 0.0),
                    "ipAE": torch.tensor(mean_ipae if mean_ipae else 100.0),  # Mean PAE across binder interfaces
                    "min_ipAE": torch.tensor(min_ipae if min_ipae else 100.0),  # Best binder interface PAE (min)
                    "mean_min_ipAE": torch.tensor(
                        mean_min_ipae if mean_min_ipae else 100.0
                    ),  # Mean of best binder interface PAE (min)
                    "mean_ipAE": torch.tensor(mean_ipae if mean_ipae else 100.0),  # Mean PAE across binder interfaces
                    "min_mean_ipAE": torch.tensor(
                        min_mean_ipae if min_mean_ipae else 100.0
                    ),  # Mean of best binder interface PAE (min)
                    "pAE": torch.tensor(pae if pae else 100.0),
                    "ipTM": torch.tensor(iptm_value if iptm_value else 0.0),
                    "pTM": torch.tensor(ptm_value if ptm_value else 0.0),
                    "ranking_score": torch.tensor(ranking_score if ranking_score else 0.0),
                    "has_clash": has_clash,
                    "min_ipSAE": torch.tensor(min_ipsae if min_ipsae else 0.0),
                    "max_ipSAE": torch.tensor(max_ipsae if max_ipsae else 0.0),
                    "avg_ipSAE": torch.tensor(avg_ipsae if avg_ipsae else 0.0),
                }
            ],
            "output_cif_path": cif_file if os.path.exists(cif_file) else None,
        }

        return prediction

    def _parse_rf3_output(self, input_file: str) -> dict[str, Any]:
        """Parse RF3 output files (legacy method).

        Args:
            input_file: Path to input file

        Returns:
            Dictionary containing parsed results
        """
        # For backward compatibility, assume it's a direct file
        return self._parse_rf3_output_from_file(input_file, self.dump_dir)


def get_default_rf3_runner(
    ckpt_path: str | None = None,
    dump_dir: str | None = None,
    rf3_path: str | None = None,
    verbose: bool = False,
    reward_weights: dict[str, float] | None = None,
    normalize_pae: bool = True,
) -> RF3RewardRunner:
    """Get default RF3 reward runner.

    Args:
        ckpt_path: Path to checkpoint file
        dump_dir: Directory to dump results (default is ./rf3_outputs)
        rf3_path: Path to RF3 executable
        verbose: If True, print debug info (file listings, JSON contents)
        reward_weights: Dict mapping metric names to scalar weights for score().
        normalize_pae: If True, divide PAE-family metrics by 31.0 in score().
    Returns:
        RF3RewardRunner instance
    """
    if dump_dir is None:
        dump_dir = "./rf3_outputs"
    return RF3RewardRunner(
        ckpt_path=ckpt_path,
        dump_dir=dump_dir,
        rf3_path=rf3_path,
        verbose=verbose,
        reward_weights=reward_weights,
        normalize_pae=normalize_pae,
    )
