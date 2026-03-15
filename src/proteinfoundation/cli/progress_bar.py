"""
Custom PyTorch Lightning progress bar callback that displays pipeline step information.

This module provides a progress bar that shows the current pipeline step (e.g., "Folding",
"Inverse Folding") by reading from environment variables set by the pipeline orchestrator.

from proteinfoundation.cli.progress_bar import PipelineProgressBar, set_pipeline_step, clear_pipeline_step

def run_design_pipeline(target_pdb: str):
    trainer = Trainer(callbacks=[PipelineProgressBar()], ...)

    # Step 1: Generate structures
    set_pipeline_step("Structure Generation", current=1, total=4)
    trainer.predict(structure_model, structure_dataloader)

    # Step 2: Fold with AF2
    set_pipeline_step("AlphaFold2 Folding", current=2, total=4)
    trainer.predict(af2_model, af2_dataloader)

    # Step 3: Inverse folding
    set_pipeline_step("Inverse Folding", current=3, total=4)
    trainer.predict(mpnn_model, mpnn_dataloader)

    # Step 4: Scoring
    set_pipeline_step("Scoring", current=4, total=4)
    trainer.predict(scoring_model, scoring_dataloader)

    clear_pipeline_step()
"""

from __future__ import annotations

import os

from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import TQDMProgressBar
from tqdm.auto import tqdm


class PipelineProgressBar(TQDMProgressBar):
    """Progress bar that displays current pipeline step in the description.

    Reads pipeline information from environment variables:
        - PIPELINE_STEP: Current step name (e.g., "Folding")
        - PIPELINE_PROGRESS: Progress indicator (e.g., "2/5")

    Example output: "[2/5] Folding - Predicting: 50%|█████     | 5/10"
    """

    # Environment variable names
    ENV_PIPELINE_STEP = "PIPELINE_STEP"
    ENV_PIPELINE_PROGRESS = "PIPELINE_PROGRESS"

    def __init__(self, refresh_rate: int = 1, process_position: int = 0) -> None:
        super().__init__(refresh_rate=refresh_rate, process_position=process_position)
        self._cached_prefix: str | None = None

    @property
    def pipeline_prefix(self) -> str:
        """Get the pipeline prefix string from environment variables.

        Returns
        -------
        str
            Formatted prefix like "[2/5] Folding" or empty string if not set.
        """
        step = os.environ.get(self.ENV_PIPELINE_STEP, "")
        if not step:
            return ""

        progress = os.environ.get(self.ENV_PIPELINE_PROGRESS, "")
        if progress:
            return f"[{progress}] {step}"
        return f"[Pipeline] {step}"

    def _format_description(self, base_desc: str = "") -> str:
        """Combine pipeline prefix with base description.

        Parameters
        ----------
        base_desc : str
            The original description text.

        Returns
        -------
        str
            Combined description with pipeline prefix.
        """
        prefix = self.pipeline_prefix
        if not prefix:
            return base_desc

        base_desc = (base_desc or "").strip().rstrip(":")
        if base_desc:
            return f"{prefix} - {base_desc}"
        return prefix

    def _update_bar(self, bar: tqdm | None, base_desc: str = "") -> None:
        """Update a progress bar's description with pipeline info.

        Parameters
        ----------
        bar : tqdm or None
            The progress bar to update.
        base_desc : str
            Base description to include.
        """
        if bar is not None:
            new_desc = self._format_description(base_desc)
            if new_desc:
                bar.set_description(new_desc)

    # Override initialization methods to add pipeline info
    def init_predict_tqdm(self) -> tqdm:
        bar = super().init_predict_tqdm()
        self._update_bar(bar, "Predicting")
        return bar

    def init_train_tqdm(self) -> tqdm:
        bar = super().init_train_tqdm()
        self._update_bar(bar, "Training")
        return bar

    def init_validation_tqdm(self) -> tqdm:
        bar = super().init_validation_tqdm()
        self._update_bar(bar, "Validating")
        return bar

    def init_test_tqdm(self) -> tqdm:
        bar = super().init_test_tqdm()
        self._update_bar(bar, "Testing")
        return bar

    def init_sanity_tqdm(self) -> tqdm:
        bar = super().init_sanity_tqdm()
        self._update_bar(bar, "Sanity Check")
        return bar

    # Update bars when stages start (in case env vars changed)
    def on_predict_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        super().on_predict_start(trainer, pl_module)
        self._update_bar(self.predict_progress_bar, "Predicting")

    def on_train_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        super().on_train_start(trainer, pl_module)
        self._update_bar(self.main_progress_bar, "Training")

    def on_validation_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        super().on_validation_start(trainer, pl_module)
        self._update_bar(self.val_progress_bar, "Validating")

    def on_test_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        super().on_test_start(trainer, pl_module)
        self._update_bar(self.test_progress_bar, "Testing")


def set_pipeline_step(step: str, current: int | None = None, total: int | None = None) -> None:
    """Set the current pipeline step for progress bar display.

    Parameters
    ----------
    step : str
        Name of the current step (e.g., "Folding", "Inverse Folding").
    current : int, optional
        Current step number (1-indexed).
    total : int, optional
        Total number of steps.

    Examples
    --------
    >>> set_pipeline_step("Folding", current=2, total=5)
    # Progress bars will show: "[2/5] Folding - ..."

    >>> set_pipeline_step("Structure Prediction")
    # Progress bars will show: "[Pipeline] Structure Prediction - ..."
    """
    os.environ[PipelineProgressBar.ENV_PIPELINE_STEP] = step

    if current is not None and total is not None:
        os.environ[PipelineProgressBar.ENV_PIPELINE_PROGRESS] = f"{current}/{total}"
    elif PipelineProgressBar.ENV_PIPELINE_PROGRESS in os.environ:
        del os.environ[PipelineProgressBar.ENV_PIPELINE_PROGRESS]


def clear_pipeline_step() -> None:
    """Clear the pipeline step information from environment variables."""
    for key in [
        PipelineProgressBar.ENV_PIPELINE_STEP,
        PipelineProgressBar.ENV_PIPELINE_PROGRESS,
    ]:
        if key in os.environ:
            del os.environ[key]
