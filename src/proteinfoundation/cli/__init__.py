"""
Proteina CLI utilities and tools.

This package provides command-line interface tools for the Proteina project:
    - search: Binder search pipeline orchestration
    - progress_bar: Custom progress bar with pipeline step display
    - startup: Quiet startup utilities to suppress warnings

Note: Imports are lazy to keep CLI startup fast. Import specific modules directly:
    from proteinfoundation.cli.progress_bar import PipelineProgressBar
    from proteinfoundation.cli.startup import quiet_startup
"""

# Lazy imports - don't load heavy dependencies at package import time
# This keeps CLI commands like `complexa target list` fast

__all__ = [
    "PipelineProgressBar",
    "clear_pipeline_step",
    "quiet_startup",
    "set_pipeline_step",
]


def __getattr__(name):
    """Lazy import for CLI utilities to avoid loading PyTorch Lightning at startup."""
    if name == "quiet_startup":
        from proteinfoundation.cli.startup import quiet_startup

        return quiet_startup
    elif name == "PipelineProgressBar":
        from proteinfoundation.cli.progress_bar import PipelineProgressBar

        return PipelineProgressBar
    elif name == "set_pipeline_step":
        from proteinfoundation.cli.progress_bar import set_pipeline_step

        return set_pipeline_step
    elif name == "clear_pipeline_step":
        from proteinfoundation.cli.progress_bar import clear_pipeline_step

        return clear_pipeline_step
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
