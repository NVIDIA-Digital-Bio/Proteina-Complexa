import logging
import os
import warnings


def quiet_startup() -> None:
    """Suppress noisy warnings and logging during startup."""

    # ==========================================================================
    # Set environment variables to suppress warnings BEFORE any imports
    # ==========================================================================

    os.environ["PYTHONWARNINGS"] = "ignore"

    # JAX-specific: suppress deprecation warnings
    os.environ["JAX_DEPRECATION_WARNINGS"] = "0"

    # Suppress XLA/JAX logging noise
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    os.environ["JAX_LOG_COMPILES"] = "0"

    # ==========================================================================
    # Blanket ignore all warnings first
    # ==========================================================================

    warnings.filterwarnings("ignore")

    # Also simplefilter to catch everything
    warnings.simplefilter("ignore")

    # ==========================================================================
    # Suppress specific warning messages (for any that slip through)
    # ==========================================================================

    warnings.filterwarnings("ignore", message=r".*predict_dataloader.*num_workers.*")
    warnings.filterwarnings("ignore", message=r".*tensorboardX.*removed as a dependency.*")
    warnings.filterwarnings(
        "ignore",
        message=r"The pynvml package is deprecated",
        category=FutureWarning,
        module=r"torch\.cuda",
    )

    # Suppress JAX deprecation warnings (by message, since they're raised from colabdesign)
    warnings.filterwarnings(
        "ignore",
        message=r".*jax\.tree_map is deprecated.*",
        category=DeprecationWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message=r".*jax\.tree_flatten is deprecated.*",
        category=DeprecationWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message=r".*jax\.tree_unflatten is deprecated.*",
        category=DeprecationWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message=r".*backend and device argument on jit is deprecated.*",
        category=DeprecationWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message=r".*jax\.numpy\.clip is deprecated.*",
        category=DeprecationWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message=r".*Passing arguments.*to jax\.numpy\.clip.*",
        category=DeprecationWarning,
    )

    # Suppress optax/jax deprecation warnings
    warnings.filterwarnings("ignore", message=r".*optax\.dpsgd.*", category=DeprecationWarning)
    warnings.filterwarnings("ignore", message=r".*optax\.noisy_sgd.*", category=DeprecationWarning)
    warnings.filterwarnings("ignore", category=DeprecationWarning, module=r"jax.*")
    warnings.filterwarnings("ignore", category=DeprecationWarning, module=r"optax.*")

    # Suppress SWIG deprecation warnings (SwigPyPacked, SwigPyObject, swigvarlink)
    # These come from <frozen importlib._bootstrap>
    warnings.filterwarnings("ignore", message=r".*builtin type Swig.*")
    warnings.filterwarnings("ignore", message=r".*builtin type swig.*")
    warnings.filterwarnings("ignore", message=r".*swigvarlink.*")
    warnings.filterwarnings("ignore", message=r".*SwigPyPacked.*")
    warnings.filterwarnings("ignore", message=r".*SwigPyObject.*")
    warnings.filterwarnings("ignore", message=r".*has no __module__ attribute.*")
    warnings.filterwarnings("ignore", category=DeprecationWarning, module=r"importlib.*")

    # Suppress all deprecation warnings from colabdesign (where jax calls happen)
    warnings.filterwarnings("ignore", category=DeprecationWarning, module=r"colabdesign.*")

    # Suppress invalid escape sequence warnings
    warnings.filterwarnings("ignore", message=r".*invalid escape sequence.*")

    # Suppress gearnet deprecation warnings
    warnings.filterwarnings("ignore", module=r".*gearnet.*")

    # Suppress atomworks environment variable warnings
    warnings.filterwarnings("ignore", message=r".*CCD_MIRROR_PATH.*")
    warnings.filterwarnings("ignore", message=r".*PDB_MIRROR_PATH.*")

    # Catch-all for remaining deprecation warnings from our codebase
    warnings.filterwarnings("ignore", category=DeprecationWarning, module=r"proteinfoundation.*")

    # ==========================================================================
    # Suppress noisy loggers
    # ==========================================================================

    logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
    logging.getLogger("lightning").setLevel(logging.ERROR)
    logging.getLogger("atomworks").setLevel(logging.WARNING)

    # ==========================================================================
    # Suppress atomworks environment variable print statements
    # ==========================================================================

    # Set dummy env vars to prevent atomworks from printing warnings
    if "CCD_MIRROR_PATH" not in os.environ:
        os.environ["CCD_MIRROR_PATH"] = ""
    if "PDB_MIRROR_PATH" not in os.environ:
        os.environ["PDB_MIRROR_PATH"] = ""
