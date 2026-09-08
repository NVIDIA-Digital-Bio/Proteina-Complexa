import os,jax

# --- jax compatibility shim (Blackwell/B200 support needs jax with CUDA 12.8) ---
# The AF2 code below uses the top-level jax.tree_* aliases, which were removed in
# jax >= 0.4.31 (fully gone by 0.6/0.11). Restore them from jax.tree_util so this
# modified colabdesign works unchanged on the newer jax required for sm_100 GPUs.
# No-op on older jax where the aliases still exist.
import jax.tree_util as _jtu
for _n in ("tree_map", "tree_flatten", "tree_unflatten", "tree_leaves"):
    if not hasattr(jax, _n):
        setattr(jax, _n, getattr(_jtu, _n))

# disable triton_gemm for jax versions > 0.3
if int(jax.__version__.split(".")[1]) > 3:
  os.environ["XLA_FLAGS"] = "--xla_gpu_enable_triton_gemm=false"

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from .shared.utils import clear_mem
from .af.model import mk_af_model
from .tr.model import mk_tr_model
from .mpnn.model import mk_mpnn_model

# backward compatability
mk_design_model = mk_afdesign_model = mk_af_model
mk_trdesign_model = mk_tr_model