# Reward Models

Reward models evaluate protein sequences and structures during binder design. They provide scores (and optionally gradients) used by search algorithms to guide optimization.

## Architecture Overview

```
BaseRewardModel (ABC)
├── score(pdb_path, requires_grad, **kwargs) -> Dict   # abstract, must implement
├── extract_results(aux) -> Dict                        # default: return aux
├── _check_capabilities(requires_grad, save_pdb)        # call at top of score()
├── _validate_score_output(result)                      # static, checks return format
└── cleanup() -> None
```

All reward models must return a dict with:
- **reward**: `Dict[str, torch.Tensor]` — component scores (e.g. plddt, pae, n_hbonds)
- **grad**: `Dict[str, torch.Tensor]` — gradients w.r.t. sequence/structure (empty if no gradients)
- **total_reward**: `torch.Tensor` — scalar total reward

## Model Types

### Folding Models

Predict 3D structures from sequences. Can optionally save refolded PDBs for downstream non-folding models.

| Model | Gradients | save_pdb | Description |
|-------|-----------|----------|-------------|
| AF2RewardModel | ✓ | ✓ | AlphaFold2 via ColabDesign |
| RF3RewardRunner | ✗ | ✗ | RoseTTAFold3 via CLI |

### Non-folding Models

Compute metrics over existing structures. Can use generated or refolded structures via `structure_source` when used in `CompositeRewardModel`.

| Model | Gradients | structure_source | Description |
|-------|-----------|-------------------|-------------|
| BioinformaticsRewardModel | ✗ | ✓ | SC, dSASA, interface metrics |
| TmolRewardModel | ✓ | ✗ | Interface energy (H-bond, electrostatic) |

## Base Helpers

In `base_reward.py`:

- **`ensure_tensor(value)`** — Convert scalars/arrays to `torch.Tensor`
- **`standardize_reward(reward, grad, total_reward, **extra)`** — Build standardized result dict

---

## Adding a Custom Reward Model

### 1. Create the module

Create `src/proteinfoundation/rewards/my_reward.py`:

```python
from typing import Any, Dict, Optional

import torch

from proteinfoundation.rewards.base_reward import (
    BaseRewardModel,
    standardize_reward,
    ensure_tensor,
)


class MyRewardModel(BaseRewardModel):
    """Description of your reward model."""

    # Set capability flags (defaults inherited from BaseRewardModel are all False)
    IS_FOLDING_MODEL = False      # True for folding models (AF2, Boltz2, PTX, RF3)
    SUPPORTS_GRAD = False         # True if score() can compute gradients
    SUPPORTS_SAVE_PDB = False     # True if score() can save a predicted PDB

    def __init__(self, weight: float = 1.0, structure_source: Optional[str] = None, **kwargs):
        super().__init__()
        self.weight = weight
        # For non-folding models: key of folding model whose structure to use
        self.structure_source = structure_source

    def score(self, pdb_path: str, requires_grad: bool = False, **kwargs) -> Dict[str, Any]:
        self._check_capabilities(requires_grad=requires_grad)

        # Your scoring logic here (binder_chain, target_chain, etc. via kwargs)
        score = self._compute_score(pdb_path, kwargs.get("binder_chain"), kwargs.get("target_chain"))

        return standardize_reward(
            reward={"my_metric": ensure_tensor(score)},
            total_reward=score * self.weight,
        )

    def _compute_score(self, pdb_path: str, binder_chain: str, target_chain: str) -> float:
        # Load PDB, compute metric, return scalar
        ...
```

### 2. Implement `score`

**Minimal signature:**
```python
def score(self, pdb_path: str, requires_grad: bool = False, **kwargs) -> Dict[str, Any]:
```

Common kwargs: `sequence`, `structure`, `binder_chain`, `target_chain`, `save_pdb`, `output_pdb_path`, etc.

**Return format (use `standardize_reward`):**
```python
{
    "reward": {"metric1": tensor, "metric2": tensor, ...},
    "grad": {"sequence": tensor, "structure": tensor, ...} or {},
    "total_reward": tensor,  # scalar
    # Optional extra keys: plddt, pae, model_specific_field
}
```

### 3. Register in CompositeRewardModel

In your config (e.g. `configs/pipeline/binder_generate.yaml`):

```yaml
reward_model:
  _target_: proteinfoundation.rewards.base_reward.CompositeRewardModel
  reward_models:
    af2:
      _target_: proteinfoundation.rewards.alphafold2_reward.AF2RewardModel
      ...
    my_reward:
      _target_: proteinfoundation.rewards.my_reward.MyRewardModel
      weight: 0.5
  weights:
    af2: 1.0
    my_reward: 0.5
```

### 4. (Optional) Use refolded structure

For non-folding models that should score the AF2-refolded structure instead of the generated one:

```python
class MyRewardModel(BaseRewardModel):
    structure_source = "af2"  # Use output from reward model key "af2"
```

`CompositeRewardModel` will pass the refolded PDB path to `score` when `structure_source` is set.

---

## Checklist for New Reward Models

- [ ] Inherit from `BaseRewardModel`
- [ ] Implement `score` with the standard signature
- [ ] Return dict with `reward`, `grad`, `total_reward`
- [ ] Override `extract_results(aux)` only if raw aux needs transformation (default is pass-through)
- [ ] Use `standardize_reward` and `ensure_tensor` for consistent formatting
- [ ] Set class attrs: `IS_FOLDING_MODEL`, `SUPPORTS_GRAD`, `SUPPORTS_SAVE_PDB` as needed
- [ ] Call `self._check_capabilities(requires_grad, save_pdb)` at top of `score()`
- [ ] Set `structure_source` in `__init__` for non-folding models that use refolded structures
- [ ] Add tests if the model has non-trivial logic
