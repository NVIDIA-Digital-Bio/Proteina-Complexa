"""Base class for all search algorithms."""

import dataclasses
from abc import ABC, abstractmethod
from collections.abc import Callable
from functools import partial
from typing import Any

import torch


@dataclasses.dataclass
class SearchContext:
    """Immutable bundle of state shared across a single search invocation.

    Previously this information was scattered across ad-hoc private
    attributes on the ``proteina`` instance (``_current_batch``,
    ``_current_predict_fn``, ``_current_ts``, ``_current_gt``,
    ``_current_sim_params``).  Grouping them here makes the data-flow
    explicit: ``build_search_context`` constructs one at the start of
    ``search()``, and it is threaded to every helper that needs it
    (``chunked_partial_simulation``, ``generate_samples_to_completion``,
    ``compute_reward_from_samples``).

    Fields mirror the exact values that ``chunked_partial_simulation``
    and ``generate_samples_to_completion`` require, so callers can
    unpack directly without reaching into ``proteina`` internals.
    """

    current_batch: dict[str, Any]
    """The conditioning batch for this predict_step (target structure,
    masks, metadata).  Named ``current_batch`` to preserve the familiar
    convention from the previous ``proteina._current_batch``."""

    predict_fn: Callable
    """Bound ``predict_for_sampling`` with the correct ``n_recycle``."""

    ts: torch.Tensor
    """Timestep schedule returned by ``fm.sample_schedule``."""

    gt: torch.Tensor
    """Ground-truth schedule returned by ``fm.sample_schedule``."""

    simulation_step_params: dict[str, dict]
    """Per-data-mode simulation step parameters from ``inf_cfg.model``."""

    nsteps: int
    """Total number of denoising steps (``inf_cfg.args.nsteps``).
    Used by ``generate_samples_to_completion`` to know the final step."""

    self_cond: bool
    """Whether self-conditioning is enabled."""

    guidance_w: float
    """Classifier-free guidance weight."""

    ag_ratio: float
    """Auto guidance ratio."""

    max_batch_size: int | None
    """Cap for ``chunked_partial_simulation`` sub-batches (None = no cap)."""

    device: torch.device
    """Device tensors live on."""


class BaseSearch(ABC):
    """Base class for all search algorithms.

    Every search algorithm receives a Proteina model instance and an
    inference config at construction, and implements ``search(batch)``
    which returns a dict with two keys:

        {"lookahead": Optional[Dict], "final": Dict}

    **final** (required): the selected output samples.  Dict with at
    least ``coors``, ``residue_type``, ``mask``, and ``metadata_tag``.
    Finals must **not** contain ``rewards`` — reward scoring for final
    samples is the responsibility of ``proteina.predict_step``, not
    the search algorithm.

    **lookahead** (optional): intermediate samples generated during
    search (e.g. rollouts at each checkpoint).  Same keys as finals
    **plus** ``rewards``, which **must** already be computed by the
    search algorithm (they are needed for selection decisions during
    search and are never re-scored).  ``None`` when the algorithm has
    no intermediate samples (e.g. single-pass).

    In short: search scores intermediates, ``proteina.predict_step``
    scores finals.

    Output layout conventions
    -------------------------
    Finals use **grouped layout** when the algorithm produces more
    samples than the input batch (all beams/replicas for sample 0
    first, then sample 1, etc.).  This matters for:

    - ``prepend_target_to_samples``: use the default
      ``repeat_mode="interleave"`` for finals.  Only internal search
      helpers (lookahead rollouts) need ``repeat_mode="tile"``.
    - ``expand_hotspot_mask`` in ``proteina.py``: pre-expands the
      mask to match grouped layout before reward scoring.
    """

    def __init__(self, proteina_instance: Any, inf_cfg: Any) -> None:
        self.proteina = proteina_instance
        self.inf_cfg = inf_cfg

    # ------------------------------------------------------------------
    # Shared setup — builds the SearchContext that replaces the old
    # proteina._current_* attributes.
    # ------------------------------------------------------------------

    def build_search_context(self, batch: dict) -> SearchContext:
        """Construct a :class:`SearchContext` from the current batch.

        Every iterative search algorithm (beam search, FK-steering, MCTS)
        previously duplicated ~15 lines of identical setup to create
        ``fn_predict_for_sampling``, ``ts``, ``gt``,
        ``simulation_step_params``, and read ``self_cond``, ``guidance_w``,
        ``ag_ratio``, ``max_batch_size``.  This method centralises that
        boilerplate so each algorithm can call it once at the top of
        ``search()`` and pass the resulting ``SearchContext`` to all
        helpers.
        """
        nsteps = self.inf_cfg.args.nsteps

        predict_fn = partial(
            self.proteina.predict_for_sampling,
            n_recycle=self.inf_cfg.get("n_recycle", 0),
        )
        ts, gt = self.proteina.fm.sample_schedule(
            nsteps=nsteps,
            sampling_model_args=self.inf_cfg.model,
        )
        simulation_step_params = {
            data_mode: self.inf_cfg.model[data_mode]["simulation_step_params"]
            for data_mode in self.proteina.fm.data_modes
        }

        max_batch_size = self.inf_cfg.search.get("max_batch_size", None)
        if max_batch_size is not None and max_batch_size <= 0:
            raise ValueError(f"max_batch_size must be > 0, got {max_batch_size}")

        return SearchContext(
            current_batch=batch,
            predict_fn=predict_fn,
            ts=ts,
            gt=gt,
            simulation_step_params=simulation_step_params,
            nsteps=nsteps,
            self_cond=self.inf_cfg.args.self_cond,
            guidance_w=self.inf_cfg.args.get("guidance_w", 1.0),
            ag_ratio=self.inf_cfg.args.get("ag_ratio", 0.0),
            max_batch_size=max_batch_size,
            device=self.proteina.device,
        )

    @abstractmethod
    def search(self, batch: dict) -> dict:
        """Run the search algorithm on a batch.

        Args:
            batch: Input batch with at least ``mask`` [batch_size, n_residues].

        Returns:
            Dict with ``"lookahead"`` (Optional[Dict]) and ``"final"`` (Dict).
        """
