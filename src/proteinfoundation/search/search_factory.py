"""
Search algorithm and refinement factory.
Extracted from Proteina class for separation of concerns.
"""

from typing import Any

from proteinfoundation.search import (
    BaseSearch,
    BeamSearch,
    BestOfNSearch,
    FKSteering,
    MCTSSearch,
    SequenceHallucination,
    SinglePassGeneration,
)


def instantiate_search(proteina: Any, inf_cfg: Any, algorithm: str) -> BaseSearch:
    """Get search algorithm instance.

    Args:
        proteina: Proteina model instance.
        inf_cfg: Inference config.
        algorithm: 'single-pass', 'best-of-n', 'beam-search', 'fk-steering', or 'mcts'.

    Returns:
        Search algorithm instance.
    """
    if algorithm == "single-pass":
        return SinglePassGeneration(proteina, inf_cfg)
    if algorithm == "best-of-n":
        return BestOfNSearch(proteina, inf_cfg)
    if algorithm == "beam-search":
        return BeamSearch(proteina, inf_cfg)
    if algorithm == "fk-steering":
        return FKSteering(proteina, inf_cfg)
    if algorithm == "mcts":
        return MCTSSearch(proteina, inf_cfg)
    raise ValueError(f"Unknown search algorithm: {algorithm}")


def instantiate_refinement(proteina: Any, inf_cfg: Any, algorithm: str) -> Any:
    """Get refinement algorithm instance.

    Args:
        proteina: Proteina model instance.
        inf_cfg: Inference config.
        algorithm: Refinement algorithm name (e.g. 'sequence_hallucination').

    Returns:
        Refinement algorithm instance.
    """
    if algorithm == "sequence_hallucination":
        return SequenceHallucination(proteina, inf_cfg)
    raise ValueError(f"Unknown refinement algorithm: {algorithm}")
