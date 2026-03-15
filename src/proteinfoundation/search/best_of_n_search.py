"""Best-of-n search algorithm for protein generation.

Batching strategy (was: ``replicas`` sequential forward passes → now: 1 batched
pass with ``max_batch_size`` chunking):

    1. ``replicate_batch`` expands every sample ``replicas`` times via
       ``repeat_interleave`` → layout ``[s0_r0, s0_r1, …, s1_r0, s1_r1, …]``
    2. If total > ``max_batch_size``, ``chunk_batch`` splits into sub-batches.
    3. Each chunk is processed in a single ``generate`` call.
    4. Results are concatenated back in order.

Estimated speedup: ``replicas×`` (e.g. replicas=10 → 10× fewer forward passes).
Actual wall-clock gain depends on GPU utilisation vs memory headroom.
"""

from loguru import logger

from proteinfoundation.search.base_search import BaseSearch
from proteinfoundation.search.search_utils import chunk_batch, replicate_batch
from proteinfoundation.search.single_pass_generation import SinglePassGeneration
from proteinfoundation.utils.tensor_utils import concat_dict_tensors


class BestOfNSearch(BaseSearch):
    """Best-of-n search: replicate the batch and generate in max-sized chunks."""

    def __init__(self, proteina_instance, inf_cfg):
        super().__init__(proteina_instance, inf_cfg)
        self.single_pass = SinglePassGeneration(proteina_instance, inf_cfg)

    def search(self, batch: dict) -> dict:
        """Expand the batch by ``replicas``, chunk to ``max_batch_size``, run.

        Returns:
            Dict with 'lookahead' (None) and 'final' (all replicas).
            Each sample's metadata_tag encodes its original sample index
            and replica index: ``bon_orig{s}_r{r}``.
        """
        replicas = self.inf_cfg.search.best_of_n.replicas
        max_batch_size = self.inf_cfg.search.get("max_batch_size", None)
        nsamples = batch["mask"].shape[0]
        total = nsamples * replicas

        logger.debug(
            f"[BestOfN] nsamples={nsamples}, replicas={replicas}, total={total}, max_batch_size={max_batch_size}"
        )

        # Build tags up-front so they stay aligned through chunking.
        # repeat_interleave layout: replicas of same sample adjacent.
        all_tags: list[str] = [f"bon_orig{s}_r{r}" for s in range(nsamples) for r in range(replicas)]

        expanded_batch = replicate_batch(batch, replicas)
        logger.info(f"[BestOfN] Expanded batch size: {expanded_batch['mask'].shape[0]}")

        if max_batch_size is not None and total > max_batch_size:
            chunks = chunk_batch(expanded_batch, max_batch_size)
            logger.debug(
                f"[BestOfN] Chunked into {len(chunks)} sub-batches (sizes {[c['mask'].shape[0] for c in chunks]})"
            )
        else:
            chunks = [expanded_batch]

        all_finals = []
        tag_offset = 0
        for chunk in chunks:
            chunk_size = chunk["mask"].shape[0]
            result = self.single_pass.search(chunk)
            chunk_final = result["final"]
            # Overwrite single-pass tags with the pre-built tags that encode
            # original sample index and replica index.  Slicing from all_tags
            # keeps tags aligned even if chunk sizes vary.
            chunk_final["metadata_tag"] = all_tags[tag_offset : tag_offset + chunk_size]
            all_finals.append(chunk_final)
            tag_offset += chunk_size

        if len(all_finals) == 1:
            final = all_finals[0]
        else:
            # concat_dict_tensors handles both tensors (torch.cat) and lists
            # (extend), so metadata_tag (list[str]) merges correctly alongside
            # tensor fields like coors/aatype.
            final = concat_dict_tensors(all_finals, dim=0)

        assert len(final["metadata_tag"]) == total, f"[BestOfN] tags ({len(final['metadata_tag'])}) != total ({total})"

        return {"lookahead": None, "final": final}
