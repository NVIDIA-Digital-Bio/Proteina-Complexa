"""Single-pass generation algorithm for protein generation.

This module provides a simple single-pass generation algorithm (no search).
"""

from proteinfoundation.search.base_search import BaseSearch
from proteinfoundation.search.search_utils import make_initial_search_tags
from proteinfoundation.utils.sample_utils import prepend_target_to_samples, sample_formatting


class SinglePassGeneration(BaseSearch):
    """Single-pass generation algorithm - one generation pass, no search."""

    def search(self, batch: dict) -> dict:
        """
        Single-pass generation step - one forward pass, no search.

        Returns:
            Dict with 'lookahead' (None) and 'final' (sample_prots). Rewards
            for final samples are computed in proteina.predict_step.
        """
        gen_samples = self.proteina.generate(batch)

        # Format the generated samples back to proteins
        sample_prots = sample_formatting(
            x=gen_samples,
            extra_info={"mask": batch["mask"]},
            ret_mode="coors37_n_aatype",
            data_modes=list(self.proteina.cfg_exp.product_flowmatcher),
            autoencoder=getattr(self.proteina, "autoencoder", None),
        )

        # 1:1 with batch — no expansion, default repeat_mode is fine.
        if batch.get("prepend_target", False) and not hasattr(self.proteina, "ligand"):
            sample_prots = prepend_target_to_samples(sample_prots, batch)

        nsamples = sample_prots["coors"].shape[0]
        sample_prots["metadata_tag"] = make_initial_search_tags("single", nsamples)

        return {"lookahead": None, "final": sample_prots}
