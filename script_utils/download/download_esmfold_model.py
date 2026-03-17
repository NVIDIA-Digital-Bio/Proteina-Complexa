#!/usr/bin/env python3
"""
Download and cache ESMFold model weights.

Pre-downloads facebook/esmfold_v1 so monomer evaluation doesn't need internet.
Run on a node with internet access before submitting cluster jobs.

Usage:
    python script_utils/download/download_esmfold_model.py
    python script_utils/download/download_esmfold_model.py --cache-dir /path/to/cache
    python script_utils/download/download_esmfold_model.py --verify
"""

import argparse
import logging
import os

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_CACHE_DIR = "./community_models/ckpts/ESMFold"
MODEL_NAME = "facebook/esmfold_v1"


def main():
    parser = argparse.ArgumentParser(description="Download and cache ESMFold model weights")
    parser.add_argument("--cache-dir", type=str, default=DEFAULT_CACHE_DIR,
                        help=f"Cache directory for model weights (default: {DEFAULT_CACHE_DIR})")
    parser.add_argument("--verify", action="store_true",
                        help="Verify model is cached (offline check, no download)")
    parser.add_argument("--token", type=str, default=None,
                        help="Hugging Face token (or set HF_TOKEN env var)")
    parser.add_argument("--dryrun", action="store_true",
                        help="Show what would be done without downloading")
    args = parser.parse_args()

    hf_token = args.token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")

    logger.info(f"Cache directory: {args.cache_dir}")
    logger.info(f"Model: {MODEL_NAME}")

    if args.dryrun:
        logger.info("[DRYRUN] Would download ESMFold to cache directory")
        return

    os.makedirs(args.cache_dir, exist_ok=True)
    # NOTE: No HF_HOME/HF_HUB_CACHE/TRANSFORMERS_CACHE env vars here — the explicit
    # cache_dir= passed to every from_pretrained() call takes precedence, making
    # env-var-based cache redirection dead code.

    try:
        from transformers import AutoTokenizer, EsmForProteinFolding
    except ImportError:
        logger.error("transformers package not installed. Run: pip install transformers")
        raise SystemExit(1)

    if args.verify:
        logger.info("Verifying ESMFold is cached (offline mode)...")
        try:
            AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=args.cache_dir, local_files_only=True)
            model = EsmForProteinFolding.from_pretrained(MODEL_NAME, cache_dir=args.cache_dir, local_files_only=True)
            logger.info("SUCCESS: ESMFold is cached and can be loaded offline!")
            logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        except Exception as e:
            logger.error("FAILED: ESMFold not cached. Run without --verify to download first.")
            logger.error(f"Error: {e}")
            raise SystemExit(1)
        return

    logger.info("Downloading ESMFold (requires internet, ~8.5GB)...")
    if hf_token:
        logger.info("Using Hugging Face token for authentication")

    try:
        logger.info("Loading tokenizer...")
        AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=args.cache_dir, token=hf_token)

        logger.info("Loading model (this may take a while on first download)...")
        model = EsmForProteinFolding.from_pretrained(MODEL_NAME, cache_dir=args.cache_dir, token=hf_token)
        logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        logger.info(f"SUCCESS: ESMFold cached to: {args.cache_dir}")

    except OSError as e:
        error_str = str(e).lower()
        if "couldn't connect" in error_str:
            logger.error("Cannot connect to Hugging Face! Run on a node with internet access.")
            raise SystemExit(1)
        elif "429" in str(e) or "rate" in error_str:
            logger.error("Rate limited (HTTP 429)! Set HF_TOKEN and retry.")
            raise SystemExit(1)
        else:
            raise


if __name__ == "__main__":
    main()
