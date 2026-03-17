#!/usr/bin/env python3
"""
Download and cache ESM model weights.

This script pre-downloads the ESM2 model to a specified cache directory.
Run this on a LOGIN NODE (with internet access) before running jobs on compute nodes.

Usage:
    # Download model (run on login node with internet)
    python script_utils/download/download_esm_model.py

    # With Hugging Face token (recommended to avoid rate limits)
    export HF_TOKEN="your_token_here"
    python script_utils/download/download_esm_model.py

    # Or pass token directly
    python script_utils/download/download_esm_model.py --token "your_token_here"

    # Verify model is cached (can run offline)
    python script_utils/download/download_esm_model.py --verify

    # Custom cache directory
    python script_utils/download/download_esm_model.py --cache-dir /path/to/cache

IMPORTANT:
- Compute nodes typically don't have internet access.
- If you get HTTP 429 errors, you need a Hugging Face token.
- Get a free token at: https://huggingface.co/settings/tokens
"""

import argparse
import logging
import os

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_CACHE_DIR = "./community_models/ckpts/ESM2"
DEFAULT_MODEL = "facebook/esm2_t33_650M_UR50D"


def main():
    parser = argparse.ArgumentParser(
        description="Download and cache ESM model weights",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Download model (requires internet - run on login node)
    python script_utils/download/download_esm_model.py

    # Verify cached model works offline
    python script_utils/download/download_esm_model.py --verify

    # Use different model
    python script_utils/download/download_esm_model.py --model facebook/esm2_t36_3B_UR50D
        """,
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=DEFAULT_CACHE_DIR,
        help=f"Cache directory for model weights (default: {DEFAULT_CACHE_DIR})",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help=f"ESM model name (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify model is cached (loads in offline mode, no internet needed)",
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="Hugging Face token (or set HF_TOKEN env var). Get one at https://huggingface.co/settings/tokens",
    )
    parser.add_argument(
        "--dryrun",
        action="store_true",
        help="Show what would be done without actually downloading",
    )
    args = parser.parse_args()

    # Get token from args or environment
    hf_token = args.token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")

    logger.info(f"Cache directory: {args.cache_dir}")
    logger.info(f"Model: {args.model}")

    if args.dryrun:
        logger.info("[DRYRUN] Would download model to cache directory")
        logger.info("[DRYRUN] No actual download performed")
        return

    os.makedirs(args.cache_dir, exist_ok=True)
    # NOTE: No HF_HOME/HF_HUB_CACHE/TRANSFORMERS_CACHE env vars here — the explicit
    # cache_dir= passed to every from_pretrained() call takes precedence, making
    # env-var-based cache redirection dead code.

    try:
        from transformers import AutoModelForMaskedLM, AutoTokenizer
    except ImportError:
        logger.error("transformers package not installed. Run: pip install transformers")
        raise SystemExit(1)

    if args.verify:
        logger.info("Verifying model is cached (offline mode)...")
        try:
            tokenizer = AutoTokenizer.from_pretrained(args.model, cache_dir=args.cache_dir, local_files_only=True)
            model = AutoModelForMaskedLM.from_pretrained(args.model, cache_dir=args.cache_dir, local_files_only=True)
            logger.info("SUCCESS: Model is cached and can be loaded offline!")
            logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        except Exception as e:
            logger.error("FAILED: Model not cached. Run without --verify to download first.")
            logger.error(f"Error: {e}")
            raise SystemExit(1)
        return

    # Download mode - requires internet
    logger.info("Downloading model (requires internet connection)...")
    if hf_token:
        logger.info("Using Hugging Face token for authentication")
    else:
        logger.warning("No HF_TOKEN set. If you get HTTP 429 errors, set a token:")
        logger.warning("  export HF_TOKEN='your_token_here'")
        logger.warning("  Get a free token at: https://huggingface.co/settings/tokens")

    try:
        logger.info("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(args.model, cache_dir=args.cache_dir, token=hf_token)
        logger.info(f"Tokenizer loaded: {type(tokenizer).__name__}")

        logger.info("Loading model (this may take a while on first download)...")
        model = AutoModelForMaskedLM.from_pretrained(args.model, cache_dir=args.cache_dir, token=hf_token)
        logger.info(f"Model loaded: {type(model).__name__}")
        logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

        logger.info(f"SUCCESS: ESM model cached to: {args.cache_dir}")
        logger.info("You can now use this model on compute nodes (offline).")

    except OSError as e:
        error_str = str(e).lower()
        if "couldn't connect" in error_str:
            logger.error("Cannot connect to Hugging Face!")
            logger.error("Make sure you're running this on a node with internet access.")
            raise SystemExit(1)
        elif "429" in str(e) or "rate" in error_str:
            logger.error("Rate limited (HTTP 429)! You need a Hugging Face token.")
            logger.error("1. Get a free token at: https://huggingface.co/settings/tokens")
            logger.error("2. Run: export HF_TOKEN='your_token_here'")
            logger.error("3. Then retry this script")
            raise SystemExit(1)
        else:
            raise


if __name__ == "__main__":
    main()
