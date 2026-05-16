#!/usr/bin/env python3
"""
One-time script to:
  1. Fetch the SHL catalog JSON
  2. Parse and filter to Individual Test Solutions
  3. Embed every product using sentence-transformers
  4. Build and save the FAISS index + metadata JSON

Run this BEFORE starting the FastAPI server:
  python scripts/build_index.py

Re-run whenever:
  - You want to refresh the catalog data
  - You change the embedding model (EMBEDDING_MODEL in .env)

Output files (configured in .env):
  data/catalog.faiss        ← FAISS vector index
  data/catalog_metadata.json ← Product metadata parallel to FAISS IDs
  data/raw_catalog.json     ← Raw fetched JSON (for debugging)
"""
import argparse
import logging
import sys
import time
from pathlib import Path

# ── Make sure the project root is on the Python path ──────────────────────
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.catalog.fetcher import fetch_catalog, save_raw_catalog, _load_raw_json
from app.catalog.indexer import build_faiss_index
from app.config import settings

# ─────────────────────────────────────────────────────────────────────────────
# Logging setup
# ─────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def main(local_json: str | None = None) -> None:
    start = time.time()

    logger.info("=" * 60)
    logger.info("SHL Catalog Index Builder")
    logger.info("=" * 60)

    # ── Step 1: Fetch catalog ────────────────────────────────────────────
    logger.info("\n[Step 1/3] Fetching SHL catalog...")
    try:
        # If local_json is provided, skip HTTP and use local file
        products = fetch_catalog(
            url=settings.catalog_json_url,
            local_json_path=local_json,
        )
    except RuntimeError as e:
        logger.error(f"Failed to fetch catalog: {e}")
        logger.error(
            "Troubleshooting:\n"
            "  1. Check your internet connection.\n"
            "  2. Verify the URL in .env: CATALOG_JSON_URL\n"
            "  3. Try passing --local-json path/to/catalog.json if you have a local copy.\n"
            "  4. Check if the SHL catalog URL has changed.\n"
        )
        sys.exit(1)

    if not products:
        logger.error("Catalog returned 0 products. Cannot build an empty index.")
        logger.error(
            "This likely means:\n"
            "  - The JSON structure doesn't match the parser's expectations.\n"
            "  - All products were filtered out as 'Job Solutions'.\n"
            "  - Try saving the raw JSON (--save-raw) and inspecting it manually.\n"
        )
        sys.exit(1)

    logger.info(f"✓ Fetched {len(products)} Individual Test Solutions.")

    # ── Save raw catalog for debugging ──────────────────────────────────
    try:
        raw = _load_raw_json(settings.catalog_json_url, local_json, timeout=30)
        save_raw_catalog(raw, "data/raw_catalog.json")
    except Exception as e:
        logger.warning(f"Could not save raw catalog: {e}")

    # ── Print catalog summary ────────────────────────────────────────────
    _print_catalog_summary(products)

    # ── Step 2: Build FAISS index ────────────────────────────────────────
    logger.info("\n[Step 2/3] Building FAISS index...")
    try:
        index, metadata = build_faiss_index(
            products=products,
            embedding_model_name=settings.embedding_model,
            faiss_index_path=settings.faiss_index_path,
            metadata_path=settings.catalog_metadata_path,
        )
    except Exception as e:
        logger.error(f"Failed to build FAISS index: {e}")
        raise

    # ── Step 3: Verify output files exist ───────────────────────────────
    logger.info("\n[Step 3/3] Verifying output files...")
    faiss_path = Path(settings.faiss_index_path)
    meta_path = Path(settings.catalog_metadata_path)

    assert faiss_path.exists(), f"FAISS index not found at {faiss_path}"
    assert meta_path.exists(), f"Metadata not found at {meta_path}"

    faiss_size_kb = faiss_path.stat().st_size / 1024
    meta_size_kb = meta_path.stat().st_size / 1024

    logger.info(f"✓ {faiss_path} ({faiss_size_kb:.0f} KB)")
    logger.info(f"✓ {meta_path} ({meta_size_kb:.0f} KB)")

    elapsed = time.time() - start
    logger.info("\n" + "=" * 60)
    logger.info(f"Index build complete in {elapsed:.1f}s")
    logger.info(f"  Products indexed: {len(products)}")
    logger.info(f"  Embedding model:  {settings.embedding_model}")
    logger.info(f"  FAISS index:      {settings.faiss_index_path}")
    logger.info(f"  Metadata:         {settings.catalog_metadata_path}")
    logger.info("=" * 60)
    logger.info("\nYou can now start the server:")
    logger.info("  uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload")


def _print_catalog_summary(products) -> None:
    """Print a breakdown of products by type code for verification."""
    from collections import Counter
    type_counts: Counter = Counter()
    for p in products:
        for code in p.test_type_codes:
            type_counts[code] += 1

    logger.info("\nCatalog breakdown by test type:")
    type_labels = {
        "P": "Personality & Behavior",
        "K": "Knowledge & Skills",
        "A": "Ability & Aptitude",
        "S": "Simulations",
        "B": "Biodata & Situational Judgment",
        "C": "Competencies",
        "D": "Development & 360",
        "E": "Assessment Exercises",
    }
    for code, label in type_labels.items():
        count = type_counts.get(code, 0)
        if count > 0:
            logger.info(f"  [{code}] {label}: {count} products")

    # Show sample products for spot-checking
    logger.info("\nSample products (first 5):")
    for i, p in enumerate(products[:5]):
        logger.info(
            f"  {i+1}. {p.name}\n"
            f"     Type: {','.join(p.test_type_codes)} | "
            f"Duration: {p.duration or 'N/A'} | "
            f"URL: {p.url}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build the FAISS index from the SHL catalog."
    )
    parser.add_argument(
        "--local-json",
        type=str,
        default=None,
        help=(
            "Path to a local catalog JSON file. "
            "Use this if you've already downloaded the catalog or have no network access."
        ),
    )
    args = parser.parse_args()
    main(local_json=args.local_json)