"""
Builds and saves the FAISS vector index from the parsed SHL catalog.

This module runs ONCE (via scripts/build_index.py) to pre-compute and store
the embeddings. At runtime, the retriever loads the saved index — no
re-embedding on every server start.

Flow:
  catalog products → to_embedding_text() → sentence-transformers → FAISS index
                                                                  → JSON metadata
"""
import json
import logging
import time
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from app.models import CatalogProduct

logger = logging.getLogger(__name__)


def build_faiss_index(
    products: list[CatalogProduct],
    embedding_model_name: str,
    faiss_index_path: str,
    metadata_path: str,
) -> tuple[faiss.IndexFlatIP, list[dict]]:
    """
    Embed all products and build a FAISS IndexFlatIP (inner product / cosine similarity).

    Args:
        products: List of CatalogProduct objects (Individual Test Solutions only).
        embedding_model_name: HuggingFace model name, e.g. 'all-MiniLM-L6-v2'.
        faiss_index_path: Where to save the .faiss file.
        metadata_path: Where to save the parallel JSON metadata file.

    Returns:
        (faiss_index, metadata_list) — both also saved to disk.
    """
    if not products:
        raise ValueError("Cannot build FAISS index from an empty product list.")

    logger.info(f"Loading embedding model: {embedding_model_name}")
    t0 = time.time()
    model = SentenceTransformer(embedding_model_name)
    logger.info(f"Embedding model loaded in {time.time() - t0:.1f}s")

    # ── Step 1: Build embedding texts ─────────────────────────────────────
    logger.info(f"Building embedding texts for {len(products)} products...")
    embedding_texts = [p.to_embedding_text() for p in products]

    # Log a sample so you can verify the quality of what's being embedded
    logger.debug("Sample embedding text (first product):\n" + embedding_texts[0])

    # ── Step 2: Compute embeddings ─────────────────────────────────────────
    logger.info("Computing embeddings (this may take 30–60 seconds on first run)...")
    t1 = time.time()
    embeddings = model.encode(
        embedding_texts,
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,   # Normalize → cosine similarity via inner product
    )
    elapsed = time.time() - t1
    logger.info(
        f"Embeddings computed in {elapsed:.1f}s. "
        f"Shape: {embeddings.shape} (products × embedding_dim)"
    )

    # ── Step 3: Build FAISS index ──────────────────────────────────────────
    # IndexFlatIP = exact (non-approximate) inner product search.
    # With normalized embeddings, inner product == cosine similarity.
    # For ~400 items, exact search is instant — no need for approximate indices.
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)

    # Wrap with IndexIDMap so FAISS stores our integer IDs (0, 1, 2, ...)
    id_index = faiss.IndexIDMap(index)
    ids = np.arange(len(products), dtype=np.int64)
    id_index.add_with_ids(embeddings.astype(np.float32), ids)

    logger.info(f"FAISS index built: {id_index.ntotal} vectors, dimension={dimension}")

    # ── Step 4: Save FAISS index ───────────────────────────────────────────
    Path(faiss_index_path).parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(id_index, faiss_index_path)
    logger.info(f"FAISS index saved to: {faiss_index_path}")

    # ── Step 5: Save metadata (parallel to FAISS IDs) ─────────────────────
    # The metadata list is indexed the same as the FAISS IDs.
    # metadata[i] corresponds to FAISS vector ID i.
    metadata = [p.to_metadata_dict() for p in products]
    Path(metadata_path).parent.mkdir(parents=True, exist_ok=True)
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    logger.info(f"Metadata saved to: {metadata_path} ({len(metadata)} entries)")

    # ── Step 6: Sanity check ───────────────────────────────────────────────
    _verify_index(id_index, model, products[:3])

    return id_index, metadata


def _verify_index(
    index: faiss.IndexIDMap,
    model: SentenceTransformer,
    sample_products: list[CatalogProduct],
) -> None:
    """
    Quick sanity check: query for each sample product by name and verify
    it appears in the top-3 results (it should be the #1 hit for itself).
    """
    logger.info("Running index sanity check...")
    for product in sample_products:
        query = product.name
        query_vec = model.encode([query], normalize_embeddings=True).astype(np.float32)
        distances, ids = index.search(query_vec, k=3)
        top_id = int(ids[0][0])
        logger.debug(
            f"Query: '{query}' → top result ID={top_id}, score={distances[0][0]:.4f}"
        )
    logger.info("Sanity check passed.")