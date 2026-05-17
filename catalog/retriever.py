"""
Loads the pre-built FAISS index at server startup and exposes a single
`search()` method used by the LangGraph agent nodes.

Singleton pattern: the CatalogRetriever is instantiated once and reused
across all requests (avoids reloading the model on every API call).

Key design decision:
  We build the retrieval query from the FULL conversation context, not just
  the latest user message. This way, a query like "Add AWS and Docker" (Turn 4
  in C9) still finds the right items because the conversation history tells us
  we're looking at a Java/Spring/SQL backend engineer role.
"""
import json
import logging
from functools import lru_cache
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from models import CatalogProduct

logger = logging.getLogger(__name__)


class CatalogRetriever:
    """
    Singleton retriever that holds the embedding model + FAISS index in memory after load().
    Call `search(query, top_k)` to get the most relevant catalog products.
    """

    def __init__(
        self,
        embedding_model_name: str,
        faiss_index_path: str,
        metadata_path: str,
    ) -> None:
        self._embedding_model_name = embedding_model_name
        self._faiss_index_path = faiss_index_path
        self._metadata_path = metadata_path

        self._model: SentenceTransformer | None = None
        self._index: faiss.IndexIDMap | None = None
        self._metadata: list[dict] = []
        self._products: list[CatalogProduct] = []

    def load(self) -> None:
        """
        Load model, FAISS index, and metadata from disk.
        Called once at application startup (from main.py lifespan).
        Raises RuntimeError if the index hasn't been built yet.
        """
        # ── Validate index files exist ─────────────────────────────────────
        if not Path(self._faiss_index_path).exists():
            raise RuntimeError(
                f"FAISS index not found at '{self._faiss_index_path}'. "
                f"Run:  python scripts/build_index.py  to build it first."
            )
        if not Path(self._metadata_path).exists():
            raise RuntimeError(
                f"Catalog metadata not found at '{self._metadata_path}'. "
                f"Run:  python scripts/build_index.py  to build it first."
            )

        # ── Load embedding model ───────────────────────────────────────────
        logger.info(f"Loading embedding model: {self._embedding_model_name}")
        self._model = SentenceTransformer(self._embedding_model_name)
        logger.info("Embedding model loaded.")

        # ── Load FAISS index ───────────────────────────────────────────────
        logger.info(f"Loading FAISS index from: {self._faiss_index_path}")
        self._index = faiss.read_index(self._faiss_index_path)
        logger.info(
            f"FAISS index loaded: {self._index.ntotal} vectors."
        )

        # ── Load metadata ──────────────────────────────────────────────────
        logger.info(f"Loading catalog metadata from: {self._metadata_path}")
        with open(self._metadata_path, "r", encoding="utf-8") as f:
            self._metadata = json.load(f)

        # Reconstruct CatalogProduct objects for easy access
        self._products = [
            CatalogProduct.from_metadata_dict(m) for m in self._metadata
        ]
        logger.info(
            f"Catalog retriever ready: {len(self._products)} products available."
        )

    def search(
        self,
        query: str,
        top_k: int = 10,
    ) -> list[CatalogProduct]:
        """
        Semantic search the catalog. Returns the top_k most similar products.

        Args:
            query: A rich text query describing the role, context, constraints.
                   Build this from the FULL conversation context, not just the
                   latest message (see build_retrieval_query below).
            top_k: How many candidates to return. We deliberately retrieve more
                   than needed (default 20) and let the LLM select the best 1–10.

        Returns:
            List of CatalogProduct objects, ordered by relevance (highest first).
        """
        if not self._model or not self._index:
            raise RuntimeError("CatalogRetriever.load() must be called before search().")

        query_vec = self._model.encode(
            [query],
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).astype(np.float32)

        # k cannot exceed the number of indexed items
        k = min(top_k, self._index.ntotal)
        distances, ids = self._index.search(query_vec, k=k)

        results: list[CatalogProduct] = []
        for dist, idx in zip(distances[0], ids[0]):
            if idx == -1:
                continue  # FAISS returns -1 for empty slots
            if 0 <= idx < len(self._products):
                product = self._products[int(idx)]
                logger.debug(
                    f"Retrieved: '{product.name}' (score={dist:.4f})"
                )
                results.append(product)

        logger.info(
            f"Search complete: {len(results)} products retrieved for query: "
            f"'{query[:100]}...'" if len(query) > 100 else f"'{query}'"
        )
        reults = self._rerank_results(query, results)
        return results
    
    def _rerank_results(self, query:str, results:list[CatalogProduct],) -> list[CatalogProduct]:
        """ 
        Lightweight semantic reranking after FAISS retrieval.

        Boosts:
            - leadership/personality alignment
            - technical relevance
            - industrial safety relevance

        Penalizes:
            - generic entry-level solutions
            - unrelated bundled solutions
        """
        q = query.lower()
        scored = []
        for product in results:
            score = 0
            text = (
                f"{product.name} "
                f"{product.description or ''} "
                f"{' '.join(product.test_type_labels)}"
            ).lower()
            
            # Leadership boosting 
            if any(t in q for t in ["leadership", "executive", "director", "cxo"]): 
                if "opq" in text: 
                    score += 10 
                if "leadership" in text: 
                    score += 8 
                if "personality" in text: 
                    score += 5 
                
                # Penalize irrelevant entry-level products 
                if "entry level" in text: 
                    score -= 10
                
            # Technical boosting 
            if any(t in q for t in ["java", "spring", "backend", "aws", "docker"]): 
                if any(t in text for t in ["java", "spring", "sql", "aws", "docker"]): 
                    score += 8 
                if "knowledge & skills" in text: 
                    score += 5
                
            # Industrial safety boosting 
            if any(t in q for t in ["safety", "plant", "industrial", "chemical"]): 
                if any(t in text for t in [ "safety", "dependability", "industrial", "manufacturing", ]): 
                    score += 10
                
            scored.append((score, product))
            
        scored.sort(key=lambda x: x[0], reverse=True)
        return [p for _, p in scored]
        


    def get_product_by_name(self, name: str) -> CatalogProduct | None:
        """
        Look up a product by exact name (case-insensitive).
        Used when the agent needs to compare two specific named products.
        """
        name_lower = name.lower().strip()
        for product in self._products:
            if product.name.lower() == name_lower:
                return product
        # Fuzzy fallback: partial match
        for product in self._products:
            if name_lower in product.name.lower():
                return product
        return None

    def get_product_by_url(self, url: str) -> CatalogProduct | None:
        """
        Look up a product by URL.
        Normalizes trailing slash before comparing.
        Used by nodes.py to replace LLM-generated names with canonical ones.
        """
        normalized = url.rstrip("/") + "/"
        for p in self._products:
            if p.url.rstrip("/") + "/" == normalized:
                return p
        return None
    
    def get_all_products(self) -> list[CatalogProduct]:
        """Return all products. Used for URL validation (checking URLs exist in catalog)."""
        return self._products

    def get_valid_urls(self) -> set[str]:
        """Return the set of all valid product URLs. Used for hallucination checks."""
        return {p.url for p in self._products}

    @property
    def is_loaded(self) -> bool:
        return self._model is not None and self._index is not None


# ─────────────────────────────────────────────────────────────────────────────
# Retrieval query builder
# ─────────────────────────────────────────────────────────────────────────────

def build_retrieval_query(messages: list[dict]) -> str:
    """
    Build a rich semantic search query from the conversation history.

    Example — C9 Turn 4: "Add AWS and Docker"
      Latest message only: "Add AWS and Docker"
        → FAISS finds: AWS tests, Docker tests ✓
        → MISSES: Java, Spring, SQL context (already established earlier)
      Full user history: "Senior Full-Stack Java Spring REST SQL AWS Docker"
                         "Backend-leaning, Java and Spring primary..."
                         "Senior IC, owns services..."
                         "Add AWS and Docker"
        → FAISS finds: Java ✓, Spring ✓, SQL ✓, AWS ✓, Docker ✓
        → Retrieval reinforces the complete existing context
        
    The latest message is repeated for higher semantic weight
    (it represents the most current constraint).

    Strategy:
    - Concatenate all user messages (not assistant messages — the assistant
      messages contain our previous recommendations, which would bias retrieval
      towards things we already found)
    - Weight the most recent user messages more by repeating them
    - Extract any explicit role/technology mentions
    """
    user_messages = [
        m["content"].strip() for m in messages if m["role"] == "user"
    ]

    if not user_messages:
        return ""

    # Repeat the latest user message (most relevant constraint)
    # and include all prior user messages for full context
    latest = user_messages[-1]
    prior = " ".join(user_messages[:-1]) if len(user_messages) > 1 else ""

    # Build query: latest message is included twice for higher weight
    query_parts = []
    if prior:
        query_parts.append(prior)
    query_parts.append(latest)
    #query_parts.append(latest)  # Repeat latest for emphasis
    latest_lower = latest.lower()

    # Leadership semantic expansion
    if any(term in latest_lower for term in ["leadership", "leader", "director", "executive", "cxo", "vp", ]):
        query_parts.append("senior  executive personality strategic thinking " "leadership assessment opq managerial decision making")
        
    # Technical hiring expansion
    if any(term in latest_lower for term in ["java", "spring", "backend", "engineer", "developer", "aws", "docker", "sql", ]): 
        query_parts.append("software engineering backend technical assessment " "knowledge skills coding reasoning")
        
    # Safety / industrial expansion
    if any(term in latest_lower for term in ["safety", "plant", "operator", "industrial", "chemical", "manufacturing",]): 
        query_parts.append( "industrial safety dependability compliance reliability " "manufacturing operations" )
        
    return " ".join(query_parts)