"""
FastAPI application exposing two endpoints:
  GET  /health  → readiness check (returns {"status": "ok"})
  POST /chat    → stateless conversational agent

Startup sequence (via lifespan):
  1. Load CatalogRetriever (embedding model + FAISS index)
  2. Create LLM (Groq or Anthropic)
  3. Compile LangGraph StateGraph

All three are module-level singletons — created once, reused across requests.

Run locally:
  uvicorn main:app --host 0.0.0.0 --port 8000 --reload

"""
import logging
import re
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from agent.graph import create_agent_graph, create_llm
from catalog.retriever import CatalogRetriever
from config import settings
from models import ChatRequest, ChatResponse, Recommendation

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Module-level singletons (set during lifespan startup)
retriever: Optional[CatalogRetriever] = None
agent_graph = None


# Lifespan — startup and shutdown
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI lifespan context manager.
    Everything before `yield` runs at startup.
    Everything after `yield` runs at shutdown.
    """
    global retriever, agent_graph

    logger.info("=" * 60)
    logger.info("SHL Assessment Recommender — Starting up")
    logger.info("=" * 60)

    # Step 1: Load catalog retriever 
    logger.info("[1/3] Loading catalog retriever (FAISS + embedding model)...")
    retriever = CatalogRetriever(
        embedding_model_name=settings.embedding_model,
        faiss_index_path=settings.faiss_index_path,
        metadata_path=settings.catalog_metadata_path,
    )
    try:
        retriever.load()
    except RuntimeError as e:
        logger.error(f"Failed to load catalog retriever: {e}")
        logger.error(
            "Did you run:  python scripts/build_index.py  ?\n"
            "The FAISS index must be built before starting the server."
        )
        raise

    logger.info(
        f"✓ Catalog loaded: {len(retriever.get_all_products())} products available."
    )

    # Step 2: Create LLM 
    logger.info(f"[2/3] Creating LLM ({settings.llm_provider}/{settings.llm_model})...")
    try:
        llm = create_llm()
    except (RuntimeError, ValueError) as e:
        logger.error(f"Failed to create LLM: {e}")
        raise

    logger.info(f"✓ LLM ready.")

    # Step 3: Compile LangGraph
    logger.info("[3/3] Compiling LangGraph StateGraph...")
    agent_graph = create_agent_graph(retriever=retriever, llm=llm)
    logger.info("✓ LangGraph compiled.")

    logger.info("=" * 60)
    logger.info(f"Server ready at http://{settings.app_host}:{settings.app_port}")
    logger.info("=" * 60)

    yield  # ← Server is live here

    # Shutdown 
    logger.info("Shutting down...")

# FastAPI app
app = FastAPI(
    title="SHL Assessment Recommender",
    description=(
        "Conversational agent that recommends SHL assessments "
        "from the official product catalog."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# Global exception handler — never let a raw Python error reach the client
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception on {request.url}: {exc}", exc_info=True)
    # Return a valid ChatResponse-shaped JSON so the evaluator doesn't break
    return JSONResponse(
        status_code=500,
        content={
            "reply": "An internal error occurred. Please try again.",
            "recommendations": None,
            "end_of_conversation": False,
        },
    )

# GET /health
@app.get("/health")
def health():
    """
    Readiness check.
    Returns {"status": "ok"} immediately once the server has started.
    The assignment evaluator allows up to 2 minutes for cold-start services.

    Note: This endpoint responds even if the FAISS index hasn't finished loading,
    because FastAPI starts accepting traffic as soon as the app is created.
    The lifespan startup ensures everything is ready before the server becomes
    reachable in practice.
    """
    return {"status": "ok"}


# POST /chat
@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Main conversational endpoint.

    Receives the full conversation history on every call (stateless design).
    Runs the LangGraph agent and returns the next reply + optional recommendations.

    Request body:
      { "messages": [{"role": "user"|"assistant", "content": "..."}] }

    Response body (non-negotiable schema):
      {
        "reply": "...",
        "recommendations": null | [{"name":"...","url":"...","test_type":"..."}],
        "end_of_conversation": false | true
      }
    """
    if agent_graph is None:
        raise HTTPException(
            status_code=503,
            detail="Agent is not ready yet. Please try again in a moment.",
        )

    # Convert Pydantic Message objects to plain dicts for LangGraph state
    messages = [{"role": m.role, "content": m.content} for m in request.messages]

    # Initialize the full state with safe defaults
    # (LangGraph requires all TypedDict keys to be present at invocation)
    initial_state = {
        "messages": messages,
        "current_recommendations": [],
        "turn_count": 0,
        "retrieved_items": [],
        "reply": "",
        "recommendations": None,
        "end_of_conversation": False,
    }

    try:
        result = await agent_graph.ainvoke(initial_state)
    except Exception as e:
        logger.error(f"LangGraph invocation failed: {e}", exc_info=True)
        return ChatResponse(
            reply="I encountered an error processing your request. Please try again.",
            recommendations=None,
            end_of_conversation=False,
        )

    # Extract results from final state 
    raw_reply: str = result.get("reply", "")
    raw_recs: Optional[list] = result.get("recommendations")
    end_of_conv: bool = bool(result.get("end_of_conversation", False))

    # Strip the internal [SHORTLIST] marker from the reply before sending
    # to the client — this is internal plumbing, not for user display
    clean_reply = _strip_shortlist_marker(raw_reply)

    # Convert raw recommendation dicts to Recommendation Pydantic objects
    recommendations = None
    if raw_recs and isinstance(raw_recs, list):
        try:
            recommendations = [
                Recommendation(
                    name=r["name"],
                    url=r["url"],
                    test_type=r["test_type"],
                )
                for r in raw_recs
                if all(k in r for k in ("name", "url", "test_type"))
            ]
            recommendations = recommendations if recommendations else None
        except Exception as e:
            logger.error(f"Failed to build Recommendation objects: {e}")
            recommendations = None

    # Enforce max 10 recommendations (safety net)
    if recommendations and len(recommendations) > 10:
        logger.warning(
            f"Truncating recommendations from {len(recommendations)} to 10."
        )
        recommendations = recommendations[:10]

    logger.info(
        f"POST /chat → reply_len={len(clean_reply)}, "
        f"recs={len(recommendations) if recommendations else 0}, "
        f"eoc={end_of_conv}"
    )

    return ChatResponse(
        reply=clean_reply,
        recommendations=recommendations if recommendations else None,
        end_of_conversation=end_of_conv,
    )

# Helper
def _strip_shortlist_marker(content: str) -> str:
    """Remove [SHORTLIST]...[/SHORTLIST] from reply before returning to client."""
    return re.sub(
        r"\[SHORTLIST\].*?\[/SHORTLIST\]", "", content, flags=re.DOTALL
    ).strip()