"""
─────────────────────────────────────────────────────────────────────────────
Assembles the LangGraph StateGraph and returns a compiled runnable.

Graph topology (linear — no conditional branching):
  START → parse_prior_state → retrieve_catalog → generate_response → END

Why linear (no conditional routing)?
  The LLM in generate_response handles ALL intent routing internally:
    - Vague query → asks clarification (recommendations: null)
    - Enough context → recommends
    - Refinement request → updates existing list
    - Compare request → gives grounded comparison
    - Off-topic → refuses

  Adding a separate classification node before generate_response would:
    a) Add latency (extra LLM call) — dangerous with the 30s timeout
    b) Risk misclassification, causing wrong routing
    c) Complicate the graph unnecessarily

  The system prompt already encodes all intent-handling logic.
  LangGraph's value here is clean state management and extensibility.
─────────────────────────────────────────────────────────────────────────────
"""
import logging

from langchain_anthropic import ChatAnthropic
from langchain_groq import ChatGroq
from langgraph.graph import END, START, StateGraph

from agent.nodes import (
    make_generate_response_node,
    make_retrieve_catalog_node,
    parse_prior_state,
)
from agent.state import AgentState
from catalog.retriever import CatalogRetriever
from config import settings

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# LLM factory
# ─────────────────────────────────────────────────────────────────────────────

def create_llm():
    """
    Create the LangChain LLM based on the configured provider.

    Groq (default):
      - Free tier, very fast (~2–4s response time)
      - Supports JSON mode → deterministic JSON output
      - Model: llama-3.3-70b-versatile (best quality on free tier)

    Anthropic (alternative):
      - Higher quality reasoning
      - Slightly slower, costs tokens
      - No native JSON mode → relies on prompt instructions + parsing
    """
    provider = settings.llm_provider.lower()

    if provider == "groq":
        if not settings.groq_api_key:
            raise RuntimeError(
                "GROQ_API_KEY is not set. "
                "Get a free key at https://console.groq.com/ "
                "and add it to your .env file."
            )
        logger.info(f"Creating Groq LLM: model={settings.llm_model}")
        return ChatGroq(
            model=settings.llm_model,
            temperature=settings.llm_temperature,
            max_tokens=settings.llm_max_tokens,
            api_key=settings.groq_api_key,
            model_kwargs={"response_format": {"type": "json_object"}},
            # json_object mode forces valid JSON output — critical for schema compliance
        )

    elif provider == "anthropic":
        if not settings.anthropic_api_key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY is not set. "
                "Add it to your .env file."
            )
        logger.info(f"Creating Anthropic LLM: model={settings.llm_model}")
        return ChatAnthropic(
            model=settings.llm_model,
            temperature=settings.llm_temperature,
            max_tokens=settings.llm_max_tokens,
            api_key=settings.anthropic_api_key,
            # Anthropic doesn't have a JSON mode in the same way —
            # we rely on system prompt + _parse_llm_json fallback parsing
        )

    else:
        raise ValueError(
            f"Unknown LLM_PROVIDER: '{provider}'. "
            f"Supported values: 'groq', 'anthropic'."
        )


# ─────────────────────────────────────────────────────────────────────────────
# Graph factory
# ─────────────────────────────────────────────────────────────────────────────

def create_agent_graph(retriever: CatalogRetriever, llm):
    """
    Build and compile the LangGraph StateGraph.

    Args:
        retriever: Pre-loaded CatalogRetriever singleton (FAISS + embedding model).
        llm: Pre-created LangChain LLM instance.

    Returns:
        Compiled LangGraph runnable — call with .ainvoke(state_dict).

    Node responsibilities:
      ┌─────────────────────┬──────────────────────────────────────────────┐
      │ parse_prior_state   │ Extract prior shortlist + count turns        │
      │ retrieve_catalog    │ FAISS semantic search → top-20 products      │
      │ generate_response   │ LLM call → reply + recommendations + eoc     │
      └─────────────────────┴──────────────────────────────────────────────┘
    """
    # Bind retriever and LLM into nodes via factory functions
    retrieve_catalog_node = make_retrieve_catalog_node(
        retriever=retriever,
        top_k=settings.retrieval_top_k,
    )
    generate_response_node = make_generate_response_node(
        retriever=retriever,
        llm=llm,
    )

    # Build the graph
    graph = StateGraph(AgentState)

    # Register nodes
    graph.add_node("parse_prior_state", parse_prior_state)
    graph.add_node("retrieve_catalog", retrieve_catalog_node)
    graph.add_node("generate_response", generate_response_node)

    # Define edges (linear flow)
    graph.add_edge(START, "parse_prior_state")
    graph.add_edge("parse_prior_state", "retrieve_catalog")
    graph.add_edge("retrieve_catalog", "generate_response")
    graph.add_edge("generate_response", END)

    compiled = graph.compile()
    logger.info("LangGraph StateGraph compiled successfully.")
    return compiled