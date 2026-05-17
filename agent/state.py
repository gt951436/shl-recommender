"""
agent/state.py
─────────────────────────────────────────────────────────────────────────────
Defines AgentState — the single shared dict that flows through every node
in the LangGraph StateGraph.

Why LangGraph StateGraph for this problem:
  The API is stateless (full history arrives on every request), but within
  ONE request we need state to flow across 3 nodes:
    parse_prior_state → retrieve_catalog → generate_response

  Each node reads from state and writes back ONLY its own fields.
  LangGraph manages this automatically — no manual passing of variables.

  The critical field is `current_recommendations`: it holds whatever was
  previously recommended (extracted from conversation history), so the
  generate_response node can UPDATE it rather than fetch 10 new items
  from scratch on every refinement turn.

State lifecycle per request:
  1. Initialized in main.py with messages + zeroed defaults
  2. parse_prior_state   → writes: current_recommendations, turn_count
  3. retrieve_catalog    → writes: retrieved_items
  4. generate_response   → writes: reply, recommendations, end_of_conversation
  5. main.py reads final state → builds ChatResponse
─────────────────────────────────────────────────────────────────────────────
"""
from typing import Optional, TypedDict


class AgentState(TypedDict):
    # ── Input ─────────────────────────────────────────────────────────────
    # Set once by main.py from the incoming ChatRequest.
    # Contains the FULL conversation history (every turn, oldest first).
    # Format: [{"role": "user"|"assistant", "content": "..."}]
    # Never modified by any node.
    messages: list[dict]

    # ── Set by Node 1: parse_prior_state ──────────────────────────────────
    # Recommendations from the PREVIOUS assistant turn (if any).
    # Format: [{"name": "...", "url": "...", "test_type": "..."}]
    # On the very first turn, this is an empty list.
    # On refinement turns (C9-style "add AWS, drop REST"), the LLM reads
    # this list and modifies it — not starting from scratch.
    current_recommendations: list[dict]

    # Total number of turns (user + assistant) seen so far.
    # Used to enforce the 8-turn cap and prevent infinite clarification loops.
    turn_count: int

    # ── Set by Node 2: retrieve_catalog ───────────────────────────────────
    # Top-K CatalogProduct metadata dicts from FAISS semantic search.
    # These become the "RETRIEVED CATALOG ITEMS" section of the LLM prompt.
    # Only products in this list can be recommended (enforced in validation).
    retrieved_items: list[dict]

    # ── Set by Node 3: generate_response ──────────────────────────────────
    # The agent's conversational reply text (goes into ChatResponse.reply).
    reply: str

    # The committed shortlist (goes into ChatResponse.recommendations).
    # None when the agent is still clarifying or refusing.
    # 1–10 items when a shortlist is committed.
    recommendations: Optional[list[dict]]

    # True only when the agent considers the task complete.
    # (User confirmed / locked in the shortlist.)
    end_of_conversation: bool