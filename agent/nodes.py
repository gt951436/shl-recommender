"""
agent/nodes.py
─────────────────────────────────────────────────────────────────────────────
The three async node functions for the LangGraph StateGraph:

  Node 1: parse_prior_state
    - Reads conversation history
    - Extracts previously recommended items (enables REFINE behavior)
    - Counts turns (for 8-turn cap awareness)

  Node 2: retrieve_catalog
    - Builds a rich semantic query from the full conversation
    - Searches FAISS index for top-K relevant products

  Node 3: generate_response
    - Injects retrieved catalog into system prompt
    - Calls LLM with full conversation history
    - Parses, validates, and de-hallucinates the JSON output

These are not plain functions — they are closures created by the factory in
graph.py, which injects the retriever and LLM as dependencies.
─────────────────────────────────────────────────────────────────────────────
"""
import json
import logging
import re
from typing import Optional

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from agent.prompts import SYSTEM_PROMPT_TEMPLATE, format_catalog_context
from agent.state import AgentState
from catalog.retriever import CatalogRetriever, build_retrieval_query

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Node 1: parse_prior_state
# ─────────────────────────────────────────────────────────────────────────────

async def parse_prior_state(state: AgentState) -> dict:
    """
    Reads the conversation history and extracts:
      - current_recommendations: what was recommended in the last assistant turn
      - turn_count: total turns so far

    Why extract current_recommendations here?
    When the user sends a REFINE request ("Add AWS, drop REST"), the
    generate_response node needs to know the EXISTING shortlist so it can
    update it rather than fetch 10 new items from scratch.

    The recommendations are embedded in the assistant's previous reply as
    a [SHORTLIST] marker (see generate_response). On the very first turn,
    or on turns where no recommendations were given, this is an empty list.
    """
    messages: list[dict] = state["messages"]

    # Count turns (user + assistant)
    user_turns = sum(1 for m in messages if m["role"] == "user")
    assistant_turns = sum(1 for m in messages if m["role"] == "assistant")
    turn_count = user_turns + assistant_turns

    # Extract current recommendations from the last assistant message that
    # contains a [SHORTLIST] marker. We embed this marker in every reply
    # that includes a shortlist (see generate_response).
    current_recommendations: list[dict] = []

    # Walk backwards through messages to find the most recent shortlist
    for msg in reversed(messages):
        if msg["role"] == "assistant":
            extracted = _extract_shortlist_from_message(msg["content"])
            if extracted:
                current_recommendations = extracted
                logger.debug(
                    f"Extracted {len(current_recommendations)} prior recommendations "
                    f"from conversation history."
                )
                break

    logger.info(
        f"parse_prior_state: turn_count={turn_count}, "
        f"prior_recs={len(current_recommendations)}"
    )

    return {
        "current_recommendations": current_recommendations,
        "turn_count": turn_count,
    }


def _extract_shortlist_from_message(content: str) -> list[dict]:
    """
    Extract the embedded [SHORTLIST] JSON from an assistant message.

    Format embedded by generate_response:
      [SHORTLIST]{"recommendations":[{"name":"...","url":"...","test_type":"..."}]}[/SHORTLIST]

    Returns [] if no shortlist marker found.
    """
    pattern = r"\[SHORTLIST\](.*?)\[/SHORTLIST\]"
    match = re.search(pattern, content, re.DOTALL)
    if not match:
        return []
    try:
        data = json.loads(match.group(1))
        return data.get("recommendations", [])
    except (json.JSONDecodeError, AttributeError):
        logger.warning("Failed to parse embedded [SHORTLIST] marker.")
        return []


# ─────────────────────────────────────────────────────────────────────────────
# Node 2: retrieve_catalog
# ─────────────────────────────────────────────────────────────────────────────

def make_retrieve_catalog_node(retriever: CatalogRetriever, top_k: int = 10):
    """
    Factory that creates the retrieve_catalog node with the retriever injected.

    Why use a factory?
    The retriever is a heavy singleton (holds the embedding model + FAISS index
    in memory). We create it once at startup and inject it via closure so every
    request reuses the same loaded instance.
    """

    async def retrieve_catalog(state: AgentState) -> dict:
        """
        Builds a semantic search query from the FULL conversation history
        and retrieves the top-K most relevant catalog products.

        Why use the full history (not just the latest message)?
        Turn 4 of C9: "Add AWS and Docker" — if we only embed this message,
        we miss the context that this is a Java/Spring/SQL Senior IC role.
        By using all user messages, FAISS finds AWS + Docker tests AND
        reinforces the existing Java/Spring/SQL items.
        """
        messages: list[dict] = state["messages"]

        query = build_retrieval_query(messages)
        logger.info(f"Retrieval query (first 120 chars): '{query[:120]}'")

        products = retriever.search(query, top_k=top_k)

        # Convert CatalogProduct objects to plain dicts for state storage
        # (TypedDict state cannot hold arbitrary objects — must be serializable)
        retrieved_items = [p.to_metadata_dict() for p in products]

        logger.info(
            f"retrieve_catalog: found {len(retrieved_items)} items. "
            f"Top 3: {[item['name'] for item in retrieved_items[:3]]}"
        )

        return {"retrieved_items": retrieved_items}

    return retrieve_catalog


# ─────────────────────────────────────────────────────────────────────────────
# Node 3: generate_response
# ─────────────────────────────────────────────────────────────────────────────

def make_generate_response_node(retriever: CatalogRetriever, llm):
    """
    Factory that creates the generate_response node with the LLM and retriever
    injected. The retriever is used here for URL validation (anti-hallucination).
    """

    async def generate_response(state: AgentState) -> dict:
        """
        The core node. Assembles the full LLM prompt and generates the response.

        Steps:
          1. Format retrieved catalog items into the system prompt
          2. Build LangChain message list (system + full conversation history)
          3. Call LLM (async)
          4. Parse JSON from response
          5. Validate all recommendation URLs against the real catalog
          6. Embed [SHORTLIST] marker into reply for next-turn extraction
          7. Return state updates
        """
        messages: list[dict] = state["messages"]
        retrieved_items: list[dict] = state["retrieved_items"]
        current_recommendations: list[dict] = state["current_recommendations"]
        turn_count: int = state.get("turn_count", 0)

        # ── Step 1: Build system prompt with catalog context ───────────────
        catalog_context = format_catalog_context(retrieved_items)

        # Inject current shortlist context if we're on a refinement turn
        if current_recommendations:
            prior_json = json.dumps(
                {"current_shortlist": current_recommendations}, indent=2
            )
            catalog_context = (
                f"EXISTING SHORTLIST FROM PREVIOUS TURN "
                f"(UPDATE this — do not start from scratch):\n"
                f"{prior_json}\n\n"
                + catalog_context
            )

        system_content = SYSTEM_PROMPT_TEMPLATE.format(
            catalog_context=catalog_context
        )

        # ── Step 2: Build LangChain message list ───────────────────────────
        # System prompt + full conversation history
        # We strip [SHORTLIST] markers from assistant messages before sending
        # to the LLM (they're internal plumbing, not for the LLM to see)
        lc_messages = [SystemMessage(content=system_content)]
        for msg in messages:
            clean_content = _strip_shortlist_marker(msg["content"])
            if msg["role"] == "user":
                lc_messages.append(HumanMessage(content=clean_content))
            else:
                lc_messages.append(AIMessage(content=clean_content))

        # ── Step 3: Call LLM ───────────────────────────────────────────────
        logger.info(
            f"generate_response: calling LLM. "
            f"turn_count={turn_count}, "
            f"lc_messages={len(lc_messages)}, "
            f"retrieved={len(retrieved_items)}"
        )
        try:
            response = await llm.ainvoke(lc_messages)
            raw_content = response.content
            logger.debug(f"LLM raw response (first 300 chars): {raw_content[:300]}")
        except Exception as e:
            logger.error(f"LLM call failed: {e}", exc_info=True)
            return _safe_fallback_state()

        # ── Step 4: Parse JSON ─────────────────────────────────────────────
        try:
            parsed = _parse_llm_json(raw_content)
        except Exception as e:
            logger.error(f"JSON parse failed: {e}. Raw: {raw_content[:500]}")
            return _safe_fallback_state(
                reply=(
                    "I had trouble formatting my response. "
                    "Could you please repeat your last message?"
                )
            )

        # ── Step 5: Extract and validate fields ────────────────────────────
        reply: str = str(parsed.get("reply", "")).strip()
        if not reply:
            reply = "I'm not sure how to respond to that. Could you clarify?"

        raw_recs = parsed.get("recommendations")
        end_of_conversation: bool = bool(parsed.get("end_of_conversation", False))

        # Validate recommendations against catalog (anti-hallucination)
        valid_recommendations = _validate_recommendations(raw_recs, retriever)

        # ── Step 6: Embed [SHORTLIST] marker in reply ──────────────────────
        # This marker is read by parse_prior_state on the NEXT request
        # to extract the current shortlist for REFINE turns.
        # It is stripped from assistant messages before they go to the LLM
        # (see _strip_shortlist_marker above) — it's internal plumbing only.
        if valid_recommendations:
            shortlist_json = json.dumps(
                {"recommendations": valid_recommendations},
                separators=(",", ":"),
            )
            reply_with_marker = (
                f"{reply}\n[SHORTLIST]{shortlist_json}[/SHORTLIST]"
            )
        else:
            reply_with_marker = reply

        logger.info(
            f"generate_response done: "
            f"reply_len={len(reply)}, "
            f"recs={len(valid_recommendations) if valid_recommendations else 0}, "
            f"end_of_conversation={end_of_conversation}"
        )

        return {
            "reply": reply_with_marker,
            "recommendations": valid_recommendations,
            "end_of_conversation": end_of_conversation,
        }

    return generate_response


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _parse_llm_json(content: str) -> dict:
    """
    Robustly parse JSON from LLM output.
    Handles:
      - Clean JSON (normal case with json_mode=True)
      - JSON wrapped in ```json ... ``` fences
      - JSON embedded in surrounding text
      - Trailing commas (common LLM mistake)
    """
    # Strip markdown code fences
    content = re.sub(r"```json\s*", "", content)
    content = re.sub(r"```\s*", "", content)
    content = content.strip()

    # Direct parse (should work with JSON mode)
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass

    # Find the outermost {...} object
    start = content.find("{")
    end = content.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = content[start:end + 1]
        # Fix trailing commas before ] or }
        candidate = re.sub(r",\s*([}\]])", r"\1", candidate)
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass

    raise ValueError(f"Could not parse JSON from LLM output: {content[:200]}")


def _validate_recommendations(
    raw_recs: Optional[list],
    retriever: CatalogRetriever,
) -> Optional[list[dict]]:
    """
    Validate LLM-generated recommendations against the real catalog.

    For each recommendation:
      1. Check that the URL exists in the catalog (anti-hallucination)
      2. Normalize the URL (ensure trailing slash, correct domain)
      3. Replace name with canonical catalog name (prevents name drift)
      4. Replace test_type with canonical codes from catalog

    If a URL is not found in the catalog, the item is SILENTLY DROPPED.
    This is the primary defense against hallucination.
    """
    if not raw_recs or not isinstance(raw_recs, list):
        return None

    valid_urls = retriever.get_valid_urls()
    # Build URL→product lookup for fast access
    url_to_product = {p.url: p for p in retriever.get_all_products()}
    # Also index by URL without trailing slash for fuzzy matching
    url_stripped = {
        u.rstrip("/"): p for u, p in url_to_product.items()
    }

    validated: list[dict] = []

    for rec in raw_recs[:10]:  # Hard cap: never more than 10
        if not isinstance(rec, dict):
            continue

        raw_url: str = str(rec.get("url", "")).strip()
        raw_name: str = str(rec.get("name", "")).strip()
        raw_type: str = str(rec.get("test_type", "")).strip()

        # Normalize: ensure trailing slash
        normalized_url = raw_url.rstrip("/") + "/"

        # Try exact match first
        product = url_to_product.get(normalized_url)
        if not product:
            # Try without trailing slash
            product = url_stripped.get(raw_url.rstrip("/"))
        if not product:
            # Last attempt: case-insensitive partial URL match
            for catalog_url, p in url_to_product.items():
                if raw_url.rstrip("/").lower() in catalog_url.lower():
                    product = p
                    break

        if product:
            # Use canonical data from catalog — not what the LLM said
            validated.append({
                "name": product.name,
                "url": product.url,
                "test_type": (
                    ",".join(product.test_type_codes)
                    if product.test_type_codes
                    else raw_type
                ),
            })
        else:
            logger.warning(
                f"Recommendation dropped — URL not in catalog: "
                f"'{raw_url}' (name: '{raw_name}')"
            )

    return validated if validated else None


def _strip_shortlist_marker(content: str) -> str:
    """Remove [SHORTLIST]...[/SHORTLIST] markers from message content."""
    return re.sub(r"\[SHORTLIST\].*?\[/SHORTLIST\]", "", content, flags=re.DOTALL).strip()


def _safe_fallback_state(reply: str = "") -> dict:
    """Return a safe state on unexpected errors — never crash the API."""
    if not reply:
        reply = (
            "I'm experiencing a technical issue. "
            "Please try again in a moment."
        )
    return {
        "reply": reply,
        "recommendations": None,
        "end_of_conversation": False,
    }