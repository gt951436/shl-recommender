"""
Fetches the SHL product catalog from the provided JSON URL and parses it
into a list of CatalogProduct objects.

Design principles:
  1. Defensive parsing — the JSON structure may differ from our assumptions.
     We inspect the top-level shape and adapt accordingly.
  2. Transparent logging — every filtering decision is logged so you can
     verify the right products are included/excluded.
  3. URL validation — every product URL is normalized to the official SHL
     domain. Catalog items without valid URLs are skipped (we cannot
     recommend them because the API spec requires real URLs).
  4. No hallucination — we add no descriptions that aren't in the source data.
"""
import json
import logging
import re
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urljoin, urlparse

import httpx

from models import CatalogProduct

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

SHL_BASE_URL = "https://www.shl.com"
CATALOG_VIEW_PREFIX = "/products/product-catalog/view/"

# Type code → human-readable label mapping.
# These are the codes visible in the 10 conversation traces.
TYPE_CODE_TO_LABEL: dict[str, str] = {
    "A": "Ability & Aptitude",
    "B": "Biodata & Situational Judgment",
    "C": "Competencies",
    "D": "Development & 360",
    "E": "Assessment Exercises",
    "K": "Knowledge & Skills",
    "P": "Personality & Behavior",
    "S": "Simulations",
}

# Keywords that indicate a product is a "Job Solution" (pre-packaged bundle)
# rather than an Individual Test Solution.
# Per the assignment: "Pre-packaged Job Solutions are out of scope."
JOB_SOLUTION_SIGNALS = [
    "job solution",
    "pre-packaged",
    "prepackaged",
    "hiring solution",
    "talent solution",
    "workforce solution",
]


# ─────────────────────────────────────────────────────────────────────────────
# Public interface
# ─────────────────────────────────────────────────────────────────────────────

def fetch_catalog(
    url: str,
    local_json_path: Optional[str] = None,
    timeout_seconds: int = 30,
) -> list[CatalogProduct]:
    """
    Main entry point. Fetches the catalog and returns parsed products.

    Args:
        url: The JSON catalog URL.
        local_json_path: If provided, load from this local file instead of
                         making an HTTP request. Useful for testing without
                         network access.
        timeout_seconds: HTTP request timeout.

    Returns:
        List of CatalogProduct objects (Individual Test Solutions only).
    """
    raw_data = _load_raw_json(url, local_json_path, timeout_seconds)
    products = _parse_catalog(raw_data)

    individual_tests = [p for p in products if p.is_individual_test]
    excluded = len(products) - len(individual_tests)

    logger.info(
        f"Catalog fetch complete: {len(individual_tests)} individual test solutions "
        f"({excluded} pre-packaged job solutions excluded)."
    )
    return individual_tests


# ─────────────────────────────────────────────────────────────────────────────
# Step 1: Load raw JSON
# ─────────────────────────────────────────────────────────────────────────────

def _load_raw_json(
    url: str,
    local_path: Optional[str],
    timeout: int,
) -> Any:
    """Load the raw catalog JSON — from disk or network."""
    if local_path and Path(local_path).exists():
        logger.info(f"Loading catalog from local file: {local_path}")
        with open(local_path, "r", encoding="utf-8") as f:
            return json.load(f)

    logger.info(f"Fetching catalog from: {url}")
    try:
        response = httpx.get(url, timeout=timeout, follow_redirects=True)
        response.raise_for_status()
        data = response.json()
        logger.info(
            f"Received catalog JSON. Top-level type: {type(data).__name__}. "
            f"Keys: {list(data.keys()) if isinstance(data, dict) else 'N/A (list)'}"
        )
        return data
    except httpx.HTTPStatusError as e:
        raise RuntimeError(
            f"HTTP {e.response.status_code} while fetching catalog from {url}. "
            f"Check that the URL is still valid."
        ) from e
    except httpx.TimeoutException:
        raise RuntimeError(
            f"Catalog fetch timed out after {timeout}s. "
            f"Check your network connection."
        )
    except json.JSONDecodeError as e:
        raise RuntimeError(
            f"The catalog URL did not return valid JSON: {e}"
        ) from e


# ─────────────────────────────────────────────────────────────────────────────
# Step 2: Parse catalog — handles multiple possible JSON shapes
# ─────────────────────────────────────────────────────────────────────────────

def _parse_catalog(data: Any) -> list[CatalogProduct]:
    """
    The SHL catalog JSON might be structured in several ways. This function
    handles the most likely shapes and logs what it found.

    Known possible shapes:
      Shape A: A list of product objects directly
               [ {"name": "...", ...}, ... ]

      Shape B: An object with a 'products' or 'data' or 'results' key
               { "products": [ {...}, ... ] }

      Shape C: An object with multiple category keys
               { "Individual Tests": [...], "Job Solutions": [...] }
    """
    product_list: list[dict] = []

    if isinstance(data, list):
        logger.info(f"Catalog shape: flat list of {len(data)} items.")
        product_list = data

    elif isinstance(data, dict):
        # Try common wrapper keys
        for key in ("products", "data", "results", "items", "catalog"):
            if key in data and isinstance(data[key], list):
                logger.info(f"Catalog shape: dict with '{key}' list of {len(data[key])} items.")
                product_list = data[key]
                break
        else:
            # Try to find any key that holds a large list (likely the products)
            list_keys = {k: v for k, v in data.items() if isinstance(v, list)}
            if list_keys:
                # Pick the key with the most items
                best_key = max(list_keys, key=lambda k: len(list_keys[k]))
                logger.info(
                    f"Catalog shape: dict with multiple list keys {list(list_keys.keys())}. "
                    f"Using '{best_key}' ({len(list_keys[best_key])} items)."
                )
                product_list = list_keys[best_key]
            else:
                # Last resort: treat the dict itself as a single product
                logger.warning(
                    "Unexpected catalog shape — treating the root object as a single product."
                )
                product_list = [data]
    else:
        raise RuntimeError(
            f"Unexpected catalog JSON type: {type(data).__name__}. Expected list or dict."
        )

    if not product_list:
        raise RuntimeError(
            "Parsed catalog is empty. The catalog URL may have changed structure."
        )

    logger.info(f"Parsing {len(product_list)} raw catalog entries...")

    products: list[CatalogProduct] = []
    skipped = 0
    for i, raw in enumerate(product_list):
        try:
            product = _parse_single_product(raw)
            if product is not None:
                products.append(product)
            else:
                skipped += 1
        except Exception as e:
            logger.warning(f"Skipping item {i} due to parse error: {e}. Raw: {str(raw)[:200]}")
            skipped += 1

    logger.info(
        f"Parsed {len(products)} valid products. Skipped {skipped} items "
        f"(missing URL or unparseable)."
    )
    return products


# ─────────────────────────────────────────────────────────────────────────────
# Step 3: Parse a single product entry
# ─────────────────────────────────────────────────────────────────────────────

def _parse_single_product(raw: dict) -> Optional[CatalogProduct]:
    """
    Parse one raw JSON object into a CatalogProduct.
    Returns None if the product should be skipped.

    We try multiple possible key names for each field because the catalog
    JSON key names are not guaranteed.
    """
    if not isinstance(raw, dict):
        return None

    # ── Name ──────────────────────────────────────────────────────────────
    name = _get_str(raw, ["name", "title", "productName", "product_name", "Name"])
    if not name:
        logger.debug(f"Skipping product with no name. Keys: {list(raw.keys())}")
        return None

    # ── URL ───────────────────────────────────────────────────────────────
    raw_url = _get_str(raw, ["url", "link", "href", "productUrl", "product_url", "URL"])
    url = _normalize_url(raw_url, name)
    if not url:
        # A product without a valid URL cannot be recommended (API requires real URLs)
        logger.debug(f"Skipping '{name}': could not resolve a valid SHL catalog URL.")
        return None

    # ── Test Type ─────────────────────────────────────────────────────────
    type_codes, type_labels = _parse_test_type(raw)

    # ── Duration ──────────────────────────────────────────────────────────
    duration = _parse_duration(raw)

    # ── Languages ─────────────────────────────────────────────────────────
    languages = _parse_languages(raw)

    # ── Description ───────────────────────────────────────────────────────
    description = _get_str(raw, ["description", "desc", "overview", "summary"])

    # ── Is Individual Test (vs Job Solution)? ─────────────────────────────
    is_individual = _is_individual_test(raw, name, description)

    return CatalogProduct(
        name=name,
        url=url,
        test_type_codes=type_codes,
        test_type_labels=type_labels,
        duration=duration,
        languages=languages,
        description=description,
        is_individual_test=is_individual,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _get_str(d: dict, keys: list[str]) -> Optional[str]:
    """Try multiple possible key names; return first non-empty string found."""
    for key in keys:
        val = d.get(key)
        if val and isinstance(val, str) and val.strip():
            return val.strip()
    return None


def _normalize_url(raw_url: Optional[str], product_name: str) -> Optional[str]:
    """
    Normalize a product URL to the canonical SHL catalog format.

    Valid URL: https://www.shl.com/products/product-catalog/view/<slug>/
    We also accept relative paths and construct the full URL.
    If we only have a product name, we generate a slug (best-effort).
    """
    if raw_url:
        parsed = urlparse(raw_url)

        # Already an absolute SHL URL
        if "shl.com" in parsed.netloc:
            # Ensure it points to the product catalog view, not some other page
            return raw_url.rstrip("/") + "/" if not raw_url.endswith("/") else raw_url

        # Relative path
        if parsed.path.startswith("/"):
            return urljoin(SHL_BASE_URL, raw_url)

        # Has a path-like structure without a hostname
        if CATALOG_VIEW_PREFIX in raw_url:
            return SHL_BASE_URL + raw_url

    # No URL provided — generate a slug from the product name (best-effort).
    # This is used when the JSON omits URLs but has names.
    slug = _name_to_slug(product_name)
    if slug:
        generated = f"{SHL_BASE_URL}{CATALOG_VIEW_PREFIX}{slug}/"
        logger.debug(f"Generated URL for '{product_name}': {generated}")
        return generated

    return None


def _name_to_slug(name: str) -> str:
    """
    Convert a product name to a URL slug.
    Example: "Core Java (Advanced Level) (New)" → "core-java-advanced-level-new"
    """
    slug = name.lower()
    slug = re.sub(r"[^a-z0-9\s-]", " ", slug)   # Remove special chars
    slug = re.sub(r"\s+", "-", slug.strip())      # Spaces → hyphens
    slug = re.sub(r"-+", "-", slug)               # Collapse multiple hyphens
    return slug.strip("-")


def _parse_test_type(raw: dict) -> tuple[list[str], list[str]]:
    """
    Extract test type codes and labels from the raw product dict.

    The JSON might store types as:
      - A single code string: "K" or "P,S" or "K,S"
      - A list of codes: ["K", "S"]
      - A list of label strings: ["Knowledge & Skills", "Simulations"]
      - An object with code + label fields
    """
    codes: list[str] = []
    labels: list[str] = []

    # Try 'type' key first (most common)
    raw_type = raw.get("type") or raw.get("testType") or raw.get("test_type") or raw.get("Type")

    if isinstance(raw_type, str):
        # Could be "K" or "P,S" or "Knowledge & Skills"
        parts = [p.strip() for p in raw_type.split(",")]
        for part in parts:
            if part in TYPE_CODE_TO_LABEL:
                codes.append(part)
                labels.append(TYPE_CODE_TO_LABEL[part])
            else:
                # It's a label — find the code
                for code, label in TYPE_CODE_TO_LABEL.items():
                    if label.lower() == part.lower():
                        codes.append(code)
                        labels.append(label)
                        break
                else:
                    # Unknown type — store as-is
                    labels.append(part)

    elif isinstance(raw_type, list):
        for item in raw_type:
            if isinstance(item, str):
                item = item.strip()
                if item in TYPE_CODE_TO_LABEL:
                    codes.append(item)
                    labels.append(TYPE_CODE_TO_LABEL[item])
                else:
                    for code, label in TYPE_CODE_TO_LABEL.items():
                        if label.lower() == item.lower():
                            codes.append(code)
                            labels.append(label)
                            break
                    else:
                        labels.append(item)

    return codes, labels


def _parse_duration(raw: dict) -> Optional[str]:
    """Extract duration as a clean string."""
    dur = _get_str(raw, ["duration", "Duration", "time", "testDuration"])
    if not dur:
        return None
    # Normalize: "25" → "25 minutes", "25 minutes" → "25 minutes"
    if re.match(r"^\d+$", dur):
        dur = f"{dur} minutes"
    return dur


def _parse_languages(raw: dict) -> list[str]:
    """Extract list of supported language strings."""
    raw_langs = raw.get("languages") or raw.get("language") or raw.get("Languages")
    if not raw_langs:
        return []
    if isinstance(raw_langs, str):
        # Could be comma-separated
        return [lang.strip() for lang in raw_langs.split(",") if lang.strip()]
    if isinstance(raw_langs, list):
        return [str(lang).strip() for lang in raw_langs if str(lang).strip()]
    return []


def _is_individual_test(raw: dict, name: str, description: Optional[str]) -> bool:
    """
    Determine whether this is an Individual Test Solution (in scope)
    vs a Pre-packaged Job Solution (out of scope).

    Rules (in priority order):
    1. If the JSON has an explicit 'solution_type' or 'category' field, trust it.
    2. If the name or description contains job-solution signals, exclude it.
    3. Default to True (include) — better to include too many than miss items.
    """
    # Check explicit fields
    solution_type = _get_str(
        raw,
        ["solution_type", "solutionType", "category", "productType", "product_type"]
    )
    if solution_type:
        lower = solution_type.lower()
        if "individual" in lower or "test solution" in lower:
            return True
        if "job solution" in lower or "pre-packaged" in lower:
            logger.debug(f"Excluding '{name}': solution_type='{solution_type}'")
            return False

    # Check name and description for job-solution signals
    search_text = f"{name} {description or ''}".lower()
    for signal in JOB_SOLUTION_SIGNALS:
        if signal in search_text:
            logger.debug(f"Excluding '{name}': matched job-solution signal '{signal}'")
            return False

    return True  # Default: include


# ─────────────────────────────────────────────────────────────────────────────
# Utility: save raw catalog JSON to disk (for debugging)
# ─────────────────────────────────────────────────────────────────────────────

def save_raw_catalog(data: Any, path: str = "data/raw_catalog.json") -> None:
    """Save the raw catalog JSON to disk for inspection."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    logger.info(f"Raw catalog saved to {path}")
