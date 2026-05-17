"""
Replay markdown traces against the running SHL recommender API.

"""

from __future__ import annotations

import json
from pathlib import Path

import requests

from trace_parser import parse_trace
from behaviour_specs import assert_behavior


API_URL = "http://localhost:8000/chat"

TRACE_DIR = Path("tests/fixtures/traces")


def replay_trace(trace_path: Path) -> bool:
    print(f"\n{'=' * 70}")
    print(f"TRACE: {trace_path.name}")
    print(f"{'=' * 70}")

    turns = parse_trace(trace_path)

    messages = []

    all_passed = True

    for turn in turns:
        print(f"\nTurn {turn.turn_number}")

        messages.append(
            {
                "role": "user",
                "content": turn.user_message,
            }
        )

        payload = {
            "messages": messages
        }

        try:
            response = requests.post(
                API_URL,
                json=payload,
                timeout=120,
            )

            response.raise_for_status()

        except Exception as e:
            print(f"FAIL: request error: {e}")
            return False

        data = response.json()

        failures = assert_behavior(
            expected_turn=turn,
            actual_response=data,
        )

        if failures:
            all_passed = False

            print("FAIL")

            for f in failures:
                print(f"  - {f}")

        else:
            print("PASS")

        assistant_reply = data.get("reply", "")

        messages.append(
            {
                "role": "assistant",
                "content": assistant_reply,
            }
        )

    return all_passed


def main():
    trace_files = sorted(
        TRACE_DIR.glob("*.md")
    )

    if not trace_files:
        print("No trace files found.")
        return

    passed = 0

    for trace_file in trace_files:
        ok = replay_trace(trace_file)

        if ok:
            passed += 1

    print(f"\n{'=' * 70}")
    print("FINAL RESULTS")
    print(f"{'=' * 70}")
    print(f"Passed: {passed}/{len(trace_files)}")


if __name__ == "__main__":
    main()