from __future__ import annotations

import json
import re
from typing import Any


def extract_json_object(text: str) -> dict[str, Any]:
    """Best-effort JSON object extractor.

    Accepts either a pure-JSON string or a response containing a single JSON
    object somewhere inside.
    """

    if not text:
        return {}

    try:
        data = json.loads(text)
        if isinstance(data, dict):
            return data
    except Exception:
        pass

    m = re.search(r"\{[\s\S]*\}", text)
    if not m:
        return {}

    try:
        data = json.loads(m.group(0))
        if isinstance(data, dict):
            return data
    except Exception:
        return {}

    return {}


def clamp(value: float, lo: float, hi: float) -> float:
    if value < lo:
        return lo
    if value > hi:
        return hi
    return value
