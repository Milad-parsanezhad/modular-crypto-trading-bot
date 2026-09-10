from __future__ import annotations

from hashlib import sha256
import json
from typing import Any


def payload_sha256(payload: dict[str, Any], *, exclude_keys: tuple[str, ...] = ()) -> str:
    """Hash the complete canonical payload except explicitly self-referential keys."""
    clean = {k: v for k, v in payload.items() if k not in set(exclude_keys)}
    raw = json.dumps(clean, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return sha256(raw).hexdigest()


def finalize_payload_hash(payload: dict[str, Any], hash_key: str) -> dict[str, Any]:
    """Set a self-excluding SHA-256 after all substantive fields are present."""
    payload[hash_key] = payload_sha256(payload, exclude_keys=(hash_key,))
    return payload
