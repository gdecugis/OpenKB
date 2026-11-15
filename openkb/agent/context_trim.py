"""Drop index.md from the query agent's context once the model has used it.

The agent reads index.md (one line per wiki page, so it grows with the wiki)
to plan its reads, and the Agents SDK then resends the whole conversation on
every model call, index included. This filter, run before each model call via
``RunConfig.call_model_input_filter``, replaces an index.md read the model has
already responded to with a short placeholder. The model sees the index once,
in full; pages it reads afterwards stay untouched.
"""

from __future__ import annotations

import json
from typing import Any

INDEX_PLACEHOLDER = (
    "[index.md was removed from the conversation to save tokens. You read it in an earlier "
    "step: rely on the page paths you noted. Read it again only if you need entries you did "
    "not note.]"
)

# Item types the model produced: anything after one of these in the input
# means the model has already had a turn with the preceding tool outputs.
_MODEL_ITEM_TYPES = {"function_call", "reasoning"}


def _field(item: Any, name: str) -> Any:
    return item.get(name) if isinstance(item, dict) else getattr(item, name, None)


def _is_index_read(item: Any) -> bool:
    if _field(item, "type") != "function_call" or _field(item, "name") != "read_file":
        return False
    raw = _field(item, "arguments")
    try:
        args = json.loads(raw) if isinstance(raw, str) else (raw or {})
    except json.JSONDecodeError:
        return False
    path = str(args.get("path", "")).strip().lstrip("./")
    return path == "index.md"


def _is_model_item(item: Any) -> bool:
    item_type = _field(item, "type")
    if item_type in _MODEL_ITEM_TYPES:
        return True
    return item_type in (None, "message") and _field(item, "role") == "assistant"


def drop_used_index_reads(items: list[Any]) -> list[Any]:
    """Return *items* with already-answered index.md outputs replaced."""
    index_call_ids = {_field(i, "call_id") for i in items if _is_index_read(i)}
    index_call_ids.discard(None)
    if not index_call_ids:
        return items
    last_model_pos = max((pos for pos, i in enumerate(items) if _is_model_item(i)), default=-1)

    trimmed: list[Any] = []
    for pos, item in enumerate(items):
        if (
            pos < last_model_pos
            and _field(item, "type") == "function_call_output"
            and _field(item, "call_id") in index_call_ids
            and _field(item, "output") != INDEX_PLACEHOLDER
        ):
            item = {**item, "output": INDEX_PLACEHOLDER} if isinstance(item, dict) else item
        trimmed.append(item)
    return trimmed


def index_trim_filter(data: Any) -> Any:
    """``RunConfig.call_model_input_filter`` applying :func:`drop_used_index_reads`."""
    from agents.run_config import ModelInputData

    return ModelInputData(
        input=drop_used_index_reads(list(data.model_data.input)),
        instructions=data.model_data.instructions,
    )
