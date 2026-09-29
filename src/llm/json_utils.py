"""Strict-JSON handling for small local models: extraction, repair, validation, retry."""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Optional, Type, TypeVar

from pydantic import BaseModel, ValidationError

from .client import LLMError, LLMProvider

log = logging.getLogger(__name__)
T = TypeVar("T", bound=BaseModel)


class JSONRepairError(ValueError):
    pass


def _balanced_block(text: str) -> Optional[str]:
    """Return the first balanced {...} or [...] block, respecting strings."""
    start = None
    stack: list[str] = []
    in_str = False
    esc = False
    for i, ch in enumerate(text):
        if start is None:
            if ch in "{[":
                start = i
                stack.append("}" if ch == "{" else "]")
            continue
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch in "{[":
            stack.append("}" if ch == "{" else "]")
        elif ch in "}]":
            if stack and stack[-1] == ch:
                stack.pop()
                if not stack:
                    return text[start:i + 1]
    if start is not None:  # truncated output: close open brackets
        tail = text[start:]
        if in_str:
            tail += '"'
        return tail + "".join(reversed(stack))
    return None


def repair_json(text: str) -> str:
    t = text.strip()
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", t, flags=re.I | re.M)
    block = _balanced_block(t)
    if block is None:
        raise JSONRepairError("no JSON object found in model output")
    t = block
    t = t.replace("“", '"').replace("”", '"')
    t = re.sub(r",\s*([}\]])", r"\1", t)                          # trailing commas
    t = re.sub(r"\bNone\b", "null", t)
    t = re.sub(r"\bTrue\b", "true", t)
    t = re.sub(r"\bFalse\b", "false", t)
    t = re.sub(r"(?<=[{,])\s*([A-Za-z_][A-Za-z0-9_]*)\s*:", r' "\1":', t)  # unquoted keys
    if "'" in t and '"' not in t:                                    # python-style dict
        t = t.replace("'", '"')
    t = re.sub(r"//[^\n]*", "", t)                                   # comments
    return t


def parse_json_lenient(text: str) -> Any:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    repaired = repair_json(text)
    try:
        return json.loads(repaired)
    except json.JSONDecodeError as exc:
        raise JSONRepairError(f"could not repair JSON: {exc}") from exc


@dataclass
class LLMCallLog:
    attempts: int = 0
    repaired: bool = False
    errors: list[str] = field(default_factory=list)
    raw_outputs: list[str] = field(default_factory=list)


def call_llm_json(llm: LLMProvider, system: str, user: str, model_cls: Type[T],
                  max_retries: int = 2, wrap_key: Optional[str] = None) -> tuple[Optional[T], LLMCallLog]:
    """Call the LLM, parse + validate the JSON against `model_cls`, retry with feedback.

    Returns (validated object or None, log). Never raises on bad model output.
    `wrap_key`: if the model returns a bare list, wrap it as {wrap_key: list}.
    """
    clog = LLMCallLog()
    schema = model_cls.model_json_schema()
    prompt = user
    for attempt in range(max_retries + 1):
        clog.attempts += 1
        try:
            raw = llm.chat(system, prompt, schema=schema)
        except LLMError as exc:
            clog.errors.append(str(exc))
            log.warning("LLM call failed (attempt %d): %s", attempt + 1, exc)
            continue
        clog.raw_outputs.append(raw[:4000])
        try:
            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                data = parse_json_lenient(raw)
                clog.repaired = True
            if isinstance(data, list) and wrap_key:
                data = {wrap_key: data}
            return model_cls.model_validate(data), clog
        except (JSONRepairError, ValidationError) as exc:
            msg = str(exc).splitlines()[0][:300]
            clog.errors.append(msg)
            log.info("Invalid JSON from LLM (attempt %d): %s", attempt + 1, msg)
            prompt = (f"{user}\n\nYOUR PREVIOUS ANSWER WAS INVALID: {msg}\n"
                      f"Return ONLY one JSON object that matches the schema. No prose, no markdown.")
    return None, clog
