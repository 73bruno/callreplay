"""Grounding: did the values the agent says come from somewhere?

A booking agent that offers "4:30" when the calendar tool never returned 4:30 has made it up,
and the caller will turn up at a time nobody is expecting. This module finds times and amounts
of money in text and normalises them, so they can be compared with what the tools returned,
what the caller said and what the prompt states (opening hours, prices).

Spoken times are ambiguous: "at five" is 05:00 or 17:00. A mention keeps every reading and is
grounded if any of them is known.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

_AMPM = r'(a\.?\s?m\.?|p\.?\s?m\.?)'
_TIME_AMPM = re.compile(rf'(?<![\d:.$€£])(\d{{1,2}})(?::(\d{{2}}))?\s*{_AMPM}(?![a-z])', re.I)
_TIME_DAYPART = re.compile(r"(?<![\d:.$€£])(\d{1,2})(?::(\d{2}))?\s+(?:o'clock\s+)?in the (morning|afternoon|evening)", re.I)
_TIME_COLON = re.compile(r'(?<![\d.$€£])(\d{1,2}):(\d{2})(?!\d)')
_NOON = re.compile(r'\b(noon|midday)\b', re.I)
_MONEY_SYMBOL = re.compile(r'([$€£])\s?(\d{1,3}(?:,\d{3})+|\d+)(?:\.(\d{1,2}))?')
_MONEY_WORD = re.compile(r'(?<![\d.])(\d+)(?:\.(\d{1,2}))?\s?(dollars|euros|pounds|usd|eur|gbp)\b', re.I)
_NUMBER = re.compile(r'(?<![\d.])(\d+)(?:\.(\d{1,2}))?(?![\d])')


@dataclass(frozen=True)
class Mention:
    kind: str                     # 'time' | 'money'
    raw: str                      # as written
    readings: frozenset[str]      # normalised: '17:30', '45.00'


def mentions(text: str, kinds: tuple[str, ...] | list[str] = ('time', 'money')) -> list[Mention]:
    """Times and amounts in a piece of text, in the order they appear."""
    found: list[tuple[int, Mention]] = []
    taken: list[tuple[int, int]] = []

    def free(m: re.Match[str]) -> bool:
        return not any(a < m.end() and m.start() < b for a, b in taken)

    def keep(m: re.Match[str], kind: str, readings: set[str]) -> None:
        if readings and free(m):
            taken.append((m.start(), m.end()))
            raw = m.group(0).strip()
            if raw.endswith('.') and not raw.endswith('.m.'):       # "4 pm." ends a sentence
                raw = raw[:-1]
            found.append((m.start(), Mention(kind, raw, frozenset(readings))))

    if 'money' in kinds:                         # first, so "$10.30" isn't read as a time
        for m in _MONEY_SYMBOL.finditer(text):
            keep(m, 'money', {_amount(m.group(2).replace(',', ''), m.group(3))})
        for m in _MONEY_WORD.finditer(text):
            keep(m, 'money', {_amount(m.group(1), m.group(2))})
    if 'time' in kinds:
        for m in _TIME_AMPM.finditer(text):
            h, mm, suffix = int(m.group(1)), int(m.group(2) or 0), m.group(3).lower()
            if 1 <= h <= 12 and mm < 60:
                h24 = h % 12 + (12 if suffix.startswith('p') else 0)
                keep(m, 'time', {f'{h24:02d}:{mm:02d}'})
        for m in _TIME_DAYPART.finditer(text):
            h, mm, part = int(m.group(1)), int(m.group(2) or 0), m.group(3).lower()
            if 1 <= h <= 12 and mm < 60:
                h24 = h % 12 + (0 if part == 'morning' else 12)
                keep(m, 'time', {f'{h24:02d}:{mm:02d}'})
        for m in _TIME_COLON.finditer(text):
            h, mm = int(m.group(1)), int(m.group(2))
            if h <= 23 and mm < 60:
                keep(m, 'time', _clock(h, mm))
        for m in _NOON.finditer(text):
            keep(m, 'time', {'12:00'})
    return [x for _, x in sorted(found, key=lambda p: p[0])]


def known_values(value: Any, kinds: tuple[str, ...] | list[str] = ('time', 'money')) -> set[str]:
    """Every normalised value inside a tool result (any JSON), a user turn or a prompt. In tool
    results a bare number can be a price ({"price": 45}), so numbers count as amounts too."""
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
    out = {r for m in mentions(text, kinds) for r in m.readings}
    if 'money' in kinds and not isinstance(value, str):
        out |= {_amount(m.group(1), m.group(2)) for m in _NUMBER.finditer(text)}
    return out


def _clock(h: int, mm: int) -> set[str]:
    """'17:30' is one time; '5:30' could be morning or afternoon."""
    if h == 0 or h >= 12:
        return {f'{h:02d}:{mm:02d}'}
    return {f'{h:02d}:{mm:02d}', f'{h + 12:02d}:{mm:02d}'}


def _amount(units: str, cents: str | None) -> str:
    return f'{int(units)}.{(cents or "0").ljust(2, "0")[:2]}'
