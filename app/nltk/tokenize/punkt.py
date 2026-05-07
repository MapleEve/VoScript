"""Small Punkt-compatible sentence span tokenizer for WhisperX.

WhisperX 3.3.1 imports ``PunktParameters`` and ``PunktSentenceTokenizer`` only
to split an already bounded segment into sentence spans. Pulling the full NLTK
distribution into the runtime introduces unrelated data/license surface, so this
module implements the small API shape WhisperX uses.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable


@dataclass
class PunktParameters:
    """Subset of NLTK's PunktParameters used by WhisperX."""

    abbrev_types: set[str] = field(default_factory=set)


class PunktSentenceTokenizer:
    """Sentence span splitter compatible with WhisperX's use of NLTK Punkt."""

    _TERMINATORS = {".", "!", "?", "。", "！", "？"}

    def __init__(self, params: PunktParameters | None = None) -> None:
        self.params = params or PunktParameters()

    def span_tokenize(self, text: str) -> Iterable[tuple[int, int]]:
        """Yield half-open sentence spans in ``text``.

        This intentionally implements conservative splitting: common
        abbreviations configured by WhisperX are not treated as sentence
        boundaries, and punctuation must be followed by whitespace or end of
        string before a split is emitted.
        """

        start = 0
        index = 0
        length = len(text)
        while index < length:
            char = text[index]
            if char not in self._TERMINATORS or self._is_abbreviation(text, index):
                index += 1
                continue

            next_index = index + 1
            while next_index < length and text[next_index] in {
                '"',
                "'",
                ")",
                "]",
                "}",
                "”",
                "’",
            }:
                next_index += 1

            if (
                next_index < length
                and char in {".", "!", "?"}
                and not text[next_index].isspace()
            ):
                index += 1
                continue

            end = next_index
            while end < length and text[end].isspace():
                end += 1

            yield (start, next_index)
            start = end
            index = end

        if start < length:
            yield (start, length)
        elif length == 0:
            return

    def _is_abbreviation(self, text: str, dot_index: int) -> bool:
        if text[dot_index] != ".":
            return False
        prefix = text[:dot_index]
        match = re.search(r"([A-Za-z]+)$", prefix)
        if not match:
            return False
        return match.group(1).lower() in self.params.abbrev_types
