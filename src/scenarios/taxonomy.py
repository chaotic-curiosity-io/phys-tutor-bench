"""Load and query the misconception taxonomy."""

from __future__ import annotations

import json
from pathlib import Path

from src.scenarios.schema import MisconceptionEntry, Prevalence


class MisconceptionTaxonomy:
    """Manages the misconception taxonomy — the knowledge backbone of PhysTutorBench."""

    def __init__(self, taxonomy_path: str | Path):
        self.taxonomy_path = Path(taxonomy_path)
        self._data: dict = {}
        self._entries: list[MisconceptionEntry] = []
        self._by_id: dict[str, MisconceptionEntry] = {}
        self._load()

    def _load(self) -> None:
        with open(self.taxonomy_path) as f:
            self._data = json.load(f)

        for raw in self._data["misconceptions"]:
            entry = MisconceptionEntry(**raw)
            self._entries.append(entry)
            self._by_id[entry.id] = entry

    @property
    def metadata(self) -> dict:
        return self._data.get("metadata", {})

    @property
    def topic_areas(self) -> dict:
        return self._data.get("topic_areas", {})

    @property
    def all_entries(self) -> list[MisconceptionEntry]:
        return list(self._entries)

    def get(self, misconception_id: str) -> MisconceptionEntry | None:
        return self._by_id.get(misconception_id)

    def by_topic(self, topic_area: str) -> list[MisconceptionEntry]:
        return [e for e in self._entries if e.topic_area == topic_area]

    def by_subtopic(self, subtopic: str) -> list[MisconceptionEntry]:
        return [e for e in self._entries if e.subtopic == subtopic]

    def by_source(self, source_instrument: str) -> list[MisconceptionEntry]:
        return [e for e in self._entries if e.source_instrument == source_instrument]

    def by_prevalence(self, prevalence: Prevalence) -> list[MisconceptionEntry]:
        return [e for e in self._entries if e.prevalence == prevalence]

    def topic_summary(self) -> dict[str, int]:
        """Return counts of misconceptions per topic area."""
        counts: dict[str, int] = {}
        for entry in self._entries:
            counts[entry.topic_area] = counts.get(entry.topic_area, 0) + 1
        return counts

    def subtopic_summary(self) -> dict[str, int]:
        """Return counts of misconceptions per subtopic."""
        counts: dict[str, int] = {}
        for entry in self._entries:
            counts[entry.subtopic] = counts.get(entry.subtopic, 0) + 1
        return counts

    def get_subtopics(self, topic_area: str) -> list[str]:
        """Return the list of subtopics for a given topic area."""
        area = self.topic_areas.get(topic_area, {})
        return area.get("subtopics", [])
