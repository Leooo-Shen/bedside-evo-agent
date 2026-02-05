"""
Evolving Memory System for Evo-Agent

The memory system stores personalized clinical insights that the agent learns
through experience. Unlike static medical knowledge, this memory evolves based
on the agent's own prediction errors and reflections.
"""

import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional


class MemoryEntry:
    """A single memory entry containing a clinical insight."""

    def __init__(
        self,
        insight: str,
        clinical_scenario: str,
        source_window: int,
        timestamp: str = None,
        insight_id: str = None,
        confidence: float = 1.0,
    ):
        """
        Initialize a memory entry.

        Args:
            insight: The learned clinical knowledge
            clinical_scenario: Brief description of the clinical context
            source_window: Which patient window this came from (window index)
            timestamp: When this insight was learned (ISO format)
            insight_id: Unique identifier (auto-generated if None)
            confidence: Relevance or importance score (0.0 to 1.0)
        """
        self.insight_id = insight_id or str(uuid.uuid4())
        self.timestamp = timestamp or datetime.now().isoformat()
        self.clinical_scenario = clinical_scenario
        self.insight = insight
        self.source_window = source_window
        self.confidence = confidence

    def to_dict(self) -> Dict:
        """Convert memory entry to dictionary."""
        return {
            "insight_id": self.insight_id,
            "timestamp": self.timestamp,
            "clinical_scenario": self.clinical_scenario,
            "insight": self.insight,
            "source_window": self.source_window,
            "confidence": self.confidence,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "MemoryEntry":
        """Create memory entry from dictionary."""
        return cls(
            insight=data["insight"],
            clinical_scenario=data["clinical_scenario"],
            source_window=data["source_window"],
            timestamp=data.get("timestamp"),
            insight_id=data.get("insight_id"),
            confidence=data.get("confidence", 1.0),
        )


class AgentMemory:
    """
    Memory system for the Evo-Agent.

    Stores and manages personalized clinical insights learned through experience.
    """

    def __init__(self, patient_id: str = None):
        """
        Initialize agent memory.

        Args:
            patient_id: Identifier for the patient (used for file naming)
        """
        self.patient_id = patient_id
        self.entries: List[MemoryEntry] = []

    def add_insight(
        self,
        insight: str,
        clinical_scenario: str,
        source_window: int,
        confidence: float = 1.0,
    ) -> MemoryEntry:
        """
        Add a new insight to memory.

        Args:
            insight: The learned clinical knowledge
            clinical_scenario: Brief description of the clinical context
            source_window: Which patient window this came from
            confidence: Relevance or importance score (0.0 to 1.0)

        Returns:
            The created MemoryEntry
        """
        entry = MemoryEntry(
            insight=insight,
            clinical_scenario=clinical_scenario,
            source_window=source_window,
            confidence=confidence,
        )
        self.entries.append(entry)
        return entry

    def get_all_insights(self) -> List[MemoryEntry]:
        """Get all memory entries in chronological order."""
        return self.entries

    def get_recent_insights(self, n: int = 5) -> List[MemoryEntry]:
        """
        Get the n most recent insights.

        Args:
            n: Number of recent insights to retrieve

        Returns:
            List of most recent MemoryEntry objects
        """
        return self.entries[-n:] if len(self.entries) >= n else self.entries

    def search_insights(self, keyword: str) -> List[MemoryEntry]:
        """
        Search for insights containing a keyword.

        Args:
            keyword: Keyword to search for (case-insensitive)

        Returns:
            List of matching MemoryEntry objects
        """
        keyword_lower = keyword.lower()
        return [
            entry
            for entry in self.entries
            if keyword_lower in entry.insight.lower() or keyword_lower in entry.clinical_scenario.lower()
        ]

    def prune_low_confidence(self, threshold: float = 0.3):
        """
        Remove insights with confidence below threshold.

        Args:
            threshold: Minimum confidence to keep (default 0.3)
        """
        original_count = len(self.entries)
        self.entries = [entry for entry in self.entries if entry.confidence >= threshold]
        removed_count = original_count - len(self.entries)
        if removed_count > 0:
            print(f"Pruned {removed_count} low-confidence insights (threshold: {threshold})")

    def to_dict(self) -> Dict:
        """Convert memory to dictionary."""
        return {
            "patient_id": self.patient_id,
            "total_insights": len(self.entries),
            "entries": [entry.to_dict() for entry in self.entries],
        }

    def to_json(self, indent: int = 2) -> str:
        """Convert memory to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    def save(self, file_path: str):
        """
        Save memory to JSON file.

        Args:
            file_path: Path to save the memory file
        """
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

        print(f"Saved memory with {len(self.entries)} insights to {file_path}")

    @classmethod
    def load(cls, file_path: str) -> "AgentMemory":
        """
        Load memory from JSON file.

        Args:
            file_path: Path to the memory file

        Returns:
            AgentMemory object
        """
        with open(file_path, "r") as f:
            data = json.load(f)

        memory = cls(patient_id=data.get("patient_id"))
        memory.entries = [MemoryEntry.from_dict(entry_data) for entry_data in data.get("entries", [])]

        print(f"Loaded memory with {len(memory.entries)} insights from {file_path}")
        return memory

    def format_for_prompt(self, max_insights: int = 10) -> str:
        """
        Format memory for inclusion in agent prompts.

        Args:
            max_insights: Maximum number of insights to include

        Returns:
            Formatted string of insights for prompt
        """
        if not self.entries:
            return "No previous insights yet."

        recent_insights = self.get_recent_insights(max_insights)

        formatted = "## Previous Clinical Insights\n\n"
        for i, entry in enumerate(recent_insights, 1):
            formatted += f"{i}. **{entry.clinical_scenario}**\n"
            formatted += f"   - {entry.insight}\n"
            formatted += f"   - (Learned at window {entry.source_window}, confidence: {entry.confidence:.2f})\n\n"

        return formatted

