"""
Managed Clinical Memory System for ICU Agent

This module implements a state-aware memory system that:
- Deduplicates entries by organ system
- Tracks status transitions (ACTIVE → RESOLVED)
- Maintains fixed capacity with intelligent pruning
- Focuses on 4-pillar clinical analysis
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional


class ClinicalMemoryEntry:
    """A single clinical memory entry with organ system focus."""

    def __init__(
        self,
        system: str,
        description: str,
        last_updated: float,
        status: str = "ACTIVE",
        entry_id: int = None,
    ):
        """
        Initialize a clinical memory entry.

        Args:
            system: Organ system (Hemodynamics, Respiratory, Renal, Neurology)
            description: Clinical observation or trend
            last_updated: Hours since admission when last updated
            status: ACTIVE or RESOLVED
            entry_id: Unique identifier (auto-generated if None)
        """
        self.id = entry_id
        self.system = system
        self.status = status
        self.description = description
        self.last_updated = last_updated

    def to_dict(self) -> Dict:
        """Convert entry to dictionary."""
        return {
            "id": self.id,
            "system": self.system,
            "status": self.status,
            "description": self.description,
            "last_updated": self.last_updated,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "ClinicalMemoryEntry":
        """Create entry from dictionary."""
        return cls(
            system=data["system"],
            description=data["description"],
            last_updated=data["last_updated"],
            status=data.get("status", "ACTIVE"),
            entry_id=data.get("id"),
        )

    def __repr__(self) -> str:
        """String representation."""
        return f"ClinicalMemoryEntry(id={self.id}, system={self.system}, status={self.status})"


class ManagedMemory:
    """
    Managed Clinical Memory System.

    Maintains a fixed-capacity memory bank with:
    - Deduplication by organ system
    - State transitions (ACTIVE → RESOLVED)
    - Intelligent pruning (prioritize ACTIVE over RESOLVED)
    """

    ORGAN_SYSTEMS = ["Hemodynamics", "Respiratory", "Renal", "Neurology"]

    def __init__(self, patient_id: str = None, max_entries: int = 5):
        """
        Initialize managed memory.

        Args:
            patient_id: Identifier for the patient
            max_entries: Maximum number of entries to maintain (default 5)
        """
        self.patient_id = patient_id
        self.max_entries = max_entries
        self.entries: List[ClinicalMemoryEntry] = []
        self._next_id = 1

    def add_or_update_entry(
        self,
        system: str,
        description: str,
        last_updated: float,
        status: str = "ACTIVE",
    ) -> ClinicalMemoryEntry:
        """
        Add new entry or update existing entry for the same organ system.

        Args:
            system: Organ system name
            description: Clinical observation
            last_updated: Hours since admission
            status: ACTIVE or RESOLVED

        Returns:
            The created or updated ClinicalMemoryEntry
        """
        # Check if entry for this system already exists
        existing_entry = self._find_entry_by_system(system)

        if existing_entry:
            # Update existing entry
            existing_entry.description = description
            existing_entry.last_updated = last_updated
            existing_entry.status = status
            return existing_entry
        else:
            # Create new entry
            entry = ClinicalMemoryEntry(
                system=system,
                description=description,
                last_updated=last_updated,
                status=status,
                entry_id=self._next_id,
            )
            self._next_id += 1
            self.entries.append(entry)

            # Prune if necessary
            self._prune_if_needed()

            return entry

    def resolve_entry(self, system: str, resolution_description: str, last_updated: float):
        """
        Mark an entry as RESOLVED with updated description.

        Args:
            system: Organ system to resolve
            resolution_description: Description of resolution
            last_updated: Hours since admission
        """
        entry = self._find_entry_by_system(system)
        if entry:
            entry.status = "RESOLVED"
            entry.description = resolution_description
            entry.last_updated = last_updated

    def _find_entry_by_system(self, system: str) -> Optional[ClinicalMemoryEntry]:
        """Find entry by organ system name."""
        for entry in self.entries:
            if entry.system.lower() == system.lower():
                return entry
        return None

    def _prune_if_needed(self):
        """
        Prune entries if exceeding max_entries.

        Pruning strategy:
        1. Prioritize keeping ACTIVE entries over RESOLVED
        2. Among same status, keep most recently updated
        """
        if len(self.entries) <= self.max_entries:
            return

        # Sort entries: ACTIVE first, then by last_updated (most recent first)
        sorted_entries = sorted(self.entries, key=lambda e: (e.status != "ACTIVE", -e.last_updated))

        # Keep only max_entries
        self.entries = sorted_entries[: self.max_entries]

    def get_all_entries(self) -> List[ClinicalMemoryEntry]:
        """Get all memory entries."""
        return self.entries

    def get_active_entries(self) -> List[ClinicalMemoryEntry]:
        """Get only ACTIVE entries."""
        return [e for e in self.entries if e.status == "ACTIVE"]

    def get_resolved_entries(self) -> List[ClinicalMemoryEntry]:
        """Get only RESOLVED entries."""
        return [e for e in self.entries if e.status == "RESOLVED"]

    def to_dict(self) -> Dict:
        """Convert memory to dictionary."""
        return {
            "patient_id": self.patient_id,
            "max_entries": self.max_entries,
            "total_entries": len(self.entries),
            "clinical_memory": [entry.to_dict() for entry in self.entries],
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

        print(f"Saved managed memory with {len(self.entries)} entries to {file_path}")

    @classmethod
    def load(cls, file_path: str) -> "ManagedMemory":
        """
        Load memory from JSON file.

        Args:
            file_path: Path to the memory file

        Returns:
            ManagedMemory object
        """
        with open(file_path, "r") as f:
            data = json.load(f)

        memory = cls(patient_id=data.get("patient_id"), max_entries=data.get("max_entries", 5))

        # Restore entries
        for entry_data in data.get("clinical_memory", []):
            entry = ClinicalMemoryEntry.from_dict(entry_data)
            memory.entries.append(entry)

        # Update next_id
        if memory.entries:
            memory._next_id = max(e.id for e in memory.entries) + 1

        print(f"Loaded managed memory with {len(memory.entries)} entries from {file_path}")
        return memory

    def format_for_prompt(self, max_insights: int = None) -> str:
        """
        Format memory for inclusion in agent prompts.

        Args:
            max_insights: Maximum number of insights to include (ignored for managed memory
                         since it already has a fixed capacity, kept for API compatibility)

        Returns:
            Formatted string of clinical memory
        """
        if not self.entries:
            return "No clinical memory entries yet."

        formatted = "## Clinical Memory Context\n\n"

        # Sort by status (ACTIVE first) and last_updated
        sorted_entries = sorted(self.entries, key=lambda e: (e.status != "ACTIVE", -e.last_updated))

        for entry in sorted_entries:
            formatted += f"{entry.id}. [Status: {entry.status}] System: {entry.system}\n"
            formatted += f"   Description: {entry.description}\n"
            formatted += f"   Last Updated: Hour {entry.last_updated:.1f}\n\n"

        return formatted

    def clear(self):
        """Clear all entries."""
        self.entries = []
        self._next_id = 1
