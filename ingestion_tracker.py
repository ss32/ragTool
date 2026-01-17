"""
Ingestion Tracker Module
Tracks file processing progress to enable resumable database creation.
"""

import json
import os
from pathlib import Path
from typing import Set, Dict, Any, Optional
from datetime import datetime


class IngestionTracker:
    """Tracks which files have been successfully ingested into the database."""

    PROGRESS_FILE = "ingestion_progress.json"

    def __init__(self, db_path: str, auto_save_threshold: int = 10):
        """Initialize the tracker.

        Args:
            db_path: Path to the database directory
            auto_save_threshold: Number of files to accumulate before auto-saving (default: 10)
        """
        self.db_path = db_path
        self.progress_file = os.path.join(db_path, self.PROGRESS_FILE)
        self._auto_save_threshold = auto_save_threshold
        self._pending_count = 0
        self._data: Dict[str, Any] = self._load()

    def _load(self) -> Dict[str, Any]:
        """Load progress data from file."""
        if os.path.exists(self.progress_file):
            try:
                with open(self.progress_file, 'r') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                return self._default_data()
        return self._default_data()

    def _default_data(self) -> Dict[str, Any]:
        """Return default progress data structure."""
        return {
            "ingested_files": [],
            "config": {},
            "started_at": None,
            "last_updated": None,
            "input_directory": None
        }

    def _save(self):
        """Save progress data to file."""
        # Ensure directory exists
        Path(self.db_path).mkdir(parents=True, exist_ok=True)
        self._data["last_updated"] = datetime.now().isoformat()
        with open(self.progress_file, 'w') as f:
            json.dump(self._data, f, indent=2)

    def get_ingested_files(self) -> Set[str]:
        """Get set of file paths that have been ingested."""
        return set(self._data.get("ingested_files", []))

    def mark_file_ingested(self, file_path: str, force_save: bool = False):
        """Mark a single file as successfully ingested.

        Args:
            file_path: Path to the file that was ingested
            force_save: If True, save immediately regardless of threshold
        """
        ingested = self._data.get("ingested_files", [])
        if file_path not in ingested:
            ingested.append(file_path)
            self._data["ingested_files"] = ingested
            self._pending_count += 1

            # Auto-save when threshold reached or forced
            if force_save or self._pending_count >= self._auto_save_threshold:
                self._save()
                self._pending_count = 0

    def mark_files_ingested(self, file_paths: list, force_save: bool = True):
        """Mark multiple files as successfully ingested.

        Args:
            file_paths: List of file paths that were ingested
            force_save: If True (default), save immediately after marking
        """
        ingested = set(self._data.get("ingested_files", []))
        new_count = len(file_paths) - len(ingested.intersection(file_paths))
        ingested.update(file_paths)
        self._data["ingested_files"] = list(ingested)
        self._pending_count += new_count

        if force_save:
            self._save()
            self._pending_count = 0

    def flush(self):
        """Force save any pending changes to disk."""
        if self._pending_count > 0:
            self._save()
            self._pending_count = 0

    def start_ingestion(self, input_directory: str, config: Dict[str, Any]):
        """Record the start of an ingestion session."""
        if self._data["started_at"] is None:
            self._data["started_at"] = datetime.now().isoformat()
        self._data["input_directory"] = input_directory
        self._data["config"] = config
        self._save()

    def get_config(self) -> Dict[str, Any]:
        """Get the configuration used for this ingestion."""
        return self._data.get("config", {})

    def get_input_directory(self) -> Optional[str]:
        """Get the input directory for this ingestion."""
        return self._data.get("input_directory")

    def get_stats(self) -> Dict[str, Any]:
        """Get ingestion statistics."""
        return {
            "ingested_count": len(self._data.get("ingested_files", [])),
            "started_at": self._data.get("started_at"),
            "last_updated": self._data.get("last_updated"),
            "input_directory": self._data.get("input_directory")
        }

    def clear(self):
        """Clear all progress data."""
        self._data = self._default_data()
        if os.path.exists(self.progress_file):
            os.remove(self.progress_file)

    def has_progress(self) -> bool:
        """Check if there is existing progress to resume."""
        return len(self._data.get("ingested_files", [])) > 0
