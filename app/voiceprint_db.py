"""Persistent voiceprint database for speaker identification."""

import json
import uuid
import numpy as np
from pathlib import Path
from datetime import datetime
from scipy.spatial.distance import cosine


class VoiceprintDB:
    def __init__(self, db_dir: str = "/data/voiceprints"):
        self.db_dir = Path(db_dir)
        self.db_dir.mkdir(parents=True, exist_ok=True)
        self.index_file = self.db_dir / "index.json"
        self.index = self._load_index()

    def _load_index(self) -> dict:
        if self.index_file.exists():
            return json.loads(self.index_file.read_text(encoding="utf-8"))
        return {"speakers": {}}

    def _save_index(self):
        self.index_file.write_text(
            json.dumps(self.index, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def add_speaker(self, name: str, embedding: np.ndarray) -> str:
        """Register a new speaker with a name and initial embedding."""
        speaker_id = f"spk_{uuid.uuid4().hex[:8]}"
        emb = embedding.reshape(1, -1)
        np.save(self.db_dir / f"{speaker_id}_samples.npy", emb)
        np.save(self.db_dir / f"{speaker_id}_avg.npy", emb.mean(axis=0))
        self.index["speakers"][speaker_id] = {
            "name": name,
            "sample_count": 1,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
        }
        self._save_index()
        return speaker_id

    def update_speaker(
        self, speaker_id: str, new_embedding: np.ndarray, name: str | None = None
    ):
        """Add a new embedding sample and recompute the average."""
        if speaker_id not in self.index["speakers"]:
            raise ValueError(f"Speaker {speaker_id} not found")

        samples_path = self.db_dir / f"{speaker_id}_samples.npy"
        samples = np.load(samples_path)
        new_emb = new_embedding.reshape(1, -1)
        samples = np.vstack([samples, new_emb])
        np.save(samples_path, samples)
        np.save(self.db_dir / f"{speaker_id}_avg.npy", samples.mean(axis=0))

        info = self.index["speakers"][speaker_id]
        info["sample_count"] = len(samples)
        info["updated_at"] = datetime.now().isoformat()
        if name is not None:
            info["name"] = name
        self._save_index()

    def delete_speaker(self, speaker_id: str):
        if speaker_id not in self.index["speakers"]:
            raise ValueError(f"Speaker {speaker_id} not found")
        for suffix in ("_samples.npy", "_avg.npy"):
            p = self.db_dir / f"{speaker_id}{suffix}"
            p.unlink(missing_ok=True)
        del self.index["speakers"][speaker_id]
        self._save_index()

    def rename_speaker(self, speaker_id: str, new_name: str):
        if speaker_id not in self.index["speakers"]:
            raise ValueError(f"Speaker {speaker_id} not found")
        self.index["speakers"][speaker_id]["name"] = new_name
        self.index["speakers"][speaker_id]["updated_at"] = datetime.now().isoformat()
        self._save_index()

    def identify(
        self, embedding: np.ndarray, threshold: float = 0.75
    ) -> tuple[str | None, str | None, float]:
        """Return (speaker_id, speaker_name, similarity) or (None, None, 0.0)."""
        if not self.index["speakers"]:
            return None, None, 0.0

        query = embedding.flatten()
        best_id, best_sim = None, -1.0

        for spk_id in self.index["speakers"]:
            avg_path = self.db_dir / f"{spk_id}_avg.npy"
            if not avg_path.exists():
                continue
            avg_emb = np.load(avg_path).flatten()
            sim = 1.0 - cosine(query, avg_emb)
            if sim > best_sim:
                best_sim = sim
                best_id = spk_id

        if best_id and best_sim >= threshold:
            return best_id, self.index["speakers"][best_id]["name"], best_sim
        return None, None, best_sim

    def list_speakers(self) -> list[dict]:
        result = []
        for spk_id, info in self.index["speakers"].items():
            result.append({"id": spk_id, **info})
        return result

    def get_speaker(self, speaker_id: str) -> dict | None:
        if speaker_id not in self.index["speakers"]:
            return None
        return {"id": speaker_id, **self.index["speakers"][speaker_id]}
