"""Persistent voiceprint database for speaker identification."""

import json
import os
import tempfile
import threading
import uuid
import numpy as np
from pathlib import Path
from datetime import datetime
from scipy.spatial.distance import cosine


class VoiceprintDB:
    """Thread-safe on-disk speaker database.

    All mutations hold ``_lock`` so two concurrent enroll / update / delete
    calls don't race on ``index.json`` (previously two simultaneous enrolls
    could silently drop one another via overlapping read-modify-write).

    ``index.json`` and the ``.npy`` files are written atomically via
    ``os.replace`` so a crash mid-write can't corrupt the index and brick
    the next startup.
    """

    def __init__(self, db_dir: str = "/data/voiceprints"):
        self.db_dir = Path(db_dir)
        self.db_dir.mkdir(parents=True, exist_ok=True)
        self.index_file = self.db_dir / "index.json"
        self._lock = threading.Lock()
        self.index = self._load_index()

    # ------------------------------------------------------------------
    # low-level atomic writers (not locked — callers must hold _lock)
    # ------------------------------------------------------------------
    def _load_index(self) -> dict:
        if self.index_file.exists():
            return json.loads(self.index_file.read_text(encoding="utf-8"))
        return {"speakers": {}}

    def _save_index(self):
        """Atomic write: json.dump to a sibling temp file, then os.replace."""
        fd, tmp = tempfile.mkstemp(dir=self.db_dir, prefix=".idx.", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(self.index, f, ensure_ascii=False, indent=2)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, self.index_file)
        except Exception:
            try:
                os.unlink(tmp)
            except FileNotFoundError:
                pass
            raise

    def _save_npy_atomic(self, path: Path, array: np.ndarray):
        """np.save via tempfile + os.replace so readers never see a torn file."""
        fd, tmp = tempfile.mkstemp(dir=self.db_dir, prefix=".npy.", suffix=".tmp")
        os.close(fd)
        try:
            np.save(tmp, array, allow_pickle=False)
            # np.save appends .npy to extensionless paths; normalize.
            written = tmp + ".npy" if not tmp.endswith(".npy") else tmp
            if written != tmp and os.path.exists(written):
                os.replace(written, path)
            else:
                os.replace(tmp, path)
        except Exception:
            for p in (tmp, tmp + ".npy"):
                try:
                    os.unlink(p)
                except FileNotFoundError:
                    pass
            raise

    # ------------------------------------------------------------------
    # public API — every mutation holds the lock
    # ------------------------------------------------------------------
    def add_speaker(self, name: str, embedding: np.ndarray) -> str:
        """Register a new speaker with a name and initial embedding."""
        speaker_id = f"spk_{uuid.uuid4().hex[:8]}"
        emb = embedding.reshape(1, -1)
        with self._lock:
            self._save_npy_atomic(self.db_dir / f"{speaker_id}_samples.npy", emb)
            self._save_npy_atomic(
                self.db_dir / f"{speaker_id}_avg.npy", emb.mean(axis=0)
            )
            now = datetime.now().isoformat()
            self.index["speakers"][speaker_id] = {
                "name": name,
                "sample_count": 1,
                "created_at": now,
                "updated_at": now,
            }
            self._save_index()
        return speaker_id

    def update_speaker(
        self, speaker_id: str, new_embedding: np.ndarray, name: str | None = None
    ):
        """Add a new embedding sample and recompute the average."""
        with self._lock:
            if speaker_id not in self.index["speakers"]:
                raise ValueError(f"Speaker {speaker_id} not found")

            samples_path = self.db_dir / f"{speaker_id}_samples.npy"
            samples = np.load(samples_path, allow_pickle=False)
            new_emb = new_embedding.reshape(1, -1)
            samples = np.vstack([samples, new_emb])
            self._save_npy_atomic(samples_path, samples)
            self._save_npy_atomic(
                self.db_dir / f"{speaker_id}_avg.npy", samples.mean(axis=0)
            )

            info = self.index["speakers"][speaker_id]
            info["sample_count"] = len(samples)
            info["updated_at"] = datetime.now().isoformat()
            if name is not None:
                info["name"] = name
            self._save_index()

    def delete_speaker(self, speaker_id: str):
        with self._lock:
            if speaker_id not in self.index["speakers"]:
                raise ValueError(f"Speaker {speaker_id} not found")
            for suffix in ("_samples.npy", "_avg.npy"):
                p = self.db_dir / f"{speaker_id}{suffix}"
                p.unlink(missing_ok=True)
            del self.index["speakers"][speaker_id]
            self._save_index()

    def rename_speaker(self, speaker_id: str, new_name: str):
        with self._lock:
            if speaker_id not in self.index["speakers"]:
                raise ValueError(f"Speaker {speaker_id} not found")
            self.index["speakers"][speaker_id]["name"] = new_name
            self.index["speakers"][speaker_id][
                "updated_at"
            ] = datetime.now().isoformat()
            self._save_index()

    # ------------------------------------------------------------------
    # read-only paths — cheap, lock not strictly required
    # ------------------------------------------------------------------
    def identify(
        self, embedding: np.ndarray, threshold: float = 0.75
    ) -> tuple[str | None, str | None, float]:
        """Return (speaker_id, speaker_name, similarity) or (None, None, 0.0)."""
        # Snapshot the speaker id list under the lock so we don't iterate
        # a mutating dict while another thread is adding / removing entries.
        with self._lock:
            if not self.index["speakers"]:
                return None, None, 0.0
            speakers_snapshot = list(self.index["speakers"].items())

        query = embedding.flatten()
        best_id, best_sim = None, -1.0

        for spk_id, _info in speakers_snapshot:
            avg_path = self.db_dir / f"{spk_id}_avg.npy"
            if not avg_path.exists():
                continue
            avg_emb = np.load(avg_path, allow_pickle=False).flatten()
            sim = 1.0 - cosine(query, avg_emb)
            if sim > best_sim:
                best_sim = sim
                best_id = spk_id

        if best_id and best_sim >= threshold:
            with self._lock:
                info = self.index["speakers"].get(best_id)
            if info is not None:
                return best_id, info["name"], best_sim
        return None, None, best_sim

    def list_speakers(self) -> list[dict]:
        with self._lock:
            return [
                {"id": spk_id, **info}
                for spk_id, info in self.index["speakers"].items()
            ]

    def get_speaker(self, speaker_id: str) -> dict | None:
        with self._lock:
            if speaker_id not in self.index["speakers"]:
                return None
            return {"id": speaker_id, **self.index["speakers"][speaker_id]}
