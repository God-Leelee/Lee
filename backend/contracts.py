from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def load_config() -> dict:
    return json.loads((ROOT / 'config.json').read_text(encoding='utf-8'))


@dataclass
class Branch:
    branch_id: str
    status: str
    score: float | None = None
    score_kind: str = 'uncalibrated_risk'
    model: str = ''
    elapsed_ms: float = 0
    evidence: list[dict[str, Any]] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)
    reason: str | None = None

    def to_dict(self):
        return asdict(self)


def atomic_json(path: Path, obj: Any):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    temp.replace(path)
