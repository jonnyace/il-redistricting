"""Download the public inputs listed in config/sources.toml and verify SHA-256."""
from __future__ import annotations

import hashlib
import tomllib
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCES = ROOT / "config" / "sources.toml"
RAW = ROOT / "data" / "raw"


def load_sources() -> dict:
    with open(SOURCES, "rb") as f:
        return tomllib.load(f)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(record: bool = False) -> dict[str, str]:
    """Download missing files, then check every hash. Returns {name: sha256}."""
    RAW.mkdir(parents=True, exist_ok=True)
    sources = load_sources()
    hashes = {}
    for name, src in sources.items():
        dest = RAW / src["file"]
        if not dest.exists():
            print(f"downloading {name}: {src['url']}")
            req = urllib.request.Request(src["url"], headers={"User-Agent": "il-redistricting/0.1"})
            with urllib.request.urlopen(req, timeout=600) as r, open(dest, "wb") as out:
                while chunk := r.read(1 << 20):
                    out.write(chunk)
        digest = sha256(dest)
        hashes[name] = digest
        expected = src.get("sha256", "")
        if expected and expected != digest and not record:
            raise SystemExit(
                f"SHA-256 mismatch for {name} ({dest.name}): expected {expected}, got {digest}. "
                "The upstream file changed; results would not be reproducible."
            )
        print(f"ok {name} sha256={digest}")
    if record:
        text = SOURCES.read_text()
        for name, digest in hashes.items():
            block_start = text.index(f"[{name}]")
            i = text.index('sha256 = ""', block_start) if 'sha256 = ""' in text[block_start:] else -1
            nxt = text.find("\n[", block_start + 1)
            if i != -1 and (nxt == -1 or i < nxt):
                text = text[:i] + f'sha256 = "{digest}"' + text[i + len('sha256 = ""'):]
        SOURCES.write_text(text)
    return hashes
