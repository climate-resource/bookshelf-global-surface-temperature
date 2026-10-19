import hashlib
import json
from pathlib import Path

import pytest

SNAPSHOTS = sorted((Path(__file__).parent.parent / "inputs").glob("*/sources.json"))


@pytest.mark.parametrize("manifest", SNAPSHOTS, ids=lambda p: p.parent.name)
def test_snapshot_files_match_their_recorded_digest(manifest):
    files = json.loads(manifest.read_text())["files"]
    for name, source in files.items():
        digest = hashlib.sha256((manifest.parent / name).read_bytes()).hexdigest()
        assert digest == source["sha256"], f"{name} changed since it was downloaded"
