#!/usr/bin/env python3
"""Hash local sources and exported artifacts without executing the experiment."""
from pathlib import Path
import hashlib
import json

root = Path(__file__).resolve().parent
paths = sorted(p for p in root.iterdir() if p.is_file() and p.suffix in {'.m', '.md', '.py', '.mat', '.csv'})
result_dir = root / 'results'
if result_dir.exists():
    paths += sorted(p for p in result_dir.iterdir() if p.is_file() and p.name != 'sha256_manifest.json')
manifest = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
result_dir.mkdir(exist_ok=True)
(result_dir / 'sha256_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
print(f'Hashed {len(manifest)} files under {root}')
