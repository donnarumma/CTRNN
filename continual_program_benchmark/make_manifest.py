#!/usr/bin/env python3
"""Hash the delivered source, frozen fixtures and results, excluding this manifest."""
from pathlib import Path
import hashlib,json
root=Path(__file__).resolve().parent
paths=sorted(p for p in root.rglob('*') if p.is_file() and p.name!='sha256_manifest.json' and '.git' not in p.parts and p.suffix not in {'.asv','.log'})
manifest={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
(root/'results').mkdir(exist_ok=True)
(root/'results/sha256_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print('Hashed',len(manifest),'source and artifact files.')
