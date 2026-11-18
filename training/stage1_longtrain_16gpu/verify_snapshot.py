#!/usr/bin/env python3
"""Read-only verification of the archived production source bytes (stdlib only)."""
import argparse
import hashlib
import json
from pathlib import Path


def verify(root, *, check_sources=False):
    root = Path(root).resolve()
    manifest = json.loads((root / 'snapshot.json').read_text())
    errors = []
    for relative, entry in manifest['files'].items():
        copied = root / relative
        if not copied.resolve().is_relative_to(root):
            errors.append(f'snapshot path escapes bundle: {relative}')
            continue
        targets = [('snapshot', copied)]
        if check_sources:
            targets.append(('source', Path(entry['source_path'])))
        for kind, path in targets:
            try:
                content = path.read_bytes()
                if len(content) != entry['size'] or hashlib.sha256(content).hexdigest() != entry['sha256']:
                    errors.append(f'{kind} bytes differ: {path}')
            except OSError as exc:
                errors.append(f'{kind} unavailable: {path}: {exc}')
    return {'ok': not errors, 'files': len(manifest['files']),
            'checked_sources': check_sources, 'errors': errors}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument('--check-sources', action='store_true',
                        help='also compare original CPFS files; mutable provenance may have advanced')
    args = parser.parse_args()
    result = verify(args.root, check_sources=args.check_sources)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if result['ok'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
