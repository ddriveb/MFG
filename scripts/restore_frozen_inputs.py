"""Restore hash-pinned, read-only model inputs to legacy loader paths.

These seven historical files define existing runtime candidates. They do not
contain formal r2 checkpoint/identity ledgers and do not authorize continuation.
Never overwrite a different local artifact, even if the input copy is valid.
"""
import hashlib
import json
from pathlib import Path


def main() -> None:
    root=Path(__file__).resolve().parents[1]
    source=root/'configs/frozen_inputs'
    entries=json.loads((source/'manifest.json').read_text())['files']
    prepared=[]
    for item in entries:
        rel=Path(item['path'])
        src=(source/rel).resolve()
        dst=(root/'artifacts'/rel).resolve()
        if not src.is_relative_to(source.resolve()) or not dst.is_relative_to((root/'artifacts').resolve()):
            raise ValueError('Frozen-input manifest path escapes its root')
        data=src.read_bytes()
        if hashlib.sha256(data).hexdigest()!=item['sha256']:
            raise ValueError(f'Frozen input hash mismatch: {rel}')
        if dst.exists() and dst.read_bytes()!=data:
            raise FileExistsError(f'Existing artifact differs; refusing overwrite: {dst}')
        prepared.append((dst,data))
    for dst,data in prepared:
        dst.parent.mkdir(parents=True,exist_ok=True)
        if not dst.exists():
            with dst.open('xb') as f:
                f.write(data)
    print(f'Verified/restored {len(prepared)} frozen runtime input files; no campaign started.')


if __name__=='__main__':
    main()
