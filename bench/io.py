"""Small local-file safety helpers. No filesystem access at import time."""
from __future__ import annotations

import os
from pathlib import Path
import tempfile


def check_output_path(output, inputs) -> Path:
    output = Path(output)
    for source in inputs:
        source = Path(source)
        if output.resolve() == source.resolve() or (
            output.exists() and source.exists() and output.samefile(source)
        ):
            raise ValueError("output path aliases an input file")
    return output


def atomic_write_text(path, text: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix=path.name + ".", suffix=".tmp", delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(text)
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
