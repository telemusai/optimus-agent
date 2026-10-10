"""Prepare NEW gold-free caller declarations from explicit scoring questions.

The opaque hash binds local artifacts only. It does not prove actual question
presentation, ground truth, or a historical run's identity. This command never
reads or rewrites predictions and refuses an existing output path.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from bench.evidence import prepare_question_declaration, read_questions
from bench.io import atomic_write_text, check_output_path
from bench.protocol import QUESTION_PROTOCOLS, validate_question_protocol


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--bench", required=True, choices=tuple(QUESTION_PROTOCOLS))
    parser.add_argument("--qs-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True,
                        help="new gold-free JSONL path; existing files are never overwritten")
    args = parser.parse_args(argv)
    try:
        inputs = sorted(args.qs_dir.glob("qs_*.jsonl"))
        output = check_output_path(args.out, inputs)
        if output.exists():
            raise ValueError("preparation requires a new output path")
        questions, _ = read_questions(args.qs_dir)
        declarations = []
        for question in questions:
            validate_question_protocol(question, args.bench)
            declarations.append(prepare_question_declaration(question))
        atomic_write_text(output, "".join(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n"
                                         for row in declarations))
    except (OSError, ValueError, KeyError, TypeError):
        parser.exit(2, "preparation rejected: check explicit scoring questions and a new output path\n")
    print(f"[prepare] {len(declarations)} new caller declarations -> {output}")


if __name__ == "__main__":
    main()
