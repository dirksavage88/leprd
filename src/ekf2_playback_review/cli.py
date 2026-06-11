from __future__ import annotations

import argparse
from pathlib import Path
import sys

from .analysis import generate_review
from .config import load_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ekf2-review", description="Generate PX4 EKF2 replay review reports")
    sub = parser.add_subparsers(dest="command", required=True)

    gen = sub.add_parser("generate", help="generate plots, summary, and optional PDF")
    gen.add_argument("--config", required=True, type=Path, help="review TOML config")
    pdf = gen.add_mutually_exclusive_group()
    pdf.add_argument("--compile-pdf", action="store_true", default=None, help="force LaTeX PDF compile")
    pdf.add_argument("--no-compile-pdf", action="store_false", dest="compile_pdf", help="skip LaTeX PDF compile")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "generate":
        config = load_config(args.config)
        artifacts = generate_review(config, compile_pdf=args.compile_pdf)
        print(f"output_dir: {artifacts.output_dir}")
        print(f"summary: {artifacts.summary_path}")
        print(f"tex: {artifacts.tex_path}")
        if artifacts.pdf_path:
            print(f"pdf: {artifacts.pdf_path}")
        else:
            print("pdf: not generated")
        for name, path in artifacts.figures.items():
            print(f"figure:{name}: {path}")
        return 0

    parser.print_help(sys.stderr)
    return 2
