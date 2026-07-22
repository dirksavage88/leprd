from __future__ import annotations

import argparse
import dataclasses
from pathlib import Path
import sys

from .analysis import generate_review
from .config import config_from_log, load_config
from .precision_landing import generate_pl_report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ekf2-review", description="Generate PX4 EKF2 replay review PDFs")
    sub = parser.add_subparsers(dest="command", required=True)

    gen = sub.add_parser("generate", help="generate plots, summary, and PDF")
    gen.add_argument("log", nargs="?", type=Path, help="path to .ulg flight log")
    gen.add_argument(
        "--config",
        type=Path,
        default=None,
        help="optional review TOML config for non-default title/output/settings",
    )
    gen.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="output directory (default: reports/<log-stem>-review)",
    )
    gen.add_argument(
        "--title",
        default=None,
        help="report title (default: PX4 EKF2 Review: <log-stem>)",
    )
    gen.add_argument(
        "--divergence-threshold",
        type=float,
        default=None,
        metavar="M",
        help="baro-SF11 divergence threshold in metres for propwash detection (overrides config)",
    )
    gen.add_argument(
        "--no-mode-shading",
        action="store_true",
        help="disable flight-mode background shading on time-series plots",
    )

    pl = sub.add_parser("pl-analysis", help="precision-landing failure analysis (optical flow, land detection, mode timeline)")
    pl.add_argument("--log", required=True, type=Path, help="path to .ulg flight log")
    pl.add_argument("--output-dir", type=Path, default=None,
                    help="output directory (default: <log_stem>_pl_analysis/ next to log)")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "generate":
        if args.config is not None and args.log is not None:
            print("error: pass either a ULog path or --config, not both", file=sys.stderr)
            return 2
        if args.config is not None:
            config = load_config(args.config)
        elif args.log is not None:
            config = config_from_log(args.log, output_dir=args.output_dir, title=args.title)
        else:
            print("error: pass a .ulg path, or use --config PATH", file=sys.stderr)
            return 2
        if args.divergence_threshold is not None:
            config = dataclasses.replace(config, divergence_threshold_m=args.divergence_threshold)
        if args.no_mode_shading:
            config = dataclasses.replace(config, shade_flight_modes=False)
        artifacts = generate_review(config)
        print(f"output_dir: {artifacts.output_dir}")
        print(f"summary: {artifacts.summary_path}")
        print(f"tex: {artifacts.tex_path}")
        print(f"pdf: {artifacts.pdf_path}")
        for name, path in artifacts.figures.items():
            print(f"figure:{name}: {path}")
        return 0

    if args.command == "pl-analysis":
        log_path = args.log.expanduser().resolve()
        if not log_path.exists():
            print(f"error: log not found: {log_path}", file=sys.stderr)
            return 1
        output_dir = args.output_dir or log_path.parent / f"{log_path.stem}_pl_analysis"
        artifacts = generate_pl_report(log_path, output_dir)
        print(f"output_dir: {artifacts.output_dir}")
        print(f"summary: {artifacts.summary_path}")
        if artifacts.tex_path:
            print(f"tex: {artifacts.tex_path}")
        if artifacts.pdf_path:
            print(f"pdf: {artifacts.pdf_path}")
        for name, path in artifacts.figures.items():
            print(f"figure:{name}: {path}")
        return 0

    parser.print_help(sys.stderr)
    return 2
