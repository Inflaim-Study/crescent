#!/usr/bin/env python3
"""Run any CRESCENT comparator's stage files sequentially in one shared namespace.

Mirrors `run_crescent.py` so every comparator under `comparators/<name>/` runs
with the exact same stage-execution contract.

Examples
--------
    python run_comparator.py --list
    python run_comparator.py cox --list
    python run_comparator.py survtrace --stages 1-4
    python run_comparator.py delphi_2m
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

COMPARATORS_DIR = Path(__file__).resolve().parent / "comparators"


def available_comparators() -> list[str]:
    return sorted(
        p.name
        for p in COMPARATORS_DIR.iterdir()
        if p.is_dir() and (p / "STAGES.txt").is_file()
    )


def load_stages(comparator: str) -> list[Path]:
    comp_dir = COMPARATORS_DIR / comparator
    stages_file = comp_dir / "STAGES.txt"
    if not stages_file.is_file():
        raise FileNotFoundError(
            f"No STAGES.txt found for comparator '{comparator}' in {comp_dir}"
        )
    lines = [
        ln.strip()
        for ln in stages_file.read_text(encoding="utf-8").splitlines()
        if ln.strip() and not ln.strip().startswith("#")
    ]
    return [comp_dir / ln for ln in lines]


def parse_stage_spec(spec: str, total: int) -> list[int]:
    selected: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-", 1)
            selected.extend(range(int(lo), int(hi) + 1))
        else:
            selected.append(int(part))
    for idx in selected:
        if idx < 1 or idx > total:
            raise ValueError(f"Stage index {idx} out of range 1..{total}")
    return selected


def main() -> None:
    p = argparse.ArgumentParser(
        description="Run CRESCENT comparator stages in a shared namespace."
    )
    p.add_argument(
        "comparator",
        nargs="?",
        help=f"Comparator name ({', '.join(available_comparators())}).",
    )
    p.add_argument(
        "--list",
        action="store_true",
        help="List available comparators or stages for a comparator.",
    )
    p.add_argument(
        "--stages",
        default=None,
        help="Stage numbers/ranges to run (1-indexed, e.g. '1-4' or '1,2,6').",
    )
    args = p.parse_args()

    if not args.comparator:
        print("Available comparators:")
        for name in available_comparators():
            print(f"  - {name}")
        return

    stages = load_stages(args.comparator)
    if args.list:
        print(f"Stages for '{args.comparator}':")
        for i, st in enumerate(stages, 1):
            print(f"  {i}. {st.relative_to(COMPARATORS_DIR.parent)}")
        return

    indices = (
        parse_stage_spec(args.stages, len(stages))
        if args.stages
        else list(range(1, len(stages) + 1))
    )
    if indices and indices[0] != 1:
        print(
            f"[WARN] --stages starts at {indices[0]} (not 1). Because stages share one "
            "namespace, skipping earlier stages may fail unless required objects are already defined.",
            file=sys.stderr,
            flush=True,
        )

    namespace = {"__name__": "__main__"}
    for idx in indices:
        stage_path = stages[idx - 1]
        print(
            f"=== [{args.comparator}] Running stage {idx}/{len(stages)}: {stage_path.name} ===",
            flush=True,
        )
        code = compile(stage_path.read_text(encoding="utf-8"), str(stage_path), "exec")
        exec(code, namespace)


if __name__ == "__main__":
    main()
