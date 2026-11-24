"""Run the CRESCENT pipeline stage by stage, in one shared namespace (as the original notebook did).

Examples
--------
Run everything:
    python run_crescent.py

Run only some stages (by number, see `--list`):
    python run_crescent.py --stages 1-4        # config to training
    python run_crescent.py --list

Settings are read from environment variables (see docs/model.md), for example:
    CRESCENT_PREPARED_DATA_DIR=/path/to/splits_229950_one_month_cap \
    EHR_CRESCENT_FINAL_OUTPUT=runs/CRESCENT_Final \
    python run_crescent.py
"""
import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
STAGES = [line.strip() for line in (ROOT / "crescent" / "STAGES.txt").read_text().splitlines() if line.strip()]


def parse_stages(spec: str) -> list[int]:
    chosen = set()
    for part in spec.split(","):
        if "-" in part:
            a, b = part.split("-")
            chosen.update(range(int(a), int(b) + 1))
        else:
            chosen.add(int(part))
    bad = [s for s in chosen if not 1 <= s <= len(STAGES)]
    if bad:
        raise SystemExit(f"Unknown stage numbers: {bad}. Use --list.")
    return sorted(chosen)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stages", default=f"1-{len(STAGES)}", help="e.g. 1-4 or 1,2,3,6")
    parser.add_argument("--list", action="store_true", help="list the stages and exit")
    args = parser.parse_args()

    if args.list:
        for i, rel in enumerate(STAGES, 1):
            print(f"{i}. {rel}")
        return

    selected = parse_stages(args.stages)
    if selected[0] != 1:
        print("Note: later stages need objects created by earlier ones; include stage 1 onwards.", file=sys.stderr)

    namespace = {"__name__": "__main__", "__file__": str(ROOT / "run_crescent.py")}
    for i in selected:
        path = ROOT / STAGES[i - 1]
        print(f"\n=== Stage {i}: {STAGES[i - 1]} ===", flush=True)
        start = time.time()
        exec(compile(path.read_text(encoding="utf-8"), str(path), "exec"), namespace)
        print(f"=== Stage {i} finished in {time.time() - start:.0f}s ===", flush=True)


if __name__ == "__main__":
    main()
