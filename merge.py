#!/usr/bin/env python
"""Combine per-job outputs from distributed HPC runs into results/all_runs.pkl.

After individual (strategy, seed) jobs have been submitted and completed via:

    python run.py --strategy <STRATEGY> --seed <SEED>

run this script to assemble the final checkpoint consumed by ``analysis.py``:

    python merge.py

The script reports which (strategy, seed) pairs are present and which are
missing before writing.  Missing pairs are omitted from the output rather than
causing an error, so partial results can still be analysed.
"""

import pickle
import re
from pathlib import Path

from src.helpers import STRATEGIES


def main() -> None:
    """Merge per-job pickle files from distributed HPC runs into a single checkpoint.

    Scans ``results/`` for all ``run_<strategy>_seed<N>.pkl`` files produced by
    individual ``python run.py --strategy S --seed N`` jobs, prints a coverage
    report showing which (strategy, seed) pairs are present vs. missing, and
    writes a merged ``results/all_runs.pkl`` containing all results alongside the
    original setup data.

    Missing pairs are omitted from the merged output rather than raising an error,
    so partial results can still be passed to ``analysis.py``.
    """
    setup_path = Path("results/setup.pkl")
    if not setup_path.exists():
        raise FileNotFoundError(
            "results/setup.pkl not found. Run `python run.py --setup-only` first."
        )

    with open(setup_path, "rb") as fh:
        setup = pickle.load(fh)
    print(f"Loaded setup from {setup_path}.")
    seeds: list[int] = setup["config"].seeds

    # ── Discover per-job pickles ───────────────────────────────────────────────
    # Discover all completed per-job pickle files matching the expected naming convention
    pattern = re.compile(r"^run_(.+)_seed(\d+)\.pkl$")
    job_results: dict[tuple[str, int], dict] = {}

    for pkl_file in sorted(Path("results").glob("run_*.pkl")):
        m = pattern.match(pkl_file.name)
        if not m:
            continue
        strategy, seed = m.group(1), int(m.group(2))
        with open(pkl_file, "rb") as fh:
            job_data = pickle.load(fh)
        job_results[(strategy, seed)] = job_data["result"]
        print(f"  Loaded {pkl_file.name}")

    # ── Coverage report ────────────────────────────────────────────────────────
    print("\nCoverage summary:")
    all_present = True
    for strategy in STRATEGIES:
        found = [s for s in seeds if (strategy, s) in job_results]
        missing = [s for s in seeds if (strategy, s) not in job_results]
        suffix = f"  [MISSING seeds: {missing}]" if missing else ""
        print(f"  {strategy}: {len(found)}/{len(seeds)} seeds{suffix}")
        if missing:
            all_present = False

    if not all_present:
        print(
            "\nWARNING: Some (strategy, seed) pairs are missing. "
            "They will be omitted from the merged output."
        )

    # ── Build all_runs ─────────────────────────────────────────────────────────
    # Only include (strategy, seed) pairs that were successfully loaded above
    all_runs: dict[str, list] = {}
    for strategy in STRATEGIES:
        runs = [
            {"seed": seed, **job_results[(strategy, seed)]}
            for seed in seeds
            if (strategy, seed) in job_results
        ]
        if runs:
            all_runs[strategy] = runs

    # ── Write merged pickle ────────────────────────────────────────────────────
    out_path = Path("results/all_runs.pkl")
    with open(out_path, "wb") as fh:
        pickle.dump(
            {**setup, "all_runs": all_runs}, fh, protocol=pickle.HIGHEST_PROTOCOL
        )

    total = sum(len(v) for v in all_runs.values())
    print(f"\nMerged {total} runs → {out_path}")


if __name__ == "__main__":
    main()
