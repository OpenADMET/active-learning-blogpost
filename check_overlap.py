"""Check whether any compound in a query CSV appears in one or more reference CSVs.

Three comparison modes per compound × per reference file:
  - raw      : as-is SMILES string equality
  - canonical: RDKit canonical SMILES
  - inchikey : InChIKey derived from SMILES

Usage examples::

    # minimal — SMILES column defaults to "SMILES" for all files
    python check_overlap.py \\
        --query data/pxr-challenge_TEST_BLINDED.csv \\
        --ref data/pxr_challenge_train.csv \\
        --ref data/chembl.csv

    # override SMILES column per reference using path:col syntax
    python check_overlap.py \\
        --query data/pxr-challenge_TEST_BLINDED.csv \\
        --query-smiles-col SMILES \\
        --ref data/pxr_challenge_train.csv \\
        --ref data/chembl.csv \\
        --ref "data/asap_potency.csv:CXSMILES"
"""

import csv  # noqa: E402
import os
from collections import defaultdict  # noqa: E402

import click  # noqa: E402
from openadmet.toolkit.chemoinformatics.rdkit_funcs import (  # noqa: E402
    canonical_smiles,
    smiles_to_inchikey,
)
from rdkit import RDLogger  # noqa: E402
from rich.console import Console  # noqa: E402
from rich.table import Table  # noqa: E402
from rich import box  # noqa: E402

console = Console(highlight=False)

RDLogger.DisableLog("rdApp.*")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse_ref(ref_str: str) -> tuple[str, str]:
    """Parse a ``path[:smiles_col]`` CLI token into a (path, column) pair.

    Parameters
    ----------
    ref_str : str
        Either a bare file path (e.g. ``"data/train.csv"``) or a
        ``path:col`` pair (e.g. ``"data/asap.csv:CXSMILES"``).
        The colon is split on the *last* occurrence so that Windows
        absolute paths (``C:\\...``) are handled correctly.

    Returns
    -------
    tuple[str, str]
        ``(path, smiles_col)`` where *smiles_col* defaults to ``"SMILES"``
        when no colon is present.
    """
    if ":" in ref_str:
        path, col = ref_str.rsplit(":", 1)
        return path.strip(), col.strip()
    return ref_str.strip(), "SMILES"


def load_smiles_col(path: str, smiles_col: str) -> list[str]:
    """Read the SMILES column from a CSV file, skipping blank entries.

    Parameters
    ----------
    path : str
        Filesystem path to the CSV file.
    smiles_col : str
        Name of the column that contains SMILES strings.

    Returns
    -------
    list[str]
        Non-empty SMILES strings in file order.
    """
    rows: list[str] = []
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            smi = row.get(smiles_col, "").strip()
            if smi:
                rows.append(smi)
    return rows


def build_lookup(smiles_list: list[str]) -> dict[str, dict[str, list[int]]]:
    """Build three lookup dicts from a list of SMILES strings.

    SMILES that cannot be parsed by RDKit produce ``None`` from the
    toolkit functions and are silently excluded from the canonical and
    InChIKey dicts (they remain in the raw dict).

    Parameters
    ----------
    smiles_list : list[str]
        Input SMILES, as loaded from a CSV column (1-based row order).

    Returns
    -------
    dict[str, dict[str, list[int]]]
        Mapping with three keys, each holding a dict from the SMILES
        representation to a list of 1-based data row numbers where it
        appears:

        ``"raw"``
            As-is SMILES strings (exact string match).
        ``"canonical"``
            RDKit canonical SMILES (normalises notation differences).
        ``"inchikey"``
            IUPAC InChIKey (representation-independent identity check).
    """
    raw: dict[str, list[int]] = {}
    can: dict[str, list[int]] = {}
    ik: dict[str, list[int]] = {}
    for row_num, smi in enumerate(smiles_list, start=1):
        raw.setdefault(smi, []).append(row_num)
        c = canonical_smiles(smi)
        if c:
            can.setdefault(c, []).append(row_num)
        k = smiles_to_inchikey(smi)
        if k:
            ik.setdefault(k, []).append(row_num)
    return {"raw": raw, "canonical": can, "inchikey": ik}


def _check(label: str, n_hits: int) -> None:
    """Print a single PASS / FAIL assertion line to stdout.

    Parameters
    ----------
    label : str
        Human-readable description of the (reference file, method) pair
        being reported (e.g. ``"vs train.csv  [canonical]"``).
    n_hits : int
        Number of overlapping compounds detected for this combination.
        Zero → PASS; any positive value → FAIL.
    """
    if n_hits == 0:
        console.print(f"  [bold green]✓[/]  {label:<55}  [green]PASS — no overlap[/]")
    else:
        console.print(
            f"  [bold red]✗[/]  {label:<55}  [red]FAIL — {n_hits} overlap(s) found[/]"
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@click.command()
@click.option(
    "--query",
    "-q",
    required=True,
    type=click.Path(exists=True, dir_okay=False),
    help="Path to the query CSV file.",
)
@click.option(
    "--query-smiles-col",
    default="SMILES",
    show_default=True,
    help="SMILES column name in the query CSV.",
)
@click.option(
    "--ref",
    "-r",
    "refs",
    required=True,
    multiple=True,
    help=(
        "Reference CSV file. Repeat for multiple files. "
        "Optionally append the SMILES column name after a colon: "
        "path/to/file.csv:SMILES_COL  (default column: SMILES)."
    ),
)
def main(
    query: str,
    query_smiles_col: str,
    refs: tuple[str, ...],
) -> None:
    """Check a query CSV for compound overlap against one or more reference CSVs.

    Comparisons are made three ways per reference file: raw SMILES string,
    RDKit canonical SMILES, and InChIKey.

    Parameters
    ----------
    query : str
        Path to the query CSV file (validated by Click to exist).
    query_smiles_col : str
        Name of the SMILES column in the query CSV.
    refs : tuple[str, ...]
        One or more ``path[:smiles_col]`` tokens for reference files.
        Parsed by :func:`_parse_ref`; SMILES column defaults to
        ``"SMILES"`` when the colon suffix is absent.
    """
    reference_files = [
        {"path": p, "smiles_col": col, "label": os.path.basename(p)}
        for p, col in (_parse_ref(r) for r in refs)
    ]

    # ---- phase 1: load query set ----
    console.rule("[bold cyan]Load query set[/]")
    console.print(f"  Reading [bold]{query}[/] (col: [italic]{query_smiles_col}[/]) ...")
    query_smiles: list[str] = []
    query_row_nums: list[int] = []
    with open(query, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row_num, row in enumerate(reader, start=1):
            smi = row.get(query_smiles_col, "").strip()
            if smi:
                query_smiles.append(smi)
                query_row_nums.append(row_num)
    console.print(f"  Loaded [bold]{len(query_smiles)}[/] compounds.\n")

    # ---- phase 2: build reference lookups ----
    console.rule("[bold cyan]Build reference lookup sets (raw / canonical / InChIKey)[/]")
    ref_lookups: dict[str, dict[str, dict[str, list[int]]]] = {}
    for ref in reference_files:
        console.print(f"  Processing [bold]{ref['label']}[/] (col: [italic]{ref['smiles_col']}[/]) ...")
        smiles_list = load_smiles_col(ref["path"], ref["smiles_col"])
        lkp = build_lookup(smiles_list)
        ref_lookups[ref["label"]] = lkp
        console.print(
            f"    [dim]{len(smiles_list)} SMILES loaded[/]  →  "
            f"{len(lkp['raw'])} raw | "
            f"{len(lkp['canonical'])} canonical | "
            f"{len(lkp['inchikey'])} InChIKey entries"
        )
    console.print()

    # ---- phase 3: compute query representations ----
    console.rule("[bold cyan]Compute canonical SMILES and InChIKeys for query compounds[/]")
    console.print(f"  Canonicalizing [bold]{len(query_smiles)}[/] query SMILES ...")
    query_can: list[str | None] = [canonical_smiles(smi) for smi in query_smiles]
    console.print(f"  Computing InChIKeys for [bold]{len(query_smiles)}[/] query SMILES ...")
    query_ik: list[str | None] = [smiles_to_inchikey(smi) for smi in query_smiles]

    n_failed_can = sum(1 for c in query_can if c is None)
    n_failed_ik = sum(1 for k in query_ik if k is None)
    if n_failed_can:
        console.print(
            f"  [yellow]WARNING:[/] {n_failed_can} query SMILES failed canonicalization "
            "(will be excluded from canonical / InChIKey checks)."
        )
    if n_failed_ik:
        console.print(f"  [yellow]WARNING:[/] {n_failed_ik} query SMILES failed InChIKey conversion.")
    if not n_failed_can and not n_failed_ik:
        console.print("  [green]All query representations computed successfully.[/]")
    console.print()

    # ---- phase 4: run overlap checks ----
    console.rule("[bold cyan]Check query compounds against each reference file[/]")

    # hits: (query_row, raw_smiles, ref_label, ref_rows_str, method)
    hits: list[tuple[int, str, str, str, str]] = []
    for i, smi_raw in enumerate(query_smiles):
        smi_can = query_can[i]
        smi_ik = query_ik[i]
        q_row = query_row_nums[i]
        for ref in reference_files:
            lkp = ref_lookups[ref["label"]]
            if smi_raw in lkp["raw"]:
                ref_rows = ", ".join(str(r) for r in lkp["raw"][smi_raw])
                hits.append((q_row, smi_raw, ref["label"], ref_rows, "raw"))
            if smi_can and smi_can in lkp["canonical"]:
                ref_rows = ", ".join(str(r) for r in lkp["canonical"][smi_can])
                hits.append((q_row, smi_raw, ref["label"], ref_rows, "canonical"))
            if smi_ik and smi_ik in lkp["inchikey"]:
                ref_rows = ", ".join(str(r) for r in lkp["inchikey"][smi_ik])
                hits.append((q_row, smi_raw, ref["label"], ref_rows, "inchikey"))

    unique_hits = sorted(set(hits))

    for ref in reference_files:
        console.print(f"\n  Reference: [bold]{ref['label']}[/]")
        for method in ("raw", "canonical", "inchikey"):
            n = sum(1 for h in unique_hits if h[2] == ref["label"] and h[4] == method)
            _check(f"  vs {ref['label']}  \\[{method}]", n)
    console.print()

    # ---- phase 5: summary report ----
    console.rule("[bold cyan]Summary report[/]")

    if not unique_hits:
        console.print("\n  [bold green]✓  OVERALL PASS — no overlap detected by any method.[/]\n")
    else:
        # Group: ref_label → (query_row, smiles) → [(ref_rows_str, method)]
        by_ref: dict[str, dict[tuple[int, str], list[tuple[str, str]]]] = defaultdict(
            lambda: defaultdict(list)
        )
        for q_row, smi_hit, ref_label, ref_rows_str, method in unique_hits:
            by_ref[ref_label][(q_row, smi_hit)].append((ref_rows_str, method))

        n_compounds = len({h[1] for h in unique_hits})
        console.print(
            f"\n  [bold red]✗  OVERALL FAIL — {len(unique_hits)} hit(s) across "
            f"{n_compounds} unique compound(s).[/]\n"
        )
        tbl = Table(box=box.SIMPLE, show_header=True, header_style="bold magenta")
        tbl.add_column("Reference", style="yellow", no_wrap=True)
        tbl.add_column("Query row", style="dim", no_wrap=True, justify="right")
        tbl.add_column("SMILES")
        tbl.add_column("Ref row(s)", style="dim", no_wrap=True, justify="right")
        tbl.add_column("Method", style="cyan", no_wrap=True)
        for ref_label, compound_map in sorted(by_ref.items()):
            first_ref = True
            for (q_row, smi_hit), rows_methods in sorted(compound_map.items()):
                first_smi = True
                for ref_rows_str, method in rows_methods:
                    tbl.add_row(
                        ref_label if first_ref and first_smi else "",
                        str(q_row) if first_smi else "",
                        smi_hit[:72] if first_smi else "",
                        ref_rows_str,
                        method,
                    )
                    first_smi = False
                    first_ref = False
        console.print(tbl)

    console.rule()


if __name__ == "__main__":
    main()
