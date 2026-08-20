"""Translate a CircuitNet ann-file CSV into a FedCircuitNet metadata CSV.

The upstream ``code_examples/CircuitNet`` pipeline consumes two-column
CSVs of the form::

    feature/<stem>.npy,label/<stem>.npy

FedCircuitNet's :class:`datasets.drc_dataset.DRCDataset` and
:class:`partitioning.iid.IIDPartitioner` instead expect a DataFrame with
at least a ``filename`` column (the on-disk stem without extension) plus
the design-configuration columns parsed out of that stem.

This script performs that translation.  It auto-detects the filename
schema:

* **CircuitNet-N28** stems match
  ``[<sample_id>-]<design>-<#macros>-c<clock>-u<util>-m<mp>-p<pm>-f<fi>``
  and produce columns:
      ``filename, sample_id, design_name, macro_count, clock,
      utilization, macro_placement, power_mesh, filler_insertion``.

* **CircuitNet-N14** stems match
  ``<design>_freq_<freq>_mp_<mp>_fpu_<util>_fpa_<ar>_p_<pm>_fi_<fi>``
  (parser ported verbatim from
  ``code_examples/CircuitNet/dataset_description_and_partitioning/feature_analysis_N14.ipynb``)
  and produce columns:
      ``filename, design_name, freq_mhz, macro_placement, utilization,
      aspect_ratio, power_mesh, filler_insertion``.

Usage
-----
    python create_metadata_files.py \\
        --input=path/to/original.csv \\
        --output=path/to/new.csv
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import sys
from typing import Dict, List, Optional


# ---------------------------------------------------------------------------
# Filename parsers
# ---------------------------------------------------------------------------

_N28_RE = re.compile(
    r"^(?:(?P<sample_id>\d+)-)?"
    r"(?P<design_name>.+?)-(?P<macro_count>\d+)"
    r"-c(?P<clock>\d+)"
    r"-u(?P<utilization>[\d.]+)"
    r"-m(?P<macro_placement>\d+)"
    r"-p(?P<power_mesh>\d+)"
    r"-f(?P<filler_insertion>\d+)$"
)

N28_COLUMNS: List[str] = [
    "filename",
    "sample_id",
    "design_name",
    "macro_count",
    "clock",
    "utilization",
    "macro_placement",
    "power_mesh",
    "filler_insertion",
]

N14_COLUMNS: List[str] = [
    "filename",
    "design_name",
    "freq_mhz",
    "macro_placement",
    "utilization",
    "aspect_ratio",
    "power_mesh",
    "filler_insertion",
]


def parse_n28(stem: str) -> Optional[Dict[str, str]]:
    m = _N28_RE.match(stem)
    if m is None:
        return None
    d = m.groupdict()
    d["filename"] = stem
    if d["sample_id"] is None:
        d["sample_id"] = ""
    return d


def parse_n14(stem: str) -> Optional[Dict[str, str]]:
    """Port of ``parse_sample_name`` from feature_analysis_N14.ipynb."""
    if "_freq_" not in stem:
        return None
    design_name, rest = stem.split("_freq_", 1)
    if not design_name:
        return None
    tokens = rest.split("_")
    # Layout: [freq, 'mp', mp, 'fpu', util, 'fpa', ar, 'p', pm, 'fi', fi]
    if len(tokens) != 11:
        return None
    for expected, idx in [("mp", 1), ("fpu", 3), ("fpa", 5), ("p", 7), ("fi", 9)]:
        if tokens[idx] != expected:
            return None
    return {
        "filename": stem,
        "design_name": design_name,
        "freq_mhz": tokens[0],
        "macro_placement": tokens[2],
        "utilization": tokens[4],
        "aspect_ratio": tokens[6],
        "power_mesh": tokens[8],
        "filler_insertion": tokens[10],
    }


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def _stem(path: str) -> str:
    """Return the basename of *path* with a trailing ``.npy`` removed."""
    base = os.path.basename(path.strip())
    if base.endswith(".npy"):
        base = base[: -len(".npy")]
    return base


def _detect_schema(sample_stems: List[str]) -> str:
    """Return ``"n14"`` if any stem carries the N14 ``_freq_`` marker,
    otherwise ``"n28"``.  Called on the first handful of rows only."""
    for s in sample_stems:
        if "_freq_" in s:
            return "n14"
    return "n28"


def translate(input_path: str, output_path: str) -> None:
    with open(input_path, newline="") as fh:
        rows = list(csv.reader(fh))
    if not rows:
        raise SystemExit(f"{input_path}: no rows to translate.")

    stems: List[str] = []
    for lineno, row in enumerate(rows, start=1):
        if not row or not row[0].strip():
            continue
        feature_stem = _stem(row[0])
        if len(row) >= 2 and row[1].strip():
            label_stem = _stem(row[1])
            if label_stem != feature_stem:
                print(
                    f"[warn] line {lineno}: feature/label stem mismatch "
                    f"({feature_stem!r} vs {label_stem!r}) -- using feature stem.",
                    file=sys.stderr,
                )
        stems.append(feature_stem)

    if not stems:
        raise SystemExit(f"{input_path}: no usable rows after filtering.")

    schema = _detect_schema(stems[: min(len(stems), 50)])
    parser = parse_n28 if schema == "n28" else parse_n14
    columns = N28_COLUMNS if schema == "n28" else N14_COLUMNS

    parsed: List[Dict[str, str]] = []
    failures = 0
    for stem in stems:
        record = parser(stem)
        if record is None:
            failures += 1
            print(f"[warn] cannot parse ({schema}): {stem}", file=sys.stderr)
            continue
        parsed.append(record)

    if not parsed:
        raise SystemExit(
            f"{input_path}: no rows parsed under schema {schema!r}."
        )

    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
    with open(output_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns)
        writer.writeheader()
        writer.writerows(parsed)

    print(
        f"[ok] {input_path} -> {output_path} "
        f"(schema={schema}, rows={len(parsed)}, failures={failures})"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--input", required=True, help="Two-column CircuitNet ann CSV.")
    ap.add_argument(
        "--output",
        required=True,
        help="Destination CSV in the FedCircuitNet metadata schema.",
    )
    args = ap.parse_args()
    translate(args.input, args.output)


if __name__ == "__main__":
    main()
