#!/usr/bin/env python3
"""Prepare deduplicated input copies without modifying source datasets."""

import argparse
from pathlib import Path

import pandas as pd
import baseline_common as C


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--wide-csv", default=C.WIDE_CSV)
    parser.add_argument("--detail-csv", default=C.DETAIL_CSV)
    parser.add_argument("--id-col", default="")
    parser.add_argument("--doc-col", default="")
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    output = Path(args.out_dir)
    output.mkdir(parents=True, exist_ok=True)

    if any(output.iterdir()):
        raise SystemExit(
            f"Prepared-input folder must be empty: {output}. "
            "Use a new run directory."
        )

    # Read IDs as strings to preserve leading zeros.
    wide = pd.read_csv(
        args.wide_csv,
        dtype=str,
        keep_default_na=False,
    )

    id_column = args.id_col or C.pick_col(
        wide, ("id", "paper_id")
    )

    if not id_column or id_column not in wide.columns:
        raise SystemExit(
            f"ID column unavailable. Columns: {list(wide.columns)}"
        )

    document_column = args.doc_col or C.pick_col(
        wide, C.DOC_COL_CANDIDATES
    )

    if not document_column or document_column not in wide.columns:
        raise SystemExit(
            "Document column unavailable. Set DOC_COL when submitting."
        )

    # Normalize surrounding whitespace without changing leading zeros.
    identifiers = wide[id_column].str.strip()

    blank = identifiers.eq("")
    if blank.any():
        records = (wide.index[blank] + 1).tolist()[:10]
        raise SystemExit(
            f"Blank IDs in column {id_column!r}, "
            f"data records: {records}. "
            "Correct these IDs or select another column using ID_COL."
        )

    # Keep FIRST occurrence of each paper ID.
    duplicate = identifiers.duplicated(keep="first")

    removed = wide.loc[duplicate].copy()
    removed.insert(
        0,
        "_lfw_source_record",
        wide.index[duplicate] + 1,
    )
    removed.to_csv(
        output / "wide_removed_duplicates.csv",
        index=False,
    )

    cleaned = wide.loc[~duplicate].copy()
    cleaned[id_column] = identifiers.loc[~duplicate]

    # Older evaluator versions prioritize the column named "id".
    # Supply the selected identifier under that name.
    if id_column != "id":
        if "id" in cleaned.columns:
            if "_lfw_original_id" in cleaned.columns:
                raise SystemExit(
                    "Reserved column already exists: _lfw_original_id"
                )
            cleaned["_lfw_original_id"] = cleaned["id"]

        cleaned["id"] = cleaned[id_column]

    if cleaned.empty:
        raise SystemExit("No papers remain in the input.")

    if cleaned[document_column].str.strip().eq("").any():
        raise SystemExit(
            "A retained paper has an empty manuscript. "
            "Correct the input before evaluation."
        )

    if not any(
        column.startswith("insert__")
        for column in cleaned.columns
    ):
        raise SystemExit(
            "No insert__<condition> columns found in the wide CSV."
        )

    cleaned.to_csv(
        output / "wide_cleaned.csv",
        index=False,
    )

    print(
        f"[input] ID column={id_column}; "
        f"source rows={len(wide)}; "
        f"duplicates removed={len(removed)}; "
        f"retained papers={len(cleaned)}",
        flush=True,
    )

    # Detail CSV can contain many conditions for each paper.
    # Deduplicate by BOTH paper_id and condition.
    detail_path = (
        Path(args.detail_csv)
        if args.detail_csv
        else None
    )

    if detail_path and detail_path.is_file():
        detail = pd.read_csv(
            detail_path,
            dtype=str,
            keep_default_na=False,
        )

        if {"paper_id", "condition"} <= set(detail.columns):
            detail["paper_id"] = detail["paper_id"].str.strip()

            duplicate_detail = detail.duplicated(
                ["paper_id", "condition"],
                keep="first",
            )

            removed_detail = detail.loc[duplicate_detail].copy()
            removed_detail.insert(
                0,
                "_lfw_source_record",
                detail.index[duplicate_detail] + 1,
            )
            removed_detail.to_csv(
                output / "detail_removed_duplicates.csv",
                index=False,
            )

            detail = detail.loc[~duplicate_detail].copy()

            print(
                "[input] Removed duplicate detail rows: "
                f"{int(duplicate_detail.sum())}",
                flush=True,
            )
        else:
            print(
                "[input] Detail CSV lacks join keys; "
                "evaluation will not join it.",
                flush=True,
            )
    else:
        print(
            "[input] Optional detail CSV not found. "
            "Using labels from the wide CSV only.",
            flush=True,
        )
        detail = pd.DataFrame(
            columns=["paper_id", "condition", "asr"]
        )

    detail.to_csv(
        output / "detail_cleaned.csv",
        index=False,
    )

    print(
        f"[input] Prepared input directory: {output.resolve()}",
        flush=True,
    )
    print(
        "[input] Original datasets unchanged. "
        "Duplicate policy: KEEP FIRST.",
        flush=True,
    )


if __name__ == "__main__":
    main()