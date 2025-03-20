#!/usr/bin/env python3
"""
Iterate over Parquet files, filter rows, rename columns,
and save valid targets as CSV files.
"""

import pandas as pd
from pathlib import Path

SRC_DIR = Path("/projects/synsight/repos/phenoseeker/data/Lit_PCBA/parquets_files")
DST_DIR = Path("/projects/synsight/repos/phenoseeker/data/Lit_PCBA/csv_files")


def process_file(file_path):
    """Process a single Parquet file and save CSV if criteria met."""
    base = file_path.name
    if base.startswith("jump_") and base.endswith(".parquet"):
        target = base[len("jump_") : -len(".parquet")]
    else:
        return

    try:
        df = pd.read_parquet(file_path)
    except Exception as exc:
        print(f"Error reading {file_path}: {exc}")
        return

    # Keep rows with tanimoto_similarity equal to 1.
    df = df[df["tanimoto_similarity"] == 1]

    # Rename columns using a multi-line dict.
    df = df.rename(
        columns={
            "smiles": "Metadata_Smiles",
            "closest_jcp": "Metadata_JCP2022",
            "Active": "role_val",
        }
    )

    # Replace 'active' with 'hit'; others become None.
    df["role_val"] = df["role_val"].apply(lambda x: "hit" if x is True else None)

    total = len(df)
    actives = df["role_val"].eq("hit").sum()

    if total >= 50 and actives >= 5:
        out_path = DST_DIR / f"{target}.csv"
        try:
            df.to_csv(out_path, index=False)
            print(
                f"Saved CSV for {target} with {total} mols and " f"{actives} actives."
            )
        except Exception as exc:
            print(f"Error saving {out_path}: {exc}")


def main():
    """Main function to process all Parquet files."""
    DST_DIR.mkdir(parents=True, exist_ok=True)
    for file_path in SRC_DIR.glob("jump_*.parquet"):
        process_file(file_path)


if __name__ == "__main__":
    main()
