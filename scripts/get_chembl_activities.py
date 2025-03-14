"""
This script was largely inspired by the following script:
    https://github.com/cfredinh/bioactive/blob/main/data_prep/prep_activity_data.py
from the repository:
    https://github.com/cfredinh/bioactive

Many thanks to cfredinh for his work and contribution.
"""

import sqlite3
import pandas as pd
import numpy as np
from pathlib import Path

n_hit = 5
n_non_hit = 50
n_measurements = 60


def load_chembl_data(chembl_db_path):
    """
    Connect to the ChEMBL SQLite database and load necessary tables.
    """
    con = sqlite3.connect(chembl_db_path)
    assay_df = pd.read_sql_query("SELECT * FROM assays", con)
    compound_df = pd.read_sql_query("SELECT * FROM compound_structures", con)
    activity_df = pd.read_sql_query("SELECT * FROM activities", con)
    return assay_df, compound_df, activity_df


def get_overlapping_compounds(compound_df, compound_meta):
    """
    Identify overlapping compounds between ChEMBL and the compound metadata.
    """
    chembl_compounds = set(compound_df.standard_inchi_key.unique())
    meta_compounds = set(compound_meta.Metadata_InChIKey.unique())
    overlapping_compounds = chembl_compounds.intersection(meta_compounds)

    df_overlap_compounds = compound_df[
        compound_df.standard_inchi_key.isin(overlapping_compounds)
    ]
    molregno_overlapping = df_overlap_compounds.molregno.values
    return df_overlap_compounds, molregno_overlapping


def process_activity_data(activity_df, molregno_overlapping):
    """
    Process the activity data for overlapping compounds:
      - Filter to type "Potency"
      - Keep records with activity_comment in ["inactive", "active", "Active",
            "Not Active"]
      - Convert activity comments to numeric labels (1 for active, -1 for inactive,
            0 otherwise)
      - Build a pivot table (label matrix) with assays as columns and compounds as rows
      - Only keep assays with at least n_hit positive and n_non_hit negative labels
    """
    # Filter activities for overlapping compounds and type "Potency"
    act_overlap = activity_df[activity_df.molregno.isin(molregno_overlapping)]
    potency_subset = act_overlap[act_overlap.standard_type == "Potency"]
    potency_subset = potency_subset[
        potency_subset.activity_comment.isin(
            ["inactive", "active", "Active", "Not Active"]
        )
    ]

    # Only keep assays with more than 100 measurements
    assay_counts = potency_subset.assay_id.value_counts()
    selected_subset = potency_subset[
        potency_subset.assay_id.isin(assay_counts[assay_counts > n_measurements].index)
    ]

    # Convert activity comments to numeric labels
    selected_subset = selected_subset.copy()
    selected_subset.loc[:, "activity_label"] = 0
    selected_subset.loc[
        selected_subset.activity_comment.isin(["Active", "active"]), "activity_label"
    ] = 1
    selected_subset.loc[
        selected_subset.activity_comment.isin(["Not Active", "inactive"]),
        "activity_label",
    ] = -1

    # Create pivot table (label matrix)
    label_matrix = selected_subset.pivot_table(
        values="activity_label",
        index="molregno",
        columns="assay_id",
        aggfunc=np.median,
    ).fillna(0)

    # Keep assays with at least 50 positive labels and 50 negative labels
    positive_assays = (label_matrix == 1).sum() > n_hit
    assays_with_pos = positive_assays[positive_assays].index
    negative_assays = (label_matrix[assays_with_pos] == -1).sum() > n_non_hit
    assays_to_keep = negative_assays[negative_assays].index

    label_matrix_subset = label_matrix[assays_to_keep]
    return label_matrix_subset


def merge_label_matrix_with_compounds(label_matrix, df_overlap_compounds):
    """
    Merge the label matrix with the overlapping compound data to add the standard
        InChI key.
    """
    label_matrix_reset = label_matrix.reset_index()
    label_matrix_merged = pd.merge(
        label_matrix_reset,
        df_overlap_compounds[["molregno", "standard_inchi_key"]],
        on="molregno",
        how="left",
    )
    return label_matrix_merged


def main(metadata_path, chembl_db_path, base_path):
    # Load compound metadata and ChEMBL data
    df_meta = pd.read_csv(metadata_path, low_memory=False)
    compound_meta = (
        df_meta[df_meta["Metadata_PlateType"] == "COMPOUND"]
        .drop_duplicates(subset="Metadata_InChI")
        .reset_index(drop=True)[
            ["Metadata_InChI", "Metadata_JCP2022", "Metadata_InChIKey"]
        ]
    )
    assay_df, compound_df, activity_df = load_chembl_data(chembl_db_path)
    assay_path = base_path / "assay.csv"
    assay_df.to_csv(assay_path, index=False)
    # Identify overlapping compounds
    df_overlap_compounds, molregno_overlapping = get_overlapping_compounds(
        compound_df, compound_meta
    )
    # Process activity data to create a label matrix
    label_matrix_subset = process_activity_data(activity_df, molregno_overlapping)

    # Merge label matrix with compound data to include InChI keys
    label_matrix_merged = merge_label_matrix_with_compounds(
        label_matrix_subset, df_overlap_compounds
    )
    output_csv = base_path / "chembl_activity_data.csv"
    # Save the result to a CSV file
    label_matrix_merged.to_csv(output_csv, index=False)
    print(f"Data saved to '{output_csv}'.")


if __name__ == "__main__":

    METADATA_PATH = "/projects/cpjump1/jump/metadata/complete_metadata.csv"
    CHEMBL_DB_PATH = "/projects/synsight/data/chembl_35/chembl_35_sqlite/chembl_35.db"
    base_path = Path("/projects/synsight/repos/phenoseeker/data/ChEMBL")
    base_path.mkdir(parents=True, exist_ok=True)
    main(METADATA_PATH, CHEMBL_DB_PATH, base_path)
