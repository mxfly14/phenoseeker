import logging
import shutil
import warnings
import pandas as pd
from pathlib import Path
from phenoseeker import (
    EmbeddingManager,
    apply_transformations,
    generate_all_pipelines,
    check_free_memory,
    cleanup_large_pipelines,
    load_config,
)


def setup_environment(config_file_path: Path) -> tuple[dict, Path]:
    """
    Load configuration, set up folders, initialize logging, and move the config file to
    the experiment folder.

    Args:
        config_file_path (Path): Path to the configuration file.

    Returns:
        dict: Loaded configuration.
        Path: Path to the results folder.
    """
    warnings.filterwarnings(action="ignore")

    # Load configuration
    config = load_config(config_file_path)

    # Set up experiment folders
    exp_parent_folder = Path(config["paths"]["exp_folder"])
    exp_name = Path(config["paths"]["exp_name"])
    exp_folder = exp_parent_folder / exp_name

    results_folder = exp_folder / "results"

    exp_folder.mkdir(parents=True, exist_ok=True)
    results_folder.mkdir(parents=True, exist_ok=True)

    # Copy config file to experiment folder
    new_config_path = exp_folder / "config_test_all_norms.yaml"
    shutil.copy(str(config_file_path), str(new_config_path))

    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    # Initialize logging inside the experiment folder
    log_file = exp_folder / "test_normalisations.log"
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    logging.info("Environment setup complete. Configuration loaded.")

    return config, results_folder


def preprocess_embeddings(config) -> EmbeddingManager:

    metadata_path = Path(config["paths"]["metadata_path"])
    well_em = EmbeddingManager(metadata_path, entity="well")

    selected_plates = config.get("selected_plates", "all")
    if selected_plates == "all":
        selected_plates = well_em.df["Metadata_Plate"].unique()

    selected_plates = [
        item for item in selected_plates if item not in well_em.no_dmso_plates
    ]

    well_em.df = well_em.df[well_em.df["Metadata_Plate"].isin(selected_plates)]

    well_em.df = well_em.df[
        well_em.df["Metadata_JCP2022"].isin(well_em.JCP_ID_controls)
    ]

    well_em.load("path_embedding", vectors_column="Embeddings_Raw")
    well_em.remove_features(threshold=10e-5, vectors_column="Embeddings_Raw")

    logging.info("Embeddings loaded and filtered.")
    logging.info(f"We have an {well_em}.")

    return well_em


def evaluate_pipeline(sequence, well_em, results_dfs, current, total) -> dict:
    """Apply transformations and evaluate the pipeline."""
    logging.info(f"Evaluating pipeline {current}/{total}: {sequence['name']}")
    try:
        col_name = apply_transformations(well_em, sequence)
        compounds_em = well_em.filter_and_instantiate(Metadata_Is_dmso=False)
        maps_jcp2022 = compounds_em.compute_maps(
            labels_column="Metadata_JCP2022",
            vectors_columns={f'{sequence["name"]}': col_name},
        )
        del compounds_em
        # maps_batch = well_em.compute_maps(
        #    labels_column="Metadata_Batch",
        #    vectors_columns={f'{sequence["name"]}': col_name},
        # )
        maps_source = well_em.compute_maps(
            labels_column="Metadata_Source",
            vectors_columns={f'{sequence["name"]}': col_name},
        )
        # maps_plate = well_em.compute_maps(
        #    labels_column="Metadata_Plate",
        #    vectors_columns={f'{sequence["name"]}': col_name},
        # )

        results_dfs["jcp2022"].append(maps_jcp2022)
        # results_dfs["batch"].append(maps_batch)
        results_dfs["source"].append(maps_source)
        # results_dfs["plate"].append(maps_plate)

        logging.info(f"Pipeline '{sequence['name']}' evaluated successfully.")
    except Exception as e:
        logging.error(f"Error in pipeline {sequence['name']}: {e}")
    return results_dfs


def main():
    """Main function to execute the normalization and evaluation."""

    logging.info("Process started.")

    # Setup environment and load configuration
    config_file_path = Path("configs/config_test_all_norms.yaml")
    config, results_folder = setup_environment(config_file_path)

    methods = config.get("methods", [])
    max_combinations = config.get("max_combinations", 100)
    n_methods_max = config.get("n_methods_max", len(methods))

    logging.info("Configuration and environment setup successfully.")

    # Load embeddings
    well_em = preprocess_embeddings(config)

    transformation_sequences = generate_all_pipelines(
        methods,
        n_methods_max,
        max_combinations,
    )

    logging.info(f"{len(transformation_sequences)} normalisation pipelines generated.")

    # Initialize results DataFrames
    results_dfs = {
        "jcp2022": [],
        # "batch": [],
        "source": [],
        # "plate": [],
    }

    results_dfs = evaluate_pipeline(
        {"name": "Raw", "transformations": []},
        well_em,
        results_dfs,
        1,
        len(transformation_sequences) + 1,
    )

    for idx, sequence in enumerate(transformation_sequences, start=2):
        results_dfs = evaluate_pipeline(
            sequence,
            well_em,
            results_dfs,
            idx,
            len(transformation_sequences) + 1,
        )
        well_em.distance_matrices = {}
        free_memory = check_free_memory()
        logging.info(
            f"Free memory after pipeline {idx}/{len(transformation_sequences) + 1}: {free_memory:.2f} GB"  # NOQA
        )
        if free_memory < 50:  # Less that 50GB of available RAM
            cleanup_large_pipelines(well_em, 2)

    for name, dfs in results_dfs.items():
        df = pd.concat(dfs, ignore_index=False, axis=1)
        df = df.loc[:, ~df.columns.duplicated()]
        df.to_csv(results_folder / f"maps_{name}.csv", index=False)

    logging.info("MAPs results saved.")
    logging.info("Process completed successfully.")


if __name__ == "__main__":
    main()
