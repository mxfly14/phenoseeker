import logging
import itertools
import pandas as pd
from pathlib import Path
from phenoseeker import (
    EmbeddingManager,
    apply_transformations,
    load_config,
    setup_environment,
    generate_sequences,
    get_method_variations,
    generate_sequence_name,
    check_free_memory,
)


def cleanup_large_pipelines(well_em: EmbeddingManager, n: int | None = 2):
    """Remove embedding columns from pipelines with more than two operations."""
    logging.info("Cleaning up large pipelines to free memory.")
    columns_to_remove = [
        col
        for col in well_em.df.columns
        if "Embeddings" in col and col.count("__") >= n
    ]
    logging.info(f"Removing {len(columns_to_remove)} columns: {columns_to_remove}")
    well_em.df.drop(columns=columns_to_remove, inplace=True)


def evaluate_pipeline(sequence, well_em, results_dfs, current, total) -> dict:
    """Apply transformations and evaluate the pipeline."""
    logging.info(f"Evaluating pipeline {current}/{total}: {sequence['name']}")
    try:
        col_name = apply_transformations(well_em, sequence)
        compounds_em = well_em.filter_and_instantiate(Metadata_Is_Control=False)
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
    try:
        # Setup logging
        log_file = Path("./tmp/test_all_norms.log")
        log_file.parent.mkdir(exist_ok=True)
        logging.basicConfig(
            filename=log_file,
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
        )

        logging.info("Process started.")

        # Load configuration
        config_file_path = Path("configs/config_test_all_norms.yaml")
        config = load_config(config_file_path)

        methods = config.get("methods", [])
        max_combinations = config.get("max_combinations", 100)
        n_methods_max = config.get("n_methods_max", len(methods))

        # Setup environment
        results_folder, metadata_path = setup_environment(config, config_file_path)

        logging.info("Configuration loaded.")

        # Load embeddings
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

        # Generate and evaluate pipelines
        method_sequences = generate_sequences(methods, n_methods_max)
        transformation_sequences = []

        for method_sequence in method_sequences:
            method_variations_list = []
            for method in method_sequence:
                method_variations_list.append(get_method_variations(method))
            sequence_variations = itertools.product(*method_variations_list)
            for seq_variation in sequence_variations:
                name = generate_sequence_name(seq_variation)
                sequence = {
                    "name": name,
                    "transformations": [dict(t) for t in seq_variation],
                }
                transformation_sequences.append(sequence)

        if len(transformation_sequences) >= max_combinations:
            transformation_sequences = transformation_sequences[:max_combinations]
        logging.info(
            f"{len(transformation_sequences)} normalisation pipelines generated."
        )

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
            df.to_csv(results_folder / f"consolidated_metrics_{name}.csv", index=False)

        logging.info("Consolidated results saved.")
        logging.info("Process completed successfully.")

    except Exception as e:
        logging.error(f"An error occurred: {e}")
        raise e


if __name__ == "__main__":
    main()
