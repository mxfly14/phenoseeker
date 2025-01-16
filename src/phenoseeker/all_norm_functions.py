import logging
import itertools
import pandas as pd

from .embedding_manager import EmbeddingManager


def apply_transformations(
    well_em: EmbeddingManager,
    sequence: dict,
    starting_embedding_name: str | None = "Embeddings_Raw",
) -> str:
    """
    Apply a sequence of transformations to the embeddings in the EmbeddingManager.

    Args:
        well_em (EmbeddingManager): Instance of the EmbeddingManager.
        sequence (dict): Transformation sequence containing a name and a list of
            transformations. Each transformation : a dict with keys 'method' and
            'params'.
        starting_embedding_name (str, optional): The name of the embedding to start with
            Defaults to 'Embeddings_Raw'.

    Returns:
        str: The name of the final embedding after applying all transformations.
    """
    # Start with the specified initial embedding name
    current_embedding_name = starting_embedding_name
    sequence_name = sequence["name"]

    for transformation in sequence["transformations"]:
        method_name = transformation.get("method")
        params = transformation.get("params") or {}

        # Generate the suffix for the current transformation
        suffix = get_suffix(method_name, params)
        produces_new_embedding = suffix != ""
        save_embedding_name = (
            f"{current_embedding_name}__{suffix}"
            if produces_new_embedding
            else current_embedding_name
        )

        # Check if this full transformation suite has already been applied
        if produces_new_embedding and save_embedding_name in well_em.embeddings:
            logging.info(
                f"Seq : '{sequence_name}', '{save_embedding_name}' already exists."
            )
            current_embedding_name = save_embedding_name
            continue

        try:
            logging.info(
                f"Seq : '{sequence_name}', applying '{method_name}' with parameters: {params}"  # noqa
            )
            method = getattr(well_em, method_name)
            method_params = {"embeddings_name": current_embedding_name, **params}
            if produces_new_embedding:
                method_params["new_embeddings_name"] = save_embedding_name
            method(**method_params)
            current_embedding_name = save_embedding_name or current_embedding_name

        except Exception as e:
            logging.error(
                f"Seq : '{sequence_name}', an error occurred while applying '{method_name}': {e}"  # noqa
            )
            break

    return current_embedding_name


def create_embedding_dict(df: pd.DataFrame, prefix: str | None = "Embeddings_"):
    if df.empty:
        print("La DataFrame est vide.")
        return {}
    embedding_columns = [col for col in df.columns if col.startswith(prefix)]
    if not embedding_columns:
        print(f"Aucune colonne ne commence par '{prefix}'.")
        return {}
    embedding_dict = {col.replace(prefix, ""): col for col in embedding_columns}
    return embedding_dict


def get_suffix(method_name, params):
    if method_name == "apply_inverse_normal_transform":
        return "Int"

    elif method_name == "apply_robust_Z_score":
        center_by = params.get("center_by", "")
        reduce_by = params.get("reduce_by", "")
        center_abbrev = "m" if center_by == "median" else "M"
        reduce_abbrev = (
            "i" if reduce_by == "iqrs" else ("m" if reduce_by == "mad" else "s")
        )
        suffix = f"rZ{center_abbrev}{reduce_abbrev}"
        if params.get("use_control"):
            suffix += "_C"
        return suffix

    elif method_name == "apply_rescale":
        scale = params.get("scale", "")
        return "Res01" if scale == "0-1" else "Res11" if scale == "-1-1" else ""

    elif method_name == "apply_spherizing_transform":
        s_method = params.get("method", "ZCA")
        suffix = s_method  # This should handle "PCA", "ZCA", "PCA-cor", "ZCA-cor"
        if params.get("norm_embeddings"):
            suffix += "_N"
        if params.get("use_control"):
            suffix += "_C"
        return suffix

    elif method_name == "apply_median_polish":
        return "MedPol"

    return ""


def get_method_variations(method):
    variations = []
    if method == "apply_inverse_normal_transform":
        variations.append({"method": method})
    elif method == "apply_robust_Z_score":
        center_by_options = ["mean", "median"]
        reduce_by_options = ["iqrs", "std"]
        use_control_options = [True, False]
        for center_by in center_by_options:
            for reduce_by in reduce_by_options:
                for use_control in use_control_options:
                    variations.append(
                        {
                            "method": method,
                            "params": {
                                "center_by": center_by,
                                "reduce_by": reduce_by,
                                "use_control": use_control,
                            },
                        }
                    )
    elif method == "apply_rescale":
        scale_options = ["0-1", "-1-1"]
        for scale in scale_options:
            variations.append({"method": method, "params": {"scale": scale}})
    elif method == "apply_spherizing_transform":
        spherizing_methods = ["PCA", "ZCA", "ZCA-cor"]
        norm_embeddings_options = [True, False]
        use_control_options = [True, False]
        for s_method in spherizing_methods:
            for norm_embeddings in norm_embeddings_options:
                for use_control in use_control_options:
                    variations.append(
                        {
                            "method": method,
                            "params": {
                                "method": s_method,
                                "norm_embeddings": norm_embeddings,
                                "use_control": use_control,
                            },
                        }
                    )
    elif method == "apply_median_polish":
        # No additional parameters are required for median polish.
        variations.append({"method": method})
    return variations


def generate_sequence_name(sequence):
    name_parts = []
    for t in sequence:
        method_abbrev = ""
        method = t["method"]
        params = t.get("params", {})
        if method == "apply_inverse_normal_transform":
            method_abbrev = "Int"
        elif method == "apply_robust_Z_score":
            method_abbrev = "rZ"
            center_by = params.get("center_by", "")
            reduce_by = params.get("reduce_by", "")
            center_abbrev = "m" if center_by == "median" else "M"
            reduce_abbrev = (
                "i" if reduce_by == "iqrs" else ("m" if reduce_by == "mad" else "s")
            )
            method_abbrev += center_abbrev + reduce_abbrev
            if params.get("use_control"):
                method_abbrev += "_C"
        elif method == "apply_rescale":
            scale = params.get("scale", "")
            method_abbrev = "Res" + ("01" if scale == "0-1" else "11")
        elif method == "apply_spherizing_transform":
            s_method = params.get("method", "")
            norm_embeddings = params.get("norm_embeddings", False)
            use_control = params.get("use_control", False)
            method_abbrev = s_method
            if norm_embeddings:
                method_abbrev += "_N"
            if use_control:
                method_abbrev += "_C"
        elif method == "apply_median_polish":
            method_abbrev = "MedPol"
        name_parts.append(method_abbrev)
    name = "__".join(name_parts)
    return name


def generate_sequences(methods: list[str], n_methods_max: int) -> list:
    """Generate sequences of transformations."""
    method_sequences = []
    for n in range(1, n_methods_max + 1):
        permutations = list(itertools.permutations(methods, n))
        for perm in permutations:
            if "apply_Z_score" in perm and "apply_robust_Z_score" in perm:
                continue
            if "apply_inverse_normal_transform" in perm and "apply_rescale" in perm:
                continue
            method_sequences.append(perm)
    return method_sequences


def generate_all_pipelines(
    methods: list[str],
    n_methods_max: int,
    max_combinations: int,
) -> list[dict]:
    """
    Generate all possible normalization pipelines given a list of methods, the maximum
    number of methods per sequence, and the maximum number of combinations to return.

    Args:
        methods (List[str]): List of normalization methods.
        n_methods_max (int): Maximum number of methods per pipeline sequence.
        max_combinations (int): Maximum number of pipelines to generate.

    Returns:
        List[Dict]: List of pipeline sequences. Each sequence is a dictionary with a
                    "name" and a list of "transformations".
    """
    # Generate all method sequences
    method_sequences = generate_sequences(methods, n_methods_max)
    transformation_sequences = []

    for method_sequence in method_sequences:
        # Get all variations for each method in the sequence
        method_variations = [
            get_method_variations(method) for method in method_sequence
        ]
        # Compute all combinations of the method variations
        for seq_variation in itertools.product(*method_variations):
            # Generate a name for the sequence
            sequence_name = generate_sequence_name(seq_variation)
            # Create the sequence dictionary
            sequence = {
                "name": sequence_name,
                "transformations": [dict(variation) for variation in seq_variation],
            }
            if "_N" in sequence_name and "Res" in sequence_name:
                continue
            transformation_sequences.append(sequence)

    # Limit the number of generated sequences
    return transformation_sequences[:max_combinations]


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
