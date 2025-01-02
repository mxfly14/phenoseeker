import logging
import shutil
import itertools
import warnings
from pathlib import Path
import pandas as pd

from .embedding_manager import EmbeddingManager


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

    return ""


def apply_transformations(
    well_em: EmbeddingManager,
    sequence,  # TODO : type this variable
    starting_embedding_col: str | None = "Embeddings_Raw",
) -> EmbeddingManager:

    # Start with the specified initial embedding column
    current_embedding_col = starting_embedding_col
    sequence_name = sequence["name"]

    for transformation in sequence["transformations"]:
        method_name = transformation.get("method")
        params = transformation.get("params") or {}

        # Generate the suffix for the current transformation
        suffix = get_suffix(method_name, params)
        produces_new_embedding = suffix != ""
        save_embedding_col = (
            current_embedding_col + "__" + suffix
            if produces_new_embedding
            else current_embedding_col
        )

        # Check if this full transformation suite has already been applied
        if produces_new_embedding and save_embedding_col in well_em.df.columns:
            logging.info(
                f"Seq : '{sequence_name}', '{save_embedding_col}' already exists."
            )
            current_embedding_col = save_embedding_col
            continue

        try:
            logging.info(
                f"Seq : '{sequence_name}', applying '{method_name}' with parameters: {params}"  # noqa
            )
            method = getattr(well_em, method_name)
            method_params = {"raw_embedding_col": current_embedding_col, **params}
            if produces_new_embedding:
                method_params["save_embedding_col"] = save_embedding_col
            method(**method_params)
            current_embedding_col = save_embedding_col or current_embedding_col

        except Exception as e:
            logging.error(
                f"Seq : '{sequence_name}', an error occurred while applying '{method_name}': {e}"  # noqa
            )
            break
    return current_embedding_col


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


def setup_environment(config, config_file_path):
    """Set up folders, logging, and move the config file to the experiment folder."""
    warnings.filterwarnings(action="ignore")

    exp_parent_folder = Path(config["paths"]["exp_folder"])
    exp_name = Path(config["paths"]["exp_name"])
    exp_folder = exp_parent_folder / exp_name
    metadata_path = Path(config["paths"]["metadata_path"])

    results_folder = exp_folder / "results"

    exp_folder.mkdir(parents=True, exist_ok=True)
    results_folder.mkdir(parents=True, exist_ok=True)

    new_config_path = exp_folder / "config_test_all_norms.yaml"
    shutil.copy(str(config_file_path), str(new_config_path))

    return results_folder, metadata_path


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
        name_parts.append(method_abbrev)
    name = "__".join(name_parts)
    return name


def generate_sequences(methods, n_methods_max):
    """Generate sequences of transformations."""
    method_sequences = []
    for n in range(1, n_methods_max + 1):
        permutations = list(itertools.permutations(methods, n))
        for perm in permutations:
            if "apply_Z_score" in perm and "apply_robust_Z_score" in perm:
                continue
            method_sequences.append(perm)
    return method_sequences
