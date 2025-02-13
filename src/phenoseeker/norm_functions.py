# Third-Party Libraries
import numpy as np
import pandas as pd
from scipy import stats
from joblib import Parallel, delayed


def inverse_normal_transform(feature: np.ndarray, c: float = 0.5) -> np.ndarray:
    """
    Apply inverse normal transformation to a given feature array of samples.
    """
    ranks = stats.rankdata(feature, method="average")
    uniform_ranks = (ranks - c) / (len(feature) - 2 * c + 1)
    return stats.norm.ppf(uniform_ranks)


def apply_int_subset(
    subset_df: pd.DataFrame,
    raw_embedding_col: str,
    save_embedding_col: str,
    indices: np.ndarray | None,
    n_jobs: int,
) -> None:
    """
    Apply inverse normal transformation to features in a specified column
    for the given indices (or all indices by default).

    :param raw_embedding_col: The name of the column containing the raw embeddings.
    :param save_embedding_col: The name of the column to save the transformed
    embeddings. If None, it defaults to 'int_{raw_embedding_col}'.
    :param indices: The indices to apply the transformation to (default is all
    indices). Or None, all indices would be used
    :param n_jobs: The number of jobs for parallel processing (default is -1 for
    all cores).
    :return: None
    """
    embeddings = np.stack(subset_df[raw_embedding_col])

    if indices is None:
        indices = range(embeddings.shape[1])

    transformed_features = Parallel(n_jobs=n_jobs)(
        delayed(inverse_normal_transform)(embeddings[:, i]) for i in indices
    )

    transformed_features = np.stack(transformed_features)

    for idx, i in enumerate(indices):
        embeddings[:, i] = transformed_features[idx, :]

    subset_df[save_embedding_col] = [embeddings[i] for i in range(len(embeddings))]
    return subset_df


def rescale(data: np.ndarray, scale: str) -> np.ndarray:
    """
    Rescale the features of the input array to be between 0 and 1 or -1 and 1.

    Parameters:
    data (np.ndarray): Input array of shape (n_samples, n_features).
    scale (str): The scale to rescale the features, either "0-1" or "-1-1".

    Returns:
    np.ndarray: Rescaled array with features between the specified scale.

    Raises:
    ValueError: If the scale parameter is not "0-1" or "-1-1".
    """
    if scale not in ["0-1", "-1-1"]:
        raise ValueError("scale must be either '0-1' or '-1-1'")

    min_vals = np.min(data, axis=0)
    max_vals = np.max(data, axis=0)
    range_vals = max_vals - min_vals

    range_vals[range_vals == 0] = 1

    if scale == "0-1":
        rescaled_data = (data - min_vals) / range_vals
    elif scale == "-1-1":
        rescaled_data = 2 * (data - min_vals) / range_vals - 1

    return rescaled_data


def median_polish(
    data: np.ndarray[np.float64],
    n_iter: int = 10,
    tol: float = 1e-6,
) -> dict:
    """
    Performs median polish on a 2-D array.

    Args:
        data: Input 2-D array (rows x columns) for median polishing.
        n_iter: Maximum number of iterations to perform. Defaults to 10.
        tol: Convergence tolerance for residual change. Defaults to 1e-6.

    Returns:
        A dictionary containing:
            - 'ave': Grand effect (overall mean after polishing).
            - 'row': Row effects.
            - 'col': Column effects.
            - 'r': Final residual matrix.
    """
    assert data.ndim == 2, "Input must be a 2D array"

    # Initialize
    data = data.copy()
    ave = np.median(data)  # Grand effect
    data -= ave
    row_effects = np.zeros(data.shape[0], dtype=np.float64)
    col_effects = np.zeros(data.shape[1], dtype=np.float64)
    previous_r = data.copy()  # To track convergence

    for iteration in range(n_iter):
        # Update row effects
        row_medians = np.median(data, axis=1)
        row_effects += row_medians
        data -= row_medians[:, None]

        # Update column effects
        col_medians = np.median(data, axis=0)
        col_effects += col_medians
        data -= col_medians

        # Update grand effect
        overall_median = np.median(data)
        ave += overall_median
        data -= overall_median

        # Convergence check
        max_residual_change = np.max(np.abs(data - previous_r))
        previous_r = data.copy()
        if max_residual_change < tol:
            break

    return {
        "ave": ave,
        "row": row_effects,
        "col": col_effects,
        "r": data,
    }
