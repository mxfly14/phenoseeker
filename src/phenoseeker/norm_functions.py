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
