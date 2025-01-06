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


def apply_spherize_subset(
    embeddings: np.ndarray,
    control_indices: list[int],
    norm_embeddings: bool,
    center_by: str = "mean",
    use_control: bool = True,
) -> np.ndarray:
    """
    Applies sphering (whitening) transformation to a subset of embeddings.

    :param embeddings: NumPy array of shape (n_samples, n_features) containing the
        embeddings.
    :param control_indices: List of indices to use as control samples for centering.
    :param norm_embeddings: If True, normalizes the transformed embeddings to unit norm.
    :param center_by: Method to center the data ('mean' or 'median').
    :param use_control: If True, uses only control samples for centering; otherwise,
        uses all samples.
    :return: NumPy array of transformed embeddings with the same shape as input
        embeddings.
    """
    if use_control:
        if not control_indices:
            raise ValueError(
                "No control indices provided, but `use_control` is set to True."
            )
        samples = embeddings[control_indices]
    else:
        samples = embeddings

    # Centering the embeddings
    if center_by == "mean":
        center = np.mean(samples, axis=0)
    elif center_by == "median":
        center = np.median(samples, axis=0)
    else:
        raise ValueError("`center_by` must be either 'mean' or 'median'.")

    centered_samples = samples - center
    cov_matrix = np.cov(centered_samples, rowvar=False)

    try:
        eigvals, eigvecs = np.linalg.eigh(cov_matrix)
    except np.linalg.LinAlgError as e:
        print(f"Eigen decomposition failed: {e}")
        raise

    # Handle small eigenvalues to ensure numerical stability
    tolerance = 1e-10
    # small_eigvals = eigvals[eigvals < tolerance]
    # if small_eigvals.size > 0:
    # print(f"Number of small eigenvalues: {len(small_eigvals)}")

    # Replace small eigenvalues with the tolerance
    eigvals = np.maximum(eigvals, tolerance)

    # Create the transformation matrix for sphering
    D_inv_sqrt = np.diag(1.0 / np.sqrt(eigvals))
    transformation_matrix = eigvecs @ D_inv_sqrt @ eigvecs.T

    # Apply the sphering transformation
    centered_embeddings = embeddings - center
    transformed_embeddings = centered_embeddings @ transformation_matrix.T

    if norm_embeddings:
        norms = np.linalg.norm(transformed_embeddings, axis=1, keepdims=True)
        small_norms = norms < tolerance
        if np.any(small_norms):
            print(f"Number of small norms before normalization: {np.sum(small_norms)}")
            norms[small_norms] = 1.0  # Prevent division by zero

        transformed_embeddings /= norms

    return transformed_embeddings, center, transformation_matrix


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
