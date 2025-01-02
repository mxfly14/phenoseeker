# Third-Party Libraries
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.neighbors import NearestNeighbors
import warnings


def calculate_statistics(data: np.ndarray) -> dict:
    """
    Calculate various descriptive statistics for the given data.

    Parameters:
    data (np.ndarray): The input data array.

    Returns:
    dict: A dictionary containing the calculated statistics.
    """
    mean = np.mean(data, axis=0).astype(np.float32)
    std = np.std(data, axis=0).astype(np.float32)
    var = np.var(data, axis=0).astype(np.float32)
    min_val = np.min(data, axis=0).astype(np.float32)
    max_val = np.max(data, axis=0).astype(np.float32)
    median = np.median(data, axis=0).astype(np.float32)
    mad = np.median(np.abs(data - median), axis=0).astype(
        np.float32
    )  # Median Absolute Deviation

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        coefficient_of_variation = np.divide(
            mad, median, out=np.zeros_like(mad), where=median != 0
        ).astype(np.float32)

    q1 = np.percentile(data, 25, axis=0).astype(np.float32)
    q3 = np.percentile(data, 75, axis=0).astype(np.float32)
    iqr = (q3 - q1).astype(np.float32)

    return {
        "mean": mean,
        "std": std,
        "var": var,
        "min": min_val,
        "max": max_val,
        "median": median,
        "mad": mad,
        "coefficient_of_variation": coefficient_of_variation,
        "iqr": iqr,
    }


def calculate_lisi_score(embeddings, labels, n_neighbors, n_jobs):
    nbrs = NearestNeighbors(
        n_neighbors=n_neighbors, algorithm="brute", n_jobs=n_jobs
    ).fit(embeddings)
    _, indices = nbrs.kneighbors(embeddings)

    def lisi_score(neighbors):
        lab_counts = np.unique(labels[neighbors], return_counts=True)[1]
        pi = lab_counts / lab_counts.sum()
        return 1 / np.sum(pi**2)

    lisi_scores = Parallel(n_jobs=n_jobs)(
        delayed(lisi_score)(neighbors) for neighbors in indices
    )
    return np.mean(lisi_scores)


def compute_reduce_center(
    df: pd.DataFrame,
    raw_embedding_col: str,
    use_control: bool,
    center_by: str,
    reduce_by: str,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the center and reduce arrays for normalization.

    Args:
        df (pd.DataFrame): DataFrame containing the data.
        raw_embedding_col (str): Column name for the raw image embeddings in the
        DataFrame.
        center_by (str): Method for centering, either 'mean' or 'median'.
        reduce_by (str): Method for reduction, either 'std', 'iqrs', or 'mad'.
        use_control (bool): Whether to use only control samples or the entire DataFrame.

    Returns:
        tuple: Arrays for center and reduce.
    """

    valid_center_by = ["mean", "median"]
    valid_reduce_by = ["std", "iqrs", "mad"]

    if center_by not in valid_center_by:
        raise ValueError(
            f"Invalid 'center_by': {center_by}. Must be one of {valid_center_by}."
        )

    if reduce_by not in valid_reduce_by:
        raise ValueError(
            f"Invalid 'reduce_by': {reduce_by}. Must be one of {valid_reduce_by}."
        )

    required_columns = [
        "Metadata_Is_Control",
        "Metadata_Row_Number",
        "Metadata_Col_Number",
        raw_embedding_col,
    ]
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Missing required column in DataFrame: {col}")

    if use_control:
        controls = df[df["Metadata_Is_Control"]]
        if controls.empty:
            raise ValueError("No control samples found in the DataFrame.")
        data_subset = controls
    else:
        data_subset = df

    max_rows = df["Metadata_Row_Number"].max()
    max_columns = df["Metadata_Col_Number"].max()
    embedding_size = df[raw_embedding_col].iloc[0].size
    data_tensor = np.zeros((max_rows, max_columns, embedding_size), dtype=np.float32)

    row_numbers = df["Metadata_Row_Number"].values - 1
    col_numbers = df["Metadata_Col_Number"].values - 1
    embeddings = np.stack(df[raw_embedding_col].values)
    data_tensor[row_numbers, col_numbers, :] = embeddings

    subset_indices = (
        data_subset["Metadata_Row_Number"].values - 1,
        data_subset["Metadata_Col_Number"].values - 1,
    )
    subset_tensors_np = data_tensor[subset_indices]

    if center_by == "mean":
        dmso_mu = np.mean(subset_tensors_np, axis=0)
    elif center_by == "median":
        dmso_mu = np.median(subset_tensors_np, axis=0)

    if reduce_by == "std":
        dmso_sigma = np.std(subset_tensors_np, axis=0)
    elif reduce_by == "iqrs":
        q1 = np.percentile(subset_tensors_np, 25, axis=0)
        q3 = np.percentile(subset_tensors_np, 75, axis=0)
        dmso_sigma = q3 - q1
    elif reduce_by == "mad":
        median = np.median(subset_tensors_np, axis=0)
        dmso_sigma = np.median(np.abs(subset_tensors_np - median), axis=0)

    return dmso_mu, dmso_sigma


def calculate_map(
    dist_matrix: np.ndarray,
    labels: np.ndarray,
    indices_with_query_label: np.ndarray,
    query_label: str,
) -> float:
    """
    Calculate the mean Average Precision (mAP) for the given labels.
    """
    mAP = 0.0
    count = len(indices_with_query_label)

    for i in indices_with_query_label:
        mask = np.ones(len(labels), dtype=bool)
        mask[i] = False
        sorted_indices = np.argsort(dist_matrix[i][mask])

        filtered_indices = np.where(mask)[0][sorted_indices]

        num_positive = count - 1
        tp = 0
        ap = 0.0

        for rank, index in enumerate(filtered_indices):
            if labels[index] == query_label:
                tp += 1
                precision_at_k = tp / (rank + 1)
                recall_at_k = tp / num_positive
                recall_at_k_prev = (tp - 1) / num_positive if tp > 1 else 0.0
                ap += (recall_at_k - recall_at_k_prev) * precision_at_k
        mAP += ap

    mAP /= count
    return mAP


def calculate_maps(
    dist_matrix: np.ndarray,
    query_label: str,
    labels: np.ndarray,
    random_maps: bool = False,
) -> tuple[str, int, float | None, float | None]:
    """
    Calculate the original and random Mean Average Precision (MAP) for a given label.

    Args:
        dist_matrix (np.ndarray): Distance matrix between elements.
        query_label (str): The label to query.
        labels (np.ndarray): Array of labels for all elements.
        random_maps (bool): Whether to compute random mAP values.

    Returns:
        tuple[str, int, Optional[float], Optional[float]]: The query label, count of
        items with the query label, original MAP value, and random MAP value.
    """
    indices_with_query_label = np.where(labels == query_label)[0]
    count = len(indices_with_query_label)

    if count <= 1:
        return query_label, count, None, None

    original_map = calculate_map(
        dist_matrix, labels, indices_with_query_label, query_label
    )

    random_map = None
    if random_maps:
        random_labels = labels.copy()
        np.random.shuffle(random_labels)
        indices_with_query_label_random = np.where(random_labels == query_label)[0]

        random_map = calculate_map(
            dist_matrix, random_labels, indices_with_query_label_random, query_label
        )

    return query_label, count, original_map, random_map
