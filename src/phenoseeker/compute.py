# Third-Party Libraries
import numpy as np
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


def calculate_lisi_score(
    embeddings: np.ndarray,
    labels: np.ndarray,
    n_neighbors: int,
    n_jobs: int,
) -> float:
    """
    Calculate LISI scores for given embeddings and labels.

    Args:
        embeddings (np.ndarray): Array of embeddings.
        labels (np.ndarray): Array of labels corresponding to embeddings.
        n_neighbors (int): Number of neighbors to consider for LISI calculation.
        n_jobs (int): Number of parallel jobs to use.

    Returns:
        float: Mean LISI score.
    """
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
    embeddings: np.ndarray,
    center_by: str,
    reduce_by: str,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the center and reduce arrays for normalization using numpy.

    Args:
        embeddings (np.ndarray): Embedding matrix.
        center_by (str): Centering method ('mean' or 'median').
        reduce_by (str): Reduction method ('std', 'iqrs', or 'mad').

    Returns:
        tuple: Arrays for center and reduce.
    """
    if center_by == "mean":
        center_array = np.mean(embeddings, axis=0)
    elif center_by == "median":
        center_array = np.median(embeddings, axis=0)
    else:
        raise ValueError("Invalid center_by method. Choose 'mean' or 'median'.")

    if reduce_by == "std":
        reduce_array = np.std(embeddings, axis=0)
    elif reduce_by == "iqrs":
        q1 = np.percentile(embeddings, 25, axis=0)
        q3 = np.percentile(embeddings, 75, axis=0)
        reduce_array = q3 - q1
    elif reduce_by == "mad":
        median = np.median(embeddings, axis=0)
        reduce_array = np.median(np.abs(embeddings - median), axis=0)
    else:
        raise ValueError("Invalid reduce_by method. Choose 'std', 'iqrs', or 'mad'.")

    return center_array, reduce_array


def calculate_map_efficient(
    dist_matrix: np.ndarray,
    labels: np.ndarray,
    indices_with_query_label: np.ndarray,
    sorted_indices: np.ndarray,
    query_label: str,
) -> float:
    """
    Efficiently calculate mean Average Precision (mAP) for a query label.

    Args:
        dist_matrix (np.ndarray): Distance matrix.
        labels (np.ndarray): Labels for all elements.
        indices_with_query_label (np.ndarray): Indices of elements with the query label.
        sorted_indices (np.ndarray): Precomputed sorted indices.
        query_label (str): The label to compute mAP for.

    Returns:
        float: Mean Average Precision.
    """
    mAP = 0.0
    count = len(indices_with_query_label)

    for i, query_idx in enumerate(indices_with_query_label):
        mask = np.ones(len(labels), dtype=bool)
        mask[query_idx] = False

        filtered_indices = sorted_indices[i][mask[sorted_indices[i]]]
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
