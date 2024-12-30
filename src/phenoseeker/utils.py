# Standard Library
import logging
import warnings
from pathlib import Path

# Third-Party Libraries
import pandas as pd
import numpy as np
import yaml
from tqdm import tqdm
from joblib import Parallel, delayed
from PIL import Image
import PIL
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import NearestNeighbors
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

warnings.filterwarnings("ignore")


def modify_first_layer(model, in_channels: int = 5):
    """
    Modifie la première couche Conv2d d'un modèle DINOv2 pour accepter un nombre
    arbitraire de canaux d'entrée. Les nouveaux canaux sont initialisés en copiant les
    poids des canaux existants.

    :param model: Le modèle DINOv2 à modifier.
    :param in_channels: Le nombre de canaux d'entrée souhaité (par défaut 5).
    """

    # Accéder à la première couche Conv2d
    original_conv = model.patch_embed.proj
    out_channels = original_conv.out_channels
    kernel_size = original_conv.kernel_size
    stride = original_conv.stride
    padding = original_conv.padding
    dilation = original_conv.dilation
    groups = original_conv.groups
    bias = original_conv.bias is not None

    # Créer une nouvelle couche Conv2d avec le nombre de canaux d'entrée souhaité
    new_conv = nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
        bias=bias,
    )

    # Copier les poids existants de l'ancienne couche
    with torch.no_grad():
        # Copier les poids pour les 3 premiers canaux
        new_conv.weight[:, :3, :, :] = original_conv.weight

        # Copier les poids des 2 premiers canaux sur les nouveaux canaux (canaux 4 et 5)
        new_conv.weight[:, 3:, :, :] = original_conv.weight[:, :2, :, :]

        # Copier le biais s'il existe
        if bias:
            new_conv.bias = original_conv.bias

    # Remplacer la couche existante dans le modèle
    model.patch_embed.proj = new_conv


def calculate_mean_std(df: pd.DataFrame, batch_size=64, num_workers=4):
    # Create dataset and dataloader
    dataset = MultiChannelImageDataset(df)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    # To store sum and sum of squared values for mean and std computation
    channel_sum = torch.zeros(5)
    channel_sum_squared = torch.zeros(5)
    total_pixels = 0

    print("Calculating mean and standard deviation...")

    # Iterate over batches of images
    for images, _ in tqdm(dataloader):
        # Images shape: (batch_size, 5, H, W) - batch of 5-channel images

        batch_pixels = (
            images.shape[0] * images.shape[2] * images.shape[3]
        )  # batch_size * height * width
        total_pixels += batch_pixels

        # Sum of pixel values for each channel (sum along batch, height, width)
        channel_sum += images.sum(dim=[0, 2, 3])

        # Sum of squared pixel values for each channel (sum along batch, height, width)
        channel_sum_squared += (images**2).sum(dim=[0, 2, 3])

    # Calculate mean and std
    mean = channel_sum / total_pixels
    std = torch.sqrt((channel_sum_squared / total_pixels) - (mean**2))

    print(f"Mean: {mean}")
    print(f"Std: {std}")

    return mean, std


class MultiChannelImageDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        metadata_cols: list[str],
        images_cols: list[str],
        transform: transforms = None,
        image_size: tuple[int, int] = (768, 768),
    ):
        """
        Args:
            df (pd.DataFrame): DataFrame with image paths and metadata.
            metadata_cols (list): List of column names for metadata.
            images_cols (list): List of column names for image paths.
            transform (callable, optional): Optional transform to be applied on an image
            image_size (tuple): Size of fallback tensors for invalid/missing images.
        """
        self.df = df
        self.metadata_cols = metadata_cols
        self.images_cols = images_cols
        self.transform = transform
        self.image_size = image_size

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        img_paths = self.df.iloc[idx][self.images_cols].values
        image_channels = []

        for img_path in img_paths:
            try:
                img = Image.open(img_path).convert("L")
                img_tensor = transforms.ToTensor()(img).squeeze(0)

            except (FileNotFoundError, PIL.UnidentifiedImageError) as e:
                logging.error(f"Error loading {img_path}: {e}")
                img_tensor = torch.zeros(self.image_size)

            except Exception as e:
                logging.error(f"Unexpected error loading {img_path}: {e}")
                # Fallback to a zero tensor for any other error
                img_tensor = torch.zeros(self.image_size)
            image_channels.append(img_tensor)

        image = torch.stack(image_channels, dim=0)
        if self.transform:
            image = self.transform(image)

        metadata = self.df.iloc[idx][self.metadata_cols].to_dict()
        return image, metadata


def load_config(config_path: Path) -> dict:
    """
    Load configuration from a YAML file.
    Args:
        config_path (Path): Path to the configuration YAML file.
    Returns:
        dict: Configuration as a dictionary.
    """
    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration file not found at {config_path}")
    with config_path.open("r") as file:
        config = yaml.safe_load(file)
    return config


def plot_distrib_mix_max(stats_df: pd.DataFrame, embeddings: np.ndarray):
    highest_var_idx = stats_df["var"].idxmax()
    lowest_var_idx = stats_df["var"].idxmin()

    if highest_var_idx >= embeddings.shape[1] or lowest_var_idx >= embeddings.shape[1]:
        raise IndexError("Index out of bounds for the embeddings array.")

    _, axs = plt.subplots(2, 2, figsize=(12, 10))

    sns.histplot(embeddings[:, highest_var_idx], kde=True, ax=axs[0, 0])
    axs[0, 0].set_title(f"Highest Variance Feature (index: {highest_var_idx})")

    second_highest_var_idx = stats_df["var"].nlargest(2).index[1]
    if second_highest_var_idx >= embeddings.shape[1]:
        raise IndexError("Index out of bounds for the embeddings array.")

    sns.histplot(
        embeddings[:, second_highest_var_idx],
        kde=True,
        ax=axs[0, 1],
    )
    axs[0, 1].set_title(
        f"Second Highest Variance Feature (index: {second_highest_var_idx})"
    )

    sns.histplot(embeddings[:, lowest_var_idx], kde=True, ax=axs[1, 0])
    axs[1, 0].set_title(f"Lowest Variance Feature (index: {lowest_var_idx})")

    second_lowest_var_idx = stats_df["var"].nsmallest(2).index[1]
    if second_lowest_var_idx >= embeddings.shape[1]:
        raise IndexError("Index out of bounds for the embeddings array.")

    sns.histplot(embeddings[:, second_lowest_var_idx], kde=True, ax=axs[1, 1])
    axs[1, 1].set_title(
        f"Second Lowest Variance Feature (index: {second_lowest_var_idx})"
    )

    plt.tight_layout()
    plt.show()


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


def plot_lisi_scores(lisi_df, n_neighbors_list, graph_title):
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(10, 6))
    for method in lisi_df.columns:
        sns.scatterplot(x=n_neighbors_list, y=lisi_df[method], label=method)

    plt.ylabel("LISI score")
    plt.xlabel("Number of neighbours")
    plt.title(graph_title)
    plt.legend(title="Normalization Method")
    plt.show()


def tensor_median(tensor_list: list[np.ndarray]) -> np.ndarray:
    """
    Compute the median of a list of arrays.

    Parameters:
    - tensor_list: A list of arrays for which the median is computed.

    Returns:
    - median_array: The median array.
    """
    stack = np.stack(tensor_list)
    median_array = np.median(stack, axis=0)
    return median_array


def process_group_field2well(
    data: pd.DataFrame,
    vector_column: str,
    new_vector_column: str,
    cols_to_keep: list[str],
    aggregation,
):
    if aggregation == "mean":
        agg_func = np.mean
    elif aggregation == "median":
        agg_func = tensor_median
    else:
        raise ValueError(
            f"Aggregation method '{aggregation}' not recognized. It must be either 'mean' ou 'median'"  # noqa
        )

    aggregated_vector = agg_func(np.stack(data[vector_column]), axis=0)
    new_data = data[cols_to_keep].iloc[0].to_dict()
    new_data[new_vector_column] = aggregated_vector
    return new_data


def save_filtered_df_with_components(
    filtered_df: pd.DataFrame,
    reduced_embeddings: np.ndarray,
    n_components: int,
    save_path: Path,
    reduction_method: str,
):
    component_columns = [
        f"Component_{i+1}_of_{reduction_method}" for i in range(n_components)
    ]
    components_df = pd.DataFrame(reduced_embeddings, columns=component_columns)
    filtered_df = pd.concat([filtered_df, components_df], axis=1)
    save_path = Path(save_path)
    save_path.mkdir(exist_ok=True)

    csv_path = save_path / f"{reduction_method}.csv"
    filtered_df.to_csv(csv_path, index=False)


def convert_row_to_number(row: str) -> int:
    """Convert an Excel-style row label to a row number.

    Args:
        row (str): The Excel-style row label.

    Returns:
        int: The corresponding row number.
    """
    if len(row) == 1:
        return ord(row) - ord("A") + 1
    else:
        return (ord(row[0]) - ord("A") + 1) * 26 + (ord(row[1]) - ord("A") + 1)


def test_distributions(
    column_index: int,
    continuous_distributions: list,
    embeddings: np.ndarray,
):
    """
    continuous_distributions : list of Scipy distributions.
    see: https://docs.scipy.org/doc/scipy/reference/stats.html
    """

    dist_results = {}
    data = embeddings[:, column_index]

    if len(data) == 0:
        print(f"Warning: Empty data for column Feature_{column_index}")
        return dist_results

    for dist_name in continuous_distributions:
        try:
            dist = getattr(stats, dist_name)
            params = dist.fit(data)
            _, p_value = stats.kstest(data, dist_name, args=params)
            follows_dist = p_value >= 0.05
            dist_results[f"{dist_name}_p_value"] = p_value
            dist_results[f"follows_{dist_name}"] = follows_dist
        except Exception as e:
            dist_results[f"{dist_name}_error"] = str(e)

    return dist_results


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


def apply_Z_score_plate(
    df: pd.DataFrame, raw_embedding_col: str, save_embedding_col: str
) -> pd.DataFrame:
    """
    Apply Z-score normalization to a DataFrame using scipy.stats.zscore.

    Args:
        df (pd.DataFrame): DataFrame containing the data.
        raw_embedding_col (str): Column name for the raw image embeddings in the df.
        save_embedding_col (str): DataFrame column name for saving the normalized
            embeddings.

    Returns:
        pd.DataFrame: DataFrame with the normalized embeddings.
    """

    if df[raw_embedding_col].isnull().any():
        raise ValueError(f"Missing values found in column {raw_embedding_col}.")

    data_tensor = np.stack(df[raw_embedding_col])
    z_score_data_tensor = stats.zscore(data_tensor, axis=0)

    df[save_embedding_col] = [
        z_score_data_tensor[row_id] for row_id, _ in df.reset_index().iterrows()
    ]
    return df


def apply_robust_Z_score_plate(
    df: pd.DataFrame,
    raw_embedding_col: str,
    save_embedding_col: str,
    center_array: np.ndarray,
    reduce_array: np.ndarray,
) -> pd.DataFrame:
    """
    Apply Z-score normalization to a DataFrame using the provided center and reduce
    arrays.

    Args:
        df (pd.DataFrame): DataFrame containing the data.
        raw_embedding_col (str): Column name for the raw image embeddings in the
        DataFrame.
        save_embedding_col (str): DataFrame column name for saving the normalized
        embeddings.
        center_array (np.ndarray): Array for centering the data.
        reduce_array (np.ndarray): Array for reducing the data.

    Returns:
        pd.DataFrame: DataFrame with the normalized embeddings.
    """

    if df[raw_embedding_col].isnull().any():
        raise ValueError(f"Missing values found in column {raw_embedding_col}.")

    max_rows = df["Metadata_Row_Number"].max()
    max_columns = df["Metadata_Col_Number"].max()
    data_tensor = np.zeros(
        (max_rows, max_columns, center_array.shape[0]), dtype=np.float32
    )

    for _, row in df.iterrows():
        tensor = row[raw_embedding_col]
        if not isinstance(tensor, np.ndarray):
            raise ValueError(
                f"Invalid tensor type: {type(tensor)}. Expected np.ndarray."
            )
        data_tensor[
            row["Metadata_Row_Number"] - 1, row["Metadata_Col_Number"] - 1, :
        ] = tensor

    z_score_data_tensor = np.zeros_like(data_tensor, dtype=np.float32)
    for i in range(center_array.shape[0]):
        z_score_data_tensor[:, :, i] = (
            data_tensor[:, :, i] - center_array[i]
        ) / reduce_array[i]

    df[save_embedding_col] = [
        z_score_data_tensor[
            row["Metadata_Row_Number"] - 1, row["Metadata_Col_Number"] - 1, :
        ]
        for _, row in df.iterrows()
    ]

    return df


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


# Why not exactly as the previous ? (~1%delta) Slower than previous

# from sklearn.metrics import average_precision_score
#
#
# def calculate_map(
#    dist_matrix: np.ndarray,
#    labels: np.ndarray,
#    indices_with_query_label: np.ndarray,
#    query_label: str,
# ) -> float:
#    mAP = 0.0
#    count = len(indices_with_query_label)
#
#    for i in indices_with_query_label:
#        # Scores pour toutes les autres instances
#        scores = dist_matrix[i]
#        # Labels binaires: 1 pour les mêmes labels, 0 sinon
#        binary_labels = (labels == query_label).astype(int)
#        binary_labels[i] = 0  # Exclure la requête elle-même
#
#        # Calcul de l'average precision pour cette requête
#        ap = average_precision_score(binary_labels, -scores)
#        mAP += ap
#
#    mAP /= count
#    return mAP


def calculate_maps(
    dist_matrix: np.ndarray,
    query_label: str,
    labels: np.ndarray,
) -> tuple[str, int, float | None, float | None]:
    """
    Calculate the original and random Mean Average Precision (MAP) for a given label.

    Args:
        dist_matrix (np.ndarray): Distance matrix between elements.
        query_label (str): The label to query.
        labels (np.ndarray): Array of labels for all elements.

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

    random_labels = labels.copy()
    np.random.shuffle(random_labels)
    indices_with_query_label_random = np.where(random_labels == query_label)[0]

    random_map = calculate_map(
        dist_matrix, random_labels, indices_with_query_label_random, query_label
    )

    return query_label, count, original_map, random_map


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


def process_labels_for_heatmaps(labels: np.ndarray) -> np.ndarray:
    groups = []
    current_label = labels[0]
    count = 0

    for label in labels:
        if label == current_label:
            count += 1
        else:
            groups.append((current_label, count))
            current_label = label
            count = 1
    groups.append((current_label, count))

    new_list = [""] * len(labels)
    index = 0

    for label, group_size in groups:
        mid_index = index + group_size // 2
        new_list[mid_index] = label
        index += group_size

    return new_list


def plot_heatmaps(matrices: dict[str, np.ndarray], labels: np.ndarray | None) -> None:
    covariance_matrix = matrices["covariance_matrix"]
    correlation_matrix = matrices["correlation_matrix"]
    n = len(covariance_matrix)

    if labels is None:
        labels = ["" for _ in range(n)]
        unique_labels = []

    else:
        unique_labels = np.unique(labels)

    print(f"Plotting matrices of size {n}x{n}...")

    unique_labels = np.unique(labels)
    separators = []

    for label in unique_labels:
        indices = np.where(labels == label)[0]
        if len(indices) > 0:
            separators.append(indices[-1] + 0.5)

    _, axes = plt.subplots(1, 2, figsize=(14, 7))

    labels_to_plot = process_labels_for_heatmaps(labels)

    sns.heatmap(
        covariance_matrix,
        ax=axes[0],
        cmap="viridis",
        annot=False,
        cbar=True,
        xticklabels=labels_to_plot,
        yticklabels=labels_to_plot,
    )
    for sep in separators:
        axes[0].axhline(sep, color="black", linewidth=1)
        axes[0].axvline(sep, color="black", linewidth=1)
    axes[0].set_title("Matrice de Covariance")

    sns.heatmap(
        correlation_matrix,
        ax=axes[1],
        cmap="viridis",
        annot=False,
        cbar=True,
        xticklabels=labels_to_plot,
        yticklabels=labels_to_plot,
    )
    for sep in separators:
        axes[1].axhline(sep, color="black", linewidth=1)
        axes[1].axvline(sep, color="black", linewidth=1)
    axes[1].set_title("Matrice de Corrélation")

    plt.tight_layout()
    plt.show()


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


def plot_maps(df):

    mAP_columns = [
        col for col in df.columns if col.startswith("mAP") or col.startswith("Random")
    ]
    df_melted = df.melt(
        id_vars=["Label"],
        value_vars=mAP_columns,
        var_name="mAP Type",
        value_name="Value",
    )

    df_melted["mAP Type"] = df_melted["mAP Type"].astype(str)
    df_melted["Label"] = df_melted["Label"].astype(str)

    plt.figure(figsize=(20, 8))
    sns.scatterplot(
        x="mAP Type", y="Value", hue="Label", style="Label", data=df_melted, s=100
    )

    plt.grid(True)
    plt.title("Scatter Plot des valeurs mAP pour chaque Label", fontsize=16)
    plt.xlabel("Type de mAP", fontsize=14)
    plt.ylabel("Valeur", fontsize=14)
    plt.xticks(rotation=90, ha="center")
    plt.tight_layout()

    plt.show()
