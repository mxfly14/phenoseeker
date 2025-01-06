# Standard Library
import logging
import warnings
import psutil
from pathlib import Path

# Third-Party Libraries
import numpy as np
import pandas as pd
from scipy import stats
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm
import yaml

# PyTorch Libraries
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# Suppress warnings
warnings.filterwarnings("ignore")


def check_free_memory() -> int:
    """Check the amount of free memory in the system."""
    memory_info = psutil.virtual_memory()
    return int(memory_info.available / (1024**3))


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

            except (FileNotFoundError, UnidentifiedImageError) as e:
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
