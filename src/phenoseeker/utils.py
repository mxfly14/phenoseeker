from torch.utils.data import Dataset, DataLoader
import pandas as pd
from torchvision import transforms
import torch
import torch.nn as nn
from pathlib import Path
from tqdm import tqdm
from PIL import Image
import PIL
import yaml
import logging


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
