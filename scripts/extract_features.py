import os
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import pandas as pd
import numpy as np
import logging
from tqdm import tqdm

from phenoseeker import load_config, modify_first_layer, MultiChannelImageDataset

# Configure logging
log_file = Path("./tmp/log_extraction.txt")
log_file.parent.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(log_file, mode="w"), logging.StreamHandler()],
)


def validate_config(config: dict, required_keys: list):
    """
    Validate the presence of required keys in the configuration file.

    Args:
        config (dict): Configuration dictionary.
        required_keys (list): List of keys that must be present in the config.

    Returns:
        dict: The validated configuration with defaults where applicable.
    """
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing required configuration key: {key}")
    return config


def validate_columns(
    df: pd.DataFrame, metadata_cols: list[str], images_cols: list[str]
):
    """
    Validate that the specified columns exist in the DataFrame.

    Args:
        df (pd.DataFrame): The input DataFrame.
        metadata_cols (list): List of metadata column names.
        images_cols (list): List of image column names.

    Raises:
        ValueError: If any column is missing from the DataFrame.
    """
    missing_metadata_cols = [col for col in metadata_cols if col not in df.columns]
    missing_images_cols = [col for col in images_cols if col not in df.columns]

    if missing_metadata_cols:
        raise ValueError(
            f"The following metadata columns are missing in the DataFrame: {missing_metadata_cols}"  # NOQA
        )
    if missing_images_cols:
        raise ValueError(
            f"The following image columns are missing in the DataFrame: {missing_images_cols}"  # NOQA
        )


def main_worker(model, df: pd.DataFrame, config: dict):
    """
    Main worker function for GPU-based feature extraction.
    """
    # Set device
    gpu_id = config.get("gpu_id", 0)
    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(gpu_id)

    num_workers = int(config.get("num_workers", os.cpu_count() // 2))
    batch_size = int(config.get("batch_size", 64))
    output_folder = Path(config.get("output_folder", "./tmp"))
    output_folder.mkdir(parents=True, exist_ok=True)

    metadata_cols = config.get("metadata_cols", [])
    images_cols = config.get("images_cols", [])

    if not metadata_cols or not images_cols:
        raise ValueError(
            "metadata_cols and images_cols must be defined in the configuration file."
        )

    # Validate the columns in the DataFrame
    validate_columns(df, metadata_cols, images_cols)

    # Transformation pipeline
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            # Add normalization if needed: transforms.Normalize(mean, std)
        ]
    )
    dataset = MultiChannelImageDataset(
        df, metadata_cols=metadata_cols, images_cols=images_cols, transform=transform
    )
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=2,
    )

    logging.info(f"Using device: {device}")
    logging.info(f"Dataset size: {len(dataset)}")
    logging.info(f"Number of batches: {len(dataloader)}")

    model = model.to(device)
    model.eval()

    features_list = []
    metadata_list = []

    logging.info("Starting feature extraction...")
    with torch.no_grad():
        for images, metadata in tqdm(dataloader, desc="Feature Extraction"):
            images = images.to(device, non_blocking=True)
            with torch.amp.autocast("cuda"):
                outputs = model(images)

            features_list.append(outputs.cpu().numpy())
            metadata_list.append(metadata)

    metadata_df = pd.concat([pd.DataFrame(d) for d in metadata_list], ignore_index=True)
    features = np.concatenate(features_list, axis=0)

    np.save(output_folder / "extracted_features.npy", features)
    metadata_df.to_csv(output_folder / "metadata.csv", index=False)
    logging.info("Feature extraction completed.")


if __name__ == "__main__":
    # Load configuration
    config_path = Path("./configs/config_extraction.yaml")
    config = load_config(config_path)
    validate_config(
        config,
        required_keys=["parquet_path", "output_folder", "metadata_cols", "images_cols"],
    )

    parquet_path = config.get("parquet_path")
    if not Path(parquet_path).is_file():
        raise FileNotFoundError(f"Parquet file not found: {parquet_path}")

    logging.info("Loading dataset and initializing model...")
    try:
        df = pd.read_parquet(parquet_path).sample(n=50)
    except Exception as e:
        logging.error(f"Failed to load parquet file: {e}")
        raise

    try:
        dinov2 = torch.hub.load("facebookresearch/dinov2", "dinov2_vitg14")
    except Exception as e:
        logging.error(f"Failed to load DINOv2 model: {e}")
        raise

    modify_first_layer(dinov2, in_channels=5)
    main_worker(dinov2, df, config)
