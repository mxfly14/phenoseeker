import random
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from joblib import Parallel, delayed
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import pairwise_distances
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
import umap
import warnings

from tqdm_joblib import tqdm_joblib

from .transform import Spherize
from .plotting import (
    plot_distrib_mix_max,
    plot_lisi_scores,
    plot_heatmaps,
    plot_maps,
)
from .compute import (
    compute_reduce_center,
    calculate_statistics,
    calculate_lisi_score,
    calculate_map_efficient,
)
from .norm_functions import (
    inverse_normal_transform,
    rescale,
    median_polish,
)
from .utils import (
    convert_row_to_number,
    test_distributions,
)


warnings.filterwarnings("ignore")


class EmbeddingManager:
    """
    Class to aggregate, normalize, and visualize embeddings.
    """

    def __init__(
        self,
        df: pd.DataFrame | Path,
        entity: str,
    ) -> None:

        if isinstance(df, Path):
            if df.suffix == ".csv":
                self.df = pd.read_csv(df)
            elif df.suffix == ".parquet":
                self.df = pd.read_parquet(df)
            else:
                raise ValueError(
                    "Unsupported file format. Please provide a CSV or Parquet file."
                )
        elif isinstance(df, pd.DataFrame):
            self.df = df
        else:
            raise TypeError(
                "df should be either a pandas DataFrame or a Path to a CSV/Parquet file."  # Noqa
            )

        #    self.df = self.df.sample(frac=1, ignore_index=True)
        #    Shuffle the DataFrame TODO: remove it
        self.entity = entity

        self.JCP_ID_poscon = [
            "JCP2022_085227",
            "JCP2022_037716",
            "JCP2022_025848",
            "JCP2022_046054",
            "JCP2022_035095",
            "JCP2022_064022",
            "JCP2022_050797",
            "JCP2022_012818",
        ]
        self.JCP_ID_controls = self.JCP_ID_poscon + ["JCP2022_033924"]
        # Initialize storage for embeddings as a dictionary
        self.embeddings = {}
        self.distance_matrices = {}

        if self.entity == "well":
            self.find_dmso_controls()

        if self.entity == "well" or self.entity == "image":
            if "Metadata_Row" not in self.df.columns:
                self.df[["Metadata_Row", "Metadata_Col"]] = self.df[
                    "Metadata_Well"
                ].apply(lambda x: pd.Series(self._well_to_row_col(x)))

            self.convert_row_col_to_number()

    def __len__(self) -> int:
        return len(self.df)

    def __repr__(self) -> str:
        return f"Embedding Manager with {len(self.df)} {self.entity}s embeddings"

    @staticmethod
    def _well_to_row_col(well: str) -> tuple[int, int]:
        row_part = "".join([ch for ch in well if ch.isalpha()])
        col_part = "".join([ch for ch in well if ch.isdigit()])

        row_index = 0
        for char in row_part:
            row_index = row_index * 26 + (ord(char.upper()) - ord("A") + 1)

        col_index = int(col_part)

        return row_index, col_index

    def load(
        self,
        embedding_name: str,
        embeddings_file: str | Path,
        metadata_file: str | Path | None = None,
        dtype: type | None = np.float64,
    ) -> None:
        """
        Load embeddings and associated metadata, align and store them in self.embeddings

        Args:
            embedding_name (str): Key for the loaded embeddings.
            embeddings_file (str | Path): Path to the .npy embeddings file.
            metadata_file (str | Path): Path to the metadata parquet file.
            dtype (type, optional): Data type for the embeddings. Defaults to np.float64
        """
        if metadata_file is not None:
            metadata_file = Path(metadata_file)
            if metadata_file.suffix != ".parquet":
                raise ValueError("Provided metadata path is not a valid parquet file.")
            loaded_metadata = pd.read_parquet(metadata_file)
        else:
            loaded_metadata = self.df

        embeddings_file = Path(embeddings_file)
        if not embeddings_file.exists() or embeddings_file.suffix != ".npy":
            raise ValueError("Provided embeddings path is not a valid .npy file.")
        embeddings = np.load(embeddings_file)

        # Définir la clé d'unicité selon l'entité
        if self.entity == "well":
            key_columns = ["Metadata_Well", "Metadata_Plate", "Metadata_Source"]
        elif self.entity == "image":
            key_columns = [
                "Metadata_Well",
                "Metadata_Plate",
                "Metadata_Source",
                "Metadata_Site",
            ]  # noqa
        elif self.entity == "compound":
            key_columns = ["Metadata_JCP2022"]
        else:
            raise ValueError(f"Unknown entity type: {self.entity}")

        # Comparer et aligner les métadonnées
        merged_metadata = self.df.merge(
            loaded_metadata, on=key_columns, how="inner", suffixes=("", "_loaded")
        )

        if len(merged_metadata) != len(self.df):
            warnings.warn(
                f"Metadata mismatch: using only {len(merged_metadata)} rows present in both metadata files."  # noqa
            )
            self.df = merged_metadata.drop(
                columns=[
                    col for col in merged_metadata.columns if col.endswith("_loaded")
                ]
            )

        # Réordonner les embeddings selon la même clé que self.df
        loaded_metadata["order_key"] = loaded_metadata[key_columns].apply(tuple, axis=1)
        self.df["order_key"] = self.df[key_columns].apply(tuple, axis=1)

        reorder_index = (
            self.df["order_key"]
            .map({k: i for i, k in enumerate(loaded_metadata["order_key"])})
            .values
        )

        # Réordonner les embeddings
        reordered_embeddings = embeddings[reorder_index.astype(int)]

        self.embeddings[embedding_name] = reordered_embeddings.astype(dtype)

        # Nettoyer la colonne temporaire
        self.df.drop(columns=["order_key"], inplace=True)

    def get_embeddings(self, embedding_name: str) -> np.ndarray:
        """
        Retrieve embeddings by their name.

        Args:
            embedding_name (str): The key corresponding to the desired embeddings.

        Returns:
            np.ndarray: The requested embeddings.
        """
        if embedding_name not in self.embeddings:
            raise KeyError(f"Embeddings with name '{embedding_name}' not found.")

        return self.embeddings[embedding_name]

    def convert_row_col_to_number(self) -> None:
        """
        Convert the Excel-style row and column labels to row and column numbers,
        and add the results to the DataFrame as new columns.
        """
        if "Metadata_Row_Number" not in self.df.columns:
            if not pd.api.types.is_integer_dtype(self.df["Metadata_Row"]):
                self.df["Metadata_Row_Number"] = self.df["Metadata_Row"].apply(
                    convert_row_to_number
                )
            else:
                self.df["Metadata_Row_Number"] = self.df["Metadata_Row"]

        if "Metadata_Col_Number" not in self.df.columns:
            self.df["Metadata_Col_Number"] = self.df["Metadata_Col"].apply(int)

    def compute_features_stats(
        self,
        embedding_name: str,
        plot: bool = False,
        n_jobs: int = -1,
    ) -> pd.DataFrame:
        """
        Compute statistical features for the specified embedding.

        Args:
            embedding_name (str): Name of the embedding to compute statistics for.
            plot (bool, optional): Whether to generate a plot. Defaults to False.
            n_jobs (int, optional): Number of parallel jobs. Defaults to -1.

        Returns:
            pd.DataFrame: DataFrame containing the computed statistics.
        """
        if embedding_name not in self.embeddings:
            raise ValueError(f"Embedding '{embedding_name}' not found.")

        embeddings = self.embeddings[embedding_name]

        def calculate_and_format_stats(subset_indices, plate_label):
            subset_embeddings = embeddings[subset_indices]
            control_indices = self.df.loc[subset_indices, "Metadata_Is_dmso"].values

            embeddings_control = subset_embeddings[control_indices]

            stats = calculate_statistics(subset_embeddings)
            stats_control = calculate_statistics(embeddings_control)
            stats_all = {**stats, **{k + "_dmso": v for k, v in stats_control.items()}}

            stats_df = pd.DataFrame(stats_all)
            stats_df.reset_index(inplace=True)
            stats_df.rename(columns={"index": "feature_index"}, inplace=True)
            stats_df.insert(0, "Metadata_Plate", plate_label)

            return stats_df

        # Compute global statistics
        global_stats_df = calculate_and_format_stats(range(len(self.df)), "all")

        # Compute per-plate statistics
        plates = self.df["Metadata_Plate"].unique()
        plate_stats_dfs = Parallel(n_jobs=n_jobs)(
            delayed(calculate_and_format_stats)(
                self.df[self.df["Metadata_Plate"] == plate].index, plate
            )
            for plate in tqdm(plates, desc="Calculating statistics for each plate")
        )

        all_stats_df = pd.concat([global_stats_df] + plate_stats_dfs, ignore_index=True)

        self.stats_df = all_stats_df

        if plot:
            plot_distrib_mix_max(global_stats_df, embeddings)

        return all_stats_df

    def plot_features_distributions(
        self,
        embedding_name: str,
        filter_dict: dict | None = None,
        feature_indices: list | None = None,
        bins: int | None = 10,
        log_scale: bool | None = False,
        hue_column: str | None = None,
    ) -> None:
        """
        Plot feature distributions for the specified embedding.

        Args:
            embedding_name (str): Name of the embedding to plot distributions for.
            filter_dict (dict, optional): Filters to apply to the DataFrame. Defaults
                to None.
            feature_indices (list, optional): Indices of features to plot. Defaults to
                None.
            bins (int, optional): Number of bins for the histogram. Defaults to 10.
            log_scale (bool, optional): Whether to use a logarithmic scale.
                Defaults to False.
            hue_column (str, optional): Column to use for coloring the histogram.
                Defaults to None.
        """
        if embedding_name not in self.embeddings:
            raise ValueError(f"Embedding '{embedding_name}' not found.")

        embeddings = self.embeddings[embedding_name]

        if filter_dict:
            filtered_df = self.df.copy()
            for key, values in filter_dict.items():
                if key not in filtered_df.columns:
                    raise ValueError(f"Column '{key}' not found in the DataFrame.")
                if isinstance(values, str):
                    values = [values]
                filtered_df = filtered_df[filtered_df[key].isin(values)]

            filtered_indices = filtered_df.index
            embeddings = embeddings[filtered_indices]
        else:
            filtered_df = self.df.copy()

        num_features = embeddings.shape[1]

        if feature_indices is None:
            feature_indices = [random.randint(0, num_features - 1)]
            n = 1
        else:
            n = len(feature_indices)

        _, axs = plt.subplots(n, 1, figsize=(10, 5 * n))
        if n == 1:
            axs = [axs]

        for i, feature_idx in enumerate(feature_indices):
            data = pd.DataFrame({"value": embeddings[:, feature_idx]})

            if hue_column and hue_column in filtered_df.columns:
                data[hue_column] = filtered_df.loc[filtered_df.index, hue_column].values
                sns.histplot(
                    data=data,
                    x="value",
                    hue=hue_column,
                    kde=True,
                    bins=bins,
                    log_scale=log_scale,
                    ax=axs[i],
                )
            else:
                sns.histplot(
                    data["value"],
                    kde=True,
                    bins=bins,
                    log_scale=log_scale,
                    ax=axs[i],
                )

            axs[i].set_title(f"Feature Distribution (index: {feature_idx})")

        plt.tight_layout()
        plt.show()

    def remove_features(
        self,
        embedding_name: str,
        threshold: float = 0.0,
        metrics: str | None = "std",
        dmso_only: bool = True,
        by_plate: bool = True,
    ) -> None:
        """
        Remove features from the specified embedding based on a threshold.

        Args:
            embedding_name (str): Name of the embedding to process.
            threshold (float, optional): Threshold for feature removal. Defaults to 0.0.
            metrics (str, optional): Metric to compute for feature evaluation ('std',
                'iqrs', 'mad'). Defaults to 'std'.
            dmso_only (bool, optional): Whether to consider only DMSO controls. Defaults
                to True.
            by_plate (bool, optional): Whether to process by plate. Defaults to True.
        """
        if embedding_name not in self.embeddings:
            raise ValueError(f"Embedding '{embedding_name}' not found.")

        embeddings = self.embeddings[embedding_name]
        plates = self.df["Metadata_Plate"].unique() if by_plate else ["all_data"]

        remove_mask = np.zeros(embeddings.shape[1], dtype=bool)

        for plate in tqdm(plates, desc="Processing plates"):
            if plate == "all_data":
                plate_indices = self.df.index
            else:
                plate_indices = self.df[self.df["Metadata_Plate"] == plate].index

            subset_embeddings = embeddings[plate_indices]

            if dmso_only:
                control_indices = self.df.loc[plate_indices, "Metadata_Is_dmso"]
                subset_embeddings = subset_embeddings[control_indices]

            if metrics == "std":
                feature_metric = subset_embeddings.std(axis=0)
            elif metrics == "iqrs":
                feature_metric = np.subtract(
                    *np.percentile(subset_embeddings, [75, 25], axis=0)
                )
            elif metrics == "mad":
                median = np.median(subset_embeddings, axis=0)
                feature_metric = np.median(np.abs(subset_embeddings - median), axis=0)
            else:
                raise ValueError(
                    "Unsupported metric. Choose from 'std', 'iqrs', or 'mad'."
                )

            remove_mask |= feature_metric <= threshold

        self.embeddings[embedding_name] = embeddings[:, ~remove_mask]
        num_features_removed = remove_mask.sum()
        print(f"Number of features removed: {num_features_removed}")

    def plot_dimensionality_reduction(
        self,
        embedding_name: str,
        reduction_method: str = "PCA",
        color_by: str | None = None,
        filter_dict: dict | None = None,
        n_components: int = 2,
        random_state: int = 42,
        save_path: Path | None = None,
    ) -> None:
        """
        Plot dimensionality reduction for the specified embedding.

        Args:
            embedding_name (str): Name of the embedding to reduce and plot.
            reduction_method (str, optional): Dimensionality reduction method ('PCA',
                't-SNE', 'UMAP'). Defaults to 'PCA'.
            color_by (str, optional): Column to use for coloring the plot. Defaults None
            filter_dict (dict, optional): Filters to apply to the df. Defaults to None.
            n_components (int, optional): Number of components for reduction. Defaults 2
            random_state (int, optional): Random state for reproducibility. Defaults 42.
            save_path (Path, optional): Path to save the reduced components.
                Defaults to None.
        """
        if embedding_name not in self.embeddings:
            raise ValueError(f"Embedding '{embedding_name}' not found.")

        embeddings = self.embeddings[embedding_name]

        if filter_dict:
            filtered_df = self.df.copy()
            for key, values in filter_dict.items():
                if key not in filtered_df.columns:
                    raise ValueError(f"Column '{key}' not found in the DataFrame.")
                if isinstance(values, str):
                    values = [values]
                filtered_df = filtered_df[filtered_df[key].isin(values)]

            filtered_indices = filtered_df.index
            embeddings = embeddings[filtered_indices]
        else:
            filtered_df = self.df.copy()

        if reduction_method == "PCA":
            reducer = PCA(n_components=n_components, random_state=random_state)
            reduced_embeddings = reducer.fit_transform(embeddings)
            explained_variance = reducer.explained_variance_ratio_

            plt.figure(figsize=(10, 4))
            plt.bar(range(1, n_components + 1), explained_variance, alpha=0.6)
            plt.xlabel("Principal Components")
            plt.ylabel("Explained Variance Ratio")
            plt.title("Explained Variance by Principal Components")
            plt.show()

            x_label = (
                f"Component 1 ({explained_variance[0]*100:.2f}% of Explained Variance)"
            )
            y_label = (
                f"Component 2 ({explained_variance[1]*100:.2f}% of Explained Variance)"
            )
        elif reduction_method == "t-SNE":
            reducer = TSNE(n_components=n_components, random_state=random_state)
            reduced_embeddings = reducer.fit_transform(embeddings)
            x_label = "Component 1"
            y_label = "Component 2"
        elif reduction_method == "UMAP":
            reducer = umap.UMAP(n_components=n_components, random_state=random_state)
            reduced_embeddings = reducer.fit_transform(embeddings)
            x_label = "Component 1"
            y_label = "Component 2"
        else:
            raise ValueError(
                f"Reduction method '{reduction_method}' not recognized. Choose from 'PCA', 't-SNE', or 'UMAP'."  # noqa
            )

        plt.figure(figsize=(10, 8))
        if color_by and color_by in self.df.columns:
            sns.scatterplot(
                x=reduced_embeddings[:, 0],
                y=reduced_embeddings[:, 1],
                hue=filtered_df[color_by],
                palette="Set1",
                s=50,
                alpha=0.8,
            )
            plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        else:
            plt.scatter(
                reduced_embeddings[:, 0], reduced_embeddings[:, 1], s=50, alpha=0.8
            )
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.title(f"Dimensionality Reduction using {reduction_method}")
        plt.tight_layout()
        plt.show()

        if save_path:
            reduced_df = pd.DataFrame(
                reduced_embeddings,
                columns=[f"Component_{i+1}" for i in range(n_components)],
            )
            reduced_df = pd.concat(
                [filtered_df.reset_index(drop=True), reduced_df], axis=1
            )
            reduced_df.to_csv(save_path, index=False)

    def apply_robust_Z_score(
        self,
        embeddings_name: str,
        new_embeddings_name: str | None = "robust_Z_score",
        use_control: bool | None = True,
        center_by: str | None = "mean",
        reduce_by: str | None = "std",
        n_jobs: int | None = -1,
    ) -> None:
        """
        Apply the robust Z-score normalization to each plate in the embeddings.

        Args:
            embedding_name (str): Name of the embedding to process.
            new_embeddings_name (str, optional): Name for the normalized embedding.
                Defaults to 'robust_Z_score'.
            use_control (bool, optional): Use control samples for computing the
                transformation. Defaults to True.
            center_by (str, optional): Centering method ('mean' or 'median').
                Defaults to 'mean'.
            reduce_by (str, optional): Reduction method ('std', 'iqrs', or 'mad').
                Defaults to 'std'.
            n_jobs (int, optional): Number of parallel jobs. Defaults to -1.
        """
        if embeddings_name not in self.embeddings:
            raise ValueError(f"Embedding '{embeddings_name}' not found.")

        embeddings = self.embeddings[embeddings_name]
        plates = self.df["Metadata_Plate"].unique()
        normalized_embeddings = np.empty_like(embeddings)

        def process_plate(plate):
            plate_indices = self.df[self.df["Metadata_Plate"] == plate].index
            plate_embeddings = embeddings[plate_indices]

            if use_control:
                control_indices = self.df.loc[plate_indices, "Metadata_Is_dmso"]
                control_embeddings = plate_embeddings[control_indices.values]
            else:
                control_embeddings = plate_embeddings

            center_array, reduce_array = compute_reduce_center(
                control_embeddings,
                center_by,
                reduce_by,
            )

            normalized_plate_embeddings = (
                plate_embeddings - center_array
            ) / reduce_array

            return plate_indices, normalized_plate_embeddings

        results = Parallel(n_jobs=n_jobs)(
            delayed(process_plate)(plate)
            for plate in tqdm(plates, desc="Normalizing plates with robust Z-score")
        )

        for plate_indices, normalized_plate_embeddings in results:
            normalized_embeddings[plate_indices] = normalized_plate_embeddings

        self.embeddings[new_embeddings_name] = normalized_embeddings

    def find_dmso_controls(self) -> None:
        if "Metadata_Is_dmso" not in self.df.columns:
            if "Metadata_InChI" in self.df.columns:
                self.df["Metadata_Is_dmso"] = self.df["Metadata_InChI"].apply(
                    lambda inchi: "InChI=1S/C2H6OS/c1-4(2)3/h1-2H3" == inchi
                )
            else:
                raise KeyError("Metadata_InChI column not found")

    def test_feature_distributions(
        self,
        embedding_name: str,
        continuous_distributions: list[str] | None = [
            "norm",
            "lognorm",
            "expon",
            "loggamma",
            "gamma",
            "beta",
            "chi2",
            "cauchy",
            "logistic",
            "pareto",
            "t",
            "skewnorm",
            "alpha",
        ],
        n_jobs: int | None = -1,
    ) -> pd.DataFrame:
        """
        Test feature distributions for the specified embedding.

        Args:
            embedding_name (str): Name of the embedding to test.
            continuous_distributions (list[str], optional): List of distributions to
                test against. Defaults to a predefined list.
            n_jobs (int, optional): Number of parallel jobs. Defaults to -1.

        Returns:
            pd.DataFrame: DataFrame containing distribution test results.
        """
        if embedding_name not in self.embeddings:
            raise ValueError(f"Embedding '{embedding_name}' not found.")

        embeddings = self.embeddings[embedding_name]
        results = pd.DataFrame(
            index=[f"Feature_{i}" for i in range(embeddings.shape[1])]
        )

        def local_test_distribution(column_index: int):
            return test_distributions(
                column_index, continuous_distributions, embeddings
            )

        results_list = Parallel(n_jobs=n_jobs)(
            delayed(local_test_distribution)(i)
            for i in tqdm(
                range(embeddings.shape[1]),
                desc="Testing features on classical distributions",
            )
        )

        for i, result in enumerate(results_list):
            for key, value in result.items():
                results.loc[f"Feature_{i}", key] = value

        return results

    def compute_lisi(
        self,
        labels_column: str,
        embeddings_names: list[str],
        n_neighbors_list: list[int] | None = [10, 15, 20, 30, 40, 50, 75, 100, 150],
        graph_title: str | None = "LISI scores for various aggregation pipelines",
        n_jobs: int = -1,
        random_lisi: bool = False,
        plot: bool = False,
    ) -> pd.DataFrame:
        """
        Compute the LISI scores for multiple embeddings and plot the results.

        Args:
            labels_column (str): Column name containing the labels.
            embeddings_names (list[str]): List of embedding names to compute LISI for.
            n_neighbors_list (list[int], optional): List of neighbor counts for LISI.
                Defaults to [10, 15, 20, 30, 40, 50, 75, 100, 150].
            graph_title (str, optional): Title for the LISI scores plot.
                Defaults to None.
            n_jobs (int, optional): Number of parallel jobs. Defaults to -1.
            plot (bool, optional): Whether to plot the LISI scores. Defaults to False.

        Returns:
            pd.DataFrame: DataFrame with LISI scores for each embedding and overall mean
        """
        if labels_column not in self.df.columns:
            raise ValueError(f"Column '{labels_column}' not found in the DataFrame.")

        labels = self.df[labels_column].values
        lisi_scores = {}

        for embedding_name in embeddings_names:
            if embedding_name not in self.embeddings:
                raise ValueError(f"Embedding '{embedding_name}' not found.")

            embeddings = self.embeddings[embedding_name]
            if random_lisi:
                lisi_scores[f"Ideal mixing ({embedding_name})"] = []

                random_labels = np.random.permutation(labels)
                for n_neighbors in tqdm(
                    n_neighbors_list,
                    desc=f"Calculating ideal mixing LISI scores for {embedding_name}",
                ):
                    score = calculate_lisi_score(
                        embeddings, random_labels, n_neighbors, n_jobs
                    )
                    lisi_scores[f"Ideal mixing ({embedding_name})"].append(score)

            lisi_scores[embedding_name] = []
            for n_neighbors in tqdm(
                n_neighbors_list, desc=f"Calculating LISI scores for {embedding_name}"
            ):
                score = calculate_lisi_score(embeddings, labels, n_neighbors, n_jobs)
                lisi_scores[embedding_name].append(score)

        lisi_df = pd.DataFrame(lisi_scores, index=n_neighbors_list)

        if plot:
            plot_lisi_scores(lisi_df, n_neighbors_list, graph_title)

        return lisi_df

    def plot_distance_matrix(
        self,
        embedding_name: str,
        distance: str = "cosine",
        sort_by: str | None = None,
        label_by: str | None = None,
        cmap: str = "coolwarm",
        n_jobs: int = -1,
        dtype: type = np.float32,
        fontsize: int = 8,
        similarity: bool = False,
    ) -> None:
        """
        Plot the distance or similarity matrix for the specified embedding.

        Args:
            embedding_name (str): Name of the embedding to plot distances for.
            distance (str, optional): Distance metric used to compute the matrix.
                Defaults to 'cosine'.
            sort_by (str, optional): Column name to sort the samples by.
                Defaults to None.
            label_by (str, optional): Column name to use for labels.
                Defaults to None.
            cmap (str, optional): Colormap for the heatmap. Defaults to 'coolwarm'.
            n_jobs (int, optional): Number of parallel jobs for distance computation.
                Defaults to -1.
            dtype (type, optional): Data type for the embeddings.
                Defaults to np.float32.
            fontsize (int, optional): Font size for the annotations. Defaults to 8.
            similarity (bool, optional): If True, plot similarity instead of distance.
                Defaults to False.
        """
        # Check if the matrix exists, otherwise compute it
        matrix_key = f"{'cosine_similarity' if similarity else distance + '_distance'}_matrix_{embedding_name}"  # noqa
        if matrix_key not in self.distance_matrices:
            print(f"Matrix '{matrix_key}' not found. Computing it now...")
            self.compute_distance_matrix(
                embedding_name, distance, n_jobs, dtype, similarity
            )

        # Retrieve the matrix
        matrix = self.distance_matrices[matrix_key]

        # Sort the matrix if sort_by is provided
        if sort_by is not None:
            if sort_by in self.df.columns:
                sorted_df = self.df.sort_values(by=sort_by)
                matrix = matrix[sorted_df.index][:, sorted_df.index]
            else:
                print(f"Warning: {sort_by} is not a valid column. No sorting applied.")
                sorted_df = self.df
        else:
            sorted_df = self.df

        # Determine labels
        if label_by is not None:
            if label_by in self.df.columns:
                ids = sorted_df[label_by].values
            else:
                print(
                    f"Warning: {label_by} is not a valid column. Using index as labels."
                )
                ids = sorted_df.index
        else:
            ids = sorted_df.index

        # Plot the heatmap
        _, ax = plt.subplots(figsize=(10, 8))
        cax = ax.imshow(matrix, cmap=cmap, interpolation="none")
        ax.set_title(
            f"{'Cosine Similarity' if similarity else distance.capitalize()} Matrix for {embedding_name}",  # noqa
            fontsize=14,
        )
        ax.set_xticks(range(len(ids)))
        ax.set_yticks(range(len(ids)))
        ax.set_xticklabels(ids, rotation=90, fontsize=fontsize)
        ax.set_yticklabels(ids, fontsize=fontsize)

        # Remove white lines between pixels
        ax.set_xticks([], minor=True)
        ax.set_yticks([], minor=True)
        ax.grid(False)

        # Add values in the cells
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                value = f"{matrix[i, j]:.2f}"  # Format to 2 decimal places
                ax.text(
                    j,
                    i,
                    value,
                    ha="center",
                    va="center",
                    fontsize=fontsize,
                    color="white",
                )

        # Add a color bar
        plt.colorbar(cax, ax=ax)

        plt.tight_layout()
        plt.show()

    def compute_distance_matrix(
        self,
        embedding_name: str,
        distance: str | None = "cosine",
        n_jobs: int | None = -1,
        dtype: type | None = np.float32,
        similarity: bool | None = False,
    ) -> None:
        """
        Compute a distance or similarity matrix for the specified embedding.

        Args:
            embedding_name (str): Name of the embedding to compute distances for.
            distance (str, optional): Distance metric to use. Defaults to 'cosine'.
            n_jobs (int, optional): Number of parallel jobs. Defaults to -1.
            dtype (type, optional): Data type for the embeddings.
                Defaults to np.float32.
            similarity (bool, optional): If True, compute similarity instead of
                distance. Defaults to False.
        """
        valid_distances = [
            "euclidean",
            "manhattan",
            "chebyshev",
            "minkowski",
            "cosine",
            "correlation",
            "jaccard",
            "mahalanobis",
            "callable",
        ]

        if distance not in valid_distances:
            raise ValueError(f"Distance metric '{distance}' is not supported.")

        if embedding_name not in self.embeddings:
            raise ValueError(f"Embedding '{embedding_name}' not found.")

        embeddings = self.embeddings[embedding_name].astype(dtype)
        distances = pairwise_distances(embeddings, metric=distance, n_jobs=n_jobs)

        if similarity and distance == "cosine":
            # Convert cosine distance to cosine similarity
            similarity_matrix = 1 - distances
            matrix_key = f"cosine_similarity_matrix_{embedding_name}"
            self.distance_matrices[matrix_key] = similarity_matrix
        else:
            matrix_key = f"{distance}_distance_matrix_{embedding_name}"
            self.distance_matrices[matrix_key] = distances

    def hierarchical_clustering_and_visualization(
        self,
        embedding_name: str,
        distance: str = "cosine",
        threshold: float = 0.5,
        similarity: bool = True,
    ):
        """
        Perform hierarchical clustering on molecules and visualize the results.

        Args:
            embedding_name (str): Name of the embedding to use.
            distance (str, optional): Distance metric used to compute the matrix.
                Defaults to 'cosine'.
            threshold (float): Threshold to determine clusters from the dendrogram.
            similarity (bool, optional): If True, use similarity matrix instead of
                distance. Defaults to True.
        """
        # Check if the matrix exists, otherwise compute it
        matrix_key = f"{'cosine_similarity' if similarity else distance + '_distance'}_matrix_{embedding_name}"  # noqa
        if matrix_key not in self.distance_matrices:
            print(f"Matrix '{matrix_key}' not found. Computing it now...")
            self.compute_distance_matrix(
                embedding_name, distance, similarity=similarity
            )

        # Retrieve the matrix
        matrix = self.distance_matrices[matrix_key]

        # Convertir la matrice de similarité en matrice de dissimilarité si nécessaire
        if similarity:
            dissimilarity = 1 - matrix
        else:
            dissimilarity = matrix

        # Créer un linkage pour le clustering hiérarchique
        linkage_matrix = linkage(dissimilarity, method="average")

        # Afficher le dendrogramme
        plt.figure(figsize=(10, 7))
        dendrogram(
            linkage_matrix,
            labels=self.df["Metadata_Molecule_ID"].values,
            leaf_rotation=90,
        )
        plt.title("Dendrogramme de clustering hiérarchique")
        plt.xlabel("Molécules")
        plt.ylabel("Distance")
        plt.axhline(y=threshold, color="r", linestyle="--", label=f"Seuil {threshold}")
        plt.legend()
        plt.show()

        # Déterminer les clusters à partir du seuil
        clusters = fcluster(linkage_matrix, t=threshold, criterion="distance")
        self.df["Cluster"] = clusters

    def compute_maps(
        self,
        labels_column: str,
        embeddings_names: list[str] | str,
        distance: str = "cosine",
        n_jobs: int = -1,
        weighted: bool = False,
        random_maps: bool = False,
        plot: bool = False,
        dtype: type | None = np.float32,
    ) -> pd.DataFrame:
        """
        Compute the mean average precision (mAP) for given embeddings and labels.

        Args:
            labels_column (str): Column name containing the labels.
            embeddings_names (list[str]): List of embedding names to compute mAP for.
            distance (str, optional): Distance metric to use. Defaults to 'cosine'.
            n_jobs (int, optional): Number of jobs for parallel processing.
                Defaults to -1.
            weighted (bool, optional): Weight the mAP by label frequency.
                Defaults to False.
            random_maps (bool, optional): Compute random mAP values. Defaults to False.
            plot (bool, optional): Whether to plot the mAP results. Defaults to True.
            dtype (type, optional): Data type for distance matrix computation.
                Defaults to np.float32.

        Returns:
            pd.DataFrame: DataFrame with mAP and random mAP for each label and the
            overall mean mAP.
        """
        if labels_column not in self.df.columns:
            raise ValueError(f"Column '{labels_column}' not found in the DataFrame.")

        labels = self.df[labels_column].values
        unique_labels = np.unique(labels)

        combined_results = {}
        label_frequencies = {label: np.sum(labels == label) for label in unique_labels}
        total_labels = len(labels)
        label_weights = {
            label: total_labels / freq for label, freq in label_frequencies.items()
        }
        max_weight = max(label_weights.values())
        label_weights = {
            label: weight / max_weight for label, weight in label_weights.items()
        }

        if isinstance(embeddings_names, str):
            embeddings_names = [embeddings_names]

        for embedding_name in embeddings_names:
            if embedding_name not in self.embeddings:
                raise ValueError(f"Embedding '{embedding_name}' not found.")

            # Compute or retrieve distance matrix
            if (
                f"{distance}_distance_matrix_{embedding_name}"
                not in self.distance_matrices
            ):
                self.compute_distance_matrix(embedding_name, distance, n_jobs, dtype)

            dist_matrix = self.distance_matrices[
                f"{distance}_distance_matrix_{embedding_name}"
            ]

            def compute_maps_label(query_label):
                indices_with_query_label = np.where(labels == query_label)[0]
                count = len(indices_with_query_label)
                if count <= 1:
                    return query_label, count, None, None

                sorted_indices = np.argsort(
                    dist_matrix[indices_with_query_label], axis=1
                )
                mAP = calculate_map_efficient(
                    labels,
                    indices_with_query_label,
                    sorted_indices,
                    query_label,
                )

                random_map = None
                if random_maps:
                    random_labels = labels.copy()
                    np.random.shuffle(random_labels)
                    random_indices = np.where(random_labels == query_label)[0]
                    random_map = calculate_map_efficient(
                        random_labels,
                        random_indices,
                        sorted_indices,
                        query_label,
                    )

                return query_label, count, mAP, random_map

            label_map_results = Parallel(n_jobs=n_jobs)(
                delayed(compute_maps_label)(query_label)
                for query_label in tqdm(
                    unique_labels, desc=f"Calculating mAP for {embedding_name}"
                )
            )

            for query_label, num_queries, mean_ap, mean_random_ap in label_map_results:
                if num_queries <= 1:
                    continue

                if query_label not in combined_results:
                    combined_results[query_label] = {
                        "Label": query_label,
                        "Number of Queries": int(num_queries),
                    }

                combined_results[query_label][f"mAP ({embedding_name})"] = mean_ap
                if random_maps:
                    combined_results[query_label][
                        f"Random mAP ({embedding_name})"
                    ] = mean_random_ap

        final_results = pd.DataFrame.from_dict(combined_results, orient="index")

        numeric_cols = final_results.select_dtypes(include=[np.number]).columns
        mean_map = final_results[numeric_cols].mean().to_frame().T
        mean_map["Label"] = "Mean mAP"

        if weighted:
            final_results["Weight"] = final_results["Label"].map(label_weights)
            weighted_map = (
                (
                    final_results.drop(columns="Number of Queries")
                    .select_dtypes(include=[np.number])
                    .mul(final_results["Weight"], axis=0)
                    .sum()
                    / final_results["Weight"].sum()
                )
                .to_frame()
                .T
            )
            weighted_map["Label"] = "Weighted Mean mAP"
            final_results = pd.concat(
                [final_results, mean_map, weighted_map], ignore_index=True
            )
        else:
            final_results = pd.concat([final_results, mean_map], ignore_index=True)

        final_results = final_results[
            ["Label"] + [col for col in final_results.columns if col != "Label"]
        ]
        if plot:
            plot_maps(final_results)

        return final_results

    def filter_and_instantiate(self, **filter_criteria) -> "EmbeddingManager":
        """
        Filter the DataFrame and associated embeddings based on criteria, and return
        a new instance of the class with the filtered data and embeddings.

        Args:
            **filter_criteria: Key-value pairs specifying the filtering conditions
                for the DataFrame.

        Returns:
            EmbeddingManager: A new instance of the class with filtered data and
            embeddings.

        Raises:
            ValueError: If any filter key is not a column in the DataFrame.
        """
        # Validate filter criteria keys
        for key in filter_criteria.keys():
            if key not in self.df.columns:
                raise ValueError(
                    f"Filter key '{key}' is not a column in the DataFrame."
                )

        # Filter the DataFrame based on provided criteria
        filtered_df = self.df
        for key, values in filter_criteria.items():
            if isinstance(values, list):
                filtered_df = filtered_df[filtered_df[key].isin(values)]
            else:
                filtered_df = filtered_df[filtered_df[key] == values]

        # Check if the filtered DataFrame is empty
        if filtered_df.empty:
            raise ValueError("No data matches the filter criteria.")

        # Filter the embeddings based on the filtered DataFrame indices
        filtered_indices = filtered_df.index
        filtered_embeddings = {
            name: embeddings[filtered_indices]
            for name, embeddings in self.embeddings.items()
        }

        # Create a new instance with the filtered data
        new_instance = EmbeddingManager(
            df=filtered_df.reset_index(drop=True),
            entity=self.entity,
        )

        # Assign filtered embeddings to the new instance
        new_instance.embeddings = filtered_embeddings

        return new_instance

    def grouped_embeddings(
        self,
        group_by: str,
        embeddings_to_aggregate: list[str] | None = None,
        cols_to_keep: list[str] | None = None,
        aggregation: str = "mean",
        n_jobs: int = -1,
    ) -> "EmbeddingManager":
        """
        Create a new instance of the class with grouped DataFrame and aggregated
        embeddings.

        Args:
            group_by (str): The entity to group by ('well' or 'compound').
            embeddings_to_aggregate (list[str] | None): Names of embeddings
                to aggregate. If None, all embeddings are aggregated.
            cols_to_keep (list[str], optional): Columns to keep in the resulting
                DataFrame.
            aggregation (str, optional): Aggregation method ('mean' or 'median').
                Defaults to 'mean'.
            n_jobs (int, optional): Number of parallel jobs to use. Defaults to -1.

        Returns:
            EmbeddingManager: A new instance with grouped DataFrame and aggregated
                embeddings.
        """
        if group_by in ["image", "well"]:
            group_by_columns = [
                "Metadata_Source",
                "Metadata_Plate",
                "Metadata_Row",
                "Metadata_Col",
            ]
            if cols_to_keep is None:
                cols_to_keep = [
                    "Metadata_Source",
                    "Metadata_Plate",
                    "Metadata_Well",
                    "Metadata_InChI",
                    "Metadata_Batch",
                    "Metadata_Is_dmso",
                ]
        elif group_by == "compound":
            if "Metadata_InChI_ID" not in self.df.columns:
                ids, _ = pd.factorize(self.df["Metadata_InChI"])
                self.df["Metadata_InChI_ID"] = ids
            group_by_columns = ["Metadata_InChI_ID"]
            if cols_to_keep is None:
                cols_to_keep = ["Metadata_InChI", "Metadata_Is_dmso"]
        else:
            raise ValueError(
                f"Group by '{group_by}' is not implemented. Use 'well' or 'compound'."
            )

        grouped = self.df.groupby(group_by_columns)
        grouped_indices = grouped.indices
        grouped_df = grouped[cols_to_keep].first().reset_index()

        def aggregate_group(
            embedding_array: np.ndarray, indices: list[int]
        ) -> np.ndarray:
            vectors = embedding_array[indices]
            if aggregation == "mean":
                return vectors.mean(axis=0)
            elif aggregation == "median":
                return np.median(vectors, axis=0)
            else:
                raise ValueError("Invalid aggregation method. Use 'mean' or 'median'.")

        # Determine embeddings to aggregate
        embeddings_keys = (
            embeddings_to_aggregate
            if embeddings_to_aggregate
            else list(self.embeddings)
        )

        new_instance = EmbeddingManager(df=grouped_df, entity=group_by)
        new_instance.embeddings = {}

        for key in embeddings_keys:
            if key not in self.embeddings:
                raise ValueError(f"Embedding '{key}' not found in self.embeddings.")

            emb_array = self.embeddings[key]
            aggregated = np.array(
                Parallel(n_jobs=n_jobs)(
                    delayed(aggregate_group)(emb_array, grouped_indices[group])
                    for group in tqdm(grouped.groups, desc=f"Aggregating {key}")
                )
            )
            new_instance.embeddings[key] = aggregated

        return new_instance

    def apply_inverse_normal_transform(
        self,
        embeddings_name: str,
        new_embeddings_name: str | None = None,
        indices: np.ndarray | None = None,
        n_jobs: int | None = -1,
    ) -> None:
        """
        Apply inverse normal transformation to features in a specified embedding.

        Args:
            embeddings_name (str): Name of the embedding to transform.
            new_embeddings_name (str, optional): Name for the transformed embedding.
                Defaults to 'int_{embeddings_name}'.
            indices (np.ndarray | None): Indices to apply the transformation to.
                If None, all indices are used.
            n_jobs (int | None): Number of parallel jobs to use. Defaults to -1.

        Returns:
            None: Modifies `self.embeddings` in place with the transformed embedding.
        """
        if embeddings_name not in self.embeddings:
            raise ValueError(
                f"Embedding '{embeddings_name}' not found in self.embeddings."
            )

        if new_embeddings_name is None:
            new_embeddings_name = f"int_{embeddings_name}"

        embeddings = self.embeddings[embeddings_name]

        if indices is None:
            indices = np.arange(embeddings.shape[1])

        plates = self.df["Metadata_Plate"].unique()
        plate_indices_mapping = {
            plate: self.df[self.df["Metadata_Plate"] == plate].index for plate in plates
        }

        def transform_plate(plate):
            plate_indices = plate_indices_mapping[plate]
            plate_embeddings = embeddings[plate_indices]

            transformed_plate_embeddings = plate_embeddings.copy()
            for column_index in indices:
                transformed_plate_embeddings[:, column_index] = (
                    inverse_normal_transform(plate_embeddings[:, column_index])
                )

            return plate_indices, transformed_plate_embeddings

        transformed_embeddings = np.zeros_like(embeddings)

        results = Parallel(n_jobs=n_jobs)(
            delayed(transform_plate)(plate)
            for plate in tqdm(plates, desc="Applying INT by plate")
        )
        for plate_indices, transformed_plate_embeddings in results:
            transformed_embeddings[plate_indices] = transformed_plate_embeddings

        self.embeddings[new_embeddings_name] = transformed_embeddings

    @staticmethod
    def _compute_covariance_and_correlation(
        embeddings: np.ndarray, by_sample: bool = True
    ) -> dict:
        """
        Calcule les matrices de covariance et de corrélation à partir des embeddings.

        Args:
            embeddings (np.ndarray): Un tableau numpy de forme (n_samples, n_features).

        Returns:
            dict: Un dictionnaire contenant les matrices de covariance et de corrélation
        """

        return {
            "covariance_matrix": np.cov(embeddings, rowvar=by_sample),
            "correlation_matrix": np.corrcoef(embeddings, rowvar=by_sample),
        }

    def plot_covariance_and_correlation(
        self,
        embeddings_name: str | None = "Embeddings_mean",
        by_sample: bool | None = False,
        use_dmso: bool | None = True,
        dmso_only: bool | None = False,
        sort_by: str | None = "Metadata_Source",
    ) -> None:
        """
        Create visualizations of covariance and correlation matrices.

        Args:
            embeddings_name (str): The name of the embedding to use.
            by_sample (bool): If True, calculate by sample. Defaults to False.
            use_dmso (bool): If True, filter the data based on DMSO controls.
            dmso_only (bool): If True, display only DMSO samples.
            sort_by (str): The column name to sort the samples by.
        """
        if embeddings_name not in self.embeddings:
            raise ValueError(
                f"Embedding '{embeddings_name}' not found in self.embeddings."
            )

        if use_dmso:
            if dmso_only:
                filtered_df = self.df[self.df["Metadata_Is_dmso"]]
            else:
                filtered_df = self.df
        else:
            filtered_df = self.df[~self.df["Metadata_Is_dmso"]]

        embeddings = self.embeddings[embeddings_name]

        if sort_by is not None:
            if sort_by in filtered_df.columns:
                filtered_df = filtered_df.sort_values(by=sort_by)
                embeddings = embeddings[filtered_df.index]
            else:
                print(f"Warning: {sort_by} is not a valid column. No sorting applied.")
        else:
            embeddings = embeddings[filtered_df.index]

        print("Computing matrices...")
        print(f"Embeddings shape: {embeddings.shape}")
        matrices = self._compute_covariance_and_correlation(embeddings, by_sample)

        labels = (
            filtered_df[sort_by].values if sort_by is not None and by_sample else None
        )

        plot_heatmaps(matrices, labels)

    def apply_rescale(
        self,
        embeddings_name: str,
        new_embeddings_name: str | None = None,
        scale: str | None = "0-1",
        n_jobs: int | None = -1,
    ) -> None:
        """
        Apply rescaling to the specified embedding and store the result.

        Args:
            embeddings_name (str): Name of the embedding to rescale.
            new_embeddings_name (str, optional): Name for the rescaled embedding.
                Defaults to 'rescale_{embeddings_name}'.
            scale (str, optional): Rescaling method. Defaults to '0-1'.
            n_jobs (int | None): Number of parallel jobs to use. Defaults to -1.

        Returns:
            None: Modifies `self.embeddings` in place with the rescaled embedding.
        """
        if embeddings_name not in self.embeddings:
            raise ValueError(
                f"Embedding '{embeddings_name}' not found in self.embeddings."
            )

        if new_embeddings_name is None:
            new_embeddings_name = f"rescale_{embeddings_name}"

        embeddings = self.embeddings[embeddings_name]

        def rescale_plate(plate_indices):
            plate_embeddings = embeddings[plate_indices]
            return rescale(plate_embeddings, scale)

        plates = self.df["Metadata_Plate"].unique()
        plate_indices_mapping = {
            plate: self.df[self.df["Metadata_Plate"] == plate].index for plate in plates
        }

        rescaled_embeddings = np.zeros_like(embeddings)

        results = Parallel(n_jobs=n_jobs)(
            delayed(rescale_plate)(plate_indices_mapping[plate])
            for plate in tqdm(plates, desc="Rescaling features by plates")
        )
        for plate, rescaled_plate_embeddings in zip(plates, results):
            rescaled_embeddings[plate_indices_mapping[plate]] = (
                rescaled_plate_embeddings
            )

        self.embeddings[new_embeddings_name] = rescaled_embeddings

    def apply_spherizing_transform(
        self,
        embeddings_name: str,
        new_embeddings_name: str | None = None,
        method: str | None = "ZCA",
        norm_embeddings: bool = True,
        use_control: bool = True,
        n_jobs: int = -1,
    ) -> None:
        """
        Apply a sphering (whitening) transformation to embeddings and store the result.

        Args:
            embeddings_name (str): Name of the embedding to transform.
            new_embeddings_name (str, optional): Name for the transformed embedding.
                Defaults to '{embeddings_name}_spherize'.
            method (str, optional): Sphering method ('PCA', 'ZCA', 'PCA-cor', 'ZCA-cor')
                Defaults to 'ZCA'.
            norm_embeddings (bool, optional): Whether to normalize embeddings to unit
                norm after sphering. Defaults to True.
            use_control (bool, optional): Whether to use control samples for computing
                the sphering transformation. Defaults to True.
            n_jobs (int, optional): Number of parallel jobs to use. Defaults to -1.

        Returns:
            None: Modifies `self.embeddings` in place with the transformed embedding.
        """
        if embeddings_name not in self.embeddings:
            raise ValueError(
                f"Embedding '{embeddings_name}' not found in self.embeddings."
            )

        if new_embeddings_name is None:
            new_embeddings_name = f"{embeddings_name}_spherize"

        embeddings = self.embeddings[embeddings_name]

        def sphering_plate(plate_indices):
            plate_embeddings = embeddings[plate_indices]
            if use_control:
                control_indices = [
                    i for i in plate_indices if self.df.iloc[i]["Metadata_Is_dmso"]
                ]
                if not control_indices:
                    raise ValueError(
                        f"No control samples found for plate: {self.df.loc[plate_indices[0], 'Metadata_Plate']}."  # noqa
                    )
                subset_embeddings = embeddings[control_indices]
            else:
                subset_embeddings = plate_embeddings

            # Initialize and fit the spherize transformer
            spherize = Spherize(epsilon=1e-5, center=True, method=method)
            spherize.fit(subset_embeddings)

            # Transform embeddings
            transformed_embeddings = spherize.transform(plate_embeddings)

            # Normalize if requested
            if norm_embeddings:
                norms = np.linalg.norm(transformed_embeddings, axis=1, keepdims=True)
                norms[norms < 1e-10] = 1.0  # Avoid division by zero
                transformed_embeddings /= norms

            return transformed_embeddings

        plates = self.df["Metadata_Plate"].unique()
        plate_indices_mapping = {
            plate: self.df[self.df["Metadata_Plate"] == plate].index.to_numpy()
            for plate in plates
        }

        sphered_embeddings = np.zeros_like(embeddings)

        results = Parallel(n_jobs=n_jobs)(
            delayed(sphering_plate)(plate_indices_mapping[plate])
            for plate in tqdm(
                plates, desc="Normalizing plates with Spherize transformation"
            )
        )
        for plate, transformed_plate_embeddings in zip(plates, results):
            sphered_embeddings[plate_indices_mapping[plate]] = (
                transformed_plate_embeddings
            )

        self.embeddings[new_embeddings_name] = sphered_embeddings

    def save_to_folder(
        self,
        folder_path: Path,
        embeddings_name: str = "all",
    ) -> None:
        """
        Save the DataFrame as a Parquet file and embeddings as NPY files.

        Args:
            folder_path (Path): Path to the folder where files will be saved.
            embeddings_name (str, optional): Name of the embedding to save. Use 'all'
                to save all embeddings. Defaults to 'all'.

        Returns:
            None
        """
        folder_path.mkdir(parents=True, exist_ok=True)

        # Save DataFrame as Parquet
        parquet_path = folder_path / "metadata.parquet"
        self.df.to_parquet(parquet_path, index=False)

        # Save embeddings
        if embeddings_name == "all":
            for name, embedding in self.embeddings.items():
                npy_path = folder_path / f"{name}.npy"
                np.save(npy_path, embedding)
        elif embeddings_name in self.embeddings:
            npy_path = folder_path / f"{embeddings_name}.npy"
            np.save(npy_path, self.embeddings[embeddings_name])
        else:
            raise ValueError(
                f"Embedding '{embeddings_name}' not found in self.embeddings."
            )

    def apply_median_polish(
        self,
        embeddings_name: str,
        new_embeddings_name: str = "Embeddings_MedianPolish",
        n_jobs: int = -1,
    ) -> None:
        """
        Apply median polish to each component of embeddings for each plate.

        Args:
            embeddings_name (str): Name of the embedding to process.
            new_embeddings_name (str, optional): Name for the adjusted embeddings.
                Defaults to 'Embeddings_MedianPolish'.
            n_jobs (int, optional): Number of parallel jobs to use. Defaults to -1.

        Returns:
            None: Updates `self.embeddings` in place with the adjusted embeddings.
        """
        if embeddings_name not in self.embeddings:
            raise ValueError(
                f"Embedding '{embeddings_name}' not found in self.embeddings."
            )

        embeddings = self.embeddings[embeddings_name]

        # Ensure adjusted_embeddings is writable
        adjusted_embeddings = np.zeros_like(embeddings, dtype=embeddings.dtype)

        def process_plate(plate):
            """
            Process a single plate: Apply median polish and return adjusted embeddings.
            """
            plate_indices = self.df[self.df["Metadata_Plate"] == plate].index
            plate_embeddings = embeddings[plate_indices]

            max_rows = self.df.loc[plate_indices, "Metadata_Row_Number"].max()
            max_columns = self.df.loc[plate_indices, "Metadata_Col_Number"].max()

            n_components = plate_embeddings.shape[1]
            data_tensor = np.zeros((max_rows, max_columns, n_components))

            for i, idx in enumerate(plate_indices):
                row = self.df.loc[idx, "Metadata_Row_Number"] - 1
                col = self.df.loc[idx, "Metadata_Col_Number"] - 1
                data_tensor[row, col, :] = plate_embeddings[i]

            adjusted_data_tensor = np.zeros_like(data_tensor)
            for i in range(n_components):
                component_data = data_tensor[:, :, i]
                result = median_polish(component_data)
                adjusted_data_tensor[:, :, i] = (
                    result["ave"] + result["row"][:, None] + result["col"][None, :]
                )

            # Map adjusted data back to the embedding array
            plate_adjusted_embeddings = np.zeros_like(plate_embeddings)
            for i, idx in enumerate(plate_indices):
                row = self.df.loc[idx, "Metadata_Row_Number"] - 1
                col = self.df.loc[idx, "Metadata_Col_Number"] - 1
                plate_adjusted_embeddings[i] = adjusted_data_tensor[row, col, :]

            return plate_indices, plate_adjusted_embeddings

        plates = self.df["Metadata_Plate"].unique()

        with tqdm_joblib(tqdm(total=len(plates), desc="Applying Median Polish")):
            results = Parallel(n_jobs=n_jobs)(
                delayed(process_plate)(plate) for plate in plates
            )

        # Combine results from all plates
        for plate_indices, plate_adjusted_embeddings in results:
            adjusted_embeddings[plate_indices] = plate_adjusted_embeddings

        # Store the adjusted embeddings
        self.embeddings[new_embeddings_name] = adjusted_embeddings
