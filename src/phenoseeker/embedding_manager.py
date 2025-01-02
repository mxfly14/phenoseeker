import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from joblib import Parallel, delayed
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import pairwise_distances
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
    calculate_maps,
)
from .norm_functions import (
    apply_Z_score_plate,
    apply_robust_Z_score_plate,
    apply_int_subset,
    apply_spherize_subset,
    rescale,
)
from .utils import (
    save_filtered_df_with_components,
    process_group_field2well,
    convert_row_to_number,
    test_distributions,
)

METADATA_PREFIX = "Metadata_"  # TODO: should we keep this ?
EMBEDDING_PREFIX = "Embeddings_"

warnings.filterwarnings("ignore")


class EmbeddingManager:
    """
    Class that has a lot of methods to aggregate, normalise or visualize embeddings
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
                "df should be either a pandas DataFrame or a Path to a CSV/Parquet file."  # noqa
            )

        self.df = self.df.sample(frac=1, ignore_index=True)

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
        self.no_dmso_plates = [
            "Dest210823-174240",
            "Dest210628-162003",
            "Dest210823-174422",
        ]
        self.embedding_columns = {
            col
            for col in self.df.columns
            if EMBEDDING_PREFIX in col  # TODO: this should be a method
        }
        self.distance_matrices = {}
        self.find_dmso_controls()

        if self.entity == "well":
            self.df[["Metadata_Row", "Metadata_Col"]] = self.df["Metadata_Well"].apply(
                lambda x: pd.Series(self._well_to_row_col(x))
            )

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
        path_column: str,
        n_cpus: int | None = -1,
        vectors_column: str | None = "Embeddings_mean",
    ) -> None:

        paths = self.df[path_column]

        def load_and_convert(path):
            if not Path(path).exists():
                print(f"Path does not exist: {path}")
                return None
            try:
                tensor = torch.load(path, weights_only=True)
                return tensor.numpy()
            except Exception as e:
                print(f"Error loading {path}: {e}")
                return None

        results = Parallel(n_jobs=n_cpus)(
            delayed(load_and_convert)(path)
            for path in tqdm(paths, desc="Loading embeddings")
        )

        self.df[vectors_column] = results

    def compute_features_stats(
        self,
        vectors_column: str = "Embeddings_mean",
        plot: bool = False,
        n_jobs: int = -1,
    ) -> pd.DataFrame:
        if vectors_column not in self.df.columns:
            raise ValueError(f"Column {vectors_column} not found in the DataFrame.")

        def calculate_and_format_stats(df_subset, plate_label):
            embeddings = np.stack(df_subset[vectors_column])

            embeddings_control = np.stack(
                df_subset[df_subset["Metadata_Is_Control"]][vectors_column]
            )

            stats = calculate_statistics(embeddings)
            stats_control = calculate_statistics(embeddings_control)
            stats_all = {**stats, **{k + "_dmso": v for k, v in stats_control.items()}}

            stats_df = pd.DataFrame(stats_all)
            stats_df.reset_index(inplace=True)
            stats_df.rename(columns={"index": "feature_index"}, inplace=True)
            stats_df.insert(0, "Metadata_Plate", plate_label)

            return stats_df

        global_stats_df = calculate_and_format_stats(self.df, "all")

        plates = self.df["Metadata_Plate"].unique()
        plate_stats_dfs = Parallel(n_jobs=n_jobs)(
            delayed(calculate_and_format_stats)(
                self.df[self.df["Metadata_Plate"] == plate], plate
            )
            for plate in tqdm(plates, desc="Calculating statistics for each plate")
        )

        all_stats_df = pd.concat([global_stats_df] + plate_stats_dfs, ignore_index=True)

        self.stats_df = all_stats_df

        if plot:
            plot_distrib_mix_max(
                global_stats_df,
                np.stack(self.df[vectors_column]),
            )

        return all_stats_df

    def plot_features_distributions(
        self,
        vectors_column: str = "Embeddings_mean",
        filter_dict: dict | None = None,
        feature_indices: list | None = None,
        bins: int | None = 10,
        log_scale: bool | None = False,
        hue_column: str | None = None,
    ) -> None:
        if vectors_column not in self.df.columns:
            raise ValueError(f"Column {vectors_column} not found in the DataFrame.")

        if filter_dict:
            for key in filter_dict.keys():
                if key not in self.df.columns:
                    raise ValueError(f"Column {key} not found in the DataFrame.")
            filter_mask = pd.Series([True] * len(self.df))

            for col, values in filter_dict.items():
                if isinstance(values, str):
                    values = [values]
                df_ = self.df[col].isin(values)
                filter_mask = filter_mask & df_

            filtered_df = self.df.loc[filter_mask].reset_index(drop=True).copy()
            embeddings = np.stack(filtered_df[vectors_column])

        else:
            filtered_df = self.df.copy()
            embeddings = np.stack(filtered_df[vectors_column])

        num_features = embeddings.shape[1]

        if feature_indices is None:
            n = 1
            feature_indices = [random.randint(0, num_features - 1)]
        else:
            n = len(feature_indices)

        _, axs = plt.subplots(n, 1, figsize=(10, 5 * n))
        if n == 1:
            axs = [axs]

        for i, feature_idx in enumerate(feature_indices):
            data = pd.DataFrame({"value": embeddings[:, feature_idx]})

            if hue_column and hue_column in filtered_df.columns:
                data[hue_column] = filtered_df[hue_column].values
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
        threshold: float = 0.0,
        metrics: str | None = "std",
        vectors_column: str | None = "Embeddings_mean",
        dmso_only: bool | None = True,
        by_plate: bool | None = True,
    ) -> None:
        if vectors_column not in self.df.columns:
            raise ValueError(
                f"Column {vectors_column} does not exist in the DataFrame."
            )
        if by_plate:
            plates = self.df["Metadata_Plate"].unique()
        else:
            plates = ["all_data"]  # TODO implement the rest of the function for this

        remove_mask = np.zeros(self.df[vectors_column].iloc[0].shape[0], dtype=bool)

        for plate in tqdm(plates, desc="Processing plates"):
            if dmso_only:
                embeddings_plate = np.stack(
                    list(
                        self.df[
                            (self.df["Metadata_Plate"] == plate)
                            & (self.df["Metadata_Is_Control"])
                        ][vectors_column]
                    )
                )
            else:
                embeddings_plate = np.stack(
                    list(self.df[self.df["Metadata_Plate"] == plate][vectors_column])
                )

            if metrics == "std":
                feature_metric = embeddings_plate.std(axis=0)
            if metrics == "iqrs":
                feature_metric = np.subtract(
                    *np.percentile(embeddings_plate, [75, 25], axis=0)
                )
            if metrics == "mad":
                median = np.median(embeddings_plate, axis=0).astype(np.float32)
                feature_metric = np.median(
                    np.abs(embeddings_plate - median), axis=0
                ).astype(np.float32)
            remove_mask |= feature_metric <= threshold

        filtered_embeddings = np.stack(
            [emb[~remove_mask] for emb in self.df[vectors_column]]
        )

        self.df[vectors_column] = list(filtered_embeddings)
        num_features_removed = remove_mask.sum()
        print(f"Number of features removed: {num_features_removed}")

    def plot_dimensionality_reduction(
        self,
        vectors_column: str | None = "Embeddings_mean",
        reduction_method: str | None = "PCA",
        color_by: str | None = None,
        filter_dict: dict | None = None,
        n_components: int | None = 2,
        random_state: int | None = 42,
        save_path: Path | None = None,
    ) -> None:
        if vectors_column not in self.df.columns:
            raise ValueError(f"Column {vectors_column} not found in the DataFrame.")

        if filter_dict:
            for key in filter_dict.keys():
                if key not in self.df.columns:
                    raise ValueError(f"Column {key} not found in the DataFrame.")
            filter_mask = pd.Series([True] * len(self.df))

            for col, values in filter_dict.items():
                if isinstance(values, str):
                    values = [values]
                df_ = self.df[col].isin(values)
                filter_mask = filter_mask & df_

            filtered_df = self.df.loc[filter_mask].reset_index(drop=True).copy()
            embeddings = np.stack(filtered_df[vectors_column])

        else:
            filtered_df = self.df.copy()
            embeddings = np.stack(filtered_df[vectors_column])

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
                f"Reduction method {reduction_method} not recognized. Choose from 'PCA', 't-SNE', or 'UMAP'."  # noqa:E501
            )

        plt.figure(figsize=(10, 8))
        if color_by and color_by in self.df.columns:
            filtered_df[color_by] = pd.Categorical(
                filtered_df[color_by],
                categories=sorted(filtered_df[color_by].unique()),
                ordered=True,
            )
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
        if color_by:
            metadata = color_by.split("_")[1]
            plt.title(
                f"Dimensionality Reduction using {reduction_method} for {metadata}s"
            )
        plt.tight_layout()
        plt.show()

        if save_path:
            save_filtered_df_with_components(
                filtered_df,
                reduced_embeddings,
                n_components,
                save_path,
                reduction_method,
            )

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

    def apply_Z_score(
        self,
        raw_embedding_col: str,
        save_embedding_col: str | None = "Z_score",
        use_control: bool | None = True,
        n_jobs: int | None = -1,
    ) -> None:
        """
        Apply the Z-score normalization to each plate in the DataFrame in
        parallel using joblib.

        Args:
            raw_embedding_col (str): Column name for the raw image embeddings in the
                DataFrame.
            save_embedding_col (str, optional): DataFrame column name for saving the
                Z-score normalized embeddings. Defaults to 'Z_score'.
            use_control (bool, optional): Whether to use control samples for computing
                the transformation. If True, only control samples are used to compute
                center_by and reduce_by. Defaults to True.
            n_jobs (int, optional): Number of jobs to run in parallel. Defaults to -1.
        """

        def apply_robust_Z_score_to_subset(plate_df):
            return apply_Z_score_plate(
                plate_df,
                raw_embedding_col,
                save_embedding_col,
            )

        plates = self.df["Metadata_Plate"].unique()

        results = Parallel(n_jobs=n_jobs)(
            delayed(apply_robust_Z_score_to_subset)(
                self.df[self.df["Metadata_Plate"] == plate].copy()
            )
            for plate in tqdm(plates, desc="Normalising plates with Z-score")
        )

        self.df = pd.concat(results, ignore_index=True)

    def apply_robust_Z_score(
        self,
        raw_embedding_col: str,
        save_embedding_col: str | None = "robust_Z_score",
        use_control: bool | None = True,
        center_by: str | None = "mean",
        reduce_by: str | None = "std",
        n_jobs: int | None = -1,
    ) -> None:
        """
        Apply the robust Z-score normalization to each plate in the DataFrame in
        parallel using joblib.

        Args:
            raw_embedding_col (str): Column name for the raw image embeddings in the
                DataFrame.
            save_embedding_col (str, optional): DataFrame column name for saving the
                robust Z-score normalized embeddings. Defaults to 'robust_Z_score'.
            n_jobs (int, optional): Number of jobs to run in parallel. Defaults to -1.
            use_control (bool, optional): Whether to use control samples for computing
                the transformation. If True, only control samples are used to compute
                center_by and reduce_by. Defaults to True.
            center_by (str, optional): Method for centering, either 'mean' or 'median'.
            reduce_by (str, optional): Method for reduction, either 'std', 'iqrs', or
            'mad'.
        """

        def apply_robust_Z_score_to_subset(plate_df):
            center_array, reduce_array = compute_reduce_center(
                plate_df,
                raw_embedding_col,
                use_control,
                center_by,
                reduce_by,
            )
            return apply_robust_Z_score_plate(
                plate_df,
                raw_embedding_col,
                save_embedding_col,
                center_array,
                reduce_array,
            )

        plates = self.df["Metadata_Plate"].unique()
        results = Parallel(n_jobs=n_jobs)(
            delayed(apply_robust_Z_score_to_subset)(
                self.df[self.df["Metadata_Plate"] == plate].copy()
            )
            for plate in tqdm(plates, desc="Normalising plates with robust Z-score")
        )
        self.df = pd.concat(results, ignore_index=True)

    def find_dmso_controls(self) -> None:
        if "Metadata_Is_Control" not in self.df.columns:
            if "Metadata_InChI" in self.df.columns:
                self.df["Metadata_Is_Control"] = self.df["Metadata_InChI"].apply(
                    lambda inchi: "InChI=1S/C2H6OS/c1-4(2)3/h1-2H3" == inchi
                )
            else:
                raise KeyError("Metadata_InChI column not found")

    def test_feature_distributions(
        self,
        vectors_column: str | None = "Embeddings_mean",
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

        embeddings = np.stack(self.df[vectors_column])

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
        vectors_columns: dict | None = {"Current Embeddings": "Embeddings_mean"},
        n_neighbors_list: list | None = [10, 15, 20, 30, 40, 50, 75, 100, 150],
        graph_title: str | None = "LISI scores for various aggregation pipelines",
        n_jobs=-1,
        plot=False,
    ) -> pd.DataFrame:
        """
        Compute the LISI scores for multiple embedding columns and plot the results.

        :param vectors_columns: Dictionary with embedding names as keys and column
        names as values.
        :param labels_column: Column name containing the labels.
        :param n_neighbors_list: List of neighbor counts to use for LISI calculation.
        :param graph_title: Title for the LISI scores plot.
        :param n_jobs: Number of jobs for parallel processing.
        :param plot: Whether to plot the LISI scores.
        :return: DataFrame with LISI scores for each label and the overall mean LISI.
        """
        for method, column in vectors_columns.items():
            if column not in self.df.columns:
                raise ValueError(f"Column {column} not found in the DataFrame.")

        if labels_column not in self.df.columns:
            raise ValueError(f"Column {labels_column} not found in the DataFrame.")

        labels = self.df[labels_column].values
        lisi_scores = {}

        for method, column in vectors_columns.items():
            lisi_scores[f"Ideal mixing ({column})"] = []
            random_labels = np.random.permutation(labels)
            embeddings = np.stack(self.df[column])

            for n_neighbors in tqdm(
                n_neighbors_list,
                desc=f"Calculating ideal mixing LISI scores for {method}",
            ):
                lisi_score = calculate_lisi_score(
                    embeddings, random_labels, n_neighbors, n_jobs
                )
                lisi_scores[f"Ideal mixing ({column})"].append(lisi_score)

            lisi_scores[method] = []
            for n_neighbors in tqdm(
                n_neighbors_list, desc=f"Calculating LISI scores for {method}"
            ):
                lisi_score = calculate_lisi_score(
                    embeddings, labels, n_neighbors, n_jobs
                )
                lisi_scores[method].append(lisi_score)

        lisi_df = pd.DataFrame(lisi_scores, index=n_neighbors_list)

        if plot:
            plot_lisi_scores(lisi_df, n_neighbors_list, graph_title)

        return lisi_df

    def compute_distance_matrix(
        self,
        vectors_column: str | None = "Embeddings_mean",
        distance: str | None = "cosine",
        n_jobs: int | None = -1,
    ) -> None:
        if distance not in [
            "euclidean",
            "manhattan",
            "chebyshev",
            "minkowski",
            "cosine",
            "correlation",
            "jaccard",
            "mahalanobis",
            "callable",
        ]:
            raise ValueError(f"Distance metric '{distance}' is not supported.")

        embeddings = np.stack(self.df[vectors_column])
        distances = pairwise_distances(embeddings, metric=distance, n_jobs=n_jobs)

        self.distance_matrices[f"{distance}_distance_matrix_{vectors_column}"] = (
            distances
        )

    def compute_maps(
        self,
        labels_column: str,
        vectors_columns: dict | None = None,
        distance: str | None = "cosine",
        n_jobs: int | None = -1,
        weighted: bool | None = False,
        random_maps: bool | None = False,
        plot: bool | None = True,
    ) -> pd.DataFrame:
        """
        Compute the mean average precision (mAP) for a given distance matrix and label
        column for multiple embedding columns.

        :param vectors_columns: Dictionary with embedding names as keys and column
        names as values.
        :param labels_column: Column name containing the labels.
        :param distance: Distance metric to use (default is 'cosine').
        :param n_jobs: Number of jobs for parallel processing.
        :param weighted: Boolean indicating whether to weight the mAP by label
        frequency.
        :param random_maps: Boolean indicating whether to compute random mAP values.
        :return: DataFrame with mAP and random mAP for each label and the overall mean
        mAP.
        """

        if vectors_columns is None:
            vectors_columns = {"Current Embeddings": "Embeddings_mean"}

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

        for emb_name, vectors_column in vectors_columns.items():
            try:
                if (
                    f"{distance}_distance_matrix_{vectors_column}"
                    not in self.distance_matrices
                ):
                    self.compute_distance_matrix(vectors_column, distance)

                dist_matrix = self.distance_matrices[
                    f"{distance}_distance_matrix_{vectors_column}"
                ]
                if dist_matrix.shape[0] > 30000:
                    self.distance_matrices.clear()

                def compute_maps_label(query_label):
                    return calculate_maps(
                        dist_matrix, query_label, np.array(labels), random_maps
                    )

                label_map_results = Parallel(n_jobs=n_jobs)(
                    delayed(compute_maps_label)(query_label)
                    for query_label in tqdm(
                        unique_labels, desc=f"Calculating mAP for {emb_name}"
                    )
                )

                for (
                    query_label,
                    num_queries,
                    mean_ap,
                    mean_random_ap,
                ) in label_map_results:
                    if num_queries <= 1:
                        continue

                    if query_label not in combined_results:
                        combined_results[query_label] = {
                            "Label": query_label,
                            "Number of Queries": int(num_queries),
                        }

                    combined_results[query_label][f"mAP ({emb_name})"] = mean_ap
                    if random_maps:
                        combined_results[query_label][
                            f"Random mAP ({emb_name})"
                        ] = mean_random_ap
            except ValueError as e:
                print(f"Erreur computing map for {emb_name}: {str(e)}")
                continue

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

    def filter_and_instantiate(self, **filter_criteria):
        filtered_df = self.df
        for key, values in filter_criteria.items():
            if key in filtered_df.columns:
                if isinstance(values, list):
                    filtered_df = filtered_df[filtered_df[key].isin(values)]
                else:
                    filtered_df = filtered_df[filtered_df[key] == values]

        new_instance = EmbeddingManager(
            df=filtered_df,
            entity=self.entity,
        )

        return new_instance

    def grouped_embeddings(
        self,
        group_by: str,
        vector_column: str | None = "Embeddings_mean",
        new_vector_column: str | None = "Embeddings_mean",
        cols_to_keep: list[str] | None = None,
        aggregation: str | None = "mean",
        n_jobs: int | None = -1,
    ):
        """
        Create a new instance of the class with a grouped DataFrame.

        :param group_by: The column to group by ('well' ou 'compound').
        :param vector_column: The column containing the embedding vectors.
        :param new_vector_column: The column name for the new aggregated embedding.
        :param cols_to_keep: List of columns to keep in the resulting DataFrame.
        :param aggregation: The aggregation method to use ('mean' ou 'median').
        :param n_jobs: Number of parallel jobs to use (-1 for all available cores).
        :return: A new instance of the class with the grouped DataFrame.
        """
        if group_by == "well":
            group_by_columns = ["Metadata_Source", "Metadata_Plate", "Metadata_Well"]
            if cols_to_keep is None:
                cols_to_keep = [
                    "Metadata_Source",
                    "Metadata_Plate",
                    "Metadata_Well",
                    "Metadata_InChI",
                    "Metadata_PlateType",
                    "Metadata_Batch",
                    "Metadata_Is_Control",
                ]

        elif group_by == "compound":
            ids, _ = pd.factorize(self.df["Metadata_InChI"])
            self.df["Metadata_InChI_ID"] = ids
            group_by_columns = "Metadata_InChI_ID"
            if cols_to_keep is None:
                cols_to_keep = ["Metadata_InChI", "Metadata_Is_Control"]

        else:
            raise ValueError(
                f"Group by '{group_by}' is not implemented. It should be 'well' or 'compound'"  # noqa
            )

        def aggr_function(data):
            try:
                return process_group_field2well(
                    data, vector_column, new_vector_column, cols_to_keep, aggregation
                )
            except Exception as e:
                print(f"Error processing data: {data}, error: {e}")
                return None

        grouped = self.df.groupby(group_by_columns)

        results_with_none = Parallel(n_jobs=n_jobs)(
            delayed(aggr_function)(data)
            for _, data in tqdm(
                grouped, desc=f"Grouping by {group_by} using {aggregation} aggregation"
            )
        )

        results = [result for result in results_with_none if result is not None]

        if len(results_with_none) > len(results):
            print(f"There is {len(results_with_none) - len(results)} None {group_by}")

        new_df = pd.DataFrame(results)
        new_em = EmbeddingManager(new_df, group_by)
        return new_em

    def apply_inverse_normal_transform(
        self,
        raw_embedding_col: str,
        save_embedding_col: str | None = None,
        indices: np.ndarray | None = None,
        n_jobs: int | None = -1,
    ) -> None:

        if save_embedding_col is None:
            save_embedding_col = f"int_{raw_embedding_col}"

        plates = self.df["Metadata_Plate"].unique()
        results = [
            apply_int_subset(
                self.df[self.df["Metadata_Plate"] == plate].copy(),
                raw_embedding_col,
                save_embedding_col,
                indices,
                n_jobs,
            )
            for plate in tqdm(plates, desc="Normalising plates with INT")
        ]
        self.df = pd.concat(results, ignore_index=True)

    def apply_sphering(
        self,
        raw_embedding_col: str,
        save_embedding_col: str | None = None,
        center_by: str = "mean",
        norm_embeddings: bool = True,
        use_control: bool = True,
        n_jobs: int | None = -1,
    ) -> None:
        """
        Applies sphering (whitening) transformation to embeddings and saves
            the transformed embeddings.

        :param raw_embedding_col: Name of the column containing raw embeddings.
        :param save_embedding_col: Name of the column to save the sphered embeddings.
            If None, appends '_spherize' to the raw column name.
        :param center_by: Method to center the data ('mean' or 'median').
        :param norm_embeddings: If True, normalizes the transformed embeddings
            to unit norm.
        :param use_control: If True, uses control samples for centering.
        :param n_jobs: Number of parallel jobs to run. -1 means using all processors.
        :return: None. The DataFrame `self.df` is modified in place.
        """
        if save_embedding_col is None:
            save_embedding_col = f"{raw_embedding_col}_spherize"

        # Ensure required columns exist in the DataFrame
        required_columns = [raw_embedding_col, "Metadata_Is_Control", "Metadata_Plate"]
        missing_columns = [
            col for col in required_columns if col not in self.df.columns
        ]
        if missing_columns:
            raise ValueError(
                f"The following required columns are missing in the DataFrame: {missing_columns}"  # NOQA
            )

        # Get unique plates for parallel processing
        plates = self.df["Metadata_Plate"].unique()

        def apply_sphering_for_plate(plate_df: pd.DataFrame) -> pd.DataFrame:
            """
            Applies sphering to a single plate's embeddings.

            :param plate_df: DataFrame containing data for a single plate.
            :return: DataFrame with the sphered embeddings added.
            """
            try:
                plate_df = plate_df.reset_index(drop=True)
                embeddings = np.stack(plate_df[raw_embedding_col].values)

                if use_control:
                    control_indices = plate_df.index[
                        plate_df["Metadata_Is_Control"]
                    ].tolist()
                    if not control_indices:
                        raise ValueError(
                            f"No control samples found for plate {plate_df['Metadata_Plate'].iloc[0]}"  # NOQA
                        )
                else:
                    control_indices = []

                new_embeddings, center, transformation_matrix = apply_spherize_subset(
                    embeddings=embeddings,
                    control_indices=control_indices,
                    norm_embeddings=norm_embeddings,
                    center_by=center_by,
                    use_control=use_control,
                )

                plate_df[save_embedding_col] = [
                    embedding.tolist() for embedding in new_embeddings
                ]

                return plate_df
            except Exception as e:
                print(
                    f"Error spherizing plate {plate_df['Metadata_Plate'].iloc[0]}: {e}"
                )
                return plate_df

        results = Parallel(n_jobs=n_jobs)(
            delayed(apply_sphering_for_plate)(
                self.df[self.df["Metadata_Plate"] == plate].copy()
            )
            for plate in tqdm(plates, desc="Normalizing plates with Spherize")
        )

        self.df = pd.concat(results, ignore_index=True)

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
        by_sample: bool | None = False,
        use_dmso: bool | None = True,
        dmso_only: bool | None = False,
        embedding_col: str | None = "Embeddings_mean",
        sort_by: str | None = None,
    ) -> None:
        """
        Crée des visualisations des matrices de covariance et de corrélation.

        Args:
            use_dmso (bool): Si True, filtrer les données en fonction des contrôles.
            dmso_only (bool): Si True, n'afficher que les échantillons DMSO.
            embedding_col (str): Le nom de la colonne contenant les embeddings.
            sort_by (str): Le nom de la colonne pour trier les échantillons.
        """
        if use_dmso:
            if dmso_only:
                filtered_df = self.df[self.df["Metadata_Is_Control"]]
            else:
                filtered_df = self.df
        else:
            filtered_df = self.df[~self.df["Metadata_Is_Control"]]

        if sort_by is not None and sort_by in filtered_df.columns:
            filtered_df = filtered_df.sort_values(by=sort_by)
        elif sort_by is not None and sort_by not in filtered_df.columns:
            print(f"Warning: {sort_by} is not a valid column. No sorting applied.")

        print("Computing matrices...")
        embeddings = np.stack(filtered_df[embedding_col])

        print(f"Embeddings shape: {embeddings.shape}")
        matrices = self._compute_covariance_and_correlation(embeddings, by_sample)
        labels = (
            filtered_df[sort_by].values if sort_by is not None and by_sample else None
        )

        plot_heatmaps(matrices, labels)

    def apply_rescale(
        self,
        raw_embedding_col: str,
        save_embedding_col: str | None = None,
        scale: str | None = "0-1",
        n_jobs: int | None = -1,
    ) -> None:

        if save_embedding_col is None:
            save_embedding_col = f"rescale_{raw_embedding_col}"

        def apply_rescale_for_multi_threading(df: pd.DataFrame) -> pd.DataFrame:
            embeddings = np.stack(df[raw_embedding_col])
            new_embeddings = rescale(embeddings, scale)
            df[save_embedding_col] = list(new_embeddings)
            return df

        plates = self.df["Metadata_Plate"].unique()

        results = Parallel(n_jobs=n_jobs)(
            delayed(apply_rescale_for_multi_threading)(
                self.df[self.df["Metadata_Plate"] == plate].copy()
            )
            for plate in tqdm(plates, desc="Rescaling features by plates")
        )

        self.df = pd.concat(results, ignore_index=True)

    @staticmethod
    def _apply_spherizing_subset(
        embeddings: np.ndarray,
        control_indices: list[int],
        norm_embeddings: bool,
        method: str | None,
        use_control: bool | None = True,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Applies sphering (whitening) transformation using the Spherize class to a
        subset of embeddings.

        Parameters:
            embeddings (np.ndarray): Array of shape (n_samples, n_features) containing
            the embeddings.
            control_indices (list[int]): List of indices to use as control samples for
            centering.
            norm_embeddings (bool): Whether to normalize embeddings to unit norm after
            sphering.
            center_by (str): Method to center the data ('mean' or 'median').
            use_control (bool): Whether to use control samples for centering.

        Returns:
            tuple: (transformed_embeddings, center, transformation_matrix)
        """
        if method is None:
            method = "ZCA"  # TODO Choose the best method once yoou know it
        else:
            if method not in ["PCA", "ZCA", "PCA-cor", "ZCA-cor"]:
                raise ValueError(
                    "Wrong method, should be either PCA, ZCA, PCA-cor or ZCA-cor"
                )

        if use_control and not control_indices:
            raise ValueError(
                "No control indices provided, but `use_control` is set to True."
            )

        # Select control samples if use_control is True
        if use_control:
            subset_embeddings = embeddings[control_indices]
        else:
            subset_embeddings = embeddings

        # Initialize the Spherize transformer
        spherize = Spherize(epsilon=1e-5, center=True, method=method)

        # Fit the Spherize transformer on the subset
        spherize.fit(subset_embeddings)

        # Transform all embeddings using the fitted transformer
        transformed_embeddings = spherize.transform(embeddings)

        # Retrieve the center used for sphering
        if spherize.method in ["PCA-cor", "ZCA-cor"]:
            center = spherize.standard_scaler.mean_
        else:
            if spherize.center:
                center = spherize.mean_centerer.mean_
            else:
                center = np.zeros(embeddings.shape[1])

        # Retrieve the transformation matrix
        transformation_matrix = spherize.W

        # Normalize embeddings to unit norm if requested
        if norm_embeddings:
            tolerance = 1e-10
            norms = np.linalg.norm(transformed_embeddings, axis=1, keepdims=True)
            small_norms = norms < tolerance
            if np.any(small_norms):
                print(
                    f"Number of small norms before normalization: {np.sum(small_norms)}"
                )
                norms[small_norms] = 1.0  # Prevent division by zero
            transformed_embeddings /= norms

        return transformed_embeddings, center, transformation_matrix

    def apply_spherizing_transform(
        self,
        raw_embedding_col: str,
        save_embedding_col: str | None = None,
        method: str | None = "ZCA",
        norm_embeddings: bool | None = True,
        use_control: bool | None = True,
        n_jobs: int | None = 1,  # TODO understand why it works better this way...
    ) -> None:
        """
        Applies a sphering (whitening) transformation to embeddings in the DataFrame
        using the Spherize class. Processes each plate in parallel and saves the
        transformed embeddings.

        Parameters:
            raw_embedding_col (str): Column name for the raw embeddings in the df.
            save_embedding_col (str, optional): Column name to save the sphered
                embeddings. If None, appends '_spherize' to the raw column name.
                Defaults to None.
            method (str, optional): Sphering method to use. Must be one of 'PCA', 'ZCA',
                'PCA-cor', or 'ZCA-cor'. Defaults to 'ZCA'.
            norm_embeddings (bool, optional): Whether to normalize embeddings to unit
                norm after sphering. Defaults to True.
            use_control (bool, optional): Whether to use control samples for computing
                the sphering transformation. If True, only control samples are used to
                fit the sphering model. Defaults to True.
            n_jobs (int, optional): Number of parallel jobs to run. -1 uses all
                available cores. Defaults to -1.

        Returns:
            None: The DataFrame `self.df` is modified in place with the new sphered
            embeddings added in the specified `save_embedding_col`.

        Side Effects:
            - Adds a new column to `self.df` with the sphered embeddings.
            - Stores the centers and transformation matrices for each plate in
            `self.spherize_centers` and `self.spherize_transformation_matrices`,
            respectively.

        Raises:
            ValueError: If required columns are missing in `self.df`.
            ValueError: If no control samples are found for a plate when `use_control`
                is True.
            ValueError: If an invalid `method` is specified.

        Notes:
            - The method processes each unique plate in `self.df['Metadata_Plate']`
                in parallel.
            - Requires the following columns to be present in `self.df`:
                - `raw_embedding_col`
                - `'Metadata_Is_Control'` (boolean indicating control samples)
                - `'Metadata_Plate'` (identifier for plates)

            - The sphering transformation is fitted using control samples if
                `use_control` is True;
                otherwise, all samples in the plate are used.
            - Embeddings are stored as lists in the DataFrame cells for compatibility
                with downstream processes.

        Example:
            >>> self.apply_spherizing_transform(
                    raw_embedding_col='embeddings',
                    save_embedding_col='embeddings_sphered',
                    method='ZCA',
                    norm_embeddings=True,
                    use_control=True,
                    n_jobs=-1,
                )

        """
        if save_embedding_col is None:
            save_embedding_col = f"{raw_embedding_col}_spherize"

        # Ensure required columns exist in the DataFrame
        required_columns = [raw_embedding_col, "Metadata_Is_Control", "Metadata_Plate"]
        missing_columns = [
            col for col in required_columns if col not in self.df.columns
        ]
        if missing_columns:
            raise ValueError(
                f"The following required columns are missing in the DataFrame: {missing_columns}"  # NOQA
            )

        # Retrieve unique plates for parallel processing
        plates = self.df["Metadata_Plate"].unique()

        def apply_spherizing_for_plate(plate_df: pd.DataFrame) -> pd.DataFrame:
            """
            Applies sphering transformation to a single plate's embeddings.

            Parameters:
                plate_df (pd.DataFrame): DataFrame subset for a single plate.

            Returns:
                pd.DataFrame: DataFrame with the sphered embeddings added.
            """
            try:
                plate_df = plate_df.reset_index(drop=True)
                embeddings = np.stack(plate_df[raw_embedding_col].values)

                if use_control:
                    control_indices = plate_df.index[
                        plate_df["Metadata_Is_Control"]
                    ].tolist()
                    if not control_indices:
                        raise ValueError(
                            f"No control samples found for plate {plate_df['Metadata_Plate'].iloc[0]}"  # NOQA
                        )
                else:
                    control_indices = []

                transformed_embeddings, center, transformation_matrix = (
                    self._apply_spherizing_subset(
                        embeddings=embeddings,
                        control_indices=control_indices,
                        norm_embeddings=norm_embeddings,
                        method=method,
                        use_control=use_control,
                    )
                )

                plate_df[save_embedding_col] = [
                    embedding for embedding in transformed_embeddings
                ]

                return plate_df
            except Exception as e:
                print(
                    f"Error spherizing plate {plate_df['Metadata_Plate'].iloc[0]}: {e}"
                )
                return plate_df

        # Apply sphering in parallel across all plates

        with tqdm_joblib(
            tqdm(desc="Normalizing plates with Spherize", total=len(plates))
        ):
            results = Parallel(n_jobs=n_jobs)(
                delayed(apply_spherizing_for_plate)(
                    self.df[self.df["Metadata_Plate"] == plate].copy()
                )
                for plate in plates
            )

        # Concatenate all processed subsets back into the main DataFrame
        self.df = pd.concat(results, ignore_index=True)
