from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np

from .embedding_manager import EmbeddingManager

# TODO : Add the possibility to use the tanimo distance matrix in embedding_manager.py

# Tanimo matrice of distance -> EM
# from rdkit import Chem, DataStructs
# from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator


class BioproxyEvaluator:
    def __init__(
        self,
        compounds_metadata: pd.DataFrame | Path,
        embeddings_path: Path,
        screens_folders: dict[str, Path],
        embeddings_name: str = "Embeddings",
        embeddings_entity: str = "compound",
    ) -> None:
        self.global_embedding_manager = EmbeddingManager(
            compounds_metadata,
            embeddings_entity,
        )
        self.global_embedding_manager.load(embeddings_name, embeddings_path)
        self.screens_data_folders = {}
        self.screens_data_folders.update(screens_folders)
        self.screen_embedding_managers = {}
        self._load_screens_data()

    def __repr__(self) -> str:
        screens_info = ", ".join(
            f"{source}: {len(screens)} screens"
            for source, screens in self.screen_embedding_managers.items()
        )
        return f"BioproxyEvaluator with {len(self.screen_embedding_managers)} sources ({screens_info})"  # NOQA

    def _load_screens_data(self) -> None:
        """Load screen data from specified folders and preprocess it."""
        for assays_source, folder in self.screens_data_folders.items():
            self.screen_embedding_managers[assays_source] = {}
            for path in folder.glob("*.csv"):
                self._process_screen_file(assays_source, path)

    def _process_screen_file(self, assays_source: str, path: Path) -> None:
        """Process individual screen file."""
        try:
            df = pd.read_csv(path)
        except Exception as e:
            print(f"Error reading {path}: {e}")
            return

        filename = path.stem
        if "role_val" in df.columns:
            df.rename(columns={"role_val": "Metadata_Bioactivity"}, inplace=True)
        if "Unnamed: 0" in df.columns:
            df.drop(columns="Unnamed: 0", inplace=True)

        screen_embedding_manager = self.global_embedding_manager.filter_and_instantiate(
            Metadata_JCP2022=df["Metadata_JCP2022"].tolist()
        )
        df = df.drop_duplicates(subset=["Metadata_JCP2022"], keep="first")

        screen_embedding_manager.df = screen_embedding_manager.df.merge(
            df, on="Metadata_JCP2022", how="inner"
        )
        self.screen_embedding_managers[assays_source][
            filename
        ] = screen_embedding_manager

    def compute_ranking(
        self,
        source: str,
        screen: str,
        embeddings_name: str,
        JCP2022_id: str,
        distance: str = "cosine",
        plot: bool | None = False,  # TODO : add plot
    ) -> dict[str, list]:
        """
        Compute the ranking of distances for a given screen and embedding.
        """
        screen_embedding_manager = self.screen_embedding_managers[source][screen]
        screen_df = screen_embedding_manager.df

        # Vérification des distances
        matrix_key = f"{distance}_distance_matrix_{embeddings_name}"
        if matrix_key not in screen_embedding_manager.distance_matrices:
            screen_embedding_manager.compute_distance_matrix(embeddings_name, distance)

        distances = screen_embedding_manager.distance_matrices[matrix_key]

        # Vérifier la présence du JCP2022_id
        if JCP2022_id not in screen_df["Metadata_JCP2022"].values:
            raise ValueError(
                f"Metadata_JCP2022 '{JCP2022_id}' not found in screen dataset."
            )

        target_index = screen_df.loc[screen_df["Metadata_JCP2022"] == JCP2022_id].index[
            0
        ]

        results_df = screen_df.copy()
        results_df["distance_to_target"] = distances[target_index]

        # Trier en excluant la première ligne (référence)
        results_df = results_df.sort_values(by="distance_to_target").iloc[1:]

        return {
            "Metadata_JCP2022": results_df["Metadata_JCP2022"].tolist(),
            "Distance": results_df["distance_to_target"].tolist(),
            "Bioactivity": results_df["Metadata_Bioactivity"].tolist(),
        }

    def compute_enrichment_factor_for_screen(
        self,
        source: str,
        screen: str,
        embeddings_name: str,
        thresholds: list[int | float],
        mode: str = "percentage",
    ) -> pd.DataFrame:
        """Compute enrichment factors (EF) for a single screen."""
        screen_embedding_manager = self.screen_embedding_managers[source][screen]
        screen_df = screen_embedding_manager.df
        results = []
        for JCP2022_id in screen_df[screen_df["Metadata_Bioactivity"] == "hit"][
            "Metadata_JCP2022"
        ].tolist():
            ranking_result = self.compute_ranking(
                source, screen, embeddings_name, JCP2022_id
            )
            ef_dict = self._calculate_ef_for_target(ranking_result, thresholds, mode)
            for threshold, ef_values in ef_dict.items():
                results.append(
                    {
                        "Source": source,
                        "Screen": screen,
                        "Metadata_JCP2022": JCP2022_id,
                        "Threshold": threshold,
                        **ef_values,
                    }
                )
        return pd.DataFrame(results)

    def compute_enrichment_factors(
        self,
        source: str,
        embeddings_name: str,
        thresholds: list[int | float],
        mode: str = "percentage",
    ) -> pd.DataFrame:
        """Compute aggregated enrichment factors (EF) for all screens of a source."""
        all_results = []
        for screen in tqdm(
            self.screen_embedding_managers.get(source, {}).keys(),
            desc=f"Processing {source} screens",
        ):
            screen_results = self.compute_enrichment_factor_for_screen(
                source, screen, embeddings_name, thresholds, mode
            )

            if not screen_results.empty:
                summary = (
                    screen_results.groupby("Threshold")[
                        [
                            "EF",
                            "Normalized_EF",
                            "Hit Rate Selected",
                            "Hit Rate Random",
                            "N Selected",
                            "N Hits",
                            "N Compounds",
                        ]
                    ]
                    .agg(
                        {
                            "EF": ["mean", "median", "max"],
                            "Normalized_EF": ["mean", "median", "max"],
                            "Hit Rate Selected": ["mean", "median", "max"],
                            "Hit Rate Random": "mean",
                            "N Selected": ["mean", "median", "max"],
                            "N Hits": "mean",
                            "N Compounds": "mean",
                        }
                    )
                    .reset_index()
                )
                summary.insert(0, "Source", source)
                summary.insert(1, "Screen", screen)
                all_results.append(summary)

        return (
            pd.concat(all_results, ignore_index=True) if all_results else pd.DataFrame()
        )

    def _calculate_ef_for_target(
        self,
        ranking_result: dict,
        thresholds: list[int | float],
        mode: str = "percentage",
    ) -> dict:
        """
        Calculate the enrichment factor for a given target.
        """
        n_total = len(ranking_result["Metadata_JCP2022"])
        n_hits = ranking_result["Bioactivity"].count("hit")
        hit_rate_random = n_hits / n_total if n_total > 0 else 0

        ef_dict = {}
        for threshold in thresholds:
            if mode == "percentage":
                n_selected = max(1, int(n_total * threshold / 100))
                selected_hits = ranking_result["Bioactivity"][:n_selected].count("hit")
            else:
                selected_hits = sum(
                    1
                    for dist, bio in zip(
                        ranking_result["Distance"], ranking_result["Bioactivity"]
                    )
                    if dist <= threshold and bio == "hit"
                )
                n_selected = sum(
                    1 for dist in ranking_result["Distance"] if dist <= threshold
                )

            hit_rate_selected = selected_hits / n_selected if n_selected > 0 else 0
            ef = hit_rate_selected / hit_rate_random if hit_rate_random > 0 else 0

            # Normalization of EF
            max_ef = (
                min(1, n_hits / n_selected) / hit_rate_random if n_selected != 0 else 0
            )
            norm_ef = (ef / max_ef) * 100 if max_ef != 0 else 0
            n_total = len(ranking_result["Metadata_JCP2022"])
            ef_dict[threshold] = {
                "EF": ef,
                "Normalized_EF": norm_ef,
                "Hit Rate Selected": hit_rate_selected,
                "Hit Rate Random": hit_rate_random,
                "N Selected": n_selected,
                "N Hits": n_hits,
                "N Compounds": n_total,
            }
        return ef_dict

    def plot_assays_distribution(self, assay_source: str) -> None:
        """Plot the distribution of hits and molecules count per screen."""
        data = {}
        for screen, screen_manager in self.screen_embedding_managers[
            assay_source
        ].items():
            hit_count = (
                screen_manager.df["Metadata_Bioactivity"].value_counts().get("hit", 0)
            )
            molecule_count = len(screen_manager.df)
            data[screen] = {"hits": hit_count, "molecules": molecule_count}

        df = pd.DataFrame(data).T
        df.plot(kind="bar", figsize=(10, 6))
        plt.ylabel("Count")
        plt.title(f"Hits and Molecules Count per Screen - {assay_source}")
        plt.xticks(rotation=45)
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def load(
        self,
        embedding_name: str,
        embeddings_file: str | Path,
        metadata_file: str | Path | None = None,
        dtype: type | None = np.float64,
    ) -> None:
        """
        Load new embeddings into the global manager and update screen
        managers.

        Args:
            embedding_name (str): Key for the embeddings.
            embeddings_file (str | Path): Path to the .npy embeddings file.
            metadata_file (str | Path | None): Optional metadata file.
            dtype (type | None): Data type for embeddings.
                Defaults to np.float64.
        """
        self.global_embedding_manager.load(
            embedding_name, embeddings_file, metadata_file, dtype
        )
        global_embeds = self.global_embedding_manager.embeddings[embedding_name]
        global_df = self.global_embedding_manager.df

        id_to_index = dict(zip(global_df["Metadata_JCP2022"], global_df.index))

        for _, screens in self.screen_embedding_managers.items():
            for _, manager in screens.items():
                screen_df = manager.df
                indices = screen_df["Metadata_JCP2022"].map(id_to_index)
                if indices.isnull().any():
                    raise ValueError(
                        "One or more 'Metadata_JCP2022' ids were not found "
                        "in the global embeddings."
                    )
                indices = indices.astype(int).values
                manager.embeddings[embedding_name] = global_embeds[indices].astype(
                    dtype
                )
