# Third-Party Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


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
        labels = [" " for _ in range(n)]
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
