from pathlib import Path
from phenoseeker import EmbeddingManager

# base_path = Path("/projects/synsight/data/") # ibens
base_path = Path("/projects/synsight/data/")  # iktos

model = "openphenom"

metadata_path = base_path / Path(
    f"jump_embeddings/wells_embeddings/{model}/metadata_{model}.parquet"
)
embeddings_path = base_path / Path(
    f"jump_embeddings/wells_embeddings/{model}/embeddings_{model}.npy"
)

well_em = EmbeddingManager(metadata_path, entity="well")
well_em.load("Embeddings_Raw", embeddings_path)

well_em.apply_spherizing_transform(
    "Embeddings_Raw",
    "Embeddings__ZCA_C",
    method="ZCA",
    norm_embeddings=False,
    use_control=True,
    n_jobs=1,
)

well_em.apply_inverse_normal_transform("Embeddings__ZCA_C", "Embeddings__ZCA_C__Int")

well_em.save_to_folder(
    base_path / Path(f"jump_embeddings/wells_embeddings/{model}/{model}"),
    "Embeddings__ZCA_C__Int",
)


compounds_em = well_em.grouped_embeddings(
    group_by="compound",
    embeddings_name="Embeddings__ZCA_C__Int",
    new_embeddings_name="Embeddings_norm",
    cols_to_keep=["Metadata_JCP2022", "Metadata_InChI"],
)
compounds_em.save_to_folder(
    base_path / Path(f"jump_embeddings/compounds_embeddings/{model}/")
)
