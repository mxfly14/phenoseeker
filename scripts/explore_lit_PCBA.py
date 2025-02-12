import pandas as pd
from tqdm import tqdm
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
import os
import numpy as np

# df_phenom = pd.read_parquet('/home/maxime/data/jump_embeddings/metadata_dinov2_g.parquet') # noqa
df_phenom = pd.read_parquet(
    "/projects/synsight/data/jump_embeddings/wells_embeddings/openphenom/metadata_openphenom.parquet"  # noqa
)

df_jump = (
    df_phenom[["Metadata_JCP2022", "Metadata_InChI"]].drop_duplicates().reset_index()
)
mg = AllChem.GetMorganGenerator(radius=2, fpSize=2048, includeChirality=False)


def inchi_to_fp(inchi):
    """Convert InChI string to RDKit Morgan fingerprint."""
    mol = Chem.MolFromInchi(inchi)
    if mol:
        return mg.GetFingerprint(mol)
    return None


def smiles_to_fp(smiles):
    """Convert SMILES to RDKit fingerprint."""
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        return mg.GetFingerprint(mol)
    return None


def compute_similarity(query_smi, list_of_fps_jump):
    """Compute Tanimoto similarity between a query InChI and a list of InChIs."""

    if query_smi is None:
        raise ValueError("Invalid query")
    query_fp = smiles_to_fp(query_smi)
    return DataStructs.BulkTanimotoSimilarity(query_fp, list(list_of_fps_jump))


def load_smi_files(base_path):
    """
    Charge les fichiers actives.smi et inactives.smi d'un dossier et retourne un
    dictionnaire de DataFrames.

    Args:
        base_path (str): Le chemin vers le dossier contenant les sous-dossiers avec les
        fichiers .smi.

    Returns:
        dict: Un dictionnaire où chaque clé est le nom du sous-dossier et la valeur est
        un DataFrame.
    """
    data_dict = {}

    # Liste tous les sous-dossiers
    for folder in os.listdir(base_path):
        folder_path = os.path.join(base_path, folder)

        # Vérifie si c'est bien un dossier
        if os.path.isdir(folder_path):
            actives_file = os.path.join(folder_path, "actives.smi")
            inactives_file = os.path.join(folder_path, "inactives.smi")

            all_data = []

            # Lire actives.smi
            if os.path.exists(actives_file):
                df_actives = pd.read_csv(
                    actives_file, sep=" ", names=["smiles", "id_lit_pcba"]
                )
                df_actives["Active"] = True
                all_data.append(df_actives)

            # Lire inactives.smi
            if os.path.exists(inactives_file):
                df_inactives = pd.read_csv(
                    inactives_file, sep=" ", names=["smiles", "id_lit_pcba"]
                )
                df_inactives["Active"] = False
                all_data.append(df_inactives)

            # Si on a des données, on les stocke
            if all_data:
                data_dict[folder] = pd.concat(all_data, ignore_index=True)

    return data_dict


# Chemin vers ton dossier "data"
base_path = "../data"

# Charger les données
data_dict = load_smi_files(base_path)

df_jump["Fps"] = [
    inchi_to_fp(inchi) for inchi in tqdm(df_jump["Metadata_InChI"].to_list())
]
df_jump.dropna(subset="Fps", inplace=True)

# On récupère la liste des empreintes (fingerprints) du dataset jump.
list_of_fps_jump = df_jump["Fps"].tolist()

unique_smiles = set()
for key, df in tqdm(data_dict.items()):
    # Ajoute toutes les valeurs uniques de la colonne 'smiles'
    unique_smiles.update(df["smiles"].unique())

print(f"Nombre de molécules uniques trouvées : {len(unique_smiles)}")

# --- 2. Calcul du meilleur voisin pour chaque molécule unique ---

# Récupérer la liste des empreintes (fingerprints) du dataset jump.
list_of_fps_jump = df_jump["Fps"].tolist()

# Dictionnaire pour stocker le mapping : smiles -> (meilleur id, similarité)
smiles_to_best_match = {}

for query_smiles in tqdm(unique_smiles, desc="Processing unique smiles"):
    # Calculer les similarités entre la molécule de requête et toutes celles de df_jump.
    # La fonction compute_similarity doit retourner une liste ou un array de similarités
    similarities = compute_similarity(query_smiles, list_of_fps_jump)
    similarities = np.array(similarities)

    # Identifier l'indice de la similarité maximale
    best_index = np.argmax(similarities)
    best_similarity = similarities[best_index]

    # Récupérer l'identifiant correspondant dans df_jump (colonne "Metadata_JCP2022")
    best_jcp_id = df_jump.iloc[best_index]["Metadata_JCP2022"]

    # Stocker le résultat dans le dictionnaire
    smiles_to_best_match[query_smiles] = (best_jcp_id, best_similarity)

# --- 3. Réaffecter les informations dans chaque DataFrame de data_dict ---

for key, df in tqdm(data_dict.items()):
    print(f"\n📂 {key} - Ajout des colonnes 'closest_jcp' et 'tanimoto_similarity'")

    # Pour chaque ligne, récupérer depuis le mapping les valeurs calculées
    df["closest_jcp"] = df["smiles"].apply(lambda s: smiles_to_best_match[s][0])
    df["tanimoto_similarity"] = df["smiles"].apply(lambda s: smiles_to_best_match[s][1])

    # Sauvegarder le DataFrame enrichi au format Parquet
    output_filename = f"jump_{key}.parquet"
    df.to_parquet(output_filename, index=False)
    print(f"✅ {output_filename} sauvegardé.")
