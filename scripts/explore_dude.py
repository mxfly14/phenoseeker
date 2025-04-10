import pandas as pd
from tqdm import tqdm
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
import os
import numpy as np

df_jump = pd.read_parquet(
    "/home/maxime/data/jump_embeddings/dinov2_g/compounds/metadata.parquet"
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
            actives_file = os.path.join(folder_path, "actives_final.ism")
            inactives_file = os.path.join(folder_path, "decoys_final.ism")

            all_data = []

            # Lire actives.smi
            if os.path.exists(actives_file):
                df_actives = pd.read_csv(
                    actives_file, sep=" ", names=["smiles", "id_dude", "ChEMBL_id"]
                )
                df_actives["Active"] = True
                all_data.append(df_actives[["smiles", "id_dude", "Active"]])

            # Lire inactives.smi
            if os.path.exists(inactives_file):
                df_inactives = pd.read_csv(
                    inactives_file, sep=" ", names=["smiles", "id_dude"]
                )
                df_inactives["Active"] = False
                all_data.append(df_inactives)

            # Si on a des données, on les stocke
            if all_data:
                data_dict[folder] = pd.concat(all_data, ignore_index=True)

    return data_dict


# Chemin vers ton dossier "data"
base_path = "../data/DUDE"

# Charger les données
data_dict = load_smi_files(base_path)

df_jump["Fps"] = [
    inchi_to_fp(inchi) for inchi in tqdm(df_jump["Metadata_InChI"].to_list())
]
df_jump.dropna(subset="Fps", inplace=True)

list_of_fps_jump = df_jump["Fps"].tolist()

# --- 1. Collecter les molécules uniques ---
unique_smiles = set()
for key, df in tqdm(data_dict.items(), desc="Collecting unique smiles"):
    unique_smiles.update(df["smiles"].unique())

print(f"Nombre de molécules uniques trouvées : {len(unique_smiles)}")

# --- 2. Calcul du meilleur voisin pour chaque molécule unique ---
list_of_fps_jump = df_jump["Fps"].tolist()

# --- 3. Réaffecter les informations dans chaque DataFrame de data_dict ---

# for key in tqdm(reversed(list(data_dict.keys())), desc="Processing each key"):
for key in tqdm(data_dict.keys(), desc="Processing each key"):
    df = data_dict[key]
    output_filename = f"{base_path}/parquets_files/jump_{key}.parquet"

    # Vérifier si le fichier existe déjà
    if os.path.exists(output_filename):
        print(f"📂 {key} - Fichier {output_filename} existe déjà. Sauter cette clé.")
        continue
    # Créer un fichier Parquet vide ou avec des données minimales
    try:
        empty_df = pd.DataFrame(columns=df.columns)
        empty_df.to_parquet(output_filename, index=False)
        print(f"📂 {key} - Fichier Parquet initialisé : {output_filename}")
    except Exception as e:
        print(
            f"Erreur lors de l'initialisation du fichier Parquet pour la clé {key}: {e}"
        )
        continue

    print(f"\n📂 {key} - Ajout des colonnes 'closest_jcp' et 'tanimoto_similarity'")

    smiles_to_best_match = {}
    try:
        for query_smiles in tqdm(
            df["smiles"].unique(), desc=f"Processing smiles for {key}"
        ):
            similarities = compute_similarity(query_smiles, list_of_fps_jump)
            similarities = np.array(similarities)

            best_index = np.argmax(similarities)
            best_similarity = similarities[best_index]
            best_jcp_id = df_jump.iloc[best_index]["Metadata_JCP2022"]

            smiles_to_best_match[query_smiles] = (best_jcp_id, best_similarity)

        df["closest_jcp"] = df["smiles"].apply(lambda s: smiles_to_best_match[s][0])
        df["tanimoto_similarity"] = df["smiles"].apply(
            lambda s: smiles_to_best_match[s][1]
        )

        # Vérifier et convertir les types de données si nécessaire
        for col in df.columns:
            if df[col].dtype == "object":
                df[col] = df[col].astype(str)

        df.to_parquet(output_filename, index=False)
        print(f"✅ {output_filename} sauvegardé.")
    except Exception as e:
        print(f"Erreur lors du traitement de la clé {key}: {e}")
