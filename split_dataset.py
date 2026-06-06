"""
split_dataset.py — Genera el split desarrollo/holdout del dataset CUAD completo.

Ejecutar UNA SOLA VEZ después de filtro_contratos.py y filtro_labels.py:
    python split_dataset.py

Salida: split.csv  (columnas: filename, contract_type, split)
  - split == "dev"     → 70% — usado para experimentos e hipótesis
  - split == "holdout" → 30% — reservado para validación final

NO volver a ejecutar este script una vez generado split.csv,
ya que cambiaría la asignación dev/holdout e invalidaría el experimento.
"""

import os
import re
import pandas as pd
from sklearn.model_selection import train_test_split

RANDOM_SEED = 42
DEV_RATIO = 0.70
DATASET_DIR = "./Dataset_CUAD_Completo"
OUTPUT_CSV = "./split.csv"
MIN_GROUP_SIZE = 2  # grupos con menos contratos se agrupan en "Other"


def extract_contract_type(filename: str) -> str:
    """
    Extrae el tipo de contrato del nombre de archivo CUAD.

    Los nombres siguen patrones como:
      COMPANYNAME_DATE-EX-X.X-CONTRACT TYPE.txt
      CompanyName_DATE_10-K_EX-10.1_ID_EX-10.1_Contract Type Agreement.txt

    Se toma el último segmento significativo (después del último guion o
    underscore antes de .txt) y se normaliza a Title Case.
    """
    name = os.path.splitext(filename)[0]

    # Intenta extraer la parte de tipo de contrato:
    # Busca la última ocurrencia de un término con "Agreement", "Contract",
    # "Amendment", "Lease", "License", "Non-Compete", etc.
    match = re.search(
        r'[\-_]([A-Za-z][A-Za-z ,\-&]+(?:Agreement|Contract|Amendment|'
        r'Lease|License|Non-Compete|Covenant|Arrangement|Guarantee|'
        r'Indenture|Note|Plan|Policy|Protocol|Schedule|Settlement|'
        r'Understanding|Warrant)[A-Za-z ,\-&0-9]*)\s*$',
        name,
        re.IGNORECASE
    )
    if match:
        contract_type = match.group(1).strip()
        # Normaliza múltiples tipos en el nombre (e.g. "Co-Branding Agreement_ Agency Agreement")
        # toma solo el primer tipo mencionado
        contract_type = re.split(r'[_|]', contract_type)[0].strip()
        return contract_type.title()

    # Fallback: toma el último token separado por guion o underscore
    parts = re.split(r'[-_]', name)
    for part in reversed(parts):
        part = part.strip()
        if len(part) > 4 and not re.match(r'^(EX|10|8K|S1|F1|N2|DEF|SC|ex)\b', part, re.I):
            return part.title()

    return "Other"


def main():
    if os.path.exists(OUTPUT_CSV):
        print(f"⚠️  {OUTPUT_CSV} ya existe. Este script no debe ejecutarse dos veces.")
        print("   Si realmente querés regenerar el split, borrá split.csv manualmente primero.")
        return

    archivos = [f for f in os.listdir(DATASET_DIR) if f.endswith(".txt")]
    if not archivos:
        print(f"❌ No se encontraron archivos .txt en '{DATASET_DIR}'.")
        print("   Ejecutá primero: python filtro_contratos.py")
        return

    print(f"Contratos encontrados: {len(archivos)}")

    df = pd.DataFrame({"filename": archivos})
    df["contract_type"] = df["filename"].apply(extract_contract_type)

    # Agrupa tipos con muy pocos contratos en "Other" para que la estratificación funcione
    type_counts = df["contract_type"].value_counts()
    rare_types = type_counts[type_counts < MIN_GROUP_SIZE].index
    df.loc[df["contract_type"].isin(rare_types), "contract_type"] = "Other"

    print(f"\nDistribución de tipos de contrato (top 15):")
    print(df["contract_type"].value_counts().head(15).to_string())
    print(f"\nTotal de tipos distintos: {df['contract_type'].nunique()}")

    # Split estratificado por tipo de contrato
    df_dev, df_holdout = train_test_split(
        df,
        test_size=1 - DEV_RATIO,
        random_state=RANDOM_SEED,
        stratify=df["contract_type"],
    )

    df_dev = df_dev.copy()
    df_holdout = df_holdout.copy()
    df_dev["split"] = "dev"
    df_holdout["split"] = "holdout"

    df_final = pd.concat([df_dev, df_holdout]).sort_values("filename").reset_index(drop=True)
    df_final.to_csv(OUTPUT_CSV, index=False)

    print(f"\n✅ Split generado y guardado en '{OUTPUT_CSV}'")
    print(f"   dev     : {len(df_dev)} contratos ({len(df_dev)/len(df)*100:.1f}%)")
    print(f"   holdout : {len(df_holdout)} contratos ({len(df_holdout)/len(df)*100:.1f}%)")
    print(f"   Total   : {len(df_final)} contratos")
    print(f"   Seed    : {RANDOM_SEED}")
    print(f"\n⚠️  No vuelvas a ejecutar este script. split.csv es el registro definitivo del split.")


if __name__ == "__main__":
    main()
