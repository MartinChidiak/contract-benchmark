import pandas as pd
import os

# 1. Path configuration
# Ejecutar desde el root del proyecto: python filtro_labels.py
ruta_csv_maestro = "./master_clauses.csv"
ruta_carpeta_filtrada = "./Dataset_Filtrado_Tesis"
ruta_csv_salida = "./ground_truth.csv"

# 2. Columns to extract from master_clauses.csv
# These match exactly the measurable fields in schema.py
columnas_interes = [
    "Filename",
    "Parties",                              "Parties-Answer",
    "Agreement Date",                       "Agreement Date-Answer",
    "Effective Date",                       "Effective Date-Answer",
    "Expiration Date",                      "Expiration Date-Answer",
    "Renewal Term",                         "Renewal Term-Answer",
    "Notice Period To Terminate Renewal",   "Notice Period To Terminate Renewal- Answer",
    "Governing Law",                        "Governing Law-Answer",
    "Anti-Assignment",                      "Anti-Assignment-Answer",
    "Audit Rights",                         "Audit Rights-Answer",
    "Cap On Liability",                     "Cap On Liability-Answer",
    "Termination For Convenience",          "Termination For Convenience-Answer",
    "Liquidated Damages",                   "Liquidated Damages-Answer",
]

print("Loading master CSV...")
df_maestro = pd.read_csv(ruta_csv_maestro, low_memory=False)
print(f"Master CSV loaded: {len(df_maestro)} rows, {len(df_maestro.columns)} columns")

# 3. Get local filenames without .txt extension
nombres_locales_limpios = [
    os.path.splitext(f)[0]
    for f in os.listdir(ruta_carpeta_filtrada)
    if f.endswith('.txt')
]
print(f"Found {len(nombres_locales_limpios)} .txt files in local folder.")

# 4. Normalize the Filename column in the master CSV (remove any extension)
df_maestro['Filename_Sin_Extension'] = df_maestro['Filename'].str.replace(
    r'\.[a-zA-Z0-9]+$', '', regex=True
)

# 5. Filter rows that match local files
df_filtrado = df_maestro[df_maestro['Filename_Sin_Extension'].isin(nombres_locales_limpios)]
print(f"Matched {len(df_filtrado)} contracts from local folder in the master CSV.")

# 6. Verify all desired columns exist in the master CSV
columnas_faltantes = [col for col in columnas_interes if col not in df_filtrado.columns]
if columnas_faltantes:
    print(f"⚠️  WARNING: These columns were not found in master_clauses.csv and will be skipped:")
    for col in columnas_faltantes:
        print(f"   - {col}")

columnas_presentes = [col for col in columnas_interes if col in df_filtrado.columns]
df_final = df_filtrado[columnas_presentes].copy()

# 7. Save
df_final.to_csv(ruta_csv_salida, index=False)
print(f"\n✅ Done!")
print(f"   Contracts matched: {len(df_final)}")
print(f"   Columns included:  {len(df_final.columns)}")
print(f"   Saved to:          {ruta_csv_salida}")