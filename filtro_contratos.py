import os
import shutil

# 1. Configuración de rutas
# Ejecutar desde el root del proyecto: python filtro_contratos.py
ruta_origen = "./Contratos_txt"
ruta_destino = "./Dataset_Filtrado_Tesis"

# Crear la carpeta de destino si no existe
if not os.path.exists(ruta_destino):
    os.makedirs(ruta_destino)

# 2. Palabras clave basadas en nuestro Top 10 de relevancia
# Estos términos aparecen en los nombres de los archivos de CUAD
temas_relevantes = [
    "Service Agreement",
    "Affiliate Agreement",
    "Maintenance Agreement",
    "Non-Compete",
    "Management Agreement"
]

print("Iniciando filtrado de contratos...")

contador = 0
for archivo in os.listdir(ruta_origen):
    # Verificamos si el archivo es un .txt y si contiene alguna de las palabras clave
    if archivo.endswith(".txt"):
        if any(tema.lower() in archivo.lower() for tema in temas_relevantes):
            shutil.copy(os.path.join(ruta_origen, archivo), ruta_destino)
            contador += 1

print(f"¡Listo! Se copiaron {contador} contratos a la carpeta de destino.")