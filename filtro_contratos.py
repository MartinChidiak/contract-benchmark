import os
import shutil

# 1. Configuración de rutas
# Ejecutar desde el root del proyecto: python filtro_contratos.py
ruta_origen = "./Contratos_txt"
ruta_destino = "./Dataset_CUAD_Completo"

# Crear la carpeta de destino si no existe
if not os.path.exists(ruta_destino):
    os.makedirs(ruta_destino)

print("Copiando todos los contratos CUAD...")

contador = 0
for archivo in os.listdir(ruta_origen):
    if archivo.endswith(".txt"):
        shutil.copy(os.path.join(ruta_origen, archivo), ruta_destino)
        contador += 1

print(f"¡Listo! Se copiaron {contador} contratos a '{ruta_destino}'.")