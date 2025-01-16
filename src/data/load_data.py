# Importar datos de kaggle
import kagglehub
import os
import pandas as pd

# Descargar el conjunto de datos desde KaggleHub
path = kagglehub.dataset_download("apoorvaappz/global-super-store-dataset")

# Buscar el archivo CSV dentro del directorio descargado
for filename in os.listdir(path):
    if filename.endswith(".csv"):
        file_path = os.path.join(path, filename)
        break

# Verificar si se encontro el archivo csv
if file_path is None:
    raise FileNotFoundError("No se encontró un archivo CSV en el dataset descargado.")
    
# Cargar el archivo CSV en un DataFrame
data_raw = pd.read_csv(file_path, sep=',', encoding='latin1')

# Definir la ruta de guardado
ruta = "data/raw/data_raw.csv"

# Crear la carpeta si no existe
os.makedirs(os.path.dirname(ruta), exist_ok=True)

# Guardar el archivo en la carpeta raw
data_raw.to_csv(ruta, index=False, encoding='latin1')

print(F"Archivo guardado en: {ruta}")