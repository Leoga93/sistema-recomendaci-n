# Importar datos de kaggle
import kagglehub

# Download latest version
path = kagglehub.dataset_download("apoorvaappz/global-super-store-dataset")

print("Path to dataset files:", path)