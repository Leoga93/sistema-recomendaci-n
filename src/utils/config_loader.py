import yaml

def load_config(path="config.yaml"):
    """Carga el archivo de configuración YAML."""
    with open(path, "r") as file:
        return yaml.safe_load(file)
