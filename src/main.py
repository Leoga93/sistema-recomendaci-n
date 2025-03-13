from recomendador.predict import SVDRecommenderPredict
from utils.config_loader import load_config
import datetime
import logging
import pandas as pd
import json
import os

config = load_config()
logging.basicConfig(level=config["logging"]["level"],
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt= "%Y-%m-%d %H:%M:%S",
                    handlers=[logging.FileHandler(config["logging"]["file_main"]),
                              logging.StreamHandler()])

logger = logging.getLogger(__name__)

class recomendacionPipeline:
    
    def __init__(self, predictor):
        self.predictor = predictor
    
    def pipeline_recomendacion(self, user, productos, top=config["recommender"]["top_n"]):
        """
        Pipeline completo para realizar una recomendación.
        
        Args:
            user (pd.Series): Datos del usuario.
            productos (dict): Diccionario de productos con nombre e ID.
            top (int): Número de productos a recomendar.
        
        Returns:
            list: Lista de productos recomendados.
        """
        try:
            self.predictor.cargar_modelo()
            self.predictor.transformar(user)
            self.predictor.reconstruir_matrix()
            self.predictor.convertir_dataframe()
            productos_recomendados = self.predictor.recomendar_productos(user, self.predictor.df_prediction, productos, top=config["recommender"]["top_n"])
            recomendacion = {
                "user": user.name,
                "recomendaciones": productos_recomendados.to_dict()
            }
            objeto_json = json.dumps(recomendacion, indent=2)
            
            output_dir = "recomendaciones"
            os.makedirs(output_dir, exist_ok=True)
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            file_name = os.path.join(output_dir, f"recomendacion_user_{user.name}_{timestamp}.json")
            
            with open(file_name, "w") as file:
                file.write(objeto_json)
            
            logger.info(f"Recomendación guardada en {file_name}")
            
            return productos_recomendados
        
        except Exception as e:
            logger.exception(f"Error en el pipeline de recomendación: {e}")

base_dir = os.getcwd()

url = os.path.join(base_dir, config["data"]["interactions_path"])
data = pd.read_csv(url, sep=',', encoding='utf-8', index_col=0)
user = data.iloc[1]

path = os.path.join(base_dir, config["data"]["products"])
with open(path, "r") as file:
    productos = json.load(file)

predictor = SVDRecommenderPredict()

pipeline = recomendacionPipeline(predictor)
recomendaciones = pipeline.pipeline_recomendacion(user, productos, top=6)
print(recomendaciones)