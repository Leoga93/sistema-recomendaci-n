from utils.config_loader import load_config
from sklearn.decomposition import TruncatedSVD
import joblib
import logging
import numpy as np
import pandas as pd
import os
import json

config = load_config()
model_path = config["model"]["path"]
logs_level = config["logging"]["level"]
log_file = config["logging"]["file_predictor"]

os.makedirs("logs", exist_ok=True)

logging.basicConfig(level=logs_level,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt= "%Y-%m-%d %H:%M:%S",
                    handlers=[logging.FileHandler(log_file),
                              logging.StreamHandler()])

logger = logging.getLogger(__name__)

class SVDRecommenderPredict:
    """
    Clase para realizar predicciones con un modelo SVD entrenado.
    """
        
    def __init__(self, model_path=model_path):
        """Inicializa la clase SVDRecommenderPredict.
        
        Args:
            model_path (str): Ruta al archivo del modelo entrenado.
        """
        self.model_path = model_path
        self.model = None
        self.U = None
        self.s = None
        self.Vt = None
        self.matriz_prediction = None
        self.df_prediction = None
        self.data = None
    
    def cargar_modelo(self):
        """
        Carga el modelo entrenado desde el archivo especificado.
        """
        logger.info("Cargando modelo SVD...")
        if not os.path.exists(self.model_path):
            logger.error(f"El archivo {self.model_path} no existe.")
            raise FileNotFoundError(f"El archivo {self.model_path} no existe.")
            
        try:
            self.model = joblib.load(self.model_path)
            logger.info("Modelo cargado con éxito.")
        except Exception as e:
            logger.exception(f"Error inesperado al cargar el modelo")
            raise 
    
    def transformar(self, data):
        """Transforma los datos de entrada usando el modelo SVD cargado.
        
        Args:
            data (array-like): Datos nuevos para transformar.
        
        Returns:
            tupla: Matrices U, S y Vt calculadas a partir de los datos transformados.
        """
        logger.info("Transformando datos con SVD...")
        
        if self.model is None:
            logger.error("Primero debe cargar el modelo.")
            raise FileNotFoundError("Primero debe cargar el modelo.")
            
        try:
            self.data = data
            data_items = np.reshape(data, (1, -1))
            self.U = self.model.transform(data_items)
            self.s = np.diag(self.model.singular_values_)
            self.Vt = self.model.components_
            logger.info("Transformación completada con éxito.")
            return self.U, self.s, self.Vt
        except Exception as e:
            logger.exception(f"Error al transformar los datos.")
            raise 

    def reconstruir_matrix(self):
        """
        Reconstruye la matriz original a partir de los datos transformados.
        """
        logger.info("Reconstruyendo matriz...")
        if self.U is None or self.Vt is None:
            logger.error("Primero debe ejecutar el método transformar.")
            raise ValueError("Primero debe ejecutar el método transformar.")
            
        try:
            self.matriz_prediction = np.dot(self.U, self.Vt)
            logger.info("Matriz reconstruida con éxito.")
        except Exception as e:
            logger.exception(f"Error al reconstruir la matriz.")
            raise
        
    def convertir_dataframe(self):
        """
        Convierte la matriz reconstruida a un DataFrame de pandas.
        """
        
        logger.info("Convirtiendo matriz reconstruida en DataFrame")
        
        if self.matriz_prediction is None or self.data is None:
            logger.error("Primero debe ejecutar el método reconstruir_matrix.")
            raise ValueError("Primero debe ejecutar el método reconstruir_matrix.")
        
        if not isinstance(self.data, pd.Series):
            logger.error("El objeto data debe ser de tipo pd.Series")
            raise TypeError("El objeto data debe ser de tipo pd.Series")
            
        try :         
            self.df_prediction = pd.DataFrame(
                self.matriz_prediction, 
                index = [self.data.name], 
                columns = self.data.index
                )
            print("Conversión completada.")
        except Exception as e:
            logger.exception(f"Error al convertir la matriz a DataFrame.")
            raise

    def recomendar_productos(self, items_user, df_prediction, productos, top=5):
        """Recomienda productos al usuario basado en las predicciones del modelo SVD.

        Args:
            items_user (pd.Series): Serie con las interacciones del usuario (longitud 2375).
            df_prediction (pd.DataFrame): DataFrame de predicciones.
            productos (dict): Diccionario de productos con nombre e ID.
            top (int, optional): Número de productos a recomendar. Por defecto 5.

        Returns:
            pd.Series: Productos recomendados con sus puntuaciones. 
        """
        if len(items_user) != 2375:
            logger.error("El tamaño de items_user debe ser 2375.")
            raise ValueError("El tamaño de items_user debe ser 2375.")
        
        if self.df_prediction is None:
            logger.error("Primero debe ejecutar el método convertir_dataframe.")
            raise ValueError("Primero debe ejecutar el método convertir_dataframe.")
        
        if len(productos) == 0:
            logger.error("La lista de productos no puede estar vacía.")
            raise ValueError("La lista de productos no puede estar vacía.")
        
        if top < 1:
            logger.error("El valor de top debe ser mayor o igual a 1.")
            raise ValueError("El valor de top debe ser mayor o igual a 1.")
        
        try:
            print("Recomendando productos...")
            items_user = pd.Series(items_user)
            
            # Se obtiene los productos comprados por el usuario para no incluirlos en las predicciones
            prod_comprados = list(items_user[items_user > 0].index)
            
            predicciones_user = df_prediction.loc[items_user.name]   
            
            # Obtener las puntuaciones predichar para el usuario
            scores = predicciones_user.drop(prod_comprados, errors='ignore')
            
            # Ordenar descendente las puntuacioens
            recommended_products = scores.sort_values(ascending=False).head(top)
            
            recommended_products.index = recommended_products.index.map(lambda x: productos.get(x, 'Desconocido'))
            print("Recomendaciones generadas exitosamente.")
            return recommended_products
        except Exception as e:
            logger.exception(f"Error al recomendar productos.")
            raise 
        