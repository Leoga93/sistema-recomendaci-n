from sklearn.decomposition import TruncatedSVD
import numpy as np
import pandas as pd
import os
import joblib

class trainModel():
    def __init__(
        self, data, n_components=443, n_iter=7, random_state=42, tol=0.0, 
        n_oversamples=13, power_iteration_normalizer="LU"
    ):
        self.data = data
        self.n_components = n_components
        self.n_iter = n_iter
        self.random_state = random_state
        self.tol = tol
        self.n_oversamples = n_oversamples
        self.power_iteration_normalizer = power_iteration_normalizer

        # Inicializar modelo TruncatedSVD
        self.svd = TruncatedSVD(
        n_components=self.n_components, 
        n_iter=self.n_iter, 
        random_state=self.random_state, 
        tol=self.tol, 
        n_oversamples=self.n_oversamples, 
        power_iteration_normalizer=self.power_iteration_normalizer
        )

        # Variables para almacenar los resultados del modelo
        self.U = None
        self.s = None
        self.Vt = None
        self.matriz_prediction = None
    
    def ajustar(self):
        """Ajusta el modelo SVD a los datos de entrada."""
        print("Ajustando modelo SVD...")
        try:
            self.svd.fit(self.data)
            print("Modelo ajustado.")
        except Exception as e:
            print("Error al ajustar el modelo:", e)
    
    def save_model(self, path="models/svd_model.pkl"):
        # Guarda el modelo entrenado
        if self.svd is None:
            raise ValueError("Primero debe ajustar el modelo.")
        
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        try:
            print("Guardando modelo...")
            joblib.dump(self.svd, path)
            print("El modelo se guardo en el archivo:", path)
        except Exception as e:
            print("Error al guardar el modelo:", e)
        
def main():
    data = pd.read_csv(
        "C:/Users/USUARIO/Desktop/Proyectos/Sistema_recomendacion/data/processed/users_matriz_items.csv", 
        encoding="utf-8", sep=',', index_col=0)
    model = trainModel(data)
    model.ajustar()
    model.save_model()

main()