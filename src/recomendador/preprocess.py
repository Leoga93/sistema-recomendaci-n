import pandas as pd
import numpy as np


class preprocessor:
    def __ini__(self, expected_length):
        self.expected_length = expected_length
    
    def validate_input(self, ratings_array):
        if len(ratings_array) != self.expected_length:
            raise ValueError(
                f"La longitud del array debe ser {self.expected_length}, pero se recibió {len(ratings_array)}"
            )
        if not set(np.unique(ratings_array)).issubset({0, 1}):
            raise ValueError("El array solo puede contener valores binarios (0 y 1).")
        return ratings_array
    
    def to_dataframe(self, ratings_array):
        return pd.DataFrame([ratings_array])
    
    def remove_user_without_interactions(self, matrix):
        return matrix.loc[(matrix !=0).any(axis=1)]