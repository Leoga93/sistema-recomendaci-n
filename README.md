# Sistema de recomendación de filtro colaborativo 
En este proyecto se realiza un sistema de recomendación de filtro colaborativo aplicando Truncatedsvd de sklearn en una matriz de interacción de usuarios y productos. Se hace uso de procesamiento de datos, DVC y Machine Learning.

## Descripción del proyecto
El objetivo de este sistema es ofrecer recomendaciones personalizadas de productos a los usuarios, basado en su comportamiento y preferencias previas. Utiliza métodos de filtrado colaborativo y procesamiento avanzado de datos. Además, se proporcionan los notebooks donde se realiza una comprensión del negocio y un análisis exploratorio de datos. Además, este proyecto se enfocó en utilizar el manejo de errores, excepciones, logs, POO. 

## Características

- Filtrado colaborativo basado en similitud de usuarios.
- Procesamiento y limpieza de datos.
- Generación de recomendaciones personalizadas.
- Integración modular con otros sistemas.

## Comprensión del negocio
Se realiza un análisis para comprender el funcionamiento y rentabilidad de la tienda global.

### Análisis de clientes
* Frecuencia de compra de los clientes: Se evalúa cuantas veces compran los clientes
* Ingresos y rentabilidad: Se analiza si los clientes frecuentes contribuyen con más ingresos y beneficios
* Segmentación de rentabilidad: Se identifican los segmenos de clientes más rentables
* Distribución geográfica: Se examina como se distribuyen los clientes en los países

### Análisis de producto
* Ventas por País: Se identifican los países con mayores ventas
* Top 5 Productos más rentables: Se identifican los cinco productos con mayores ventas
* Relación entre precio y ventas
* Tiempo de entrega: Se cálcula el tiempo de entrega promedio por estado y se representa gráficamente


## Estructura del proyecto
data/ - Conjunto de datos (Gestionado con DVC)
notebooks/ - (Comprensión del negocio, Análisis exploratorio de datos)
src/ - Código fuente
main.py - Script principal
utils/ - Funciones de utilidad
main.py - Script principalp para ejecutar el sistema
congig.yaml - Configuración general del proyecto
requirements.txt - Dependencias del proyecto
README.md documentación de proyecto

## Instalación 
1. Clona el repositorio:
'''bash
git clone https://github.com/Leoga93/sistema-recomendaci-n.git
cd sistema-recomendacion

2. Crea y activa un entorno virtual:
python -m venv venv
source venv/bin/activate  # En Linux/Mac
venv\Scripts\activate     # En Windows

2. Instala las Dependencias:
pip install -r requirements.txt

3. Recuperar los datos con DVC
dvc pull

## USO
python src/main.py

Hacer predicciones 
python src/predict.py

## Tecnologías usadas
* Scikit-learn
* Python
* Pandas
* Numpy
* Matplotlib
* Seaborn
* Plotly
* DVC
* GIT
